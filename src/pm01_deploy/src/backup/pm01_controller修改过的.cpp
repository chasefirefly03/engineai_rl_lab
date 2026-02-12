#include <chrono>
#include <memory>
#include <rclcpp/logging.hpp>
#include <rclcpp/rclcpp.hpp>
#include <thread>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <Eigen/Core>


#include "components/message_handler.hpp"
#include "pm01_controller.hpp"


pm01_controller::pm01_controller():Node("pm01_controller"){
    this->declare_parameter<std::string>("config_file", "");
    this->get_parameter("config_file", config_file);

    if (config_file.empty()) {
        RCLCPP_ERROR(this->get_logger(), "Config file parameter 'config_file' is not set!");
        // Determine a default logic or just let it fail if critical. 
        // Given the code does LoadFile immediately, we should probably return or throw. 
        // But since constructor can't return, throwing is appropriate or error logging.
    } else {
        RCLCPP_INFO(this->get_logger(), "Loading config from: %s", config_file.c_str());
    }

    YAML::Node yaml_node = YAML::LoadFile(config_file);

	default_joint_pos = yaml_node["default_joint_pos"].as<std::vector<float>>();
    joint_kp = yaml_node["joint_kp"].as<std::vector<float>>();
    joint_kd = yaml_node["joint_kd"].as<std::vector<float>>();
    observation_scale_linear_vel = yaml_node["observation_scale_linear_vel"].as<float>();
    observation_scale_base_ang_vel = yaml_node["observation_scale_base_ang_vel"].as<float>();
    observation_scale_base_quat_w = yaml_node["observation_scale_base_quat_w"].as<float>();   
    observation_scale_joint_pos = yaml_node["observation_scale_joint_pos"].as<float>();
    observation_scale_joint_vel = yaml_node["observation_scale_joint_vel"].as<float>();
    num_observations = yaml_node["num_observations"].as<float>();
    num_actions = yaml_node["num_actions"].as<float>();
    action_scale = yaml_node["action_scale"].as<float>();
    action_clip_limit = yaml_node["action_clip_limit"].as<float>();

    control_frequency = yaml_node["control_frequency"].as<float>();
    cycle_time = yaml_node["cycle_time"].as<float>();

    policy_file = yaml_node["policy_file"].as<std::string>();

    info_get_action_output = yaml_node["info_get_action_output"].as<bool>();
    info_get_joint_command_output = yaml_node["info_get_joint_command_output"].as<bool>();
    info_get_obs = yaml_node["info_get_obs"].as<bool>();

    obs.setZero(num_observations);
	act.setZero(num_actions);

    module = torch::jit::load(policy_file);
}

bool pm01_controller::Initialize() {
    try {
        // Initialize message handler
        message_handler_ = std::make_shared<MessageHandler>(shared_from_this());
        joint_command_ = std::make_shared<interface_protocol::msg::JointCommand>();
        message_handler_->Initialize();
        // Wait for first motion state
        while (!message_handler_->GetLatestMotionState() ||
               !message_handler_->GetLatestImu() ||
               !message_handler_->GetLatestJointState() ||
                message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") {
        rclcpp::spin_some(shared_from_this());
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, "Waiting for joint bridge state and sensor data...");
        }
        RCLCPP_INFO(get_logger(), "Already in joint bridge state");
        // Get initial joint positions
        auto initial_state = message_handler_->GetLatestJointState();
        if (!initial_state) {
        RCLCPP_ERROR(get_logger(), "Failed to get initial joint state");
        return false;
        }
        RCLCPP_INFO(get_logger(), "Starting control loop");
        control_timer_ = create_timer(std::chrono::duration<float>(1.0/control_frequency),
                                         std::bind(&pm01_controller::ControlCallback, this));        
        return true;
    } catch (const std::exception& e) {
        RCLCPP_ERROR(get_logger(), "Failed to initialize: %s", e.what());
        return false;
    }
}

void pm01_controller::ControlCallback() {   
    if (message_handler_->GetLatestMotionState()->current_motion_task != "joint_bridge") return;
    
    auto joint_state = message_handler_->GetLatestJointState();
    
    // Calculate dt based on joint state header stamp for sim time sync
    rclcpp::Time current_sim_time(joint_state->header.stamp);
    static rclcpp::Time last_sim_time(0, 0, RCL_ROS_TIME); 

    float control_dt_ = 0.0f;
    // Initial run or time reset
    if (last_sim_time.nanoseconds() == 0 || current_sim_time < last_sim_time) {
        control_dt_ = 1.0 / control_frequency;
    } else {
        control_dt_ = (current_sim_time - last_sim_time).seconds();
    }
    last_sim_time = current_sim_time;

    // Safety clamp: if dt is too large (lag) or invalid, revert to default
    if (control_dt_ > 0.05f || control_dt_ <= 0.0f) {
        control_dt_ = 1.0 / control_frequency;
    }

    global_phase_ += control_dt_ / cycle_time;
    global_phase_ -= static_cast<int>(global_phase_);

    joint_pos = Eigen::Map<const Eigen::VectorXd>(joint_state->position.data(), joint_state->position.size()).cast<float>();
    joint_vel = Eigen::Map<const Eigen::VectorXd>(joint_state->velocity.data(), joint_state->velocity.size()).cast<float>();

    auto imu = message_handler_->GetLatestImu();
    Eigen::Quaternionf base_quat(
        (float)imu->quaternion.w, 
        (float)imu->quaternion.x, 
        (float)imu->quaternion.y, 
        (float)imu->quaternion.z);
    
    // Vector for observation loop [w, x, y, z]
    Eigen::Vector4f base_quat_w(base_quat.w(), base_quat.x(), base_quat.y(), base_quat.z());

    Eigen::Vector3f base_ang_vel = Eigen::Vector3f(imu->angular_velocity.x, imu->angular_velocity.y, imu->angular_velocity.z).cast<float>();

    auto velocity_commands_ = message_handler_->GetLatestBodyVelCmd();
    Eigen::Vector3f velocity_commands;
    if (velocity_commands_) {
        velocity_commands = Eigen::Vector3f(velocity_commands_->linear_velocity[0],velocity_commands_->linear_velocity[1],velocity_commands_->yaw_velocity).cast<float>();
    } else {
        velocity_commands.setZero();
    }
  
    Eigen::Vector3f gravity_world(0.0f, 0.0f, -9.81f);
    // Pass quaternion to projected_gravity_b
    Eigen::Vector3f projected_gravity = projected_gravity_b(base_quat, gravity_world);
  
    Eigen::Vector2f gait_phase(std::sin(2 * M_PI * global_phase_), std::cos(2 * M_PI * global_phase_));

    // +-----------+---------------------------------+-----------+
    // |   Index   | Name                            |   Shape   |
    // +-----------+---------------------------------+-----------+
    // |     0     | joint_pos                       |   (24,)   |
    // |     1     | joint_vel                       |   (24,)   |
    // |     2     | actions                         |   (24,)   |
    // |     3     | base_ang_vel                    |    (3,)   |
    // |     4     | base_quat_w                     |    (4,)   |
    // |     5     | velocity_commands               |    (3,)   |
    // |     6     | projected_gravity               |    (3,)   |
    // |     7     | gait_phase                      |    (2,)   |
    // +-----------+---------------------------------+-----------+
    for (int i = 0; i < 24; i++)
    {
        // joint_pos
        obs(i) = (joint_pos[i] - default_joint_pos[i]) * observation_scale_joint_pos;
        // joint_vel
        obs(i+24) = joint_vel[i] * observation_scale_joint_vel;
    }
    // actions
    obs.segment(48, 24) = act;
    for (int i = 0; i < 3; i++)
    {
        // base_ang_vel
        obs(72 + i) = base_ang_vel[i] * observation_scale_base_ang_vel;
    }
    for (int i = 0; i < 4; i++)
    {
        // base_quat_w
        obs(75 + i) = base_quat_w[i] * observation_scale_base_quat_w;
    }
    obs.segment(79, 3) = velocity_commands;
    obs.segment(82, 3) = projected_gravity;
    obs.segment(85, 2) = gait_phase;

    // policy forward
    torch::Tensor torch_tensor = torch::from_blob(obs.data(), {1, obs.size()}, torch::kFloat).clone();
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(torch_tensor);
    torch::Tensor output_tensor = module.forward(inputs).toTensor();
    std::memcpy(act.data(), output_tensor.data_ptr<float>(), output_tensor.size(1) * sizeof(float));
    
    for(int i=0; i<act.size(); ++i) {
        if(act(i) > action_clip_limit) act(i) = action_clip_limit;
        if(act(i) < -action_clip_limit) act(i) = -action_clip_limit;
    }

    joint_command_->position.resize(24);
    for (int i = 0; i < 24; i++){
        joint_command_->position[i] = act(i) * action_scale + default_joint_pos[i];
    }
    
    // joint_command_->position = std::vector<double>(act.data(), act.data() + act.size());
    joint_command_->velocity = std::vector<double>(act.size(), 0.0);
    joint_command_->feed_forward_torque = std::vector<double>(act.size(), 0.0);
    joint_command_->torque = std::vector<double>(act.size(), 0.0);
    joint_command_->stiffness = std::vector<double>(joint_kp.data(), joint_kp.data() + joint_kp.size());
    joint_command_->damping = std::vector<double>(joint_kd.data(), joint_kd.data() + joint_kd.size());
    joint_command_->parallel_parser_type = interface_protocol::msg::ParallelParserType::RL_PARSER;
    
    message_handler_->PublishJointCommand(*joint_command_);
    
    if (info_get_obs){std::cout << "obs: \n" << obs.transpose() << std::endl;};
    if (info_get_action_output){std::cout << "action: \n" << act.transpose() << std::endl;};
    if (info_get_joint_command_output){std::cout << "joint_command_output: \n" << joint_command_->position << std::endl;};
}

Eigen::Vector3f pm01_controller::projected_gravity_b(Eigen::Quaternionf quat, Eigen::Vector3f vec) {
    float w = quat.w();
    float x = quat.x();
    float y = quat.y();
    float z = quat.z();
    
    // reshape to (N, 3) for multiplication -> vec is already (3)
    // extract components from quaternions
    Eigen::Vector3f xyz(x, y, z);
    // t = xyz.cross(vec, dim=-1) * 2
    Eigen::Vector3f t = xyz.cross(vec) * 2;
    // (vec - quat[:, 0:1] * t + xyz.cross(t, dim=-1))
    return vec - w * t + xyz.cross(t);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
	auto controller = std::make_shared<pm01_controller>();
    if (controller->Initialize()) {
        rclcpp::spin(controller);
    }

    rclcpp::shutdown();
	return 0;
}




