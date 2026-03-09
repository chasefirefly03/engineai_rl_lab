import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import argparse

config_file="/home/ubuntu/workspace/pm01/deploy/pm01_deploy/src/pm01_deploy/config/param/pm01_mujoco.yaml"
motion_file=""

anchor_body_name="torso_link"
MOTION_BODY_INDEX = 0

def load_motion(motion_file):                
    motion = np.load(motion_file)            
    motion_pos = motion["body_pos_w"]        
    motion_quat = motion["body_quat_w"]      
    motion_input_pos = motion["joint_pos"]   
    motion_input_vel = motion["joint_vel"]   
    return motion_pos, motion_quat, motion_input_pos, motion_input_vel 

def subtract_frame_transforms_mujoco(pos_a, quat_a, pos_b, quat_b):  # 计算 A 到 B 的相对位姿  
    rotm_a = np.zeros(9)  # 初始化旋转矩阵扁平数组  
    mujoco.mju_quat2Mat(rotm_a, quat_a)  # 四元数转旋转矩阵  
    rotm_a = rotm_a.reshape(3, 3)  # 重塑为 3x3 矩阵  
    rel_pos = rotm_a.T @ (pos_b - pos_a)  # 计算相对位置向量  
    rel_quat = quaternion_multiply((quat_a), quat_b)  # 计算相对旋转四元数  
    rel_quat = rel_quat / np.linalg.norm(rel_quat)  # 归一化四元数  
    return rel_pos, rel_quat  # 返回相对位置和相对姿态 

def quaternion_conjugate(q):  # 计算四元数共轭  
    return np.array([q[0], -q[1], -q[2], -q[3]])  # 标量不变向量取负 

def quaternion_multiply(q1, q2):  # 计算四元数乘积  
    w1, x1, y1, z1 = q1  # 解包第一个四元数  
    w2, x2, y2, z2 = q2  # 解包第二个四元数  
    
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2  # 计算结果 w 分量  
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2  # 计算结果 x 分量  
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2  # 计算结果 y 分量  
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2  # 计算结果 z 分量  
    
    return np.array([w, x, y, z])  # 返回乘积四元数 

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str, help="config file name in the config folder")
    # args = parser.parse_args()
    # config_file = args.config_file

    xml_to_policy = [0, 6, 12, 1, 7, 13, 18, 23, 2, 8, 14, 19, 3, 9, 15, 20, 4, 10, 16, 21, 5, 11, 17, 22]
    policy_to_xml = [0, 3, 8, 12, 16, 20, 1, 4, 9, 13, 17, 21, 2, 5, 10, 14, 18, 22, 6, 11, 15, 19, 23, 7]

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_file"]
        xml_path = config["robot_xml_path"]
        get_info = config["get_info"]

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["joint_kp"], dtype=np.float32)
        kds = np.array(config["joint_kd"], dtype=np.float32)

        default_angles = np.array(config["default_joint_pos"], dtype=np.float32)
        
        ang_vel_scale = config["observation_scale_base_ang_vel"]
        dof_pos_scale = config["observation_scale_joint_pos"]
        dof_vel_scale = config["observation_scale_joint_vel"]
        action_scale = config["action_scale"]
        base_quat_w_scale = config["observation_scale_base_quat_w"]
        # cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_observations"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        period = np.array(config["cycle_time"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)   

    timestep = 0
    motion_body_index = MOTION_BODY_INDEX
    body_name = anchor_body_name
    body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body_name)  # 获取锚点 ID 
    motion_pos, motion_quat, motion_input_pos, motion_input_vel = load_motion(motion_file)



    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                position = d.xpos[body_id]  # 获取仿真中锚点刚体位置  
                quaternion = d.xquat[body_id]  # 获取仿真中锚点刚体姿态  
                motion_input = np.concatenate(  # 拼接目标关节位置与速度  
                    (motion_input_pos[timestep, :], motion_input_vel[timestep, :]),  # 使用 npz 原始关节顺序  
                    axis=0,  # 进行拼接  
                )  # 拼接结束  
                motion_pos_current = motion_pos[timestep, motion_body_index, :]  # 读取当前参考锚点动作位置  
                motion_quat_current = motion_quat[timestep, motion_body_index, :]  # 读取当前参考锚点动作姿态  
                _, anchor_quat = subtract_frame_transforms_mujoco(  # 计算相对锚点位置与姿态  
                    position, quaternion, motion_pos_current, motion_quat_current  # 输入位姿参数  
                )  # 获取相对位置与旋转结果  
                anchor_ori = np.zeros(6)  # 初始化锚点旋转矩阵  
                mujoco.mju_quat2Mat(anchor_ori, anchor_quat)  # 仿真锚点相对于参考锚点的相对姿态四元数转旋转矩阵  
                anchor_ori = anchor_ori.reshape(3, 3)[:, :2]  # 取旋转矩阵前两列  
                motion_anchor_ori_b = anchor_ori.reshape(-1,)  # 展平为向量  




                joint_pos = d.qpos[7:31]
                joint_vel = d.qvel[6:30]

                base_ang_vel = d.qvel[3:6]
                base_quat_w = d.qpos[3:7]

                velocity_commands = cmd
                projected_gravity = get_gravity_orientation(base_quat_w)
                # gait_phase

                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                qj = (joint_pos - default_angles) * dof_pos_scale
                dqj = joint_vel * dof_vel_scale

                obs[:24] = qj[xml_to_policy]
                obs[24:48] = dqj[xml_to_policy]               
                obs[48:72] = action[xml_to_policy]
                obs[72:75] = base_ang_vel * ang_vel_scale
                obs[75:79] = base_quat_w * base_quat_w_scale
                obs[79:82] = velocity_commands
                obs[82:85] = projected_gravity
                obs[85:87] = np.array([sin_phase, cos_phase])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = action[policy_to_xml]
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                if get_info:
                    print("--------------------------------")
                    print("joint_pos        \n", (obs[0:24]))
                    print("joint_vel        \n", (obs[24:48]))
                    print("actions          \n",(obs[48:72]))
                    print("base_ang_vel     \n",(obs[72:75]))
                    print("base_quat_w      \n",(obs[75:79]))
                    print("velocity_commands\n",(obs[79:82]))
                    print("projected_gravity\n",(obs[82:85]))
                    print("gait_phase       \n",(obs[85:87]))
                    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep * 20 - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)