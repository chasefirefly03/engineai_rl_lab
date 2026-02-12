import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import argparse


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
    config_file="/home/ubuntu/workspace/pm01/deploy/pm01_deploy/src/pm01_deploy/config/pm01_mujoco.yaml"
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

    # Get actuator gears
    actuator_gears = m.actuator_gear[:, 0]

    # Initialize state
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    else:
        d.qpos[2] = 0.82
        d.qpos[7:] = default_angles

    # load policy
    policy = torch.jit.load(policy_path)        

    with mujoco.viewer.launch_passive(m, d) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau / actuator_gears
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:

                joint_pos = d.qpos[7:]
                joint_vel = d.qvel[6:]
                # last_action
                base_ang_vel = d.sensordata[10:13]
                base_quat_w = d.sensordata[0:4]
                velocity_commands = cmd
                projected_gravity = get_gravity_orientation(base_quat_w)
                # gait_phase

                count = counter * simulation_dt
                phase = count % period / period
                sin_phase = np.sin(2 * np.pi * phase)
                cos_phase = np.cos(2 * np.pi * phase)

                for i in range(24):
                    obs[i] = (joint_pos[i] - default_angles[i]) * dof_pos_scale
                    obs[i + 24] = joint_vel[i] * dof_vel_scale
                
                obs[48:72] = action

                for i in range(3):
                    obs[72 + i] = base_ang_vel[i] * ang_vel_scale
                
                for i in range(4):
                    obs[75 + i] = base_quat_w[i] * base_quat_w_scale

                obs[79:82] = velocity_commands
                obs[82:85] = projected_gravity
                obs[85:87] = np.array([sin_phase, cos_phase])

                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles
                # target_dof_pos = action * 0.5 + default_angles

                if get_info:
                    print("--------------------------------")
                    # print("joint_pos\n",joint_pos)
                    # print("joint_vel\n",joint_vel)
                    # print("base_ang_vel\n",base_ang_vel)
                    # print("base_quat_w\n",base_quat_w)
                    # print("obs\n",obs)
                    print("action\n",action)
                    print("default_angles\n",default_angles)
                    print("target_dof_pos\n",target_dof_pos)
                    # print("obs[82:85]\n",obs[82:85]) 


            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)