import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml
import argparse
from scipy.spatial.transform import Rotation as R


def get_root_local_rot_tan_norm(root_quat_w):
    # Input: root_quat_w is (w, x, y, z)
    # Scipy Rotation uses (x, y, z, w), so we must reorder
    r = R.from_quat([root_quat_w[1], root_quat_w[2], root_quat_w[3], root_quat_w[0]])
    
    # yaw quaternion
    euler = r.as_euler('zyx')
    yaw = euler[0]
    yaw_rot = R.from_euler('z', yaw)
    
    # root_quat_local = yaw_inv * root_quat
    root_rot_local_r = yaw_rot.inv() * r
    root_rotm_local = root_rot_local_r.as_matrix()
    
    tan_vec = root_rotm_local[:, 0]
    norm_vec = root_rotm_local[:, 2]
    return np.concatenate([tan_vec, norm_vec])

def get_key_body_pos_b(m, d, key_body_names):
    key_body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name) for name in key_body_names]
    # Check if any ID is -1 (not found)
    if -1 in key_body_ids:
        print(f"Warning: Some key bodies not found: {key_body_names}")
    
    key_body_pos_w = np.array([d.xpos[i] for i in key_body_ids]) # (M, 3)
    root_pos_w = d.qpos[0:3]
    root_quat_w = d.qpos[3:7] # w, x, y, z
    
    # Input: root_quat_w is (w, x, y, z)
    # Convert to scipy (x, y, z, w) for calculation
    r = R.from_quat([root_quat_w[1], root_quat_w[2], root_quat_w[3], root_quat_w[0]])
    
    # key_body_pos_b = quat_apply_inverse(root_quat, key_body_pos_w - root_pos_w)
    delta_pos = key_body_pos_w - root_pos_w
    key_body_pos_b = r.inv().apply(delta_pos)
    
    return key_body_pos_b.flatten()

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("config_file", type=str, help="config file name in the config folder")
    # args = parser.parse_args()
    # config_file = args.config_file
    config_file="/home/ubuntu/workspace/pm01/deploy/pm01_deploy/src/pm01_deploy/config/pm01_mujoco_amp.yaml"

    KEY_BODY_NAMES = [
        "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R",
        "LINK_ELBOW_YAW_L",
        "LINK_ELBOW_YAW_R",
        "LINK_SHOULDER_ROLL_L",
        "LINK_SHOULDER_ROLL_R",
    ]

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

        num_actions = 24
        num_obs = 102 # Dimensions of a single observation frame
        history_length = 5
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    # history buffer: (5, 102) - Row 0 is the newest observation
    obs_history_buffer = np.zeros((history_length, num_obs), dtype=np.float32)

    counter = 0

    # Load robot model
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # load policy
    policy = torch.jit.load(policy_path)        

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

                joint_pos = d.qpos[7:31]
                joint_vel = d.qvel[6:30]

                # Policy expects:
                # 0. base_ang_vel (3)
                # 1. root_local_rot_tan_norm (6)
                # 2. velocity_commands (3)
                # 3. joint_pos (24)
                # 4. joint_vel (24)
                # 5. actions (24)
                # 6. key_body_pos_b (18)

                base_quat_w = d.qpos[3:7] # w, x, y, z
                
                # 0. base_ang_vel (3) - in base frame
                base_ang_vel_w = d.qvel[3:6]
                r_w = R.from_quat([base_quat_w[1], base_quat_w[2], base_quat_w[3], base_quat_w[0]])
                base_ang_vel_b = r_w.inv().apply(base_ang_vel_w)
                
                # 1. root_local_rot_tan_norm (6)
                root_local_rot = get_root_local_rot_tan_norm(base_quat_w) # Returns (6,)

                # 2. velocity_commands (3)
                velocity_commands = cmd

                # 3. joint_pos (24)
                qj = (joint_pos - default_angles) * dof_pos_scale
                
                # 4. joint_vel (24)
                dqj = joint_vel * dof_vel_scale

                # 5. actions (24) is 'action' variable (previous action)
                
                # 6. key_body_pos_b (18)
                key_body_pos_b = get_key_body_pos_b(m, d, KEY_BODY_NAMES) # Returns (18,)

                
                # Construct observation
                obs_list = []
                obs_list.append(base_ang_vel_b * ang_vel_scale)
                obs_list.append(root_local_rot)
                obs_list.append(velocity_commands)
                obs_list.append(qj[xml_to_policy])
                obs_list.append(dqj[xml_to_policy])
                obs_list.append(action[xml_to_policy])
                obs_list.append(key_body_pos_b)

                obs_current = np.concatenate(obs_list)
                
                # update history buffer (index 0 is newest)
                obs_history_buffer[1:] = obs_history_buffer[:-1]
                obs_history_buffer[0] = obs_current

                # Re-order history for policy (term by term)
                # The policy expects term-wise history stacking: [term1_t...term1_t-4, term2_t...term2_t-4, ...]
                obs_parts = []
                indices = [0, 3, 9, 12, 36, 60, 84, 102]
                for i in range(len(indices) - 1):
                    # Slice the term across all history steps and flatten
                    term_history = obs_history_buffer[:, indices[i]:indices[i+1]].flatten()
                    obs_parts.append(term_history)
                
                obs_final = np.concatenate(obs_parts)

                obs_tensor = torch.from_numpy(obs_final).float().unsqueeze(0)
                # policy inference
                action = policy(obs_tensor).detach().numpy().squeeze()
                action = action[policy_to_xml]
                # transform action to target_dof_pos
                target_dof_pos = action * action_scale + default_angles

                # if get_info:
                if True:
                    print("--------------------------------")
                    # print("base_ang_vel    \n", obs_final[0:15])
                    # print("root_local_rot  \n", obs_final[15:45])
                    print("root_local_rot  \n", obs_final[15:21])
                    # print("velocity_cmd    \n", obs_final[45:60])
                    # print("joint_pos       \n", obs_final[60:180])
                    # print("joint_vel       \n", obs_final[180:300])
                    # print("actions         \n", obs_final[300:420])
                    # print("key_body_pos_b  \n", obs_final[420:510])
                    

                # Pick up changes to the physics state, apply perturbations, update options from GUI.
                viewer.sync()

                # Rudimentary time keeping, will drift relative to wall clock.
                time_until_next_step = m.opt.timestep * 20 - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)