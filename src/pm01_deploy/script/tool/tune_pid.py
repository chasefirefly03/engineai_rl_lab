
import time
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import os

# Configuration Paths
# Assuming this script is in .../script/ and config is in .../config/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Correct path to config based on workspace structure
CONFIG_FILE = "/home/ubuntu/workspace/pm01/deploy/pm01_deploy/src/pm01_deploy/config/pm01_mujoco.yaml"

def pd_control(target_q, q, kp, target_dq, dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp + (target_dq - dq) * kd

def main():
    print(f"Loading config from: {CONFIG_FILE}")
    if not os.path.exists(CONFIG_FILE):
        print(f"Error: Config file not found at {CONFIG_FILE}")
        return

    with open(CONFIG_FILE, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    xml_path = config["robot_xml_path"]
    simulation_dt = config["simulation_dt"]
    
    # --- PID TUNING SECTION --------------------------
    # Modify these parameters or the values below to tune stiffness and damping
    
    # 1. Scaling factors (apply to all joints)
    KP_SCALE = 1.0
    KD_SCALE = 1.0

    # 2. Base values from config (you can also manually replace these lists)
    # The config has default values:
    # joint_kp: [200, 150, 150, 200, 100, 100, ...] for 24 joints
    kps = np.array(config["joint_kp"], dtype=np.float32) * KP_SCALE
    kds = np.array(config["joint_kd"], dtype=np.float32) * KD_SCALE
    
    # Example: Manually override leg pitch joint Kp (indices 0 and 6 normally, check mapping)
    # kps[0] = 300.0 
    
    default_angles = np.array(config["default_joint_pos"], dtype=np.float32)
    # -------------------------------------------------

    print(f"Loading XML Model: {xml_path}")
    if not os.path.exists(xml_path):
         print(f"Error: XML file not found at {xml_path}")
         return

    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Get actuator gears for normalization (Crucial for correct torque application)
    actuator_gears = m.actuator_gear[:, 0]
    print(f"Actuator gears loaded: {actuator_gears}")

    # Initialize Robot State
    if m.nkey > 0:
        mujoco.mj_resetDataKeyframe(m, d, 0)
        print("Initialized robot state from Keyframe 0 (Home pose)")
    else:
        print("No keyframe found. Using default joint positions.")
        d.qpos[7:] = default_angles
        d.qpos[2] = 0.82  # Approximate standing height

    # Target is to hold default angles
    target_dof_pos = default_angles.copy()

    print("\nStarting Simulation for PID Tuning...")
    print("Press ESC in viewer to exit.")
    
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()

            # PD Control to hold position
            # target_dq is 0 (holding still)
            target_dq = np.zeros_like(kds)
            current_q = d.qpos[7:]
            current_dq = d.qvel[6:]

            tau = pd_control(target_dof_pos, current_q, kps, target_dq, current_dq, kds)
            
            # Apply Gear Ratio Correction
            # d.ctrl[:] = tau / actuator_gears
            
            mujoco.mj_step(m, d)
            viewer.sync()

            # Time keeping
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
