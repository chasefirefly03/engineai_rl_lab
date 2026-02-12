import mujoco
import numpy as np
import yaml
import os

# 关节名称映射
# 这是从 yaml 文件注释中提取的，用于对应显示
joint_names_yaml = [
    "J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R",
    "J12_WAIST_YAW",
    "J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L", "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R", "J22_ELBOW_YAW_R",
    "J23_HEAD_YAW"
]

def check_initial_angles():
    config_file = "/home/ubuntu/workspace/pm01/deploy/pm01_deploy/src/pm01_deploy/config/pm01_mujoco.yaml"
    
    if not os.path.exists(config_file):
        print(f"Error: Config file not found at {config_file}")
        return

    print(f"Loading config from: {config_file}")
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    # 1. 从 YAML 获取默认角度 (Control Target)
    default_angles = np.array(config["default_joint_pos"], dtype=np.float32)
    print("\n" + "="*80)
    print("--- 1. Default Joint Angles from YAML Config (Control Target) ---")
    print("="*80)
    
    if len(default_angles) != len(joint_names_yaml):
        print(f"Warning: Number of angles in YAML ({len(default_angles)}) does not match expected joint names count ({len(joint_names_yaml)})")
    
    for i, angle in enumerate(default_angles):
        name = joint_names_yaml[i] if i < len(joint_names_yaml) else f"Joint_{i}"
        print(f"{i:2d}: {name:<25} : {angle:.4f}")

    # Check for default velocities in YAML
    print("\n" + "="*80)
    print("--- 1.1. Default Joint Velocities from YAML Config ---")
    print("="*80)
    if "default_joint_vel" in config:
        default_vels = np.array(config["default_joint_vel"], dtype=np.float32)
        for i, vel in enumerate(default_vels):
            name = joint_names_yaml[i] if i < len(joint_names_yaml) else f"Joint_{i}"
            print(f"{i:2d}: {name:<25} : {vel:.4f}")
    else:
        print("Key 'default_joint_vel' not found in YAML. Assuming target velocity is 0.0 for all joints.")

    # 2. 尝试从 XML 获取关节信息 (Mujoco Model Default)
    xml_path = config["robot_xml_path"]
    print("\n" + "="*140)
    print(f"--- 2. Joint Info from Mujoco Model ({os.path.basename(xml_path)}) ---")
    print("="*140)
    print(f"XML Path: {xml_path}")
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    try:
        m = mujoco.MjModel.from_xml_path(xml_path)
        d = mujoco.MjData(m)
    except Exception as e:
        print(f"Failed to load Mujoco model: {e}")
        return

    # m.qpos0 是 XML 中定义的默认位置 (或默认 0)
    qpos0 = m.qpos0
    
    print(f"Model nq (coordinates): {m.nq}, nu (actuators): {m.nu}, njnt (joints): {m.njnt}")
    print("-" * 140)
    print(f"{'ID':<4} | {'Joint Name':<25} | {'Type':<8} | {'Range (Min, Max)':<20} | {'qpos0':<20} | {'qvel0':<10} | {'Limited'}")
    print("-" * 140)

    for i in range(m.njnt):
        j_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
        if j_name is None:
            j_name = f"Joint_{i}"
            
        j_type = m.jnt_type[i]
        j_qposadr = m.jnt_qposadr[i]
        j_dofadr = m.jnt_dofadr[i]
        
        type_str = "UNKNOWN"
        val_str = ""
        vel_str = ""
        range_str = ""
        limited_str = str(m.jnt_limited[i] == 1)

        j_range = m.jnt_range[i]
        range_str = f"[{j_range[0]:.3f}, {j_range[1]:.3f}]"
        
        if j_type == mujoco.mjtJoint.mjJNT_FREE:
            type_str = "FREE"
            val_str = f"{qpos0[j_qposadr:j_qposadr+3]}..." # show position only partly
            vel_str = f"{d.qvel[j_dofadr:j_dofadr+3]}..."
            range_str = "N/A"
            limited_str = "N/A"
        elif j_type == mujoco.mjtJoint.mjJNT_BALL:
            type_str = "BALL"
            val_str = f"{qpos0[j_qposadr:j_qposadr+4]}"
            vel_str = f"{d.qvel[j_dofadr:j_dofadr+3]}"
            range_str = "N/A"
            limited_str = "N/A"
        elif j_type == mujoco.mjtJoint.mjJNT_SLIDE:
            type_str = "SLIDE"
            val_str = f"{qpos0[j_qposadr]:.4f}"
            vel_str = f"{d.qvel[j_dofadr]:.4f}"
        elif j_type == mujoco.mjtJoint.mjJNT_HINGE:
            type_str = "HINGE"
            val_str = f"{qpos0[j_qposadr]:.4f}"
            vel_str = f"{d.qvel[j_dofadr]:.4f}"
            
        print(f"{i:<4} | {j_name:<25} | {type_str:<8} | {range_str:<20} | {val_str:<20} | {vel_str:<10} | {limited_str}")

if __name__ == "__main__":
    check_initial_angles()
