import mujoco
import numpy as np
import sys

# Replace with your actual XML path if different
xml_path = "/home/ubuntu/workspace/pm01/deploy/pm01_deploy/src/pm01_deploy/resource/pm_v2.xml"

try:
    m = mujoco.MjModel.from_xml_path(xml_path)
except Exception as e:
    print(f"Error loading model from {xml_path}: {e}")
    sys.exit(1)

print(f"Model ID: {m.nq} qpos, {m.nv} qvel, {m.nu} actuators")

print("\n=== Joints (d.qpos / d.qvel) ===")
qpos_offset = 0
qvel_offset = 0
for i in range(m.njnt):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i)
    jtype = m.jnt_type[i]
    
    # joint type: 0:free, 1:ball, 2:slide, 3:hinge
    if jtype == 0: # free
        q_len, v_len = 7, 6
        desc = "Free (Base)"
    elif jtype == 1: # ball
        q_len, v_len = 4, 3
        desc = "Ball"
    elif jtype == 2: # slide
        q_len, v_len = 1, 1
        desc = "Slide"
    elif jtype == 3: # hinge
        q_len, v_len = 1, 1
        desc = "Hinge"
    else:
        q_len, v_len = 0, 0
        desc = "Unknown"

    if name is None:
        name = "None"
    print(f"Joint {i}: {name:<20} | Type: {desc:<12} | qpos[{qpos_offset}:{qpos_offset+q_len}] | qvel[{qvel_offset}:{qvel_offset+v_len}]")
    qpos_offset += q_len
    qvel_offset += v_len

print("\n=== Actuators (Action Order) ===")
# Actuators correspond to the control inputs (actions)
for i in range(m.nu):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
    if name is None:
        name = "None"
    # Get associated joint
    # internal representation usually stores joint index at trnid[i, 0] for hinge/slide joints
    trnid = m.actuator_trnid[i]
    joint_id = trnid[0]
    
    # Verify if it points to a valid joint
    if joint_id >= 0 and joint_id < m.njnt:
        joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        print(f"Actuator {i}: {name:<20} -> Controls Joint: {joint_name}")
    else:
         print(f"Actuator {i}: {name:<20} -> Controls ID: {joint_id} (Not a simple joint?)")
