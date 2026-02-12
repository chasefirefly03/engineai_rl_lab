
import mujoco
import numpy as np
import yaml

config_file = "/home/ubuntu/workspace/pm01/deploy/pm01_deploy/src/pm01_deploy/config/pm01_mujoco.yaml"
with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    xml_path = config["robot_xml_path"]

m = mujoco.MjModel.from_xml_path(xml_path)

print("Number of joints:", m.njnt)
print("Number of dofs:", m.nv)
print("Number of actuators:", m.nu)

joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)]
# print("\nJoint Names in Order:")
# for i, name in enumerate(joint_names):
#     print(f"{i}: {name}")

# actuator_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
# print("\nActuator Names in Order:")
# for i, name in enumerate(actuator_names):
#     print(f"{i}: {name}")
print(joint_names)