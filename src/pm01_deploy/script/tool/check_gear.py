
import mujoco
import numpy as np
import yaml

config_file = "/home/ubuntu/workspace/pm01/deploy/pm01_deploy/src/pm01_deploy/config/pm01_mujoco.yaml"
with open(config_file, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    xml_path = config["robot_xml_path"]

m = mujoco.MjModel.from_xml_path(xml_path)
print("Actuator gears (first column):")
print(m.actuator_gear[:, 0])
