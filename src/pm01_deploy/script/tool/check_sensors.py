import mujoco
import numpy as np
import sys

# Replace with the actual XML path
xml_path = "/home/ubuntu/workspace/pm01/deploy/pm01_deploy/src/pm01_deploy/resource/pm_v2.xml"

try:
    m = mujoco.MjModel.from_xml_path(xml_path)
except Exception as e:
    print(f"Error loading model from {xml_path}: {e}")
    sys.exit(1)

print(f"Total sensor data length: {m.nsensordata}")
print(f"Number of sensors: {m.nsensor}")

print("\n=== Sensors (d.sensordata) ===")
offset = 0
for i in range(m.nsensor):
    name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i)
    if name is None:
        name = "None"
    
    stype = m.sensor_type[i]
    dim = m.sensor_dim[i]
    
    # Map sensor type to string (common types)
    # https://mujoco.readthedocs.io/en/latest/APIreference/APItypes.html#mjtsensortype
    type_map = {
        0: "touch", 1: "accelerometer", 2: "velocimeter", 3: "gyro", 
        4: "force", 5: "torque", 6: "magnetometer", 7: "rangefinder",
        8: "jointpos", 9: "jointvel", 10: "tendonpos", 11: "tendonvel",
        12: "actuatorpos", 13: "actuatorvel", 14: "actuatorfrc",
        15: "ballcquat", 16: "ballvel", 17: "ballangvel",
        18: "jointlimitpos", 19: "jointlimitvel", 20: "jointlimitfrc",
        21: "tendonlimitpos", 22: "tendonlimitvel", 23: "tendonlimitfrc",
        24: "framepos", 25: "framequat", 26: "framexaxis", 27: "frameyaxis", 28: "framezaxis",
        29: "framelinvel", 30: "frameangvel", 31: "framelinacc", 32: "frameangacc",
        33: "subtreecom", 34: "subtreelinvel", 35: "subtreeangmom",
        36: "clock", 37: "user"
    }
    
    type_str = type_map.get(stype, f"Unknown({stype})")
    
    print(f"Sensor {i}: {name:<25} | Type: {type_str:<15} | Dim: {dim} | Indices: [{offset}:{offset+dim}]")
    offset += dim
