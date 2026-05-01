import sys
import os
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def measure_gripper_offset():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    link8_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link8")
    
    mujoco.mj_forward(model, data)
    
    link6_pos = data.xpos[link6_id]
    link7_pos = data.xpos[link7_id]
    link8_pos = data.xpos[link8_id]
    gripper_center = (link7_pos + link8_pos) / 2
    
    offset = gripper_center - link6_pos
    
    print(f"link6 位置: {link6_pos}")
    print(f"link7 位置: {link7_pos}")
    print(f"link8 位置: {link8_pos}")
    print(f"夹爪中心: {gripper_center}")
    print(f"偏移 (gripper_center - link6): {offset}")


if __name__ == "__main__":
    import sys
    measure_gripper_offset()
