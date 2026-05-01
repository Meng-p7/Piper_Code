"""
Piper 机械臂基础运行脚本

使用新的模块化架构，支持仿真和真机切换
"""

import os
import sys
import numpy as np
import mujoco
import mujoco.viewer

from core.kinematics import ForwardKinematics, InverseKinematics
from core.trajectory import TrajectoryPlanner
from core.controller import SimulationController

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "scene.xml")

TARGET_XYZ = np.array([0.3, 0.2, 0.15])
SPEED = 0.02

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

fk = ForwardKinematics(model, ee_body_name="link7")
ik = InverseKinematics(model, ee_body_name="link7")
planner = TrajectoryPlanner()

controller = SimulationController(
    model,
    data,
    joint_names=[f"joint{i}" for i in range(1, 8)],
    gripper_joint_names=["gripper_left", "gripper_right"]
)

q_target, success = ik.solve_position(TARGET_XYZ)
print(f"目标XYZ: {TARGET_XYZ}")
print(f"IK求解: {'成功' if success else '失败'}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    controller.connect()
    
    q_current = data.qpos[:7].copy()
    num_steps = 200
    trajectory = planner.quintic_interpolation(q_current, q_target, num_steps)
    
    try:
        for q_cmd in trajectory:
            data.ctrl[:7] = q_cmd
            data.ctrl[7:9] = 0.04
            mujoco.mj_step(model, data)
            viewer.sync()
        
        curr_pos, _ = fk.get_ee_pose(data)
        print(f"\n到达末端: [{curr_pos[0]:.3f}, {curr_pos[1]:.3f}, {curr_pos[2]:.3f}]")
        
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    except KeyboardInterrupt:
        print("\n程序已退出")
    finally:
        controller.disconnect()
