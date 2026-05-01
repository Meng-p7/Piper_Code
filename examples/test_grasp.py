import sys
import os
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kinematics import ForwardKinematics, InverseKinematics
from core.trajectory import TrajectoryPlanner


def test_grasp():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scene.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    # 初始化
    arm_joint_names = [f"joint{i}" for i in range(1, 7)]
    ik = InverseKinematics(model, "link6", joint_names=arm_joint_names)
    planner = TrajectoryPlanner(max_velocity=0.5, max_acceleration=0.3)
    
    # 检查各部分位置
    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    link8_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link8")
    ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_ball")
    
    # 设置到观察位姿
    observe_qpos = np.array([0, 0.8, -1.0, 0, 0, 0])
    for i, name in enumerate(arm_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[jid] = observe_qpos[i]
    mujoco.mj_forward(model, data)
    
    print(f"初始 link6: {data.xpos[link6_id]}")
    print(f"初始 link7: {data.xpos[link7_id]}")
    print(f"初始 link8: {data.xpos[link8_id]}")
    print(f"初始夹爪: {(data.xpos[link7_id] + data.xpos[link8_id])/2}")
    print(f"小球位置: {data.xpos[ball_id]}")
    print()
    
    # 测试 IK 求解
    ball_pos = data.xpos[ball_id].copy()
    ball_radius = 0.015
    gripper_offset_x = 0.1345
    gripper_offset_z = 0.0118
    
    print("=== 测试抓取位置 ===")
    gripper_center_target = ball_pos.copy()
    gripper_center_target[2] = ball_pos[2] + ball_radius
    print(f"目标夹爪中心: {gripper_center_target}")
    
    link6_target = gripper_center_target.copy()
    link6_target[0] -= gripper_offset_x
    link6_target[2] -= gripper_offset_z
    print(f"目标 link6: {link6_target}")
    
    q_init = observe_qpos.copy()
    q_sol, success = ik.solve_position(link6_target, q_init=q_init, q_full=data.qpos.copy())
    print(f"IK 成功: {success}")
    print(f"求解关节角: {q_sol}")
    
    # 应用求解
    for i, name in enumerate(arm_joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[jid] = q_sol[i]
    mujoco.mj_forward(model, data)
    
    print()
    print("应用结果后:")
    print(f"link6 实际: {data.xpos[link6_id]}")
    print(f"夹爪中心: {(data.xpos[link7_id] + data.xpos[link8_id])/2}")
    print(f"小球位置: {data.xpos[ball_id]}")
    
    dist = np.linalg.norm((data.xpos[link7_id] + data.xpos[link8_id])/2 - data.xpos[ball_id])
    print(f"距离: {dist:.4f}m")


if __name__ == "__main__":
    test_grasp()
