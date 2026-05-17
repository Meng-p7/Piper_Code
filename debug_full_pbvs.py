"""完整 PBVS 抓取流程关节角度调试"""
import sys
import os
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.kinematics import InverseKinematics
from core.trajectory import TrajectoryPlanner
from core.visual_servo import PBVSController
from core.controller import SimulationController
from utils import config

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "scene.xml")
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

arm_joint_names = config.robot.joint_names
ik_ee_body_name = config.robot.ik_ee_body_name
gripper_bodies = config.robot.gripper_bodies

controller = SimulationController(model, data)
controller.connect()

ik = InverseKinematics(model, ik_ee_body_name, joint_names=arm_joint_names, gripper_bodies=gripper_bodies)
planner = TrajectoryPlanner(max_velocity=config.robot.max_velocity, max_acceleration=config.robot.max_acceleration)

# 设置观察位姿
observe_qpos = np.array(config.robot.observe_qpos)
for i, name in enumerate(arm_joint_names):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    adr = model.jnt_qposadr[jid]
    data.qpos[adr] = observe_qpos[i]
mujoco.mj_forward(model, data)

ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.grasp_demo.ball_body_name)
ball_pos = data.xpos[ball_id].copy()

link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
link8_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link8")

def get_gripper_center():
    return (data.xpos[link7_id] + data.xpos[link8_id]) / 2.0

def get_gripper_to_link6_offset():
    return data.xpos[link6_id].copy() - get_gripper_center()

pbvs = PBVSController(
    model=model,
    data=mujoco.MjData(model),
    joint_names=arm_joint_names,
    ee_body_name="link6",
    Kp=3.0,
    Ko=1.5,
    lambda_damping=0.05,
    pos_tol=1e-3,
    ori_tol=0.017,
)

def run_pbvs_step(target_pos, duration=4.0):
    """运行 PBVS 并打印关节角度"""
    print(f"\n  PBVS 接近目标: {target_pos}")
    
    # IK 预定位
    q_current = controller.get_joint_positions()
    q_full = data.qpos.copy()
    q_ik, success = ik.solve_gripper_position(target_pos, q_init=q_current, q_full=q_full)
    
    if success:
        trajectory = planner.quintic_interpolation(q_current, q_ik, num_steps=80)
        for q in trajectory:
            for j, adr in enumerate(controller.joint_qpos_adrs):
                data.qpos[adr] = q[j]
            mujoco.mj_forward(model, data)
        print(f"  IK 预定位: {np.round(q_ik, 4)}")
    
    # PBVS 闭环
    pbvs.data.qpos[:] = data.qpos.copy()
    mujoco.mj_forward(model, pbvs.data)
    
    dt = model.opt.timestep
    max_steps = int(duration / dt)
    
    for step in range(max_steps):
        q = controller.get_joint_positions()
        
        for i, adr in enumerate(pbvs.joint_qpos_adrs):
            pbvs.data.qpos[adr] = q[i]
        mujoco.mj_forward(model, pbvs.data)
        
        offset = get_gripper_to_link6_offset()
        link6_target = target_pos + offset
        
        gripper_err = np.linalg.norm(get_gripper_center() - target_pos)
        
        if step % 100 == 0 or step < 3:
            print(f"    Step {step:4d}: q=[{', '.join(f'{v:7.4f}' for v in q)}], err={gripper_err:.4f}m")
        
        if gripper_err < 5e-3:
            print(f"    收敛于第 {step} 步")
            break
        
        qvel = pbvs.compute_joint_velocity(link6_target, qpos=q)
        q_new = q + qvel * dt
        
        # 应用关节限位约束
        joint_limits = [
            (-2.618, 2.618),   # joint1
            (0.0, 3.14),       # joint2
            (-2.697, 0.0),     # joint3
            (-1.832, 1.832),   # joint4
            (-1.22, 1.22),     # joint5
            (-3.14, 3.14),     # joint6
        ]
        for i in range(6):
            q_new[i] = np.clip(q_new[i], joint_limits[i][0], joint_limits[i][1])
        
        for i, adr in enumerate(controller.joint_qpos_adrs):
            data.qpos[adr] = q_new[i]
        mujoco.mj_forward(model, data)

print("=" * 80)
print("完整 PBVS 抓取流程关节角度调试")
print("=" * 80)
print(f"小球位置: {ball_pos}")
print(f"初始关节角: {np.round(controller.get_joint_positions(), 4)}")
print(f"初始夹爪中心: {get_gripper_center()}")

# 步骤 1: 接近小球上方 15cm
print("\n" + "=" * 80)
print("步骤 1: 接近小球上方 15cm")
print("=" * 80)
approach_pos = ball_pos.copy()
approach_pos[2] += 0.15
run_pbvs_step(approach_pos, duration=4.0)

# 步骤 2: 下降到预抓取位置（球上方 3cm）
print("\n" + "=" * 80)
print("步骤 2: 下降到预抓取位置（球上方 3cm）")
print("=" * 80)
pre_grasp_pos = ball_pos.copy()
pre_grasp_pos[2] += 0.03
run_pbvs_step(pre_grasp_pos, duration=3.0)

# 步骤 3: 精确接近抓取位置（夹爪中心对准球心）
print("\n" + "=" * 80)
print("步骤 3: 精确接近抓取位置（夹爪中心对准球心）")
print("=" * 80)
grasp_pos = ball_pos.copy()
run_pbvs_step(grasp_pos, duration=4.0)

print("\n" + "=" * 80)
print("最终状态")
print("=" * 80)
print(f"最终关节角: {np.round(controller.get_joint_positions(), 4)}")
print(f"最终夹爪中心: {get_gripper_center()}")
print(f"小球位置: {ball_pos}")
print(f"偏差: {np.linalg.norm(get_gripper_center() - ball_pos):.4f}m")
