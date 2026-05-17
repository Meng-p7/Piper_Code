#!/usr/bin/env python3
"""PBVS 运动过程关节角度调试脚本 - 打印每个控制步的关节角度

流程：观察位姿 -> IK预定位 -> PBVS微调 -> 打印全过程关节角度
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import mujoco

from core.controller import SimulationController
from core.kinematics import InverseKinematics
from core.visual_servo.pbvs import PBVSController

# 关节名称和限位
JOINT_NAMES = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
JOINT_LIMITS = [
    (-2.618, 2.618),   # joint1
    (0.0, 3.14),       # joint2
    (-2.697, 0.0),     # joint3
    (-1.832, 1.832),   # joint4
    (-1.22, 1.22),     # joint5
    (-3.14, 3.14),     # joint6
]

def print_joint_angles(q, step, label=""):
    """打印关节角度"""
    print(f"\n{'='*70}")
    print(f"步骤 {step} {label}")
    print(f"{'='*70}")
    print(f"{'关节':<10} {'角度(rad)':<12} {'角度(deg)':<12} {'限位范围':<25} {'状态':<10}")
    print(f"{'-'*70}")
    
    for i in range(6):
        angle = q[i]
        angle_deg = np.degrees(angle)
        low, high = JOINT_LIMITS[i]
        if angle < low:
            status = "✗ 低于下限"
        elif angle > high:
            status = "✗ 高于上限"
        else:
            status = "✓ 正常"
        print(f"{JOINT_NAMES[i]:<10} {angle:<12.4f} {angle_deg:<12.1f} [{low:>6.3f}, {high:>6.3f}] {status:<10}")

def move_to_qpos(data, model, q_target, pbvs, steps=50):
    """平滑移动到目标关节位置"""
    q_current = np.array([data.qpos[adr] for adr in pbvs.joint_qpos_adrs])
    for i in range(steps):
        alpha = i / (steps - 1)
        q_interp = q_current + (q_target - q_current) * alpha
        for j, adr in enumerate(pbvs.joint_qpos_adrs):
            data.qpos[adr] = q_interp[j]
        mujoco.mj_forward(model, data)

def main():
    model_path = "models/scene.xml"
    
    print("加载模型...")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    arm_joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    
    # 初始化控制器
    controller = SimulationController(model, data)
    controller.connect()
    
    # 初始化 IK
    ik = InverseKinematics(model, 'link6', joint_names=arm_joint_names)
    
    # 初始化 PBVS
    pbvs = PBVSController(
        model=model,
        data=data,
        joint_names=arm_joint_names,
        ee_body_name='link6',
        Kp=1.0,
        Ko=1.0,
        lambda_damping=0.05,
        pos_tol=0.01,
        ori_tol=0.05,
    )
    
    # 1. 设置到观察位姿
    observe_qpos = np.array([0, 0.8, -1.0, 0, 0, 0])
    for i, adr in enumerate(pbvs.joint_qpos_adrs):
        data.qpos[adr] = observe_qpos[i]
    mujoco.mj_forward(model, data)
    
    q_current = np.array([data.qpos[adr] for adr in pbvs.joint_qpos_adrs])
    print_joint_angles(q_current, 0, "观察位姿")
    
    # 2. 获取小球位置
    ball_pos = data.body('target_ball').xpos.copy()
    print(f"\n小球位置: {ball_pos}")
    
    # 计算 link6 到夹爪中心的偏移
    link6_pos = data.body('link6').xpos.copy()
    gripper_center = (data.body('link7').xpos + data.body('link8').xpos) / 2
    offset = gripper_center - link6_pos
    link6_target = ball_pos + offset
    print(f"link6 目标位置: {link6_target}")
    print(f"偏移量: {offset}")
    
    # 3. IK 预定位
    print(f"\n{'='*70}")
    print("IK 预定位...")
    print(f"{'='*70}")
    
    q_ik, success = ik.solve_gripper_position(ball_pos, q_init=q_current)
    
    if success:
        print(f"IK 解: {np.round(q_ik, 3)}")
        print_joint_angles(q_ik, 0, "IK预定位后")
        
        # 平滑移动到 IK 解
        move_to_qpos(data, model, q_ik, pbvs, steps=30)
        q_current = q_ik.copy()
    else:
        print("IK 求解失败，使用观察位姿")
    
    # 4. PBVS 微调
    dt = model.opt.timestep
    max_steps = 300
    step_count = 0
    
    print(f"\n开始 PBVS 微调 (dt={dt:.4f}s, 最大步数: {max_steps})")
    
    while step_count < max_steps:
        step_count += 1
        
        # 同步当前状态到 PBVS
        for i, adr in enumerate(pbvs.joint_qpos_adrs):
            pbvs.data.qpos[adr] = q_current[i]
        mujoco.mj_forward(model, pbvs.data)
        
        # 计算关节速度
        qvel = pbvs.compute_joint_velocity(link6_target, None)
        
        # 积分更新关节位置
        q_new = q_current + qvel * dt
        
        # 应用关节限位
        for i in range(6):
            q_new[i] = np.clip(q_new[i], JOINT_LIMITS[i][0], JOINT_LIMITS[i][1])
        
        # 打印关键步骤
        if step_count <= 3 or step_count % 10 == 0:
            print_joint_angles(q_new, step_count)
            
            # 检查收敛
            current_pos = pbvs.data.xpos[pbvs.ee_body_id].copy()
            pos_error = np.linalg.norm(current_pos - link6_target)
            print(f"  当前位置: {current_pos}")
            print(f"  位置误差: {pos_error:.4f} m (阈值: {pbvs.pos_tol})")
            
            if pos_error < pbvs.pos_tol:
                print(f"\n{'='*70}")
                print(f"✓ 收敛！步数: {step_count}")
                print_joint_angles(q_new, step_count, "收敛时")
                print(f"{'='*70}")
                break
        
        # 更新当前关节位置
        q_current = q_new.copy()
        
        # 更新仿真
        for i, adr in enumerate(pbvs.joint_qpos_adrs):
            data.qpos[adr] = q_new[i]
        mujoco.mj_forward(model, data)
    
    if step_count >= max_steps:
        print(f"\n{'='*70}")
        print(f"✗ 未收敛，达到最大步数 {max_steps}")
        q_final = np.array([data.qpos[adr] for adr in pbvs.joint_qpos_adrs])
        print_joint_angles(q_final, step_count, "最终位置")
        print(f"{'='*70}")

if __name__ == "__main__":
    main()
