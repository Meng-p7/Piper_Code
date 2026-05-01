"""
Piper 机械臂抓取演示 (无界面测试版)
"""

import sys
import os
import numpy as np
import mujoco

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kinematics import ForwardKinematics, InverseKinematics
from core.trajectory import TrajectoryPlanner


class GraspBallDemo:
    def __init__(self, model_path="models/scene.xml"):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        arm_joint_names = [f"joint{i}" for i in range(1, 7)]
        
        self.ik_body_name = "link6"
        self.ball_body_name = "target_ball"
        
        self.fk = ForwardKinematics(self.model, self.ik_body_name)
        self.ik = InverseKinematics(self.model, self.ik_body_name, joint_names=arm_joint_names)
        self.planner = TrajectoryPlanner(max_velocity=0.5, max_acceleration=0.3)
        
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ball_body_name)
        self.ik_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ik_body_name)
        self.link7_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.link8_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
        
        self.arm_joint_ids = []
        self.arm_actuator_ids = []
        for name in arm_joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.arm_joint_ids.append(jid)
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.arm_actuator_ids.append(aid)
        
        self.gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        
        self.gripper_open_ctrl = 0.035
        self.gripper_close_ctrl = 0.0
        
        self.home_qpos = np.array([0, 1.57, -1.3485, 0, 0, 0])
        self.observe_qpos = np.array([0, 0.8, -1.0, 0, 0, 0])
        self.ball_radius = 0.015
        
        print("GraspBallDemo initialized")
    
    def get_gripper_center(self):
        return (self.data.xpos[self.link7_id] + self.data.xpos[self.link8_id]) / 2.0
    
    def get_ball_position(self):
        return self.data.xpos[self.ball_id].copy()
    
    def set_ctrl(self, q_arm, gripper_ctrl=None):
        for i, aid in enumerate(self.arm_actuator_ids):
            self.data.ctrl[aid] = q_arm[i]
        if gripper_ctrl is not None:
            self.data.ctrl[self.gripper_actuator_id] = gripper_ctrl
    
    def wait_for_settle(self, steps=20):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
    
    def move_to_qpos(self, q_target, gripper_ctrl=None, num_steps=100):
        if gripper_ctrl is None:
            gripper_ctrl = self.gripper_open_ctrl
        
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        trajectory = self.planner.quintic_interpolation(q_current, q_target, num_steps)
        
        for q in trajectory:
            for i, aid in enumerate(self.arm_actuator_ids):
                self.data.ctrl[aid] = q[i]
            self.data.ctrl[self.gripper_actuator_id] = gripper_ctrl
            mujoco.mj_step(self.model, self.data)
        
        self.wait_for_settle()
    
    def move_to_position(self, target_pos, gripper_ctrl=None, num_steps=100):
        if gripper_ctrl is None:
            gripper_ctrl = self.gripper_open_ctrl
        
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        q_full = self.data.qpos.copy()
        q_target, success = self.ik.solve_position(target_pos, q_init=q_current, q_full=q_full)
        
        if not success:
            print(f"  警告: IK 求解可能不精确")
        
        trajectory = self.planner.quintic_interpolation(q_current, q_target, num_steps)
        for q in trajectory:
            for i, aid in enumerate(self.arm_actuator_ids):
                self.data.ctrl[aid] = q[i]
            self.data.ctrl[self.gripper_actuator_id] = gripper_ctrl
            mujoco.mj_step(self.model, self.data)
        
        self.wait_for_settle()
        return q_target
    
    def grasp_object(self, num_steps=100):
        print("  闭合夹爪...")
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        
        for i in range(num_steps):
            gripper_ctrl = self.gripper_open_ctrl * (1 - i / num_steps)
            for j, aid in enumerate(self.arm_actuator_ids):
                self.data.ctrl[aid] = q_current[j]
            self.data.ctrl[self.gripper_actuator_id] = gripper_ctrl
            for _ in range(3):
                mujoco.mj_step(self.model, self.data)
        
        self.wait_for_settle(30)
        
        gripper_center = self.get_gripper_center()
        ball_pos = self.get_ball_position()
        dist = np.linalg.norm(gripper_center - ball_pos)
        print(f"  夹爪中心: {gripper_center}")
        print(f"  小球位置: {ball_pos}")
        print(f"  距离: {dist:.4f}m")
    
    def lift_object(self, lift_height=0.10, num_steps=100):
        print(f"  提升 {lift_height}m...")
        
        ee_pos = self.data.xpos[self.ik_body_id].copy()
        lift_pos = ee_pos.copy()
        lift_pos[2] += lift_height
        
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        q_full = self.data.qpos.copy()
        q_lift, success = self.ik.solve_position(lift_pos, q_init=q_current, q_full=q_full)
        
        if not success:
            print("  警告: 提升位姿 IK 求解可能不精确")
        
        trajectory = self.planner.quintic_interpolation(q_current, q_lift, num_steps)
        for q in trajectory:
            for i, aid in enumerate(self.arm_actuator_ids):
                self.data.ctrl[aid] = q[i]
            self.data.ctrl[self.gripper_actuator_id] = self.gripper_close_ctrl
            mujoco.mj_step(self.model, self.data)
        
        self.wait_for_settle()
    
    def release_object(self, num_steps=100):
        print("  打开夹爪...")
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        
        for i in range(num_steps):
            gripper_ctrl = self.gripper_close_ctrl + self.gripper_open_ctrl * (i / num_steps)
            for j, aid in enumerate(self.arm_actuator_ids):
                self.data.ctrl[aid] = q_current[j]
            self.data.ctrl[self.gripper_actuator_id] = gripper_ctrl
            for _ in range(3):
                mujoco.mj_step(self.model, self.data)
        
        self.wait_for_settle(30)
    
    def run_demo(self):
        print("=" * 50)
        print("Piper 机械臂抓取演示")
        print("=" * 50)
        
        print("\n步骤 1: 移动到初始位姿...")
        self.move_to_qpos(self.home_qpos, self.gripper_open_ctrl)
        print(f"  夹爪中心: {self.get_gripper_center()}")
        
        print("\n步骤 2: 低头观察位姿...")
        self.move_to_qpos(self.observe_qpos, self.gripper_open_ctrl)
        print(f"  夹爪中心: {self.get_gripper_center()}")
        
        ball_pos = self.get_ball_position()
        print(f"  小球真实位置: {ball_pos}")
        
        print("\n步骤 3: 移动到小球正上方...")
        approach_pos = np.array([0.19, 0, 0.3])
        self.move_to_position(approach_pos, self.gripper_open_ctrl)
        print(f"  夹爪中心: {self.get_gripper_center()}")
        
        print("\n步骤 4: 下降到抓取高度...")
        grasp_pos = np.array([0.19, 0, 0.08])
        print(f"  目标抓取位置: {grasp_pos}")
        self.move_to_position(grasp_pos, self.gripper_open_ctrl)
        print(f"  夹爪中心: {self.get_gripper_center()}")
        
        print("\n步骤 5: 执行抓取...")
        self.grasp_object()
        
        print("\n步骤 6: 提升物体...")
        self.lift_object(lift_height=0.10)
        print(f"  提升后夹爪中心: {self.get_gripper_center()}")
        print(f"  提升后小球位置: {self.get_ball_position()}")
        
        print("\n步骤 7: 移动到放置位置...")
        place_pos = np.array([0.3, -0.15, 0.25])
        self.move_to_position(place_pos, self.gripper_close_ctrl)
        
        print("\n步骤 8: 释放物体...")
        self.release_object()
        
        print("\n步骤 9: 返回初始位姿...")
        self.move_to_qpos(self.home_qpos, self.gripper_open_ctrl)
        
        print("\n" + "=" * 50)
        print("演示完成!")
        print("=" * 50)


def main():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scene.xml")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    demo = GraspBallDemo(model_path)
    demo.run_demo()


if __name__ == "__main__":
    main()
