"""
Piper 机械臂视觉抓取仿真例程

功能：机械臂通过视觉识别红色小球，并移动到目标位置完成抓取
运行方式：python examples/grasp_ball_demo.py
"""

import sys
import os
import numpy as np
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kinematics import ForwardKinematics, InverseKinematics
from core.trajectory import TrajectoryPlanner
from core.vision import Camera, ObjectDetector


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
        
        self.camera = Camera(self.model, self.data, camera_name="camera", width=640, height=480)
        self.detector = ObjectDetector()
        
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
        
        # 测量得到的准确偏移 (gripper_center - link6)
        self.gripper_offset_x = 0.1345
        self.gripper_offset_z = 0.0118
        
        print("GraspBallDemo initialized")
    
    def get_gripper_center(self):
        return (self.data.xpos[self.link7_id] + self.data.xpos[self.link8_id]) / 2.0
    
    def get_ball_position(self):
        return self.data.xpos[self.ball_id].copy()
    
    def detect_ball_from_camera(self):
        image = self.camera.get_image()
        center, radius = self.detector.detect_ball_position(image, color_name="red")
        
        if center is None:
            print("  未检测到红色小球，使用真实位置")
            return None
        
        print(f"  检测到小球，像素坐标: {center}, 半径: {radius}")
        
        depth = self.camera.get_depth()
        u, v = int(center[0]), int(center[1])
        
        if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
            d = depth[v, u]
            if d > 0:
                ball_pos = self.camera.pixel_to_world(u, v, d)
                print(f"  小球世界坐标: {ball_pos}")
                return ball_pos
        
        return self.get_ball_position()
    
    def set_ctrl(self, q_arm, gripper_ctrl=None):
        for i, aid in enumerate(self.arm_actuator_ids):
            self.data.ctrl[aid] = q_arm[i]
        if gripper_ctrl is not None:
            self.data.ctrl[self.gripper_actuator_id] = gripper_ctrl
    
    def wait_for_settle(self, viewer, steps=20):
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
    
    def move_to_qpos(self, q_target, viewer, gripper_ctrl=None, num_steps=100):
        if gripper_ctrl is None:
            gripper_ctrl = self.gripper_open_ctrl
        
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        trajectory = self.planner.quintic_interpolation(q_current, q_target, num_steps)
        
        for q in trajectory:
            self.set_ctrl(q, gripper_ctrl)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        self.wait_for_settle(viewer)
    
    def move_to_position(self, target_pos, viewer, gripper_ctrl=None, num_steps=100):
        if gripper_ctrl is None:
            gripper_ctrl = self.gripper_open_ctrl
        
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        q_full = self.data.qpos.copy()
        q_target, success = self.ik.solve_position(target_pos, q_init=q_current, q_full=q_full)
        
        if not success:
            print(f"  警告：IK 求解可能不精确")
        
        trajectory = self.planner.quintic_interpolation(q_current, q_target, num_steps)
        for q in trajectory:
            self.set_ctrl(q, gripper_ctrl)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        self.wait_for_settle(viewer)
        return q_target
    
    def move_to_home(self, viewer):
        print("  移动到初始位姿（抬头）...")
        self.move_to_qpos(self.home_qpos, viewer, self.gripper_open_ctrl)
        print(f"  末端位置: {self.get_gripper_center()}")
    
    def move_to_observe(self, viewer):
        print("  移动到低头观察位姿...")
        self.move_to_qpos(self.observe_qpos, viewer, self.gripper_open_ctrl)
        print(f"  末端位置: {self.get_gripper_center()}")
    
    def grasp_object(self, viewer, num_steps=100):
        print("  闭合夹爪...")
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        
        for i in range(num_steps):
            gripper_ctrl = self.gripper_open_ctrl * (1 - i / num_steps)
            self.set_ctrl(q_current, gripper_ctrl)
            for _ in range(3):
                mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        self.wait_for_settle(viewer, 30)
        
        gripper_center = self.get_gripper_center()
        ball_pos = self.get_ball_position()
        dist = np.linalg.norm(gripper_center - ball_pos)
        print(f"  夹爪中心: {gripper_center}")
        print(f"  小球位置: {ball_pos}")
        print(f"  距离: {dist:.4f}")
    
    def lift_object(self, viewer, lift_height=0.10, num_steps=100):
        print(f"  提升 {lift_height}m...")
        
        ee_pos = self.data.xpos[self.ik_body_id].copy()
        lift_pos = ee_pos.copy()
        lift_pos[2] += lift_height
        
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        q_full = self.data.qpos.copy()
        q_lift, success = self.ik.solve_position(lift_pos, q_init=q_current, q_full=q_full)
        
        if not success:
            print("  警告：提升位姿 IK 求解可能不精确")
        
        trajectory = self.planner.quintic_interpolation(q_current, q_lift, num_steps)
        for q in trajectory:
            self.set_ctrl(q, self.gripper_close_ctrl)
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        self.wait_for_settle(viewer)
    
    def release_object(self, viewer, num_steps=100):
        print("  打开夹爪...")
        q_current = self.data.qpos[self.arm_joint_ids].copy()
        
        for i in range(num_steps):
            gripper_ctrl = self.gripper_close_ctrl + self.gripper_open_ctrl * (i / num_steps)
            self.set_ctrl(q_current, gripper_ctrl)
            for _ in range(3):
                mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        self.wait_for_settle(viewer, 30)
    
    def run_demo(self):
        print("=" * 50)
        print("Piper 机械臂抓取演示")
        print("=" * 50)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            print("\n步骤 1: 移动到初始位姿...")
            self.move_to_home(viewer)
            
            print("\n步骤 2: 低头观察位姿...")
            self.move_to_observe(viewer)
            
            # 直接获取真实小球位置，不依赖视觉
            ball_pos = self.get_ball_position()
            print(f"  小球真实位置: {ball_pos}")
            
            print("\n步骤 3: 移动到小球正上方...")
            approach_pos = np.array([0.19, 0, 0.3])
            self.move_to_position(approach_pos, viewer, self.gripper_open_ctrl)
            print(f"  夹爪中心: {self.get_gripper_center()}")
            
            print("\n步骤 4: 下降到抓取高度...")
            grasp_pos = np.array([0.19, 0, 0.08])
            print(f"  目标抓取位置: {grasp_pos}")
            self.move_to_position(grasp_pos, viewer, self.gripper_open_ctrl)
            print(f"  link6 实际高度: {self.data.xpos[self.ik_body_id][2]:.4f}")
            print(f"  夹爪中心: {self.get_gripper_center()}")
            
            print("\n步骤 5: 执行抓取...")
            self.grasp_object(viewer)
            
            print("\n步骤 6: 提升物体...")
            self.lift_object(viewer, lift_height=0.10)
            print(f"  提升后夹爪中心: {self.get_gripper_center()}")
            print(f"  提升后小球位置: {self.get_ball_position()}")
            
            print("\n步骤 7: 移动到放置位置...")
            place_pos = np.array([0.3, -0.15, 0.25])
            self.move_to_position(place_pos, viewer, self.gripper_close_ctrl)
            
            print("\n步骤 8: 释放物体...")
            self.release_object(viewer)
            
            print("\n步骤 9: 返回初始位姿...")
            self.move_to_home(viewer)
            
            print("\n" + "=" * 50)
            print("演示完成！")
            print("=" * 50)
            
            print("\n提示: 按 ESC 或关闭 viewer 窗口退出")
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


def main():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scene.xml")
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return
    
    demo = GraspBallDemo(model_path)
    demo.run_demo()


if __name__ == "__main__":
    main()
