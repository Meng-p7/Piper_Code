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
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.kinematics import ForwardKinematics, InverseKinematics
from core.trajectory import TrajectoryPlanner
from core.controller import SimulationController
from core.vision import Camera, ObjectDetector
from core.calibration import HandEyeCalibration
from core.data_collection import DataRecorder


class GraspBallDemo:
    """视觉抓取小球演示类"""
    
    def __init__(self, model_path="models/scene.xml"):
        """
        初始化演示环境
        
        Args:
            model_path: MuJoCo 场景模型路径
        """
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.ee_body_name = "link7"
        self.ball_body_name = "target_ball"
        
        self.fk = ForwardKinematics(self.model, self.ee_body_name)
        self.ik = InverseKinematics(self.model, self.ee_body_name)
        self.planner = TrajectoryPlanner(max_velocity=0.5, max_acceleration=0.3)
        
        self.controller = SimulationController(
            self.model, 
            self.data,
            joint_names=[f"joint{i}" for i in range(1, 8)],
            gripper_joint_names=["gripper_left", "gripper_right"]
        )
        
        self.camera = Camera(self.model, self.data, camera_name="camera", width=640, height=480)
        self.detector = ObjectDetector()
        
        self.calibration = HandEyeCalibration()
        self.recorder = DataRecorder(save_dir="./data")
        
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ball_body_name)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, self.ee_body_name)
        
        self.gripper_open_pos = 0.04
        self.gripper_close_pos = 0.0
        
        self.home_qpos = np.array([0, -0.3, 0.5, -0.2, 0, 0, 0])
        self.grasp_qpos = None
        
        print("GraspBallDemo initialized")
    
    def get_ball_position(self):
        """获取小球的真实世界坐标"""
        ball_pos = self.data.xpos[self.ball_id].copy()
        return ball_pos
    
    def detect_ball_from_camera(self):
        """
        从相机图像中检测小球位置
        
        Returns:
            ball_world_pos: 小球的世界坐标，未检测到返回 None
        """
        image = self.camera.get_image()
        
        center, radius = self.detector.detect_ball_position(image, color_name="red")
        
        if center is None:
            print("未检测到红色小球")
            return None
        
        print(f"检测到小球，像素坐标: {center}, 半径: {radius}")
        
        depth = self.camera.get_depth()
        u, v = center
        
        if 0 <= u < depth.shape[1] and 0 <= v < depth.shape[0]:
            d = depth[v, u]
            if d > 0:
                ball_pos = self.camera.pixel_to_world(u, v, d)
                print(f"小球世界坐标: {ball_pos}")
                return ball_pos
        
        return self.get_ball_position()
    
    def move_to_home(self, viewer, speed=0.02):
        """
        移动到初始位姿
        
        Args:
            viewer: MuJoCo viewer
            speed: 移动速度
        """
        print("移动到初始位姿...")
        
        num_steps = 100
        q_current = self.data.qpos[:7].copy()
        trajectory = self.planner.quintic_interpolation(q_current, self.home_qpos, num_steps)
        
        for q_target in trajectory:
            self.data.ctrl[:7] = q_target
            self.data.ctrl[7:9] = self.gripper_open_pos
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        print("初始位姿到达")
    
    def move_to_position(self, target_pos, viewer, approach_height=0.15, speed=0.02):
        """
        移动到目标位置（带接近轨迹）
        
        Args:
            target_pos: 目标位置 [x, y, z]
            viewer: MuJoCo viewer
            approach_height: 接近高度
            speed: 移动速度
        """
        print(f"移动到目标位置: {target_pos}")
        
        approach_pos = target_pos.copy()
        approach_pos[2] += approach_height
        
        q_current = self.data.qpos[:7].copy()
        
        _, success1 = self.ik.solve_position(approach_pos, q_init=q_current)
        if not success1:
            print("警告：接近位姿 IK 求解失败，使用近似解")
        
        q_approach = self.data.qpos[:7].copy()
        
        num_steps = 100
        trajectory1 = self.planner.quintic_interpolation(q_current, q_approach, num_steps)
        
        for q_target in trajectory1:
            self.data.ctrl[:7] = q_target
            self.data.ctrl[7:9] = self.gripper_open_pos
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        _, success2 = self.ik.solve_position(target_pos, q_init=q_approach)
        if not success2:
            print("警告：目标位姿 IK 求解失败，使用近似解")
        
        q_grasp = self.data.qpos[:7].copy()
        trajectory2 = self.planner.quintic_interpolation(q_approach, q_grasp, num_steps)
        
        for q_target in trajectory2:
            self.data.ctrl[:7] = q_target
            self.data.ctrl[7:9] = self.gripper_open_pos
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        print("目标位置到达")
        return q_grasp
    
    def grasp_object(self, viewer):
        """
        执行抓取动作
        
        Args:
            viewer: MuJoCo viewer
        """
        print("执行抓取...")
        
        num_steps = 50
        for i in range(num_steps):
            gripper_pos = self.gripper_open_pos * (1 - i / num_steps)
            self.data.ctrl[7:9] = gripper_pos
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        print("抓取完成")
    
    def lift_object(self, viewer, lift_height=0.1):
        """
        提升物体
        
        Args:
            viewer: MuJoCo viewer
            lift_height: 提升高度
        """
        print(f"提升物体 {lift_height}m...")
        
        ee_pos, ee_rot = self.controller.get_ee_pose()
        lift_pos = ee_pos.copy()
        lift_pos[2] += lift_height
        
        q_current = self.data.qpos[:7].copy()
        q_lift, success = self.ik.solve_position(lift_pos, q_init=q_current)
        
        if not success:
            print("警告：提升位姿 IK 求解失败")
        
        num_steps = 100
        trajectory = self.planner.quintic_interpolation(q_current, q_lift, num_steps)
        
        for q_target in trajectory:
            self.data.ctrl[:7] = q_target
            self.data.ctrl[7:9] = self.gripper_close_pos
            mujoco.mj_step(self.model, self.data)
            viewer.sync()
        
        print("提升完成")
    
    def run_demo(self):
        """运行完整的抓取演示"""
        print("=" * 50)
        print("Piper 机械臂视觉抓取演示")
        print("=" * 50)
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            
            self.controller.connect()
            
            self.move_to_home(viewer)
            
            print("\n步骤 1: 视觉检测小球...")
            ball_pos = self.detect_ball_from_camera()
            
            if ball_pos is None:
                print("未检测到小球，使用默认位置")
                ball_pos = self.get_ball_position()
            
            print(f"小球位置: {ball_pos}")
            
            print("\n步骤 2: 移动到小球上方...")
            self.move_to_position(ball_pos, viewer, approach_height=0.15)
            
            print("\n步骤 3: 下降到抓取位置...")
            grasp_pos = ball_pos.copy()
            grasp_pos[2] = 0.03
            self.move_to_position(grasp_pos, viewer, approach_height=0.01)
            
            print("\n步骤 4: 执行抓取...")
            self.grasp_object(viewer)
            
            print("\n步骤 5: 提升物体...")
            self.lift_object(viewer, lift_height=0.15)
            
            print("\n步骤 6: 移动到放置位置...")
            place_pos = np.array([0.3, -0.2, 0.03])
            self.move_to_position(place_pos, viewer, approach_height=0.15)
            
            print("\n步骤 7: 释放物体...")
            num_steps = 50
            for i in range(num_steps):
                gripper_pos = self.gripper_close_pos + (self.gripper_open_pos) * (i / num_steps)
                self.data.ctrl[7:9] = gripper_pos
                mujoco.mj_step(self.model, self.data)
                viewer.sync()
            
            print("\n步骤 8: 返回初始位姿...")
            self.move_to_home(viewer)
            
            print("\n" + "=" * 50)
            print("演示完成！")
            print("=" * 50)
            
            self.controller.disconnect()
            
            print("\n提示: 按 ESC 或关闭 viewer 窗口退出")
            while viewer.is_running():
                mujoco.mj_step(self.model, self.data)
                viewer.sync()


def main():
    """主函数"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scene.xml")
    
    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return
    
    demo = GraspBallDemo(model_path)
    demo.run_demo()


if __name__ == "__main__":
    main()
