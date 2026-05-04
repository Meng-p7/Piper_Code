"""
PiperSim 视觉抓取仿真例程

功能：机械臂通过视觉识别红色小球，并移动到目标位置完成抓取
运行方式：python demos/grasp_ball_demo.py
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
from core.controller import SimulationController
from utils import config


class GraspBallDemo:
    def __init__(self, model_path="models/scene.xml"):
        self.model_path = model_path
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        arm_joint_names = config.robot.joint_names
        ik_ee_body_name = config.robot.ik_ee_body_name
        gripper_bodies = config.robot.gripper_bodies
        ball_body_name = config.grasp_demo.ball_body_name

        self.fk = ForwardKinematics(self.model, ik_ee_body_name)
        self.ik = InverseKinematics(self.model, ik_ee_body_name, joint_names=arm_joint_names, gripper_bodies=gripper_bodies)
        self.planner = TrajectoryPlanner(max_velocity=config.robot.max_velocity, max_acceleration=config.robot.max_acceleration)

        cam_width = config.vision.image_width
        cam_height = config.vision.image_height
        self.camera = Camera(self.model, self.data, camera_name=config.vision.camera_name, width=cam_width, height=cam_height)
        self.detector = ObjectDetector()

        self.controller = SimulationController(self.model, self.data)
        self.controller.connect()

        self.ik_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ik_ee_body_name)
        self.link7_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.link8_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ball_body_name)

        self.gripper_open_ctrl = config.robot.gripper.open_ctrl
        self.gripper_close_ctrl = config.robot.gripper.close_ctrl
        self.gripper_range = config.robot.gripper.range
        self.home_qpos = np.array(config.robot.home_qpos)
        self.observe_qpos = np.array(config.robot.observe_qpos)
        self.ball_radius = config.grasp_demo.ball_radius
        self.ball_body_name = ball_body_name

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
        self.controller.send_joint_command(q_arm)
        if gripper_ctrl is not None:
            self.controller.send_gripper_command(gripper_ctrl / self.gripper_range)

    def wait_for_settle(self, viewer, steps=20):
        for _ in range(steps):
            self.controller.step()
            self.controller.render(viewer)

    def move_to_qpos(self, q_target, viewer, gripper_ctrl=None, num_steps=100):
        if gripper_ctrl is None:
            gripper_ctrl = self.gripper_open_ctrl

        q_current = self.controller.get_joint_positions()
        trajectory = self.planner.quintic_interpolation(q_current, q_target, num_steps)

        for q in trajectory:
            self.set_ctrl(q, gripper_ctrl)
            self.controller.step()
            self.controller.render(viewer)

        self.wait_for_settle(viewer)

    def move_to_gripper_position(self, target_pos, viewer, gripper_ctrl=None, num_steps=100):
        if gripper_ctrl is None:
            gripper_ctrl = self.gripper_open_ctrl

        q_current = self.controller.get_joint_positions()
        q_full = self.data.qpos.copy()
        q_target, success = self.ik.solve_gripper_position(target_pos, q_init=q_current, q_full=q_full)

        if not success:
            print(f"  IK 未收敛，残差 > 1e-6")

        trajectory = self.planner.quintic_interpolation(q_current, q_target, num_steps)
        for q in trajectory:
            self.set_ctrl(q, gripper_ctrl)
            self.controller.step()
            self.controller.render(viewer)

        self.wait_for_settle(viewer)

        actual_center = self.get_gripper_center()
        error = np.linalg.norm(actual_center - target_pos)
        print(f"  IK目标: {target_pos}")
        print(f"  实际到达: {actual_center}")
        print(f"  误差: {error:.4f}m")
        return q_target

    def move_gripper_straight_down(self, target_z, viewer, gripper_ctrl=None, num_waypoints=15, steps_per_segment=40):
        if gripper_ctrl is None:
            gripper_ctrl = self.gripper_close_ctrl

        current = self.get_gripper_center()
        waypoints_z = np.linspace(current[2], target_z, num_waypoints)

        q_current = self.controller.get_joint_positions()
        for z in waypoints_z[1:]:
            wp = np.array([current[0], current[1], z])
            q_full = self.data.qpos.copy()
            q_target, _ = self.ik.solve_gripper_position(wp, q_init=q_current, q_full=q_full)
            traj = self.planner.quintic_interpolation(q_current, q_target, steps_per_segment)
            for q in traj:
                self.set_ctrl(q, gripper_ctrl)
                self.controller.step()
                self.controller.render(viewer)
            q_current = q_target

        self.wait_for_settle(viewer)

    def move_to_home(self, viewer):
        print("  移动到初始位姿（抬头）...")
        self.move_to_qpos(self.home_qpos, viewer, self.gripper_open_ctrl)
        print(f"  末端位置: {self.get_gripper_center()}")

    def move_to_observe(self, viewer):
        print("  移动到低头观察位姿...")
        self.move_to_qpos(self.observe_qpos, viewer, self.gripper_open_ctrl)
        print(f"  末端位置: {self.get_gripper_center()}")

    def grasp_object(self, viewer, num_steps=200, settle_steps=300):
        print("  闭合夹爪...")
        q_current = self.controller.get_joint_positions()
        close_target = self.gripper_close_ctrl

        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            gripper_ctrl = self.gripper_open_ctrl - (self.gripper_open_ctrl - close_target) * alpha
            self.set_ctrl(q_current, gripper_ctrl)
            for _ in range(4):
                self.controller.step()
            self.controller.render(viewer)

        finger_gap = 2 * close_target
        penetration = 0.04 - finger_gap
        print(f"  夹爪闭合到 ctrl={close_target:.3f} (指间距={finger_gap:.3f}m, 球直径=0.04m, 穿透={penetration:.3f}m)")
        print(f"  等待软接触变形稳定 {settle_steps} 步...")
        for _ in range(settle_steps):
            self.set_ctrl(q_current, close_target)
            for _ in range(4):
                self.controller.step()
            self.controller.render(viewer)

        gripper_center = self.get_gripper_center()
        ball_pos = self.get_ball_position()
        dist = np.linalg.norm(gripper_center - ball_pos)
        print(f"  夹爪中心: {gripper_center}")
        print(f"  小球位置: {ball_pos}")
        print(f"  夹爪中心-球心距: {dist:.4f}m")

    def lift_object(self, viewer, lift_height=0.10, num_steps=300):
        print(f"  缓慢提升 {lift_height}m ({num_steps}步)...")

        gripper_pos = self.get_gripper_center()
        lift_pos = gripper_pos.copy()
        lift_pos[2] += lift_height

        q_current = self.controller.get_joint_positions()
        q_full = self.data.qpos.copy()
        q_lift, success = self.ik.solve_gripper_position(lift_pos, q_init=q_current, q_full=q_full)

        if not success:
            print("  警告：提升位姿 IK 求解可能不精确")

        trajectory = self.planner.quintic_interpolation(q_current, q_lift, num_steps)
        for q in trajectory:
            self.set_ctrl(q, self.gripper_close_ctrl)
            self.controller.step()
            self.controller.render(viewer)

        self.wait_for_settle(viewer)

    def release_object(self, viewer, num_steps=200):
        print("  缓慢打开夹爪...")
        q_current = self.controller.get_joint_positions()

        for i in range(num_steps):
            alpha = i / (num_steps - 1)
            gripper_ctrl = self.gripper_close_ctrl + (self.gripper_open_ctrl - self.gripper_close_ctrl) * alpha
            self.set_ctrl(q_current, gripper_ctrl)
            for _ in range(5):
                self.controller.step()
            self.controller.render(viewer)

        self.wait_for_settle(viewer, 60)

    def run_demo(self):
        print("=" * 50)
        print("PiperSim 抓取演示")
        print("=" * 50)

        approach_height = config.grasp_demo.approach_height
        pre_grasp_height = config.grasp_demo.pre_grasp_height
        place_x, place_y = config.grasp_demo.place_position

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

            print("\n步骤 1: 移动到初始位姿...")
            self.move_to_home(viewer)

            print("\n步骤 2: 低头观察位姿...")
            self.move_to_observe(viewer)

            ball_pos = self.detect_ball_from_camera()
            if ball_pos is None:
                print("  视觉检测失败，使用真实位置作为后备")
                ball_pos = self.get_ball_position()
            print(f"  检测到小球位置: {ball_pos}")

            print(f"\n步骤 3: 快速移动到小球上方 {approach_height*100:.0f}cm...")
            approach_pos = ball_pos.copy()
            approach_pos[2] += approach_height
            self.move_to_gripper_position(approach_pos, viewer, self.gripper_open_ctrl, num_steps=100)

            print(f"\n步骤 4: 缓慢下降到预抓取位置（球上方 {pre_grasp_height*100:.0f}cm）...")
            pre_grasp_pos = ball_pos.copy()
            pre_grasp_pos[2] += pre_grasp_height
            self.move_to_gripper_position(pre_grasp_pos, viewer, self.gripper_open_ctrl, num_steps=150)

            print("\n步骤 5: 极慢接近抓取位置...")
            grasp_pos = ball_pos.copy()
            print(f"  小球中心: {ball_pos}")
            print(f"  目标抓取位置(夹爪中心): {grasp_pos}")
            self.move_to_gripper_position(grasp_pos, viewer, self.gripper_open_ctrl, num_steps=250)
            print(f"  夹爪中心: {self.get_gripper_center()}")
            print(f"  夹爪中心-球心偏差: {np.linalg.norm(self.get_gripper_center() - ball_pos):.4f}m")

            print("\n步骤 6: 执行抓取...")
            self.grasp_object(viewer)

            print("\n步骤 7: 微提测试握力（2cm，极慢）...")
            self.lift_object(viewer, lift_height=0.02, num_steps=500)

            print("\n步骤 8: 正式提升（8cm）...")
            self.lift_object(viewer, lift_height=0.08, num_steps=300)
            print(f"  提升后夹爪中心: {self.get_gripper_center()}")
            print(f"  提升后小球位置: {self.get_ball_position()}")

            print("\n步骤 9: 缓慢提升到安全高度...")
            safe_pos = self.get_gripper_center().copy()
            safe_pos[2] = max(safe_pos[2], 0.25)
            self.move_to_gripper_position(safe_pos, viewer, self.gripper_close_ctrl, num_steps=300)

            print("\n步骤 10: 极慢移动到放置位置上方...")
            place_above = np.array([place_x, place_y, safe_pos[2]])
            self.move_to_gripper_position(place_above, viewer, self.gripper_close_ctrl, num_steps=500)

            print("  横移后重新稳定握力...")
            q_release = self.controller.get_joint_positions()
            for _ in range(200):
                self.set_ctrl(q_release, self.gripper_close_ctrl)
                for _ in range(4):
                    self.controller.step()
                self.controller.render(viewer)

            print("\n步骤 11: 笛卡尔直线超慢下降...")
            self.move_gripper_straight_down(0.12, viewer, self.gripper_close_ctrl, num_waypoints=15, steps_per_segment=40)

            print("\n步骤 12: 缓慢释放物体...")
            self.release_object(viewer)

            print("\n步骤 13: 返回初始位姿...")
            self.move_to_home(viewer)

            print("\n" + "=" * 50)
            print("演示完成！")
            print("=" * 50)

            print("\n提示: 按 ESC 或关闭 viewer 窗口退出")
            while viewer.is_running():
                self.controller.step()
                self.controller.render(viewer)


def main():
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "scene.xml")

    if not os.path.exists(model_path):
        print(f"错误：模型文件不存在 {model_path}")
        return

    demo = GraspBallDemo(model_path)
    demo.run_demo()


if __name__ == "__main__":
    main()
