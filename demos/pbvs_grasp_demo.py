"""
PBVS 视觉伺服抓取演示

功能：机械臂通过视觉识别红色小球，使用 PBVS 闭环控制接近并抓取
运行方式：python demos/pbvs_grasp_demo.py
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
from core.visual_servo import PBVSController
from utils import config


class PBVSGraspDemo:
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

        self.link7_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.link8_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link8")
        self.ball_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, ball_body_name)

        self.gripper_open_ctrl = config.robot.gripper.open_ctrl
        self.gripper_close_ctrl = config.robot.gripper.close_ctrl
        self.gripper_range = config.robot.gripper.range
        self.home_qpos = np.array(config.robot.home_qpos)
        self.observe_qpos = np.array(config.robot.observe_qpos)
        self.ball_body_name = ball_body_name

        self.pbvs = PBVSController(
            model=self.model,
            data=mujoco.MjData(self.model),
            joint_names=arm_joint_names,
            ee_body_name=ik_ee_body_name,
            Kp=3.0,
            Ko=1.5,
            lambda_damping=0.05,
            pos_tol=1e-3,
            ori_tol=0.017,
        )

        print("PBVSGraspDemo initialized")

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

    def move_to_home(self, viewer):
        print("  移动到初始位姿...")
        self.move_to_qpos(self.home_qpos, viewer, self.gripper_open_ctrl)
        print(f"  末端位置: {self.get_gripper_center()}")

    def move_to_observe(self, viewer):
        print("  移动到观察位姿...")
        self.move_to_qpos(self.observe_qpos, viewer, self.gripper_open_ctrl)
        print(f"  末端位置: {self.get_gripper_center()}")

    def get_gripper_to_link6_offset(self):
        """计算当前位姿下夹爪中心到 link6 的偏移向量"""
        link6_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        link6_pos = self.data.xpos[link6_id].copy()
        gripper_center = self.get_gripper_center()
        return link6_pos - gripper_center  # link6 在夹爪中心前方

    def run_pbvs_approach(self, target_pos, viewer, target_ori=None, duration=5.0):
        """使用 PBVS 闭环接近目标位置

        控制 link6 到达目标位置 + 当前偏移补偿，使得夹爪中心对准目标。
        先用 IK 求解合理初始位姿避免关节翻转，再用 PBVS 微调。
        """
        print(f"  PBVS 接近目标: {target_pos}")

        # 关节限位（从模型中读取）
        joint_limits = [
            (-2.618, 2.618),   # joint1
            (0.0, 3.14),       # joint2
            (-2.697, 0.0),     # joint3
            (-1.832, 1.832),   # joint4
            (-1.22, 1.22),     # joint5
            (-3.14, 3.14),     # joint6
        ]

        # 先用 IK 求解一个合理的中间位姿，避免 PBVS 从远处直接计算导致关节翻转
        q_current = self.controller.get_joint_positions()
        q_full = self.data.qpos.copy()
        q_ik, success = self.ik.solve_gripper_position(target_pos, q_init=q_current, q_full=q_full)

        if success:
            # 用轨迹插值移动到 IK 解附近（避免瞬移）
            trajectory = self.planner.quintic_interpolation(q_current, q_ik, num_steps=80)
            for q in trajectory:
                self.set_ctrl(q, self.gripper_open_ctrl)
                self.controller.step()
                self.controller.render(viewer)
            print(f"  IK 预定位完成，关节角: {np.round(q_ik, 3)}")
        else:
            print("  IK 预定位失败，直接使用 PBVS")

        # 先同步一次当前状态
        self.pbvs.data.qpos[:] = self.data.qpos.copy()
        mujoco.mj_forward(self.model, self.pbvs.data)

        dt = self.model.opt.timestep
        max_steps = int(duration / dt)
        pos_errors = []
        ori_errors = []
        converged = False

        for step in range(max_steps):
            q = self.controller.get_joint_positions()

            # 同步到 PBVS 内部 data
            for i, adr in enumerate(self.pbvs.joint_qpos_adrs):
                self.pbvs.data.qpos[adr] = q[i]
            mujoco.mj_forward(self.model, self.pbvs.data)

            # 计算当前偏移补偿：让 link6 到达 (目标 + 偏移)，从而使夹爪中心对准目标
            offset = self.get_gripper_to_link6_offset()
            link6_target = target_pos + offset

            pos_error, ori_error = self.pbvs.compute_error(link6_target, target_ori)
            pos_errors.append(np.linalg.norm(pos_error))
            ori_errors.append(np.linalg.norm(ori_error))

            # 直接检查夹爪中心是否到位（避免目标漂移导致不收敛）
            gripper_err = np.linalg.norm(self.get_gripper_center() - target_pos)
            if gripper_err < 5e-3:
                converged = True
                print(f"  PBVS 收敛于第 {step} 步，"
                      f"link6 误差={pos_errors[-1]:.6f}m，"
                      f"夹爪中心-目标偏差={gripper_err:.6f}m")
                break

            qvel = self.pbvs.compute_joint_velocity(link6_target, target_ori, qpos=q)

            # 直接积分更新关节位置并写入仿真
            q_new = q + qvel * dt

            # 应用关节限位约束
            for i in range(6):
                q_new[i] = np.clip(q_new[i], joint_limits[i][0], joint_limits[i][1])

            for i, adr in enumerate(self.controller.joint_qpos_adrs):
                self.data.qpos[adr] = q_new[i]
            mujoco.mj_forward(self.model, self.data)

            # 同步 gripper 控制
            self.controller.send_joint_command(q_new)
            self.controller.render(viewer)

        if not converged:
            gripper_err = np.linalg.norm(self.get_gripper_center() - target_pos)
            print(f"  PBVS 未收敛，link6 误差={pos_errors[-1]:.6f}m，"
                  f"夹爪中心-目标偏差={gripper_err:.6f}m")

        return {
            "converged": converged,
            "pos_errors": np.array(pos_errors),
            "ori_errors": np.array(ori_errors),
            "final_pos_error": pos_errors[-1] if pos_errors else float("inf"),
            "final_ori_error": ori_errors[-1] if ori_errors else float("inf"),
        }

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
        print(f"  提升 {lift_height}m...")

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
        print("  打开夹爪...")
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
        print("PBVS 视觉伺服抓取演示")
        print("=" * 50)

        approach_height = config.grasp_demo.approach_height
        pre_grasp_height = config.grasp_demo.pre_grasp_height
        place_x, place_y = config.grasp_demo.place_position

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:

            print("\n步骤 1: 移动到初始位姿...")
            self.move_to_home(viewer)

            print("\n步骤 2: 移动到观察位姿...")
            self.move_to_observe(viewer)

            ball_pos = self.detect_ball_from_camera()
            if ball_pos is None:
                print("  视觉检测失败，使用真实位置")
                ball_pos = self.get_ball_position()
            print(f"  目标小球位置: {ball_pos}")

            print(f"\n步骤 3: PBVS 接近小球上方 {approach_height*100:.0f}cm...")
            approach_pos = ball_pos.copy()
            approach_pos[2] += approach_height
            self.run_pbvs_approach(approach_pos, viewer, duration=4.0)

            print(f"\n步骤 4: PBVS 下降到预抓取位置（球上方 {pre_grasp_height*100:.0f}cm）...")
            pre_grasp_pos = ball_pos.copy()
            pre_grasp_pos[2] += pre_grasp_height
            self.run_pbvs_approach(pre_grasp_pos, viewer, duration=3.0)

            print("\n步骤 5: PBVS 精确接近抓取位置...")
            grasp_pos = ball_pos.copy()
            self.run_pbvs_approach(grasp_pos, viewer, duration=4.0)
            print(f"  夹爪中心: {self.get_gripper_center()}")
            print(f"  夹爪中心-球心偏差: {np.linalg.norm(self.get_gripper_center() - ball_pos):.4f}m")

            print("\n步骤 6: 执行抓取...")
            self.grasp_object(viewer)

            print("\n步骤 7: 提升...")
            self.lift_object(viewer, lift_height=0.10, num_steps=300)

            print("\n步骤 8: 移动到放置位置上方...")
            safe_pos = self.get_gripper_center().copy()
            safe_pos[2] = max(safe_pos[2], 0.25)
            self.run_pbvs_approach(safe_pos, viewer, duration=3.0)

            print("\n步骤 9: 移动到放置位置...")
            place_pos = np.array([place_x, place_y, 0.12])
            self.run_pbvs_approach(place_pos, viewer, duration=4.0)

            print("\n步骤 10: 释放物体...")
            self.release_object(viewer)

            print("\n步骤 11: 返回初始位姿...")
            self.move_to_home(viewer)

            print("\n" + "=" * 50)
            print("PBVS 演示完成！")
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

    demo = PBVSGraspDemo(model_path)
    demo.run_demo()


if __name__ == "__main__":
    main()
