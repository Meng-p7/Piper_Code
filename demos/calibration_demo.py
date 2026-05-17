"""
PiperSim 手眼标定仿真演示（Eye-in-Hand，ChArUco 9x14 标定板）

流程与 real_calibration_demo.py 完全对齐

运行方式：
  conda activate mujoco && python demos/calibration_demo.py
  无头模式：python demos/calibration_demo.py --headless
"""

import os
import sys
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from core.kinematics import InverseKinematics
from core.trajectory import TrajectoryPlanner
from core.vision import Camera
from core.controller import SimulationController
from core.calibration import HandEyeCalibration
from utils import config

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "calibration_scene.xml",
)

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "calibration", "sim_captures"
)

JOINT_LIMITS = np.array([
    [-2.618, 2.618],
    [0.0, 3.14],
    [-2.697, 0.0],
    [-1.832, 1.832],
    [-1.22, 1.22],
    [-3.14, 3.14],
])

HOME_QPOS = np.array(config.robot.home_qpos)
CAMERA_NAME = "wrist_camera"
IK_BODY_NAME = "link7"
STEPS_PER_RAD = 400

SQUARE_POSES = [
    (0.150, -0.050, 0.280),
    (0.150,  0.050, 0.280),
    (0.250,  0.050, 0.280),
    (0.250, -0.050, 0.280),
    (0.150, -0.050, 0.280),
]

EXTRA_POSES = [
    (0.200,  0.000, 0.280),
    (0.200, -0.030, 0.280),
    (0.200,  0.030, 0.280),
]


def get_T_cam_board_from_mujoco(model, data, camera):
    board_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board")
    T_world_board = np.eye(4)
    T_world_board[:3, :3] = data.xmat[board_id].reshape(3, 3).copy()
    T_world_board[:3, 3] = data.xpos[board_id].copy()

    cam_pos, cam_rot = camera.get_camera_pose()
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = cam_rot
    T_world_cam[:3, 3] = cam_pos

    return np.linalg.inv(T_world_cam) @ T_world_board


def try_launch_viewer(model, data):
    try:
        import mujoco.viewer
        return mujoco.viewer.launch_passive(model, data)
    except Exception:
        return None


def step_to(model, data, controller, planner, q_target, viewer=None):
    q_current = controller.get_joint_positions()
    max_diff = np.max(np.abs(q_target - q_current))
    num_steps = max(300, int(max_diff * STEPS_PER_RAD))
    trajectory = planner.quintic_interpolation(q_current, q_target, num_steps)
    for q in trajectory:
        controller.send_joint_command(q)
        controller.send_gripper_command(1.0)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
    for _ in range(50):
        controller.send_joint_command(q_target)
        controller.send_gripper_command(1.0)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()


def ik_solve(ik, target_pos, q_init):
    return ik.solve_position(target_pos, q_init=q_init, max_retries=10)


def run_calibration(model, data, viewer=None, show_cam=True):
    planner = TrajectoryPlanner(
        max_velocity=config.robot.max_velocity,
        max_acceleration=config.robot.max_acceleration,
    )
    controller = SimulationController(model, data)
    controller.connect()

    camera = Camera(model, data, camera_name=CAMERA_NAME, width=640, height=480)

    ik = InverseKinematics(
        model, IK_BODY_NAME,
        joint_names=config.robot.joint_names,
        gripper_bodies=config.robot.gripper_bodies,
    )

    calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")

    step_to(model, data, controller, planner, HOME_QPOS, viewer)
    mujoco.mj_forward(model, data)

    os.makedirs(SAVE_DIR, exist_ok=True)

    all_poses = list(SQUARE_POSES) + list(EXTRA_POSES)
    collected = 0
    target = len(all_poses)

    print(f"\n采集 {target} 个姿态")
    print(f"{'序号':>4}  {'X':>7}  {'Y':>7}  {'Z':>7}  {'状态'}")
    print("-" * 45)

    q_prev = HOME_QPOS.copy()

    for i, pos in enumerate(all_poses):
        if collected >= target:
            break

        q_sol, ok = ik_solve(ik, np.array(pos), q_init=q_prev)
        if not ok:
            print(f"  {i+1:>2}   IK 失败")
            continue

        q_sol = np.clip(q_sol, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        step_to(model, data, controller, planner, q_sol, viewer)

        for _ in range(15):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

        ee_pos, ee_rot = controller.get_ee_pose()
        mujoco.mj_forward(model, data)

        x_mm = ee_pos[0] * 1000
        y_mm = ee_pos[1] * 1000
        z_mm = ee_pos[2] * 1000

        img = camera.get_image()

        collected += 1

        raw_path = os.path.join(SAVE_DIR, f"pose_{collected:02d}_raw.png")
        cv2.imwrite(raw_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        T_cam_board = get_T_cam_board_from_mujoco(model, data, camera)
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = ee_rot
        T_base_ee[:3, 3] = ee_pos
        calib.add_sample(T_base_ee, T_cam_board)

        det_path = os.path.join(SAVE_DIR, f"pose_{collected:02d}_det.png")
        cv2.imwrite(det_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        print(f"  {collected:>2}   {x_mm:>6.1f}  {y_mm:>6.1f}  {z_mm:>6.1f}  OK")

        if show_cam:
            display = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.putText(display, f"{collected}/{target} OK", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("Wrist Camera", display)
            cv2.setWindowProperty("Wrist Camera", cv2.WND_PROP_TOPMOST, 1)
            for _ in range(3):
                cv2.waitKey(1)

        q_prev = q_sol.copy()

    if show_cam:
        cv2.destroyAllWindows()

    if collected < 3:
        print(f"\n只采集 {collected} 组，不足")
        controller.disconnect()
        return

    T_result, error = calib.calibrate()

    mujoco.mj_forward(model, data)
    ee_pos_gt, ee_rot_gt = controller.get_ee_pose()
    cam_pos_gt, cam_rot_gt = camera.get_camera_pose()

    T_world_ee = np.eye(4)
    T_world_ee[:3, :3] = ee_rot_gt
    T_world_ee[:3, 3] = ee_pos_gt
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = cam_rot_gt
    T_world_cam[:3, 3] = cam_pos_gt
    T_gt = np.linalg.inv(T_world_ee) @ T_world_cam

    R_diff = T_result[:3, :3].T @ T_gt[:3, :3]
    angle_err = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    trans_err = np.linalg.norm(T_result[:3, 3] - T_gt[:3, 3])

    print(f"\n{'='*45}")
    print(f"采集: {collected}/{target} 组")
    print(f"平移: [{T_result[0,3]:.4f}, {T_result[1,3]:.4f}, {T_result[2,3]:.4f}] m")
    print(f"旋转误差: {angle_err:.3f} deg")
    print(f"平移误差: {trans_err*1000:.3f} mm")
    print(f"AX=XB:    {error:.6f}")
    print(f"照片: {SAVE_DIR}")
    print(f"{'='*45}")

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "calibration", "eye_in_hand_result.npy",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    calib.save_result(save_path)

    controller.disconnect()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    viewer = None
    if not args.headless:
        viewer = try_launch_viewer(model, data)

    try:
        run_calibration(model, data, viewer, show_cam=not args.headless)
        if viewer is not None:
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    finally:
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
