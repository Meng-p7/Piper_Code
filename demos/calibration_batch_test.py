import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from core.vision import Camera
from core.kinematics import ForwardKinematics, InverseKinematics
from core.controller import SimulationController
from core.trajectory import TrajectoryPlanner
from core.calibration import HandEyeCalibration
from utils import config

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "calibration_scene.xml",
)

SEEDS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "calibration", "calibration_seeds.npy",
)

CHECKER_COLS = 8
CHECKER_ROWS = 6

JOINT_LIMITS = np.array([
    [-2.618, 2.618],
    [0.0, 3.14],
    [-2.697, 0.0],
    [-1.832, 1.832],
    [-1.22, 1.22],
    [-3.14, 3.14],
])

HOME_QPOS = np.array([0.0, 1.57, -1.57, 0.0, 1.57, 0.0])
NUM_SAMPLES = 15
MAX_ATTEMPTS = 50
STEPS_PER_RAD = 300
CAMERA_NAME = "wrist_camera"


def detect_checkerboard(camera):
    import cv2
    image = camera.get_image()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, (CHECKER_COLS, CHECKER_ROWS), flags)
    if not ret:
        gray_eq = cv2.equalizeHist(gray)
        ret, corners = cv2.findChessboardCorners(gray_eq, (CHECKER_COLS, CHECKER_ROWS), flags)
    if not ret:
        return None

    cam_pos, cam_rot = camera.get_camera_pose()
    board_body_id = mujoco.mj_name2id(camera.model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board")
    board_pos = camera.data.xpos[board_body_id].copy()
    board_mat = camera.data.xmat[board_body_id].copy().reshape(3, 3)
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = cam_rot
    T_world_cam[:3, 3] = cam_pos
    T_world_board = np.eye(4)
    T_world_board[:3, :3] = board_mat
    T_world_board[:3, 3] = board_pos
    return np.linalg.inv(T_world_cam) @ T_world_board


def step_to(model, data, controller, planner, q_target):
    q_current = controller.get_joint_positions()
    max_diff = np.max(np.abs(q_target - q_current))
    num_steps = max(300, int(max_diff * STEPS_PER_RAD))
    trajectory = planner.quintic_interpolation(q_current, q_target, num_steps)
    for q in trajectory:
        controller.send_joint_command(q)
        controller.send_gripper_command(1.0)
        mujoco.mj_step(model, data)
    for _ in range(50):
        controller.send_joint_command(q_target)
        controller.send_gripper_command(1.0)
        mujoco.mj_step(model, data)


def run_single_calibration(model, data, seeds, run_id):
    import cv2
    planner = TrajectoryPlanner(
        max_velocity=config.robot.max_velocity,
        max_acceleration=config.robot.max_acceleration,
    )
    controller = SimulationController(model, data)
    controller.connect()
    camera = Camera(model, data, camera_name=CAMERA_NAME, width=640, height=480)
    calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")

    rng = np.random.RandomState(run_id * 1000 + 42)

    step_to(model, data, controller, planner, HOME_QPOS)

    attempt = 0
    collected = 0
    while collected < NUM_SAMPLES and attempt < MAX_ATTEMPTS:
        attempt += 1
        idx = rng.randint(len(seeds))
        seed = seeds[idx]
        noise = rng.randn(6) * np.array([0.002, 0.003, 0.003, 0.004, 0.002, 0.004])
        q_target = np.clip(seed + noise, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        step_to(model, data, controller, planner, q_target)

        T_cam_board = detect_checkerboard(camera)
        if T_cam_board is None:
            attempt += 1
            noise = rng.randn(6) * np.array([0.003, 0.005, 0.005, 0.006, 0.003, 0.006])
            q_retry = np.clip(seed + noise, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            step_to(model, data, controller, planner, q_retry)
            T_cam_board = detect_checkerboard(camera)

        if T_cam_board is None:
            continue

        ee_pos, ee_rot = controller.get_ee_pose()
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = ee_rot
        T_base_ee[:3, 3] = ee_pos
        calib.add_sample(T_base_ee, T_cam_board)
        collected += 1

    if collected < 3:
        controller.disconnect()
        return None, attempt, collected

    T_result, ax_xb_error = calib.calibrate()

    mujoco.mj_forward(model, data)
    ee_pos_gt, ee_rot_gt = controller.get_ee_pose()
    cam_pos_gt, cam_rot_gt = camera.get_camera_pose()

    T_world_ee = np.eye(4)
    T_world_ee[:3, :3] = ee_rot_gt
    T_world_ee[:3, 3] = ee_pos_gt
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = cam_rot_gt
    T_world_cam[:3, 3] = cam_pos_gt
    T_cam2gripper_gt = np.linalg.inv(T_world_ee) @ T_world_cam

    R_diff = T_result[:3, :3].T @ T_cam2gripper_gt[:3, :3]
    angle_err = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    trans_err = np.linalg.norm(T_result[:3, 3] - T_cam2gripper_gt[:3, 3]) * 1000

    controller.disconnect()
    return (angle_err, trans_err, ax_xb_error), attempt, collected


def main():
    print("=" * 60)
    print("批量标定测试 (5 次)")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    seeds = np.load(SEEDS_PATH)
    print(f"加载 {len(seeds)} 个种子")

    results = []
    for run_id in range(5):
        print(f"\n--- Run {run_id + 1} ---")
        t0 = time.time()
        result, attempts, collected = run_single_calibration(model, data, seeds, run_id)
        elapsed = time.time() - t0
        if result:
            r_err, t_err, residual = result
            print(f"  采集: {collected}/{attempts} attempts, 耗时: {elapsed:.1f}s")
            print(f"  旋转误差: {r_err:.4f} deg, 平移误差: {t_err:.4f} mm, 残差: {residual:.2e}")
            results.append((True, r_err, t_err, residual, collected, attempts, elapsed))
        else:
            print(f"  失败 (只采集 {collected}/{attempts}), 耗时: {elapsed:.1f}s")
            results.append((False, None, None, None, collected, attempts, elapsed))

    successes = sum(1 for r in results if r[0])
    print(f"\n{'=' * 60}")
    print(f"总结: {successes}/5 成功率")
    if successes > 0:
        r_errs = [r[1] for r in results if r[0]]
        t_errs = [r[2] for r in results if r[0]]
        residuals = [r[3] for r in results if r[0]]
        print(f"  旋转误差: {np.mean(r_errs):.4f} deg (max {np.max(r_errs):.4f})")
        print(f"  平移误差: {np.mean(t_errs):.4f} mm (max {np.max(t_errs):.4f})")
        print(f"  平均残差: {np.mean(residuals):.2e}")
    attempts_all = [r[5] for r in results]
    collected_all = [r[4] for r in results]
    times_all = [r[6] for r in results]
    print(f"  平均采集: {np.mean(collected_all):.1f}/{np.mean(attempts_all):.1f} attempts")
    print(f"  平均耗时: {np.mean(times_all):.1f}s")


if __name__ == "__main__":
    main()
