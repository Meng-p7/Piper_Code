"""
PiperSim 手眼标定仿真演示（Eye-in-Hand，棋盘格标定板）

功能：
  1. 使用专用标定场景 calibration_scene.xml，末端安装 wrist_camera
  2. 机械臂移动到不同姿态（多种构型覆盖）
  3. 相机拍摄棋盘格，findChessboardCorners + solvePnP 获取 T_cam_board
  4. 记录 T_base_ee
  5. 调用 calibrate() 求解 T_cam2gripper
  6. 输出精度报告 + 实时相机画面预览

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
from core.trajectory import TrajectoryPlanner
from core.vision import Camera
from core.controller import SimulationController
from core.calibration import HandEyeCalibration
from utils import config

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "calibration_scene.xml",
)

CHECKER_COLS = 8
CHECKER_ROWS = 6
SQUARE_SIZE = 0.030

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
NUM_SAMPLES = 15
MAX_ATTEMPTS = 300
STEPS_PER_RAD = 400

SEED_CONFIGS = [
    # 核心高成功率种子 (>=8/10 with noise) — 主要采样池
    np.array([0.0, 1.1, -1.0, 0.0, 1.0, 3.14]),
    np.array([0.0, 1.2, -1.0, 0.0, 1.0, 3.14]),
    np.array([0.0, 1.2, -1.1, 0.0, 0.9, 3.14]),
    np.array([0.0, 1.3, -0.8, 0.0, 0.6, 3.14]),
    np.array([-0.2, 1.1, -1.1, 0.0, 1.0, 3.14]),
    # 扩展高成功率区域
    np.array([0.0, 1.15, -1.05, 0.0, 0.95, 3.14]),
    np.array([0.0, 1.25, -0.95, 0.0, 0.85, 3.14]),
    np.array([0.0, 1.35, -0.85, 0.0, 0.55, 3.14]),
    np.array([-0.2, 1.2, -1.0, 0.0, 0.9, 3.14]),
    np.array([-0.2, 1.0, -1.05, 0.0, 1.05, 3.14]),
    # 中等成功率但提供空间多样性 — 降低采样权重
    np.array([0.0, 1.0, -0.9, 0.0, 1.1, 3.14]),
    np.array([0.0, 1.05, -1.05, 0.0, 0.95, 3.14]),
    np.array([-0.3, 1.1, -1.0, 0.0, 1.1, 3.14]),
    np.array([0.3, 1.1, -1.0, 0.0, 1.0, 3.14]),
    np.array([0.0, 0.95, -0.85, 0.0, 1.15, 3.14]),
    np.array([0.0, 1.4, -1.1, 0.0, 0.7, 3.14]),
    np.array([-0.15, 1.15, -1.1, 0.0, 0.9, 3.14]),
    np.array([0.15, 1.15, -1.0, 0.0, 1.0, 3.14]),
    np.array([0.0, 1.1, -0.8, 0.0, 1.3, 3.14]),
    np.array([0.0, 1.3, -1.15, 0.0, 0.8, 3.14]),
]

_OCV2MJ = np.diag([1.0, -1.0, -1.0])


def build_object_points():
    pts = np.zeros((CHECKER_COLS * CHECKER_ROWS, 3), np.float32)
    for i in range(CHECKER_ROWS):
        for j in range(CHECKER_COLS):
            pts[i * CHECKER_COLS + j] = [j * SQUARE_SIZE, i * SQUARE_SIZE, 0]
    return pts


def detect_checkerboard(camera, obj_points, enhance=False):
    image = camera.get_image()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, (CHECKER_COLS, CHECKER_ROWS), flags)

    # 若首次失败且启用增强，尝试对比度增强
    if not ret and enhance:
        gray_eq = cv2.equalizeHist(gray)
        ret, corners = cv2.findChessboardCorners(gray_eq, (CHECKER_COLS, CHECKER_ROWS), flags)
        if ret:
            gray = gray_eq  # 用增强后的图做 subpix

    if not ret:
        return None, image

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    fx, fy, cx, cy = camera.get_camera_params()
    camera_matrix = np.array([[fx, 0, cx],
                               [0, fy, cy],
                               [0, 0, 1]], dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        obj_points, corners, camera_matrix, np.zeros(4),
    )
    if not success:
        return None, image

    R_ocv, _ = cv2.Rodrigues(rvec)
    R_mj = _OCV2MJ @ R_ocv @ _OCV2MJ
    t_mj = _OCV2MJ @ tvec.flatten()

    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R_mj
    T_cam_board[:3, 3] = t_mj

    vis = image.copy()
    cv2.drawChessboardCorners(vis, (CHECKER_COLS, CHECKER_ROWS), corners, True)
    return T_cam_board, vis


def random_joint_config(rng):
    # 50% 概率从高成功率种子(前10个)采样，50% 从全部种子采样
    # 平衡成功率与空间多样性
    if rng.rand() < 0.5:
        seed = SEED_CONFIGS[rng.randint(10)]
    else:
        seed = SEED_CONFIGS[rng.randint(len(SEED_CONFIGS))]
    noise = rng.randn(6) * np.array([0.04, 0.06, 0.06, 0.08, 0.05, 0.06])
    q = seed + noise
    return np.clip(q, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])


def try_launch_viewer(model, data):
    try:
        import mujoco.viewer
        return mujoco.viewer.launch_passive(model, data)
    except Exception as e:
        print(f"[WARN] Viewer 启动失败: {e}")
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


def _update_cam_preview(image, step_info=""):
    if image is None:
        return
    display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if step_info:
        cv2.putText(display, step_info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Wrist Camera", display)
    cv2.waitKey(1)


def run_calibration(model, data, viewer=None, show_cam=True):
    planner = TrajectoryPlanner(
        max_velocity=config.robot.max_velocity,
        max_acceleration=config.robot.max_acceleration,
    )
    controller = SimulationController(model, data)
    controller.connect()

    camera = Camera(model, data, camera_name=CAMERA_NAME, width=640, height=480)
    calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")
    obj_points = build_object_points()

    board_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board"
    )
    mujoco.mj_forward(model, data)
    board_pos = data.xpos[board_body_id].copy()
    print(f"\n标定板中心位置: {board_pos}")

    rng = np.random.RandomState(42)

    step_to(model, data, controller, planner, HOME_QPOS, viewer)

    print(f"\n开始采集标定数据 (目标 {NUM_SAMPLES} 组, 最多 {MAX_ATTEMPTS} 次尝试)...")
    print("-" * 60)

    attempt = 0
    collected = 0

    while collected < NUM_SAMPLES and attempt < MAX_ATTEMPTS:
        attempt += 1
        q_target = random_joint_config(rng)
        step_to(model, data, controller, planner, q_target, viewer)

        for _ in range(15):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()

        T_cam_board, vis_image = detect_checkerboard(camera, obj_points)

        if show_cam:
            status = f"Attempt {attempt} | Samples {collected}/{NUM_SAMPLES}"
            _update_cam_preview(vis_image, status)

        # 局部修复重试: 若噪声导致失败，回退到纯净种子重试
        if T_cam_board is None:
            # 找到最接近当前 q_target 的种子配置
            best_idx = 0
            best_dist = float('inf')
            for idx, seed in enumerate(SEED_CONFIGS):
                d = np.sum((q_target - seed) ** 2)
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            seed_q = SEED_CONFIGS[best_idx]

            # 重试 1: 回到纯净种子 (带图像增强)
            step_to(model, data, controller, planner, seed_q, viewer)
            for _ in range(15):
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
            T_cam_board, vis_image = detect_checkerboard(camera, obj_points, enhance=True)
            attempt += 1  # 计为额外一次尝试

            if show_cam and T_cam_board is not None:
                status = f"Attempt {attempt} | Samples {collected}/{NUM_SAMPLES} (retry seed)"
                _update_cam_preview(vis_image, status)

            # 重试 2: 若种子也失败，尝试更小噪声的邻域
            if T_cam_board is None and attempt < MAX_ATTEMPTS:
                small_noise = rng.randn(6) * np.array([0.02, 0.03, 0.03, 0.04, 0.02, 0.03])
                q_retry = seed_q + small_noise
                q_retry = np.clip(q_retry, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
                step_to(model, data, controller, planner, q_retry, viewer)
                for _ in range(15):
                    mujoco.mj_step(model, data)
                    if viewer is not None:
                        viewer.sync()
                T_cam_board, vis_image = detect_checkerboard(camera, obj_points, enhance=True)
                attempt += 1

                if show_cam and T_cam_board is not None:
                    status = f"Attempt {attempt} | Samples {collected}/{NUM_SAMPLES} (retry small)"
                    _update_cam_preview(vis_image, status)

        if T_cam_board is None:
            continue

        ee_pos, ee_rot = controller.get_ee_pose()
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = ee_rot
        T_base_ee[:3, 3] = ee_pos

        calib.add_sample(T_base_ee, T_cam_board)
        collected += 1

        cam_pos, _ = camera.get_camera_pose()
        dist = np.linalg.norm(cam_pos - board_pos)
        j_cfg = controller.get_joint_positions()
        print(f"  [{collected:2d}/{NUM_SAMPLES}] (尝试 {attempt:3d}) "
              f"j1={j_cfg[0]:+.2f} "
              f"EE=[{ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}]  "
              f"cam_dist={dist:.3f}m")

    print("-" * 60)

    if show_cam:
        cv2.destroyAllWindows()

    if collected < 3:
        print(f"错误: 只采集到 {collected} 组有效样本 (至少需要 3 组)")
        print("请检查: 相机是否对准标定板 / 增加 MAX_ATTEMPTS / 调整相机安装姿态")
        controller.disconnect()
        return None, None, None, None

    print(f"采集完成: {collected} 组 / {attempt} 次尝试")

    print("\n执行手眼标定 (Park 方法)...")
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

    print("\n" + "=" * 60)
    print("标定结果")
    print("=" * 60)

    print(f"\n求解得到 T_cam2gripper (相机→末端):")
    print(f"  平移: [{T_result[0, 3]:.6f}, {T_result[1, 3]:.6f}, {T_result[2, 3]:.6f}]")
    print(f"  旋转:\n{T_result[:3, :3]}")

    print(f"\n仿真真值 T_cam2gripper:")
    print(f"  平移: [{T_cam2gripper_gt[0, 3]:.6f}, {T_cam2gripper_gt[1, 3]:.6f}, {T_cam2gripper_gt[2, 3]:.6f}]")

    R_diff = T_result[:3, :3].T @ T_cam2gripper_gt[:3, :3]
    angle_err = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    trans_err = np.linalg.norm(T_result[:3, 3] - T_cam2gripper_gt[:3, 3])

    print(f"\n精度评估 (与仿真真值对比):")
    print(f"  旋转误差:  {angle_err:.4f}°")
    print(f"  平移误差:  {trans_err * 1000:.4f} mm")
    print(f"  AX=XB 平均残差: {ax_xb_error:.2e}")

    if angle_err < 0.01 and trans_err < 1e-5:
        print(f"  ★★★ 精确")
    elif angle_err < 1.0 and trans_err < 1e-3:
        print(f"  ★★☆ 良好")
    elif angle_err < 5.0 and trans_err < 5e-3:
        print(f"  ★☆☆ 一般")
    else:
        print(f"  ☆☆☆ 较差")

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "calibration", "eye_in_hand_result.npy",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    calib.save_result(save_path)

    controller.disconnect()
    return T_result, angle_err, trans_err, ax_xb_error


def main():
    parser = argparse.ArgumentParser(description="PiperSim 手眼标定演示")
    parser.add_argument("--headless", action="store_true", help="无头模式运行")
    args = parser.parse_args()

    print("=" * 60)
    print("PiperSim 手眼标定仿真演示 (Eye-in-Hand)")
    print("标定板: 棋盘格 8x6 内角点, 方格边长 30mm")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    viewer = None
    if not args.headless:
        viewer = try_launch_viewer(model, data)
        if viewer is None:
            print("[INFO] 无法启动 viewer，自动切换到无头模式")
        else:
            print("[INFO] Viewer 已启动 (MuJoCo 3D 窗口)")
        print("[INFO] 腕部相机预览窗口将实时显示")

    try:
        run_calibration(model, data, viewer, show_cam=not args.headless)

        if viewer is not None:
            print("\n" + "=" * 60)
            print("演示完成！按 ESC 或关闭 viewer 窗口退出")
            print("=" * 60)
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    finally:
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
