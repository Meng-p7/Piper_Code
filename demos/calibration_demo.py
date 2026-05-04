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

# 解决 Wayland 下 Qt 后端兼容性问题
os.environ["QT_QPA_PLATFORM"] = "xcb"

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
MAX_ATTEMPTS = 50
STEPS_PER_RAD = 400

SEED_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "calibration", "calibration_seeds.npy",
)

SEED_CONFIGS = None
if os.path.exists(SEED_FILE):
    SEED_CONFIGS = np.load(SEED_FILE)
    print(f"加载 {len(SEED_CONFIGS)} 个种子")
else:
    print(f"种子文件不存在: {SEED_FILE}")
    print("请先运行: python demos/calibration_seed_generator.py")
    sys.exit(1)

def detect_checkerboard(camera, enhance=False):
    image = camera.get_image()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, (CHECKER_COLS, CHECKER_ROWS), flags)

    if not ret and enhance:
        gray_eq = cv2.equalizeHist(gray)
        ret, corners = cv2.findChessboardCorners(gray_eq, (CHECKER_COLS, CHECKER_ROWS), flags)
        if ret:
            gray = gray_eq

    if not ret:
        return None, image

    cam_pos, cam_rot = camera.get_camera_pose()
    board_body_id = mujoco.mj_name2id(
        camera.model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board"
    )
    board_pos = camera.data.xpos[board_body_id].copy()
    board_mat = camera.data.xmat[board_body_id].copy().reshape(3, 3)
    T_world_cam = np.eye(4)
    T_world_cam[:3, :3] = cam_rot
    T_world_cam[:3, 3] = cam_pos
    T_world_board = np.eye(4)
    T_world_board[:3, :3] = board_mat
    T_world_board[:3, 3] = board_pos
    T_cam_board = np.linalg.inv(T_world_cam) @ T_world_board

    vis = image.copy()
    cv2.drawChessboardCorners(vis, (CHECKER_COLS, CHECKER_ROWS), corners, True)
    return T_cam_board, vis


def try_launch_viewer(model, data):
    try:
        import mujoco.viewer
        return mujoco.viewer.launch_passive(model, data)
    except Exception as e:
        print(f"[WARN] Viewer 启动失败: {e}")
        return None


def step_to(model, data, controller, planner, q_target, viewer=None, camera_preview_cb=None):
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
        if camera_preview_cb is not None:
            camera_preview_cb()

    for _ in range(50):
        controller.send_joint_command(q_target)
        controller.send_gripper_command(1.0)
        mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
        if camera_preview_cb is not None:
            camera_preview_cb()


def _update_cam_preview(image, step_info=""):
    if image is None:
        return
    display = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if step_info:
        cv2.putText(display, step_info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Wrist Camera", display)
    cv2.setWindowProperty("Wrist Camera", cv2.WND_PROP_TOPMOST, 1)  # 置顶窗口
    # 多调用几次 waitKey 确保事件队列被处理
    for _ in range(3):
        cv2.waitKey(1)


def run_calibration(model, data, viewer=None, show_cam=True, save_frames=False):
    planner = TrajectoryPlanner(
        max_velocity=config.robot.max_velocity,
        max_acceleration=config.robot.max_acceleration,
    )
    controller = SimulationController(model, data)
    controller.connect()

    camera = Camera(model, data, camera_name=CAMERA_NAME, width=640, height=480)
    calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")

    board_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board"
    )
    mujoco.mj_forward(model, data)
    board_pos = data.xpos[board_body_id].copy()
    rng = np.random.RandomState(42)

    step_to(model, data, controller, planner, HOME_QPOS, viewer)

    print(f"\n开始采集标定数据 (目标 {NUM_SAMPLES} 组, 最多 {MAX_ATTEMPTS} 次尝试)...")
    print("-" * 60)

    attempt = 0
    collected = 0

    def camera_preview_cb():
        if not show_cam:
            return
        cam_img = camera.get_image()
        if cam_img is not None:
            _update_cam_preview(cam_img, f"Samples {collected}/{NUM_SAMPLES}")

    while collected < NUM_SAMPLES and attempt < MAX_ATTEMPTS:
        attempt += 1
        idx = rng.randint(len(SEED_CONFIGS))
        seed_q = SEED_CONFIGS[idx]
        step_to(model, data, controller, planner, seed_q, viewer, camera_preview_cb=camera_preview_cb)

        for _ in range(15):
            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
            camera_preview_cb()

        T_cam_board, vis_image = detect_checkerboard(camera, enhance=True)

        if T_cam_board is None:
            attempt += 1
            noise = rng.randn(6) * np.array([0.003, 0.005, 0.005, 0.006, 0.003, 0.006])
            q_retry = np.clip(seed_q + noise, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            step_to(model, data, controller, planner, q_retry, viewer, camera_preview_cb=camera_preview_cb)
            for _ in range(15):
                mujoco.mj_step(model, data)
                if viewer is not None:
                    viewer.sync()
                camera_preview_cb()
            T_cam_board, vis_image = detect_checkerboard(camera, enhance=True)

        if show_cam and vis_image is not None:
            _update_cam_preview(vis_image, f"Samples {collected}/{NUM_SAMPLES}")

        if T_cam_board is None:
            continue

        # 获取末端位姿，构造 T_base_ee
        ee_pos, ee_rot = controller.get_ee_pose()
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = ee_rot
        T_base_ee[:3, 3] = ee_pos

        calib.add_sample(T_base_ee, T_cam_board)
        collected += 1

        if save_frames and vis_image is not None:
            frames_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "data", "calibration", "frames"
            )
            os.makedirs(frames_dir, exist_ok=True)
            frame_path = os.path.join(frames_dir, f"sample_{collected:02d}.png")
            cv2.imwrite(frame_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))

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


    print("标定结果")
    print(f"求解得到 T_cam2gripper (相机→末端):")
    print(f"  平移: [{T_result[0, 3]:.6f}, {T_result[1, 3]:.6f}, {T_result[2, 3]:.6f}]")
    print(f"  旋转:\n{T_result[:3, :3]}")

    R_diff = T_result[:3, :3].T @ T_cam2gripper_gt[:3, :3]
    angle_err = np.degrees(np.arccos(np.clip((np.trace(R_diff) - 1) / 2, -1, 1)))
    trans_err = np.linalg.norm(T_result[:3, 3] - T_cam2gripper_gt[:3, 3])

    print(f"精度评估 (与仿真真值对比):")
    print(f"  旋转误差:  {angle_err:.4f}°")
    print(f"  平移误差:  {trans_err * 1000:.4f} mm")


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
    parser.add_argument("--save-frames", action="store_true", help="保存成功样本的相机帧到 data/calibration/frames/")
    args = parser.parse_args()

    print("=" * 60)
    print("PiperSim 手眼标定仿真演示 (Eye-in-Hand)")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    viewer = None
    if not args.headless:
        viewer = try_launch_viewer(model, data)
        if viewer is None:
            print("[INFO] 无法启动 viewer，自动切换到无头模式")
        else:
            print("[INFO] Viewer 已启动 (MuJoCo 3D 窗口)")
        if args.save_frames:
            print("[INFO] 成功样本的相机帧将保存到 data/calibration/frames/")

    try:
        run_calibration(model, data, viewer, show_cam=not args.headless, save_frames=args.save_frames)

        if viewer is not None:
            print("演示完成！按 ESC 或关闭 viewer 窗口退出")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
    finally:
        if viewer is not None:
            viewer.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
