"""
真机手眼标定 Demo（Eye-in-Hand，ChArUco 9x14 标定板）

使用方式：
  conda activate mujoco && python demos/real_calibration_demo.py
"""

import os
import sys
import time
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pyrealsense2 as rs
from piper_sdk import C_PiperInterface_V2
from core.calibration import HandEyeCalibration

CHARUCO_SQUARES_X = 9
CHARUCO_SQUARES_Y = 14
CHARUCO_SQUARE_LEN = 0.020
CHARUCO_MARKER_LEN = 0.015
CHARUCO_DICT_TYPE = cv2.aruco.DICT_5X5_100

SAVE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "calibration", "captures"
)

SQUARE_POSES = [
    (200000, -30000, 240000, -179900,     0, -179900),
    (200000,      0, 240000, -169900,     0, -179900),
    (200000,  30000, 240000, -179900, 10000, -179900),
    (255000,  30000, 240000, -179900,     0, -169900),
    (310000,  30000, 240000, -169900,     0, -169900),
    (310000,      0, 240000, -179900, -10000, -179900),
    (310000, -30000, 240000, -179900,     0,  170100),
    (255000, -30000, 240000, -169900, 10000, -179900),
]

PIPELINE = None
CAM_STOP = False


def start_camera():
    global PIPELINE
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    PIPELINE = pipeline
    time.sleep(1)
    # 恢复自动曝光，避免之前调试设置的影响
    device = pipeline.get_active_profile().get_device()
    rgb_sensor = device.first_color_sensor()
    rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)
    rgb_sensor.set_option(rs.option.enable_auto_white_balance, 1)
    print("相机已启动（自动曝光）")


def grab_frame():
    if PIPELINE is None:
        return None
    frames = PIPELINE.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None
    return np.asanyarray(color_frame.get_data())


def show_camera(image, info=""):
    global CAM_STOP
    display = image.copy()
    if info:
        cv2.putText(display, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Calibration", display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        CAM_STOP = True


def get_camera_intrinsics():
    frames = PIPELINE.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError("无法获取相机帧")
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)
    return camera_matrix, dist_coeffs


def create_charuco_board():
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LEN, CHARUCO_MARKER_LEN, dictionary,
    )
    return dictionary, board


def detect_board_pose(image, camera_matrix, dist_coeffs, dictionary, board):
    """用 ArUco 标记 solvePnP+朝向校验排除翻转解"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    ad = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = ad.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        return None, image

    obj_pts_3d = board.getObjPoints()
    board_ids = board.getIds().flatten().tolist()
    obj_points = []
    img_points = []
    for i in range(len(ids)):
        id_val = ids[i][0]
        if id_val in board_ids:
            idx = board_ids.index(id_val)
            obj_points.append(obj_pts_3d[idx])
            img_points.append(corners[i].reshape(4, 2))

    all_obj = np.vstack(obj_points).astype(np.float64)
    all_img = np.vstack(img_points).astype(np.float64)

    ok, rvec, tvec = cv2.solvePnP(all_obj, all_img, camera_matrix, dist_coeffs,
                                    flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, image

    R, _ = cv2.Rodrigues(rvec)

    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R
    T_cam_board[:3, 3] = tvec.flatten()

    print(f"  ArUco标记: {len(ids)}个", end="")
    vis = image.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
    return T_cam_board, vis


def piper_pose_to_matrix(x, y, z, rx_deg, ry_deg, rz_deg):
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    R = np.array([
        [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
        [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
        [-sy,   sx*cy,             cx*cy],
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def connect_and_enable(can_name="can0", timeout=10.0):
    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    time.sleep(0.2)
    t0 = time.time()
    while not piper.EnablePiper():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"机械臂使能超时 ({timeout}s)")
        time.sleep(0.01)
    print("机械臂使能成功")
    return piper


def get_ee_pose(piper):
    ep = piper.GetArmEndPoseMsgs().end_pose
    return ep.X_axis, ep.Y_axis, ep.Z_axis, ep.RX_axis, ep.RY_axis, ep.RZ_axis


def move_to_pose(piper, pose, mode=0x00, speed=100, wait_time=2.0):
    piper.MotionCtrl_2(0x01, mode, speed, 0x00)
    piper.EndPoseCtrl(*pose)
    t0 = time.time()
    while time.time() - t0 < wait_time:
        if CAM_STOP:
            return
        img = grab_frame()
        if img is not None:
            show_camera(img)
        time.sleep(0.03)


def main():
    global CAM_STOP

    start_camera()
    camera_matrix, dist_coeffs = get_camera_intrinsics()
    dictionary, board = create_charuco_board()
    piper = connect_and_enable("can0")

    os.makedirs(SAVE_DIR, exist_ok=True)

    calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")

    print(f"\n采集 {len(SQUARE_POSES)} 个姿态")
    input("准备好后按回车...")

    for i, pose in enumerate(SQUARE_POSES):
        if CAM_STOP:
            break

        mode = 0x00 if i == 0 else 0x02
        print(f"\n移动到点 {i+1}/{len(SQUARE_POSES)}...")
        move_to_pose(piper, pose, mode=mode, speed=100, wait_time=2.0)

        if CAM_STOP:
            break

        x, y, z, rx, ry, rz = get_ee_pose(piper)
        print(f"  末端位置: X={x}, Y={y}, Z={z}, RX={rx}, RY={ry}, RZ={rz}")

        image = grab_frame()
        if image is None:
            print("  无图像")
            continue

        raw_path = os.path.join(SAVE_DIR, f"pose_{i+1:02d}_raw.png")
        cv2.imwrite(raw_path, image)

        T_cam_board, vis = detect_board_pose(image, camera_matrix, dist_coeffs, dictionary, board)
        if T_cam_board is None:
            print("  未检测到标定板")
            show_camera(image, f"{i+1}/{len(SQUARE_POSES)} 未检测到")
            continue

        x_m, y_m, z_m = x/1000000, y/1000000, z/1000000
        rx_d, ry_d, rz_d = rx/1000, ry/1000, rz/1000
        T_base_ee = piper_pose_to_matrix(x_m, y_m, z_m, rx_d, ry_d, rz_d)
        calib.add_sample(T_base_ee, T_cam_board)
        print(f"  T_base_ee pos: [{x_m:.3f}, {y_m:.3f}, {z_m:.3f}]")
        print(f"  T_cam_board pos: [{T_cam_board[0,3]:.3f}, {T_cam_board[1,3]:.3f}, {T_cam_board[2,3]:.3f}]")

        det_path = os.path.join(SAVE_DIR, f"pose_{i+1:02d}_det.png")
        cv2.imwrite(det_path, vis)
        show_camera(vis, f"{i+1}/{len(SQUARE_POSES)} OK")
        print("  OK")

    cv2.destroyAllWindows()

    if len(calib.robot_poses) < 3:
        print(f"\n只采集 {len(calib.robot_poses)} 组，不足")
        PIPELINE.stop()
        return

    print(f"\n{'='*45}")
    print(f"采集 {len(calib.robot_poses)} 组数据:")
    for i in range(len(calib.robot_poses)):
        rp = calib.robot_poses[i]
        cp = calib.camera_poses[i]
        print(f"  [{i}] robot=({rp[0,3]:.3f}, {rp[1,3]:.3f}, {rp[2,3]:.3f})  "
              f"cam_board=({cp[0,3]:.3f}, {cp[1,3]:.3f}, {cp[2,3]:.3f})")
    print(f"{'='*45}")

    # 对齐相机姿态: 每对相邻姿态的旋转角与机械臂应一致
    print("  对齐相机姿态朝向...")
    flips = [np.diag([1, -1, -1]),   # Rx(180°)
             np.diag([-1, 1, -1]),   # Ry(180°)
             np.diag([-1, -1, 1])]   # Rz(180°)
    fixed = 0
    for i in range(len(calib.camera_poses) - 1):
        A = np.linalg.inv(calib.robot_poses[i]) @ calib.robot_poses[i+1]
        angle_A = np.degrees(np.arccos(np.clip((np.trace(A[:3,:3]) - 1) / 2, -1, 1)))
        B = calib.camera_poses[i] @ np.linalg.inv(calib.camera_poses[i+1])
        angle_B = np.degrees(np.arccos(np.clip((np.trace(B[:3,:3]) - 1) / 2, -1, 1)))
        diff = abs(angle_A - angle_B)
        if diff < 20:
            continue
        # 尝试翻转后一个姿态的不同轴线
        best = diff
        best_R = None
        for F in flips:
            R_new = calib.camera_poses[i+1][:3, :3] @ F
            B_new = calib.camera_poses[i][:3, :3] @ np.linalg.inv(R_new)
            angle_B_new = np.degrees(np.arccos(np.clip((np.trace(B_new) - 1) / 2, -1, 1)))
            new_diff = abs(angle_A - angle_B_new)
            if new_diff < best:
                best = new_diff
                best_R = R_new
        if best_R is not None:
            calib.camera_poses[i+1][:3, :3] = best_R
            fixed += 1
            print(f"    姿态 {i+1} 已翻转(diff {diff:.1f}°→{best:.1f}°)")
    if fixed == 0:
        print("    所有姿态朝向一致,无需翻转")

    T_result, error = calib.calibrate()

    print(f"\n{'='*45}")
    print(f"采集: {len(calib.robot_poses)}/{len(SQUARE_POSES)} 组")
    print(f"平移: [{T_result[0,3]:.4f}, {T_result[1,3]:.4f}, {T_result[2,3]:.4f}] m")
    print(f"AX=XB: {error:.6f}")
    print(f"\n标定结果 X (T_cam_ee):")
    print(np.array2string(T_result, precision=6, suppress_small=True))
    print(f"\n照片: {SAVE_DIR}")
    print(f"{'='*45}")
    print(f"\n照片已保存到: {SAVE_DIR}")

    save_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "calibration", "real_eye_in_hand_result.npy",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    calib.save_result(save_path)

    print("标定完成")

    PIPELINE.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
