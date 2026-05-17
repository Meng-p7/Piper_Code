"""
手眼标定验证：检测标定板位置，让机械臂末端移动到该位置上方
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
from scipy.spatial.transform import Rotation

CHARUCO_SQUARES_X = 9
CHARUCO_SQUARES_Y = 14
CHARUCO_SQUARE_LEN = 0.020
CHARUCO_MARKER_LEN = 0.015
CHARUCO_DICT_TYPE = cv2.aruco.DICT_5X5_100

CALIB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "calibration", "real_eye_in_hand_result.npy",
)


def load_calibration(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"标定结果文件不存在: {path}")
    return np.load(path)


def connect_and_enable(can_name="can0"):
    piper = C_PiperInterface_V2(can_name)
    piper.ConnectPort()
    time.sleep(0.2)
    t0 = time.time()
    while not piper.EnablePiper():
        if time.time() - t0 > 10:
            raise TimeoutError("使能超时")
        time.sleep(0.01)
    print("机械臂已使能")
    return piper


def matrix_to_sdk_pose(T):
    """4x4矩阵 → (x, y, z, rx, ry, rz) SDK单位"""
    x = T[0, 3] * 1000000
    y = T[1, 3] * 1000000
    z = T[2, 3] * 1000000
    r = Rotation.from_matrix(T[:3, :3])
    rx, ry, rz = r.as_euler('xyz', degrees=True)
    return int(x), int(y), int(z), int(rx * 1000), int(ry * 1000), int(rz * 1000)


def detect_board_pose(image, camera_matrix, dist_coeffs):
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICT_TYPE)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_LEN, CHARUCO_MARKER_LEN, dictionary,
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    params = cv2.aruco.DetectorParameters()
    ad = cv2.aruco.ArucoDetector(dictionary, params)
    corners, ids, _ = ad.detectMarkers(gray)

    if ids is None or len(ids) < 4:
        return None, None

    obj_pts_3d = board.getObjPoints()
    board_ids = board.getIds().flatten().tolist()
    obj_points, img_points = [], []
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
        return None, None

    R, _ = cv2.Rodrigues(rvec)

    T_cam_board = np.eye(4)
    T_cam_board[:3, :3] = R
    T_cam_board[:3, 3] = tvec.flatten()

    vis = image.copy()
    cv2.aruco.drawDetectedMarkers(vis, corners, ids)
    cv2.drawFrameAxes(vis, camera_matrix, dist_coeffs, rvec, tvec, 0.05)
    return T_cam_board, vis


def main():
    print("=" * 50)
    print("手眼标定验证")
    print("=" * 50)

    # 加载标定结果
    X = load_calibration(CALIB_PATH)
    # 确保旋转矩阵行列式为 +1
    if np.linalg.det(X[:3, :3]) < 0:
        U, S, Vt = np.linalg.svd(X[:3, :3])
        Vt[-1, :] *= -1
        X[:3, :3] = U @ Vt
        print("(已修正旋转矩阵行列式)")
    print(f"\n标定结果 X (T_cam_ee):")
    print(np.array2string(X, precision=6, suppress_small=True))

    # 启动相机
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    time.sleep(1)
    device = pipeline.get_active_profile().get_device()
    rgb_sensor = device.first_color_sensor()
    rgb_sensor.set_option(rs.option.enable_auto_exposure, 1)
    print("相机已启动")

    # 获取内参
    for _ in range(10):
        pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

    # 连接机械臂
    piper = connect_and_enable("can0")

    # 获取当前末端位姿
    ep = piper.GetArmEndPoseMsgs().end_pose
    T_base_ee_current = np.eye(4)
    rx = math.radians(ep.RX_axis / 1000)
    ry = math.radians(ep.RY_axis / 1000)
    rz = math.radians(ep.RZ_axis / 1000)
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    T_base_ee_current[:3, :3] = np.array([
        [cy*cz, sx*sy*cz - cx*sz, cx*sy*cz + sx*sz],
        [cy*sz, sx*sy*sz + cx*cz, cx*sy*sz - sx*cz],
        [-sy,   sx*cy,             cx*cy],
    ])
    T_base_ee_current[:3, 3] = [ep.X_axis/1000000, ep.Y_axis/1000000, ep.Z_axis/1000000]
    print(f"\n当前末端位姿:\n{np.array2string(T_base_ee_current, precision=4, suppress_small=True)}")

    input("\n准备好后，将标定板放在相机视野内，按回车检测...")

    # 拍照检测标定板
    for _ in range(10):
        pipeline.wait_for_frames()
    frames = pipeline.wait_for_frames()
    image = np.asanyarray(frames.get_color_frame().get_data())

    T_cam_board, vis = detect_board_pose(image, camera_matrix, dist_coeffs)
    if T_cam_board is None:
        print("未检测到标定板！")
        pipeline.stop()
        return

    cv2.imshow("Board Detection", vis)
    print("\n标定板位姿 T_cam_board:")
    print(np.array2string(T_cam_board, precision=4, suppress_small=True))

    # 计算标定板在基座坐标系下的位姿
    # 链式变换: Base ← EE ← Camera ← Board
    # T_base_board = T_base_ee @ T_ee_cam @ T_cam_board
    # X = T_ee_cam (标定结果)
    T_base_board = T_base_ee_current @ X @ T_cam_board
    print("\n标定板在基座坐标系下的位姿 T_base_board:")
    print(np.array2string(T_base_board, precision=4, suppress_small=True))

    # 检查平移量是否合理
    t_norm = np.linalg.norm(T_base_board[:3, 3])
    if t_norm > 5.0:
        print(f"\n⚠ 平移量 {t_norm:.1f}m 异常！标定结果 X 可能不准确。")
        print("  建议重新标定：确保每次移动机械臂时标定板都清晰可见。")
        cv2.waitKey(0)
        pipeline.stop()
        cv2.destroyAllWindows()
        piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        piper.JointCtrl(0, 0, 0, 0, 0, 0)
        time.sleep(2)
        return

    # 转换为 SDK 位姿并移动到标定板上方
    target_pose = list(matrix_to_sdk_pose(T_base_board))
    # 保持与标定相同的朝向
    target_pose[3] = -179900
    target_pose[4] = 0
    target_pose[5] = -179900
    print(f"\n目标位姿 (X, Y, Z, RX, RY, RZ): {target_pose}")

    # 先抬高到安全高度再下降
    safe_pose = target_pose.copy()
    safe_pose[2] = safe_pose[2] + 60000  # 抬高6cm
    print("\n先移动到安全高度...")
    piper.MotionCtrl_2(0x01, 0x00, 80, 0x00)
    piper.EndPoseCtrl(*safe_pose)
    time.sleep(3)

    print("移动到标定板位置...")
    piper.MotionCtrl_2(0x01, 0x02, 80, 0x00)
    piper.EndPoseCtrl(*target_pose)
    time.sleep(3)

    print("验证完成！")

    # 关闭
    cv2.waitKey(0)
    pipeline.stop()
    cv2.destroyAllWindows()
    piper.MotionCtrl_2(0x01, 0x01, 30, 0x00)
    piper.JointCtrl(0, 0, 0, 0, 0, 0)
    time.sleep(2)


if __name__ == "__main__":
    main()
