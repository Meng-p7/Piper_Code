"""
标定种子生成器：关节空间采样 + 几何预筛选 + 检测验证
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import cv2
from core.vision import Camera

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "calibration_scene.xml",
)

OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "calibration", "calibration_seeds.npy",
)

CHECKER_COLS = 8
CHECKER_ROWS = 6
SQUARE_SIZE = 0.015

JOINT_LIMITS = np.array([
    [-2.618, 2.618],
    [0.0, 3.14],
    [-2.697, 0.0],
    [-1.832, 1.832],
    [-1.22, 1.22],
    [-3.14, 3.14],
])

BOARD_CENTER = np.array([0.5, 0.0, 0.001])
N_SAMPLES = 2000
DEDUP_DIST = 0.03
DETECT_MARGIN = 10

FOVY = 60
HFOV = 2 * np.degrees(np.arctan(np.tan(np.radians(FOVY) / 2) * 4 / 3))


def main():
    print("=" * 60)
    print("标定种子生成器")
    print("=" * 60)

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    camera = Camera(model, data, camera_name="wrist_camera", width=640, height=480)
    board_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "calibration_board"
    )
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "wrist_camera")

    rng = np.random.RandomState(54321)

    valid_seeds = []
    n_tried = 0
    n_geom_pass = 0

    print(f"\n目标: {N_SAMPLES} 个有效种子")

    while len(valid_seeds) < N_SAMPLES and n_tried < N_SAMPLES * 30:
        n_tried += 1
        q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        data.qpos[:6] = q
        mujoco.mj_forward(model, data)

        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].copy().reshape(3, 3)
        board_pos = data.xpos[board_body_id].copy()

        to_board = board_pos - cam_pos
        dist = np.linalg.norm(to_board)
        if dist < 0.08 or dist > 0.6:
            continue

        to_board_dir = to_board / dist
        cam_forward = -cam_mat[:, 2]
        dot = np.dot(cam_forward, to_board_dir)
        max_angle = min(HFOV / 2, FOVY / 2) * 0.75
        if dot < np.cos(np.radians(max_angle)):
            continue

        n_geom_pass += 1

        image = camera.get_image()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        ret, corners = cv2.findChessboardCorners(
            gray, (CHECKER_COLS, CHECKER_ROWS), flags
        )
        if not ret:
            continue

        h, w = gray.shape
        xs = corners[:, 0, 0]
        ys = corners[:, 0, 1]
        if (
            xs.min() < DETECT_MARGIN
            or xs.max() > w - DETECT_MARGIN
            or ys.min() < DETECT_MARGIN
            or ys.max() > h - DETECT_MARGIN
        ):
            continue

        valid_seeds.append(q.copy())

        if len(valid_seeds) % 50 == 0:
            print(f"  进度: {len(valid_seeds)}/{N_SAMPLES} "
                  f"(尝试 {n_tried}, 几何通过 {n_geom_pass})")

    print(f"\n总尝试: {n_tried}")
    print(f"几何预筛选通过: {n_geom_pass}")
    print(f"检测通过: {len(valid_seeds)}")

    if not valid_seeds:
        print("\n未找到有效种子")
        return

    seeds = np.array(valid_seeds)
    keep = [0]
    for i in range(1, len(seeds)):
        is_dup = False
        for j in keep:
            if np.linalg.norm(seeds[i] - seeds[j]) < DEDUP_DIST:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    seeds = seeds[keep]
    print(f"去重后: {len(seeds)} 个")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, seeds)
    print(f"\n保存到: {OUTPUT_PATH}")
    print(f"\n关节角统计:")
    for j in range(6):
        print(
            f"  J{j+1}: [{seeds[:, j].min():.3f}, {seeds[:, j].max():.3f}]"
            f"  mean={seeds[:, j].mean():.3f}  std={seeds[:, j].std():.3f}"
        )


if __name__ == "__main__":
    main()
