"""
标定种子生成器：基于已有种子做邻域扩展 + 检测验证
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

JOINT_LIMITS = np.array([
    [-2.618, 2.618],
    [0.0, 3.14],
    [-2.697, 0.0],
    [-1.832, 1.832],
    [-1.22, 1.22],
    [-3.14, 3.14],
])

TARGET_SEEDS = 50
DEDUP_DIST = 0.04
DETECT_MARGIN = 5
NOISE_PER_JOINT = [0.05, 0.08, 0.08, 0.10, 0.05, 0.10]
PERTURBS_PER_BASE = 200


def detect_board(model, data, camera):
    image = camera.get_image()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK
    ret, corners = cv2.findChessboardCorners(gray, (CHECKER_COLS, CHECKER_ROWS), flags)
    if not ret:
        gray_eq = cv2.equalizeHist(gray)
        ret, corners = cv2.findChessboardCorners(gray_eq, (CHECKER_COLS, CHECKER_ROWS), flags)
    if not ret:
        return False
    h, w = gray.shape
    xs = corners[:, 0, 0]
    ys = corners[:, 0, 1]
    if xs.min() < DETECT_MARGIN or xs.max() > w - DETECT_MARGIN:
        return False
    if ys.min() < DETECT_MARGIN or ys.max() > h - DETECT_MARGIN:
        return False
    return True


def dedup(seeds, dist):
    keep = [0]
    for i in range(1, len(seeds)):
        is_dup = False
        for j in keep:
            if np.linalg.norm(seeds[i] - seeds[j]) < dist:
                is_dup = True
                break
        if not is_dup:
            keep.append(i)
    return seeds[keep]


def main():
    print("=" * 60)
    print("标定种子生成器 (邻域扩展模式)")
    print("=" * 60)

    if not os.path.exists(OUTPUT_PATH):
        print(f"\n未找到已有种子文件: {OUTPUT_PATH}")
        print("请先手动用纯随机采样生成一批初始种子")
        return

    base_seeds = np.load(OUTPUT_PATH)
    print(f"加载 {len(base_seeds)} 个基础种子")

    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    camera = Camera(model, data, camera_name="wrist_camera", width=640, height=480)

    rng = np.random.RandomState(42)
    all_seeds = list(base_seeds)

    noise_scale = np.array(NOISE_PER_JOINT)

    print(f"\n目标: 扩展至 {TARGET_SEEDS} 个种子")
    print(f"每个基础种子扰动 {PERTURBS_PER_BASE} 次")

    for i, base in enumerate(base_seeds):
        if len(all_seeds) >= TARGET_SEEDS * 3:
            break
        found = 0
        for _ in range(PERTURBS_PER_BASE):
            noise = rng.randn(6) * noise_scale
            q = np.clip(base + noise, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
            data.qpos[:6] = q
            mujoco.mj_forward(model, data)
            if detect_board(model, data, camera):
                all_seeds.append(q.copy())
                found += 1
        print(f"  基础种子 {i+1}/{len(base_seeds)}: 找到 {found} 个有效邻域")

    all_seeds = np.array(all_seeds)
    print(f"\n去重前: {len(all_seeds)} 个")
    final = dedup(all_seeds, DEDUP_DIST)
    print(f"去重后: {len(final)} 个")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    np.save(OUTPUT_PATH, final)
    print(f"\n保存到: {OUTPUT_PATH}")
    print(f"\n关节角统计:")
    for j in range(6):
        print(
            f"  J{j+1}: [{final[:, j].min():.3f}, {final[:, j].max():.3f}]"
            f"  mean={final[:, j].mean():.3f}  std={final[:, j].std():.3f}"
        )


if __name__ == "__main__":
    main()
