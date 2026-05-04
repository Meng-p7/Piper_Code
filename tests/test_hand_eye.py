"""
手眼标定单元测试（合成数据验证）

测试内容：
- 旋转求解精度（通过完整 calibrate 流程验证）
- eye_in_hand 模式零噪声全流程
- eye_to_hand 模式零噪声全流程
- 样本不足抛异常
"""

import sys
import os
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.calibration.hand_eye_calibration import HandEyeCalibration


def _random_pose(rng: np.random.RandomState) -> np.ndarray:
    """生成随机 4x4 位姿矩阵"""
    T = np.eye(4)
    T[:3, :3] = Rotation.random(random_state=rng).as_matrix()
    T[:3, 3] = rng.uniform(-0.5, 0.5, 3)
    return T


def _make_eye_in_hand_data(R_x_true: np.ndarray, t_x_true: np.ndarray,
                           n: int = 5, rng: np.random.RandomState | None = None
                           ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    生成 eye-in-hand 合成数据: X = T_cam_ee

    AX=XB 其中 A = T_ee1^{-1} @ T_ee2, B = T_cam1 @ T_cam2^{-1}

    生成方法: 随机 N 个 T_base_ee[i], 计算 T_cam_target[i] = X^{-1} @ T_base_ee[i]^{-1} @ T_base_ee[0] @ X @ T_cam_target[0]
    这里简化为: 选参考帧 T_cam_target[0] = random, 然后对每个 i:
        T_cam_target[i] = X^{-1} @ T_base_ee[i]^{-1} @ T_base_ee[0] @ X @ T_cam_target[0]
    但这种太复杂。更简单的方法是直接用随机 AX=XB 反向生成:
    - 随机生成 A_i (相对机器人运动)
    - 用真值 X 计算 B_i = X^{-1} @ A_i @ X
    - 然后生成满足 A_i = T_ee_i^{-1} @ T_ee_{i+1} 的任意一致帧

    最简单：直接用已知 X 验证 AX=XB，用随机构造 robot_poses 和反推 camera_poses
    """
    if rng is None:
        rng = np.random.RandomState(42)

    X_true = np.eye(4)
    X_true[:3, :3] = R_x_true
    X_true[:3, 3] = t_x_true

    # 随机 N 个机器人末端位姿
    robot_poses = [_random_pose(rng) for _ in range(n)]

    # 随机第一个相机位姿
    camera_poses = [_random_pose(rng)]

    # 对每个后续帧，通过 AX=XB 关系反推
    for i in range(1, n):
        # A = T_ee_{i-1}^{-1} @ T_ee_i
        A = np.linalg.inv(robot_poses[i - 1]) @ robot_poses[i]
        # B = X^{-1} @ A @ X
        B = np.linalg.inv(X_true) @ A @ X_true
        # B = T_cam_{i-1} @ T_cam_i^{-1}  →  T_cam_i = B^{-1} @ T_cam_{i-1}
        T_cam_i = np.linalg.inv(B) @ camera_poses[i - 1]
        camera_poses.append(T_cam_i)

    return robot_poses, camera_poses


def _make_eye_to_hand_data(R_x_true: np.ndarray, t_x_true: np.ndarray,
                           n: int = 5, rng: np.random.RandomState | None = None
                           ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    生成 eye-to-hand 合成数据: X = T_cam_base

    A = T_ee_i @ T_ee_{i+1}^{-1}, B = T_cam_i^{-1} @ T_cam_{i+1}
    AX = XB
    """
    if rng is None:
        rng = np.random.RandomState(42)

    X_true = np.eye(4)
    X_true[:3, :3] = R_x_true
    X_true[:3, 3] = t_x_true

    robot_poses = [_random_pose(rng) for _ in range(n)]
    camera_poses = [_random_pose(rng)]

    for i in range(1, n):
        A = robot_poses[i - 1] @ np.linalg.inv(robot_poses[i])
        B = np.linalg.inv(X_true) @ A @ X_true
        # B = T_cam_{i-1}^{-1} @ T_cam_i → T_cam_i = T_cam_{i-1} @ B
        T_cam_i = camera_poses[i - 1] @ B
        camera_poses.append(T_cam_i)

    return robot_poses, camera_poses


# ─────────────────────────────────────────────────────────────────────
#  Tests
# ─────────────────────────────────────────────────────────────────────


class TestSolveRotation:
    """旋转求解精度测试（通过完整 calibrate 流程验证）"""

    def test_known_rotation_identity(self):
        """已知 R_x=I 时应精确返回 I"""
        rng = np.random.RandomState(0)
        R_x = np.eye(3)
        t_x = np.zeros(3)

        robot_poses, camera_poses = _make_eye_in_hand_data(R_x, t_x, n=8, rng=rng)

        calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")
        for rp, cp in zip(robot_poses, camera_poses):
            calib.add_sample(rp, cp)

        T_result, _ = calib.calibrate()
        rot_err = np.linalg.norm(T_result[:3, :3] - np.eye(3), 'fro')
        assert rot_err < 1e-10, f"Rotation error = {rot_err:.2e}"

    def test_known_rotation_90deg_z(self):
        """已知 R_x = Rot(z, 90°) 时应精确恢复"""
        R_x_true = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        rng = np.random.RandomState(1)
        t_x = rng.uniform(-0.2, 0.2, 3)

        robot_poses, camera_poses = _make_eye_in_hand_data(R_x_true, t_x, n=10, rng=rng)

        calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")
        for rp, cp in zip(robot_poses, camera_poses):
            calib.add_sample(rp, cp)

        T_result, _ = calib.calibrate()
        frob_err = np.linalg.norm(T_result[:3, :3] @ R_x_true.T - np.eye(3), 'fro')
        assert frob_err < 1e-6, f"Frobenius error = {frob_err:.2e}"

    def test_known_rotation_arbitrary(self):
        """已知任意 R_x 时应精确恢复"""
        rng = np.random.RandomState(2)
        R_x_true = Rotation.random(random_state=rng).as_matrix()
        t_x = rng.uniform(-0.2, 0.2, 3)

        robot_poses, camera_poses = _make_eye_in_hand_data(R_x_true, t_x, n=15, rng=rng)

        calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")
        for rp, cp in zip(robot_poses, camera_poses):
            calib.add_sample(rp, cp)

        T_result, _ = calib.calibrate()
        frob_err = np.linalg.norm(T_result[:3, :3] @ R_x_true.T - np.eye(3), 'fro')
        assert frob_err < 1e-6, f"Frobenius error = {frob_err:.2e}"


class TestEyeInHand:
    """eye_in_hand 模式端到端测试"""

    def test_perfect_zero_noise(self):
        """零噪声合成数据：error < 1e-10"""
        rng = np.random.RandomState(3)
        R_x = Rotation.random(random_state=rng).as_matrix()
        t_x = rng.uniform(-0.2, 0.2, 3)

        robot_poses, camera_poses = _make_eye_in_hand_data(R_x, t_x, n=8, rng=rng)

        calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")
        for rp, cp in zip(robot_poses, camera_poses):
            calib.add_sample(rp, cp)

        T_result, error = calib.calibrate()
        R_result = T_result[:3, :3]
        t_result = T_result[:3, 3]

        rot_err = np.linalg.norm(R_result @ R_x.T - np.eye(3), 'fro')
        trans_err = np.linalg.norm(t_result - t_x)

        assert rot_err < 1e-10, f"Rotation Frobenius error = {rot_err:.2e}"
        assert trans_err < 1e-10, f"Translation error = {trans_err:.2e}"
        assert error < 1e-10, f"Calibration error = {error:.2e}"

    def test_five_samples(self):
        """最少 5 组样本也能正确求解"""
        rng = np.random.RandomState(4)
        R_x = Rotation.random(random_state=rng).as_matrix()
        t_x = rng.uniform(-0.2, 0.2, 3)

        robot_poses, camera_poses = _make_eye_in_hand_data(R_x, t_x, n=5, rng=rng)

        calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")
        for rp, cp in zip(robot_poses, camera_poses):
            calib.add_sample(rp, cp)

        T_result, _ = calib.calibrate()
        rot_err = np.linalg.norm(T_result[:3, :3] @ R_x.T - np.eye(3), 'fro')
        assert rot_err < 1e-6


class TestEyeToHand:
    """eye_to_hand 模式端到端测试"""

    def test_perfect_zero_noise(self):
        """零噪声合成数据：error < 1e-10"""
        rng = np.random.RandomState(5)
        R_x = Rotation.random(random_state=rng).as_matrix()
        t_x = rng.uniform(-0.2, 0.2, 3)

        robot_poses, camera_poses = _make_eye_to_hand_data(R_x, t_x, n=8, rng=rng)

        calib = HandEyeCalibration(method="park", eye_mode="eye_to_hand")
        for rp, cp in zip(robot_poses, camera_poses):
            calib.add_sample(rp, cp)

        T_result, error = calib.calibrate()
        R_result = T_result[:3, :3]
        t_result = T_result[:3, 3]

        rot_err = np.linalg.norm(R_result @ R_x.T - np.eye(3), 'fro')
        trans_err = np.linalg.norm(t_result - t_x)

        assert rot_err < 1e-10, f"Rotation Frobenius error = {rot_err:.2e}"
        assert trans_err < 1e-10, f"Translation error = {trans_err:.2e}"
        assert error < 1e-10, f"Calibration error = {error:.2e}"


class TestEdgeCases:
    """边界情况测试"""

    def test_insufficient_samples(self):
        """样本不足应抛异常"""
        calib = HandEyeCalibration()
        rp = np.eye(4)
        calib.add_sample(rp, rp)
        calib.add_sample(rp, rp)
        with pytest.raises(ValueError, match="至少需要3组"):
            calib.calibrate()

    def test_invalid_eye_mode(self):
        """无效 eye_mode 应抛异常"""
        with pytest.raises(ValueError, match="eye_mode"):
            HandEyeCalibration(eye_mode="invalid")

    def test_clear_samples(self):
        """clear_samples 后样本归零"""
        calib = HandEyeCalibration()
        for _ in range(5):
            calib.add_sample(np.eye(4), np.eye(4))
        calib.clear_samples()
        assert len(calib.robot_poses) == 0
        assert len(calib.camera_poses) == 0

    def test_translation_only_motion(self):
        """纯平移为主但含微小旋转的运动也能正确标定"""
        rng = np.random.RandomState(6)
        R_x = Rotation.from_euler('x', 30, degrees=True).as_matrix()
        t_x = np.array([0.1, -0.05, 0.08])

        robot_poses = []
        camera_poses = []
        X_true = np.eye(4)
        X_true[:3, :3] = R_x
        X_true[:3, 3] = t_x

        T0 = np.eye(4)
        T0[:3, 3] = np.array([0.2, 0.1, 0.3])
        robot_poses.append(T0.copy())
        camera_poses.append(np.eye(4))

        for i in range(1, 8):
            T_ee = np.eye(4)
            T_ee[:3, 3] = T0[:3, 3] + rng.uniform(-0.3, 0.3, 3)
            T_ee[:3, :3] = Rotation.from_euler(
                'xyz', rng.uniform(-0.05, 0.05, 3)
            ).as_matrix()
            robot_poses.append(T_ee)
            A = np.linalg.inv(robot_poses[i - 1]) @ robot_poses[i]
            B = np.linalg.inv(X_true) @ A @ X_true
            camera_poses.append(np.linalg.inv(B) @ camera_poses[i - 1])

        calib = HandEyeCalibration(method="park", eye_mode="eye_in_hand")
        for rp, cp in zip(robot_poses, camera_poses):
            calib.add_sample(rp, cp)

        T_result, _ = calib.calibrate()
        rot_err = np.linalg.norm(T_result[:3, :3] @ R_x.T - np.eye(3), 'fro')
        trans_err = np.linalg.norm(T_result[:3, 3] - t_x)
        assert rot_err < 1e-6, f"Rotation error = {rot_err:.2e}"
        assert trans_err < 1e-8, f"Translation error = {trans_err:.2e}"
