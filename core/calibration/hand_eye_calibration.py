from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)


def _skew(v: np.ndarray) -> np.ndarray:
    """向量的反对称矩阵"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class HandEyeCalibration:
    """
    手眼标定模块

    支持 Eye-in-Hand 和 Eye-to-Hand 两种标定方式
    使用 AX = XB 方程求解相机与机械臂之间的变换矩阵

    Eye-in-Hand:  相机装在末端，标定板固定在世界 → X = T_cam_ee
    Eye-to-Hand:  相机固定在世界，标定板装在末端 → X = T_cam_base
    """

    def __init__(self, method: str = "tsai", eye_mode: str = "eye_in_hand") -> None:
        """
        初始化手眼标定

        Args:
            method: 标定方法 ("tsai", "park")
            eye_mode: 标定模式 ("eye_in_hand", "eye_to_hand")
        """
        if eye_mode not in ("eye_in_hand", "eye_to_hand"):
            raise ValueError("eye_mode must be 'eye_in_hand' or 'eye_to_hand'")
        self.method = method
        self.eye_mode = eye_mode
        self.robot_poses: list[np.ndarray] = []
        self.camera_poses: list[np.ndarray] = []
        self.calibration_result: Optional[np.ndarray] = None

    def add_sample(self, robot_pose: np.ndarray, camera_pose: np.ndarray) -> None:
        """
        添加一组标定样本

        Args:
            robot_pose: 机器人末端位姿 (4x4 变换矩阵 T_base_ee)
            camera_pose: 相机观测到的标定板位姿 (4x4 变换矩阵 T_cam_target)
        """
        self.robot_poses.append(robot_pose)
        self.camera_poses.append(camera_pose)

    def clear_samples(self) -> None:
        """清除所有样本"""
        self.robot_poses = []
        self.camera_poses = []

    def calibrate(self) -> tuple[np.ndarray, float]:
        """
        执行手眼标定

        Returns:
            T_cam_ee: 变换矩阵 (4x4)
            error: 标定误差（平移平均误差，单位 m）
        """
        if len(self.robot_poses) < 3:
            raise ValueError("至少需要3组样本进行标定")

        if self.method == "tsai":
            T_result, error = self._tsai_method()
        elif self.method == "park":
            T_result, error = self._park_method()
        else:
            T_result, error = self._tsai_method()

        self.calibration_result = T_result
        return T_result, error

    def _tsai_method(self) -> tuple[np.ndarray, float]:
        """
        Tsai-Lenz 手眼标定算法（Axis-Angle + SVD）

        分两步求解 AX = XB:
          1. 旋转: R_A @ R_x = R_x @ R_B → 用轴角+SVD 求解 R_x
          2. 平移: (R_A - I) @ t_x = R_x @ t_B - t_A → 最小二乘求解 t_x

        Returns:
            T_x: 4x4 变换矩阵
            error: 标定误差
        """
        n = len(self.robot_poses)

        A_list: list[np.ndarray] = []
        B_list: list[np.ndarray] = []
        t_A_list: list[np.ndarray] = []
        t_B_list: list[np.ndarray] = []

        for i in range(n - 1):
            R_g1 = self.robot_poses[i][:3, :3]
            t_g1 = self.robot_poses[i][:3, 3]
            R_g2 = self.robot_poses[i + 1][:3, :3]
            t_g2 = self.robot_poses[i + 1][:3, 3]

            R_c1 = self.camera_poses[i][:3, :3]
            t_c1 = self.camera_poses[i][:3, 3]
            R_c2 = self.camera_poses[i + 1][:3, :3]
            t_c2 = self.camera_poses[i + 1][:3, 3]

            if self.eye_mode == "eye_in_hand":
                R_A = R_g1.T @ R_g2
                t_A = R_g1.T @ (t_g2 - t_g1)
                R_B = R_c1 @ R_c2.T
                t_B = t_c1 - R_c1 @ R_c2.T @ t_c2
            else:  # eye_to_hand
                R_A = R_g1 @ R_g2.T
                t_A = t_g1 - R_g1 @ R_g2.T @ t_g2
                R_B = R_c1.T @ R_c2
                t_B = R_c1.T @ (t_c2 - t_c1)

            A_list.append(R_A)
            B_list.append(R_B)
            t_A_list.append(t_A)
            t_B_list.append(t_B)

        R_x = self._solve_rotation(A_list, B_list)
        t_x = self._solve_translation(A_list, t_A_list, t_B_list, R_x)

        T_x = np.eye(4)
        T_x[:3, :3] = R_x
        T_x[:3, 3] = t_x

        error = self._compute_error(T_x)
        return T_x, error

    def _solve_rotation(self, A_list: list[np.ndarray],
                        B_list: list[np.ndarray]) -> np.ndarray:
        """
        Tsai-Lenz 旋转求解：轴角 + SVD 方法

        对每对 (R_A, R_B)，提取旋转轴 k_A, k_B，堆叠方程:
            skew(k_A + k_B) @ k_x' = k_B - k_A

        用最小二乘法求解 Gibbs 向量 k_x'，再恢复旋转矩阵。

        Args:
            A_list: R_A 列表 (N-1 个 3x3 矩阵)
            B_list: R_B 列表 (N-1 个 3x3 矩阵)

        Returns:
            R_x: 3x3 旋转矩阵
        """
        pairs = len(A_list)

        # 收集所有有效对的旋转轴
        C_rows: list[np.ndarray] = []
        d_rows: list[np.ndarray] = []
        eps = 1e-12

        for i in range(pairs):
            rotvec_A = Rotation.from_matrix(A_list[i]).as_rotvec()
            rotvec_B = Rotation.from_matrix(B_list[i]).as_rotvec()

            angle_A = np.linalg.norm(rotvec_A)
            angle_B = np.linalg.norm(rotvec_B)

            # 跳过近似零旋转的对（轴方向不定）
            if angle_A < eps or angle_B < eps:
                continue

            k_A = rotvec_A / angle_A
            k_B = rotvec_B / angle_B

            C = _skew(k_A + k_B)  # 3x3
            d = k_B - k_A         # 3,

            C_rows.append(C)
            d_rows.append(d)

        if len(C_rows) == 0:
            return np.eye(3)

        C_stack = np.vstack(C_rows)   # (3*K, 3)
        d_stack = np.hstack(d_rows)   # (3*K,)

        # 最小二乘求解 k_x'
        k_x_prime, _, _, _ = np.linalg.lstsq(C_stack, d_stack, rcond=None)

        # 从 Gibbs 向量恢复旋转矩阵
        # k_x' = k_x * tan(θ/2), |k_x'| = tan(θ/2)
        norm_sq = np.dot(k_x_prime, k_x_prime)

        if norm_sq < eps:
            return np.eye(3)

        sk = _skew(k_x_prime)
        # R = I + 2/(1+|k_x'|^2) * (skew(k_x') + skew(k_x')^2)
        R_x = np.eye(3) + (2.0 / (1.0 + norm_sq)) * (sk + sk @ sk)

        # 确保结果是合法旋转矩阵（正交化）
        U, _, Vt = np.linalg.svd(R_x)
        R_x = U @ Vt
        if np.linalg.det(R_x) < 0:
            R_x = U @ np.diag([1, 1, -1]) @ Vt

        return R_x

    def _solve_translation(self, A_list: list[np.ndarray],
                           t_A_list: list[np.ndarray],
                           t_B_list: list[np.ndarray],
                           R_x: np.ndarray) -> np.ndarray:
        """
        求解平移向量

        对每对 (i, i+1):
            (R_A - I) @ t_x = R_x @ t_B - t_A

        Args:
            A_list: R_A 列表
            t_A_list: t_A 列表
            t_B_list: t_B 列表
            R_x: 已求解的旋转矩阵

        Returns:
            t_x: 3, 平移向量
        """
        LHS_rows: list[np.ndarray] = []
        RHS_rows: list[np.ndarray] = []

        for R_A, t_A, t_B in zip(A_list, t_A_list, t_B_list):
            LHS_rows.append(R_A - np.eye(3))        # 3x3
            RHS_rows.append(R_x @ t_B - t_A)         # 3,

        LHS = np.vstack(LHS_rows)    # (3*K, 3)
        RHS = np.hstack(RHS_rows)    # (3*K,)

        t_x, _, _, _ = np.linalg.lstsq(LHS, RHS, rcond=None)
        return t_x

    def _compute_error(self, T_x: np.ndarray) -> float:
        """
        计算标定误差

        验证 AX=XB 的残差：对每对 (i, i+1)，计算 |A @ X - X @ B| 的 Frobenius 范数。

        Args:
            T_x: 标定结果变换矩阵 (4x4)

        Returns:
            error: 平均 Frobenius 误差
        """
        X = T_x.copy()
        errors: list[float] = []

        for i in range(len(self.robot_poses) - 1):
            if self.eye_mode == "eye_in_hand":
                A = np.linalg.inv(self.robot_poses[i]) @ self.robot_poses[i + 1]
                B = self.camera_poses[i] @ np.linalg.inv(self.camera_poses[i + 1])
            else:
                A = self.robot_poses[i] @ np.linalg.inv(self.robot_poses[i + 1])
                B = np.linalg.inv(self.camera_poses[i]) @ self.camera_poses[i + 1]

            residual = A @ X - X @ B
            errors.append(np.linalg.norm(residual, 'fro'))

        return float(np.mean(errors))

    def _park_method(self) -> tuple[np.ndarray, float]:
        """Park 方法（预留）"""
        raise NotImplementedError("Park method not implemented yet")

    def collect_calibration_data(self, controller, camera,
                                 num_samples: int = 20) -> None:
        """
        自动采集标定数据（仿真环境）

        Args:
            controller: 机械臂控制器
            camera: 相机对象
            num_samples: 样本数量
        """
        logger.info("开始采集手眼标定数据，共 %d 个样本...", num_samples)

        for i in range(num_samples):
            robot_pose = np.eye(4)
            ee_pos, ee_rot = controller.get_ee_pose()
            robot_pose[:3, :3] = ee_rot
            robot_pose[:3, 3] = ee_pos

            camera_pose = np.eye(4)
            cam_pos, cam_rot = camera.get_camera_pose()
            camera_pose[:3, :3] = cam_rot
            camera_pose[:3, 3] = cam_pos

            self.add_sample(robot_pose, camera_pose)
            logger.debug("  采集样本 %d/%d", i + 1, num_samples)

        logger.info("标定数据采集完成")

    def save_result(self, filepath: str) -> None:
        """
        保存标定结果

        Args:
            filepath: 保存路径 (.npy)
        """
        if self.calibration_result is None:
            raise ValueError("请先执行标定")
        np.save(filepath, self.calibration_result)
        logger.info("标定结果已保存到 %s", filepath)

    def load_result(self, filepath: str) -> np.ndarray:
        """
        加载标定结果

        Args:
            filepath: 文件路径 (.npy)

        Returns:
            T_x: 变换矩阵 (4x4)
        """
        self.calibration_result = np.load(filepath)
        return self.calibration_result
        logger.info("标定结果已从 %s 加载", filepath)
        return self.calibration_result
    
    def transform_point(self, point_cam, T_cam_ee=None):
        """
        将相机坐标系中的点转换到末端执行器坐标系
        
        Args:
            point_cam: 相机坐标系中的点 [x, y, z]
            T_cam_ee: 变换矩阵，若为 None 则使用标定结果
            
        Returns:
            point_ee: 末端执行器坐标系中的点
        """
        if T_cam_ee is None:
            T_cam_ee = self.calibration_result
        
        if T_cam_ee is None:
            raise ValueError("未提供变换矩阵")
        
        point_homogeneous = np.append(point_cam, 1)
        point_ee_homogeneous = T_cam_ee @ point_homogeneous
        
        return point_ee_homogeneous[:3]
