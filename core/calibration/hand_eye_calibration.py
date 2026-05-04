from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from utils.logger import get_logger

logger = get_logger(__name__)

_METHOD_MAP = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


class HandEyeCalibration:
    """
    手眼标定模块

    支持 Eye-in-Hand 和 Eye-to-Hand 两种标定方式
    使用 AX = XB 方程求解相机与机械臂之间的变换矩阵

    Eye-in-Hand:  相机装在末端，标定板固定在世界 → X = T_cam_ee
    Eye-to-Hand:  相机固定在世界，标定板装在末端 → X = T_cam_base
    """

    def __init__(self, method: str = "park", eye_mode: str = "eye_in_hand") -> None:
        """
        初始化手眼标定

        Args:
            method: 标定方法 ("tsai", "park", "horaud", "andreff", "daniilidis")
            eye_mode: 标定模式 ("eye_in_hand", "eye_to_hand")
        """
        if eye_mode not in ("eye_in_hand", "eye_to_hand"):
            raise ValueError("eye_mode must be 'eye_in_hand' or 'eye_to_hand'")
        if method not in _METHOD_MAP:
            raise ValueError(f"method must be one of {list(_METHOD_MAP.keys())}")
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
        执行手眼标定（基于 OpenCV calibrateHandEye）

        Returns:
            T_cam2gripper: 变换矩阵 (4x4)
                OpenCV 返回 T_cam2gripper（从相机坐标系到夹爪坐标系的变换）
                eye_in_hand  → T_ee_cam (末端←相机)
                eye_to_hand  → T_base_cam (基座←相机)
            error: 标定误差（平均 Frobenius 范数）
        """
        if len(self.robot_poses) < 3:
            raise ValueError("至少需要3组样本进行标定")

        if self.eye_mode == "eye_in_hand":
            R_gripper2base = [p[:3, :3] for p in self.robot_poses]
            t_gripper2base = [p[:3, 3] for p in self.robot_poses]
            R_target2cam = [p[:3, :3] for p in self.camera_poses]
            t_target2cam = [p[:3, 3] for p in self.camera_poses]
        else:
            R_gripper2base = [p[:3, :3].T for p in self.robot_poses]
            t_gripper2base = [-p[:3, :3].T @ p[:3, 3] for p in self.robot_poses]
            R_target2cam = [p[:3, :3].T for p in self.camera_poses]
            t_target2cam = [-p[:3, :3].T @ p[:3, 3] for p in self.camera_poses]

        cv_method = _METHOD_MAP[self.method]

        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            R_gripper2base, t_gripper2base,
            R_target2cam, t_target2cam,
            method=cv_method,
        )

        T_result = np.eye(4)
        T_result[:3, :3] = R_cam2gripper
        T_result[:3, 3] = t_cam2gripper.flatten()

        error = self._compute_error(T_result)
        self.calibration_result = T_result
        logger.info("标定完成 [%s/%s] 误差: %.6f", self.method, self.eye_mode, error)
        return T_result, error

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

    def collect_calibration_data(
        self,
        controller,
        camera,
        target_world_pos: np.ndarray,
    ) -> None:
        """
        采集一组标定样本（当前机器人姿态下）

        读取末端执行器位姿作为 robot_pose，根据目标世界位置和
        相机位姿计算目标在相机坐标系中的位姿作为 camera_pose。

        Args:
            controller: Controller 对象（需提供 get_ee_pose()）
            camera: Camera 对象（需提供 get_camera_pose()）
            target_world_pos: 标定目标在世界坐标系中的位置 (3,)
        """
        ee_pos, ee_rot = controller.get_ee_pose()
        T_base_ee = np.eye(4)
        T_base_ee[:3, :3] = ee_rot
        T_base_ee[:3, 3] = ee_pos

        cam_pos, cam_rot = camera.get_camera_pose()
        T_world_cam = np.eye(4)
        T_world_cam[:3, :3] = cam_rot
        T_world_cam[:3, 3] = cam_pos

        T_world_target = np.eye(4)
        T_world_target[:3, 3] = np.asarray(target_world_pos).flatten()

        T_cam_target = np.linalg.inv(T_world_cam) @ T_world_target

        self.add_sample(T_base_ee, T_cam_target)
        logger.debug("采集样本 %d/%d 完成", len(self.robot_poses), len(self.robot_poses))

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
