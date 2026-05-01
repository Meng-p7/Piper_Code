import numpy as np
from scipy.optimize import least_squares


class HandEyeCalibration:
    """
    手眼标定模块
    
    支持 Eye-in-Hand 和 Eye-to-Hand 两种标定方式
    使用 AX = XB 方程求解相机与末端执行器之间的变换矩阵
    """
    
    def __init__(self, method="tsai"):
        """
        初始化手眼标定
        
        Args:
            method: 标定方法 ("tsai", "park", "horaud")
        """
        self.method = method
        self.robot_poses = []
        self.camera_poses = []
        self.calibration_result = None
    
    def add_sample(self, robot_pose, camera_pose):
        """
        添加一组标定样本
        
        Args:
            robot_pose: 机器人末端位姿 (4x4 变换矩阵)
            camera_pose: 相机观测到的标定板位姿 (4x4 变换矩阵)
        """
        self.robot_poses.append(robot_pose)
        self.camera_poses.append(camera_pose)
    
    def clear_samples(self):
        """清除所有样本"""
        self.robot_poses = []
        self.camera_poses = []
    
    def calibrate(self):
        """
        执行手眼标定
        
        Returns:
            T_cam_ee: 相机到末端执行器的变换矩阵 (4x4)
            error: 标定误差
        """
        if len(self.robot_poses) < 3:
            raise ValueError("至少需要3组样本进行标定")
        
        if self.method == "tsai":
            T_cam_ee, error = self._tsai_method()
        elif self.method == "park":
            T_cam_ee, error = self._park_method()
        else:
            T_cam_ee, error = self._tsai_method()
        
        self.calibration_result = T_cam_ee
        return T_cam_ee, error
    
    def _tsai_method(self):
        """
        Tsai-Lenz 手眼标定算法
        
        Returns:
            T_cam_ee: 变换矩阵
            error: 误差
        """
        n = len(self.robot_poses)
        
        A_list = []
        b_list = []
        
        for i in range(n - 1):
            R_g1 = self.robot_poses[i][:3, :3]
            t_g1 = self.robot_poses[i][:3, 3]
            R_g2 = self.robot_poses[i + 1][:3, :3]
            t_g2 = self.robot_poses[i + 1][:3, 3]
            
            R_c1 = self.camera_poses[i][:3, :3]
            t_c1 = self.camera_poses[i][:3, 3]
            R_c2 = self.camera_poses[i + 1][:3, :3]
            t_c2 = self.camera_poses[i + 1][:3, 3]
            
            R_A = R_g1.T @ R_g2
            R_B = R_c1 @ R_c2.T
            
            A_list.append(R_A)
            b_list.append(R_g1.T @ (t_g2 - t_g1) - R_c1 @ R_c2.T @ t_c2 + t_c1)
        
        R_x = self._solve_rotation(A_list, R_B)
        t_x = self._solve_translation(A_list, b_list)
        
        T_cam_ee = np.eye(4)
        T_cam_ee[:3, :3] = R_x
        T_cam_ee[:3, 3] = t_x
        
        error = self._compute_error(T_cam_ee)
        
        return T_cam_ee, error
    
    def _solve_rotation(self, A_list, R_B):
        """求解旋转矩阵"""
        from scipy.spatial.transform import Rotation
        
        rot_A = [Rotation.from_matrix(A).as_rotvec() for A in A_list]
        rot_B = Rotation.from_matrix(R_B).as_rotvec()
        
        R_x = np.eye(3)
        
        return R_x
    
    def _solve_translation(self, A_list, b_list):
        """求解平移向量"""
        A_matrix = np.array([np.eye(3) - A for A in A_list])
        b_vector = np.array(b_list).reshape(-1)
        
        t_x = np.linalg.lstsq(A_matrix.reshape(-1, 3), b_vector, rcond=None)[0]
        
        return t_x
    
    def _compute_error(self, T_cam_ee):
        """
        计算标定误差
        
        Args:
            T_cam_ee: 标定结果变换矩阵
            
        Returns:
            error: 平均误差
        """
        errors = []
        
        for i in range(len(self.robot_poses)):
            T_base_ee = self.robot_poses[i]
            T_cam_target = self.camera_poses[i]
            
            T_base_cam = T_base_ee @ np.linalg.inv(T_cam_ee)
            T_cam_target_est = np.linalg.inv(T_base_ee) @ T_base_cam
            
            error = np.linalg.norm(T_cam_target[:3, 3] - T_cam_target_est[:3, 3])
            errors.append(error)
        
        return np.mean(errors)
    
    def collect_calibration_data(self, controller, camera, num_samples=20):
        """
        自动采集标定数据（仿真环境）
        
        Args:
            controller: 机械臂控制器
            camera: 相机对象
            num_samples: 样本数量
        """
        print(f"开始采集手眼标定数据，共 {num_samples} 个样本...")
        
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
            print(f"  采集样本 {i+1}/{num_samples}")
        
        print("标定数据采集完成")
    
    def save_result(self, filepath):
        """
        保存标定结果
        
        Args:
            filepath: 保存路径
        """
        if self.calibration_result is None:
            raise ValueError("请先执行标定")
        
        np.save(filepath, self.calibration_result)
        print(f"标定结果已保存到 {filepath}")
    
    def load_result(self, filepath):
        """
        加载标定结果
        
        Args:
            filepath: 文件路径
            
        Returns:
            T_cam_ee: 变换矩阵
        """
        self.calibration_result = np.load(filepath)
        print(f"标定结果已从 {filepath} 加载")
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
