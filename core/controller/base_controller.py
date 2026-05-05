from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np


class BaseController(ABC):
    """机械臂控制器基类：定义仿真和真机通用接口"""
    
    def __init__(self, num_joints: int = 6, gripper_dof: int = 1) -> None:
        """
        初始化控制器
        
        Args:
            num_joints: 机械臂关节数
            gripper_dof: 夹爪自由度数
        """
        self.num_joints = num_joints
        self.gripper_dof = gripper_dof
        self.total_dof = num_joints + gripper_dof
        self.is_connected = False
    
    @abstractmethod
    def connect(self) -> None:
        """连接到机械臂（仿真环境或真机）"""
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """断开连接"""
        pass
    
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """
        获取当前关节角度
        
        Returns:
            qpos: 关节角度数组
        """
        pass
    
    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        """
        获取当前关节速度
        
        Returns:
            qvel: 关节速度数组
        """
        pass
    
    @abstractmethod
    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """
        获取末端执行器位姿
        
        Returns:
            position: 末端位置 [x, y, z]
            orientation: 末端旋转矩阵 [3x3]
        """
        pass
    
    @abstractmethod
    def send_joint_command(self, q_target: np.ndarray) -> None:
        """
        发送关节位置命令
        
        Args:
            q_target: 目标关节角度
        """
        pass
    
    @abstractmethod
    def send_joint_velocity(self, qvel: np.ndarray) -> None:
        """
        发送关节速度命令
        
        Args:
            qvel: 关节速度数组
        """
        pass
    
    @abstractmethod
    def send_gripper_command(self, position: float) -> None:
        """
        发送夹爪位置命令
        
        Args:
            position: 夹爪位置，0 表示闭合，1 表示打开
        """
        pass
    
    @abstractmethod
    def step(self) -> None:
        """执行一步控制"""
        pass
    
    def move_to_joint_position(self, q_target: np.ndarray, speed: float = 0.02) -> None:
        """
        平滑移动到目标关节位置
        
        Args:
            q_target: 目标关节角度
            speed: 插值速度
        """
        q_current = self.get_joint_positions()
        q_cmd = (1 - speed) * q_current + speed * q_target
        self.send_joint_command(q_cmd)
    
    def run_loop(self, duration: float, dt: float | None = None,
                 callback: callable | None = None) -> None:
        """
        实时控制循环
        
        Args:
            duration: 运行总时长（秒）
            dt: 控制周期（秒），默认用模型 timestep
            callback: 每步回调函数，签名为 callback(controller, t) -> None
        """
        if dt is None:
            dt = 0.002
        
        steps = int(duration / dt)
        for i in range(steps):
            t = i * dt
            if callback is not None:
                callback(self, t)
            self.step()
    
    def open_gripper(self) -> None:
        """打开夹爪"""
        self.send_gripper_command(1.0)
    
    def close_gripper(self) -> None:
        """闭合夹爪"""
        self.send_gripper_command(0.0)
