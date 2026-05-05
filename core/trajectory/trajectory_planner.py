from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


class TrajectoryPlanner:
    """轨迹规划器：生成平滑的关节空间或笛卡尔空间轨迹"""
    
    def __init__(self, max_velocity: float = 0.5, max_acceleration: float = 0.3) -> None:
        """
        初始化轨迹规划器
        
        Args:
            max_velocity: 最大速度
            max_acceleration: 最大加速度
        """
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
    
    def linear_interpolation(self, q_start: np.ndarray, q_end: np.ndarray, num_steps: int) -> np.ndarray:
        """
        线性插值
        
        Args:
            q_start: 起始关节角度
            q_end: 目标关节角度
            num_steps: 插值步数
            
        Returns:
            trajectory: 轨迹数组 [num_steps, num_joints]
        """
        trajectory = np.linspace(q_start, q_end, num_steps)
        return trajectory
    
    def cubic_interpolation(self, q_start: np.ndarray, q_end: np.ndarray, num_steps: int,
                            v_start: float = 0, v_end: float = 0) -> np.ndarray:
        """
        三次多项式插值
        
        Args:
            q_start: 起始关节角度
            q_end: 目标关节角度
            num_steps: 插值步数
            v_start: 起始速度
            v_end: 终止速度
            
        Returns:
            trajectory: 轨迹数组 [num_steps, num_joints]
        """
        t = np.linspace(0, 1, num_steps)
        t2 = t ** 2
        t3 = t ** 3
        
        a0 = q_start
        a1 = v_start
        a2 = 3 * (q_end - q_start) - 2 * v_start - v_end
        a3 = -2 * (q_end - q_start) + v_start + v_end
        
        trajectory = a0 + a1 * t + a2 * t2 + a3 * t3
        return trajectory
    
    def quintic_interpolation(self, q_start: np.ndarray, q_end: np.ndarray, num_steps: int) -> np.ndarray:
        """
        五次多项式插值（位置和速度、加速度连续）
        
        Args:
            q_start: 起始关节角度
            q_end: 目标关节角度
            num_steps: 插值步数
            
        Returns:
            trajectory: 轨迹数组 [num_steps, num_joints]
        """
        t = np.linspace(0, 1, num_steps).reshape(-1, 1)
        t2 = t ** 2
        t3 = t ** 3
        t4 = t ** 4
        t5 = t ** 5
        
        trajectory = q_start + (q_end - q_start) * (10 * t3 - 15 * t4 + 6 * t5)
        return trajectory
    
    def trapezoidal_velocity(self, q_start: np.ndarray, q_end: np.ndarray,
                             max_vel: float | None = None, max_acc: float | None = None,
                             dt: float = 0.002) -> tuple[np.ndarray, np.ndarray]:
        """
        梯形速度规划
        
        Args:
            q_start: 起始关节角度
            q_end: 目标关节角度
            max_vel: 最大速度
            max_acc: 最大加速度
            dt: 时间步长
            
        Returns:
            trajectory: 轨迹数组
            velocities: 速度数组
        """
        if max_vel is None:
            max_vel = self.max_velocity
        if max_acc is None:
            max_acc = self.max_acceleration
        
        dq = q_end - q_start
        distance = np.abs(dq)
        
        t_acc = max_vel / max_acc
        d_acc = 0.5 * max_acc * t_acc ** 2
        
        if 2 * d_acc >= distance:
            t_acc = np.sqrt(distance / max_acc)
            t_const = 0
            t_dec = t_acc
        else:
            t_const = (distance - 2 * d_acc) / max_vel
            t_dec = t_acc
        
        total_time = t_acc + t_const + t_dec
        num_steps = int(total_time / dt)
        
        trajectory = np.zeros((num_steps, len(q_start)))
        velocities = np.zeros((num_steps, len(q_start)))
        
        for i in range(num_steps):
            t = i * dt
            for j in range(len(q_start)):
                d_acc_j = float(d_acc[j]) if hasattr(d_acc, '__getitem__') else float(d_acc)
                if dq[j] >= 0:
                    if t <= t_acc:
                        trajectory[i, j] = q_start[j] + 0.5 * max_acc * t ** 2
                        velocities[i, j] = max_acc * t
                    elif t <= t_acc + t_const:
                        trajectory[i, j] = q_start[j] + d_acc_j + max_vel * (t - t_acc)
                        velocities[i, j] = max_vel
                    else:
                        t_dec_actual = t - t_acc - t_const
                        trajectory[i, j] = q_end[j] - 0.5 * max_acc * (t_dec - t_dec_actual) ** 2
                        velocities[i, j] = max_acc * (t_dec - t_dec_actual)
                else:
                    if t <= t_acc:
                        trajectory[i, j] = q_start[j] - 0.5 * max_acc * t ** 2
                        velocities[i, j] = -max_acc * t
                    elif t <= t_acc + t_const:
                        trajectory[i, j] = q_start[j] - d_acc_j - max_vel * (t - t_acc)
                        velocities[i, j] = -max_vel
                    else:
                        t_dec_actual = t - t_acc - t_const
                        trajectory[i, j] = q_end[j] + 0.5 * max_acc * (t_dec - t_dec_actual) ** 2
                        velocities[i, j] = -max_acc * (t_dec - t_dec_actual)
        
        return trajectory, velocities
    
    def cartesian_linear(self, pos_start: np.ndarray, pos_end: np.ndarray, num_steps: int,
                         orientation_start: np.ndarray | None = None,
                         orientation_end: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        笛卡尔空间直线插值（位置线性 + 旋转 SLERP）
        
        Args:
            pos_start: 起始位置 [x, y, z]
            pos_end: 目标位置 [x, y, z]
            num_steps: 插值步数
            orientation_start: 起始旋转矩阵 [3x3]
            orientation_end: 目标旋转矩阵 [3x3]
            
        Returns:
            positions: 位置轨迹 [num_steps, 3]
            orientations: 旋转矩阵轨迹 [num_steps, 3, 3]
        """
        positions = np.linspace(pos_start, pos_end, num_steps)
        
        if orientation_start is None:
            orientation_start = np.eye(3)
        if orientation_end is None:
            orientation_end = np.eye(3)
        
        # 使用四元数 SLERP 进行旋转插值，保证中间帧为合法旋转矩阵
        rot_start = Rotation.from_matrix(orientation_start)
        rot_end = Rotation.from_matrix(orientation_end)
        slerp = Slerp([0, 1], Rotation.concatenate([rot_start, rot_end]))
        times = np.linspace(0, 1, num_steps)
        orientations = slerp(times).as_matrix()
        
        return positions, orientations
    
    def generate_approach_trajectory(self, current_pos: np.ndarray, target_pos: np.ndarray,
                                      approach_distance: float = 0.1, num_steps: int = 50) -> np.ndarray:
        """
        生成接近目标的轨迹（先移动到目标上方，再下降到目标）
        
        Args:
            current_pos: 当前位置
            target_pos: 目标位置
            approach_distance: 接近距离
            num_steps: 每段轨迹的步数
            
        Returns:
            full_trajectory: 完整轨迹
        """
        approach_pos = target_pos.copy()
        approach_pos[2] += approach_distance
        
        trajectory1 = self.quintic_interpolation(current_pos, approach_pos, num_steps)
        trajectory2 = self.quintic_interpolation(approach_pos, target_pos, num_steps)
        
        full_trajectory = np.vstack([trajectory1, trajectory2])
        return full_trajectory
