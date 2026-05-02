"""
轨迹规划器单元测试

测试内容：
- 各插值方法的端点一致性
- SLERP 旋转矩阵正交性
- 梯形速度约束
- 接近轨迹结构
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trajectory import TrajectoryPlanner


class TestLinearInterpolation:
    """线性插值测试"""

    def test_linear_endpoints(self):
        planner = TrajectoryPlanner()
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 2.0])
        traj = planner.linear_interpolation(start, end, num_steps=10)
        assert traj.shape == (10, 2)
        assert np.allclose(traj[0], start)
        assert np.allclose(traj[-1], end)

    def test_linear_shape(self):
        planner = TrajectoryPlanner()
        start = np.array([0.0, 0.5, 1.0])
        end = np.array([1.0, 1.5, 2.0])
        traj = planner.linear_interpolation(start, end, num_steps=100)
        assert traj.shape == (100, 3)


class TestQuinticInterpolation:
    """五次多项式插值测试"""

    def test_quintic_endpoints(self):
        planner = TrajectoryPlanner()
        start = np.array([0.0, 0.0])
        end = np.array([1.0, 2.0])
        traj = planner.quintic_interpolation(start, end, num_steps=50)
        assert np.allclose(traj[0], start, atol=1e-10)
        assert np.allclose(traj[-1], end, atol=1e-10)

    def test_quintic_monotonic_scalar(self):
        """标量五次插值应严格单调"""
        planner = TrajectoryPlanner()
        start = np.array([0.0])
        end = np.array([1.0])
        traj = planner.quintic_interpolation(start, end, num_steps=100)
        diffs = np.diff(traj.flatten())
        assert np.all(diffs >= -1e-12), "Quintic should be monotonic"


class TestCubicInterpolation:
    """三次多项式插值测试"""

    def test_cubic_endpoints(self):
        planner = TrajectoryPlanner()
        start = np.array([0.0])
        end = np.array([1.0])
        traj = planner.cubic_interpolation(start, end, num_steps=50)
        assert np.allclose(traj[0], start)
        assert np.allclose(traj[-1], end)


class TestCartesianLinear:
    """笛卡尔空间线性插值测试"""

    def test_cartesian_positions_linear(self):
        planner = TrajectoryPlanner()
        start = np.array([0.0, 0.0, 0.0])
        end = np.array([1.0, 2.0, 3.0])
        positions, _ = planner.cartesian_linear(start, end, num_steps=10)
        # 位置应均匀等间距
        diffs = np.diff(positions, axis=0)
        assert np.allclose(diffs, diffs[0])

    def test_cartesian_slerp_orthogonality(self):
        """SLERP 中间帧旋转矩阵应保持正交"""
        from scipy.spatial.transform import Rotation
        planner = TrajectoryPlanner()
        R_start = np.eye(3)
        R_end = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        _, orientations = planner.cartesian_linear(
            np.zeros(3), np.ones(3), num_steps=10,
            orientation_start=R_start, orientation_end=R_end
        )
        for i, R in enumerate(orientations):
            ortho_err = np.max(np.abs(R @ R.T - np.eye(3)))
            assert ortho_err < 1e-10, (
                f"Step {i}: orthogonality error = {ortho_err:.2e}")

    def test_cartesian_default_orientations(self):
        """默认（None）旋转应为全单位矩阵"""
        planner = TrajectoryPlanner()
        _, orientations = planner.cartesian_linear(
            np.zeros(3), np.ones(3), num_steps=5
        )
        assert orientations.shape == (5, 3, 3)
        for R in orientations:
            assert np.allclose(R, np.eye(3))


class TestTrapezoidalVelocity:
    """梯形速度规划测试"""

    def test_trapezoidal_endpoints(self):
        planner = TrajectoryPlanner(max_velocity=0.5, max_acceleration=0.3)
        start = np.array([0.0])
        end = np.array([2.0])
        traj, vel = planner.trapezoidal_velocity(start, end)
        assert np.allclose(traj[0], start, atol=0.01)
        assert np.allclose(traj[-1], end, atol=0.01)

    def test_trapezoidal_velocity_limit(self):
        planner = TrajectoryPlanner(max_velocity=0.3, max_acceleration=0.5)
        start = np.array([0.0])
        end = np.array([2.0])
        _, vel = planner.trapezoidal_velocity(start, end)
        max_vel = np.max(np.abs(vel))
        assert max_vel <= planner.max_velocity + 1e-6, (
            f"Max velocity {max_vel:.3f} exceeds limit {planner.max_velocity}")


class TestApproachTrajectory:
    """接近轨迹测试"""

    def test_approach_structure(self):
        planner = TrajectoryPlanner()
        current = np.array([0.0, 0.0, 0.0])
        target = np.array([0.0, 0.0, -0.2])
        traj = planner.generate_approach_trajectory(
            current, target, approach_distance=0.1, num_steps=25
        )
        assert traj.shape == (50, 3)
        assert np.allclose(traj[-1], target, atol=1e-6)
