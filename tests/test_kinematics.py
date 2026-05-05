"""
运动学模块单元测试

测试内容：
- 正运动学输出形状
- FK→IK→FK 往返一致性
- 夹爪中心 IK 求解精度
- 多初始值 IK 收敛性
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# FK/IK 需要 MuJoCo 模型，使用 test_scene.xml
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "test_scene.xml")


def _load_model():
    """加载 MuJoCo 模型（pytest fixture helper）"""
    import mujoco
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


def _make_fk():
    from core.kinematics import ForwardKinematics
    model = _load_model()
    return ForwardKinematics(model, ee_body_name="link6")


def _make_ik():
    from core.kinematics import InverseKinematics
    model = _load_model()
    joint_names = [f"joint{i}" for i in range(1, 7)]
    return InverseKinematics(model, ee_body_name="link6",
                             joint_names=joint_names,
                             gripper_bodies=["link7", "link8"])


class TestForwardKinematics:
    """正运动学测试"""

    def test_fk_output_shapes(self):
        """验证 FK 输出形状正确"""
        fk = _make_fk()
        q = np.array([0.5, 0.8, -1.2, 0.3, 0.1, -0.2])
        pos, ori = fk.compute(q)
        assert pos.shape == (3,)
        assert ori.shape == (3, 3)

    def test_fk_consistency_same_q(self):
        """同一 q 多次调用返回一致结果"""
        fk = _make_fk()
        q = np.array([0.0, 0.5, -0.5, 0.0, 0.0, 0.0])
        pos1, ori1 = fk.compute(q)
        pos2, ori2 = fk.compute(q)
        assert np.allclose(pos1, pos2)
        assert np.allclose(ori1, ori2)

    def test_fk_different_q(self):
        """不同 q 产生不同结果"""
        fk = _make_fk()
        q1 = np.zeros(6)
        q2 = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])
        pos1, _ = fk.compute(q1)
        pos2, _ = fk.compute(q2)
        assert not np.allclose(pos1, pos2)


class TestInverseKinematics:
    """逆运动学测试"""

    def test_ik_solve_position_success(self):
        """IK 能收敛到可达位置"""
        fk = _make_fk()
        ik = _make_ik()
        q_orig = np.array([0.3, 0.6, -0.9, 0.2, 0.1, -0.1])
        pos, _ = fk.compute(q_orig)
        q_solved, success = ik.solve_position(pos, q_init=np.zeros(6))
        assert success, f"IK failed to converge (residual > threshold)"
        pos_back, _ = fk.compute(q_solved)
        error = np.linalg.norm(pos_back - pos)
        assert error < 1e-3, f"FK→IK→FK position error = {error:.2e} > 1e-3"

    def test_fk_ik_roundtrip_multiple(self):
        """多组随机 q 的 FK→IK→FK 往返测试"""
        fk = _make_fk()
        ik = _make_ik()
        rng = np.random.RandomState(42)
        test_qs = [
            np.zeros(6),
            np.array([0.5, 1.0, -1.0, 0.5, 0.3, -0.2]),
            np.array([-0.3, 0.2, -0.5, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.5, -1.5, 0.8, 0.5, -0.4]),
            rng.uniform(-1.0, 1.0, 6),
        ]
        for i, q_orig in enumerate(test_qs):
            pos, _ = fk.compute(q_orig)
            q_solved, success = ik.solve_position(pos, q_init=np.zeros(6))
            assert success, f"Case {i}: IK failed"
            pos_back, _ = fk.compute(q_solved)
            error = np.linalg.norm(pos_back - pos)
            assert error < 1e-3, f"Case {i}: FK→IK→FK error = {error:.2e}"

    def test_ik_gripper_position(self):
        """验证 solve_gripper_position 精度"""
        fk = _make_fk()
        ik = _make_ik()
        q_orig = np.array([0.2, 0.7, -1.0, 0.1, 0.0, 0.0])
        pos, _ = fk.compute(q_orig)
        # 夹爪中心在 link6 下方约 0.135m
        gripper_target = pos + np.array([0.02, -0.01, -0.135])
        q_solved, success = ik.solve_gripper_position(gripper_target,
                                                       q_init=q_orig)
        assert success, "Gripper IK failed to converge"

    def test_ik_multiple_starts(self):
        """不同初始 q 收敛到同一位置"""
        fk = _make_fk()
        ik = _make_ik()
        pos_target = np.array([0.15, 0.1, 0.05])
        starts = [
            np.zeros(6),
            np.array([0.5, 0.5, -0.5, 0.0, 0.0, 0.0]),
            np.array([-0.2, 0.3, -0.3, 0.1, 0.0, 0.0]),
        ]
        results = []
        for q_init in starts:
            q_solved, success = ik.solve_position(pos_target, q_init=q_init)
            assert success
            pos, _ = fk.compute(q_solved)
            results.append(pos.copy())
        # 所有结果应在目标附近
        for pos in results:
            assert np.linalg.norm(pos - pos_target) < 0.01

    def test_ik_no_pose_change(self):
        """q 已对应目标时 IK 返回近似相同的 q"""
        fk = _make_fk()
        ik = _make_ik()
        q_orig = np.array([0.3, 0.6, -0.9, 0.2, 0.1, -0.1])
        pos, _ = fk.compute(q_orig)
        q_solved, success = ik.solve_position(pos, q_init=q_orig)
        assert success
        pos_back, _ = fk.compute(q_solved)
        assert np.linalg.norm(pos_back - pos) < 1e-3
