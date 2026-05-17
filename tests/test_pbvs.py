"""
PBVS（基于位置的视觉伺服）单元测试

测试内容：
- 误差计算正确性
- 阻尼伪逆控制律输出形状
- 收敛判断逻辑
- PBVS 闭环收敛性（从偏移位置到目标）
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "test_scene.xml")


def _load_model():
    import mujoco
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


def _make_pbvs():
    from core.visual_servo import PBVSController
    import mujoco
    model = _load_model()
    data = mujoco.MjData(model)
    joint_names = [f"joint{i}" for i in range(1, 7)]
    return PBVSController(
        model=model,
        data=data,
        joint_names=joint_names,
        ee_body_name="link6",
        Kp=2.0,
        Ko=1.0,
        lambda_damping=0.05,
    )


class MockController:
    """用于测试 PBVS 闭环的模拟控制器"""

    def __init__(self, model, data, joint_names):
        import mujoco
        self.model = model
        self.data = data
        self.joint_names = joint_names
        self.joint_qpos_adrs = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_qpos_adrs.append(model.jnt_qposadr[jid])

    def get_joint_positions(self):
        q = np.zeros(len(self.joint_qpos_adrs))
        for i, adr in enumerate(self.joint_qpos_adrs):
            q[i] = self.data.qpos[adr]
        return q

    def get_joint_velocities(self):
        qvel = np.zeros(len(self.joint_qpos_adrs))
        for i, adr in enumerate(self.joint_qpos_adrs):
            qvel[i] = self.data.qvel[adr]
        return qvel

    def send_joint_command(self, q_target):
        """直接设置关节位置并更新运动学"""
        import mujoco
        for i, adr in enumerate(self.joint_qpos_adrs):
            self.data.qpos[adr] = q_target[i]
        mujoco.mj_forward(self.model, self.data)

    def send_joint_velocity(self, qvel):
        """通过位置命令近似实现速度控制"""
        dt = self.model.opt.timestep
        q_current = self.get_joint_positions()
        q_target = q_current + qvel * dt
        self.send_joint_command(q_target)

    def step(self):
        """跳过物理积分，因为 send_joint_command 已直接更新位置"""
        pass

    def send_gripper_command(self, position):
        pass


class TestPBVSBasic:
    """PBVS 基础功能测试"""

    def test_error_computation(self):
        """误差计算：当前位置与目标位置的差"""
        pbvs = _make_pbvs()
        import mujoco
        model = _load_model()
        data = mujoco.MjData(model)

        # 设置一个已知关节角
        q = np.array([0.3, 0.6, -0.9, 0.2, 0.1, -0.1])
        for i, adr in enumerate(pbvs.joint_qpos_adrs):
            data.qpos[adr] = q[i]
        mujoco.mj_forward(model, data)

        # 将当前状态复制到 PBVS
        pbvs.data.qpos[:] = data.qpos.copy()
        mujoco.mj_forward(model, pbvs.data)

        current_pos = data.xpos[pbvs.ee_body_id].copy()
        target_pos = current_pos + np.array([0.05, 0.03, -0.02])

        pos_error, ori_error = pbvs.compute_error(target_pos)

        assert pos_error.shape == (3,)
        assert ori_error.shape == (3,)
        assert np.linalg.norm(pos_error - (target_pos - current_pos)) < 1e-6

    def test_convergence_true(self):
        """当前位姿等于目标位姿时应收敛"""
        pbvs = _make_pbvs()
        import mujoco
        model = _load_model()
        data = mujoco.MjData(model)

        q = np.array([0.3, 0.6, -0.9, 0.2, 0.1, -0.1])
        for i, adr in enumerate(pbvs.joint_qpos_adrs):
            data.qpos[adr] = q[i]
        mujoco.mj_forward(model, data)

        pbvs.data.qpos[:] = data.qpos.copy()
        mujoco.mj_forward(model, pbvs.data)

        current_pos = data.xpos[pbvs.ee_body_id].copy()
        current_mat = data.xmat[pbvs.ee_body_id].copy().reshape(3, 3)

        assert pbvs.is_converged(current_pos, current_mat)

    def test_convergence_false(self):
        """目标位姿远离当前位姿时不应收敛"""
        pbvs = _make_pbvs()
        import mujoco
        model = _load_model()
        data = mujoco.MjData(model)

        q = np.zeros(6)
        for i, adr in enumerate(pbvs.joint_qpos_adrs):
            data.qpos[adr] = q[i]
        mujoco.mj_forward(model, data)

        pbvs.data.qpos[:] = data.qpos.copy()
        mujoco.mj_forward(model, pbvs.data)

        target_pos = np.array([0.5, 0.3, 0.4])
        assert not pbvs.is_converged(target_pos)

    def test_joint_velocity_shape(self):
        """关节速度输出形状正确"""
        pbvs = _make_pbvs()
        import mujoco
        model = _load_model()
        data = mujoco.MjData(model)

        q = np.array([0.3, 0.6, -0.9, 0.2, 0.1, -0.1])
        for i, adr in enumerate(pbvs.joint_qpos_adrs):
            data.qpos[adr] = q[i]
        mujoco.mj_forward(model, data)

        pbvs.data.qpos[:] = data.qpos.copy()
        mujoco.mj_forward(model, pbvs.data)

        target_pos = data.xpos[pbvs.ee_body_id].copy() + np.array([0.02, 0.01, -0.01])
        qvel = pbvs.compute_joint_velocity(target_pos)

        assert qvel.shape == (6,)

    def test_damped_pinv_avoids_singularity(self):
        """阻尼伪逆在奇异附近应产生有限输出"""
        pbvs = _make_pbvs()
        import mujoco
        model = _load_model()
        data = mujoco.MjData(model)

        # 接近奇异位姿（腕部关节对齐）
        q = np.array([0.0, 1.57, -1.57, 0.0, 0.0, 0.0])
        for i, adr in enumerate(pbvs.joint_qpos_adrs):
            data.qpos[adr] = q[i]
        mujoco.mj_forward(model, data)

        pbvs.data.qpos[:] = data.qpos.copy()
        mujoco.mj_forward(model, pbvs.data)

        target_pos = data.xpos[pbvs.ee_body_id].copy() + np.array([0.05, 0.0, 0.0])
        qvel = pbvs.compute_joint_velocity(target_pos)

        assert np.all(np.isfinite(qvel))
        assert np.linalg.norm(qvel) < 100.0  # 速度不应过大


class TestPBVSConvergence:
    """PBVS 闭环收敛测试"""

    def test_pbvs_converges_to_target(self):
        """PBVS 从偏移位置收敛到目标位置"""
        from core.visual_servo import PBVSController
        import mujoco

        model = _load_model()
        data = mujoco.MjData(model)
        joint_names = [f"joint{i}" for i in range(1, 7)]

        # 设置初始关节角
        q_init = np.array([0.3, 0.6, -0.9, 0.2, 0.1, -0.1])
        for i, name in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = q_init[i]
        mujoco.mj_forward(model, data)

        # 目标位置：当前位置偏移 5cm
        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        current_pos = data.xpos[ee_id].copy()
        target_pos = current_pos + np.array([0.05, -0.03, 0.02])

        pbvs = PBVSController(
            model=model,
            data=mujoco.MjData(model),
            joint_names=joint_names,
            ee_body_name="link6",
            Kp=3.0,
            Ko=1.5,
            lambda_damping=0.05,
            pos_tol=1e-3,
            ori_tol=0.017,
            max_iter=3000,
        )

        controller = MockController(model, data, joint_names)

        result = pbvs.run_pbvs_loop(
            controller=controller,
            target_pos=target_pos,
            duration=6.0,
            dt=model.opt.timestep,
        )

        assert result["converged"], (
            f"PBVS 未收敛，最终位置误差={result['final_pos_error']:.6f}m"
        )
        assert result["final_pos_error"] < 1e-3

    def test_pbvs_large_offset(self):
        """PBVS 从较大偏移（10cm）收敛"""
        from core.visual_servo import PBVSController
        import mujoco

        model = _load_model()
        data = mujoco.MjData(model)
        joint_names = [f"joint{i}" for i in range(1, 7)]

        q_init = np.array([0.0, 0.8, -1.0, 0.0, 0.0, 0.0])
        for i, name in enumerate(joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = q_init[i]
        mujoco.mj_forward(model, data)

        ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
        current_pos = data.xpos[ee_id].copy()
        target_pos = current_pos + np.array([0.10, 0.0, -0.05])

        pbvs = PBVSController(
            model=model,
            data=mujoco.MjData(model),
            joint_names=joint_names,
            ee_body_name="link6",
            Kp=2.0,
            Ko=1.0,
            lambda_damping=0.05,
            pos_tol=1e-3,
            ori_tol=0.017,
            max_iter=5000,
        )

        controller = MockController(model, data, joint_names)

        result = pbvs.run_pbvs_loop(
            controller=controller,
            target_pos=target_pos,
            duration=10.0,
            dt=model.opt.timestep,
        )

        assert result["converged"], (
            f"PBVS 大偏移未收敛，最终位置误差={result['final_pos_error']:.6f}m"
        )
        assert result["final_pos_error"] < 1e-3
