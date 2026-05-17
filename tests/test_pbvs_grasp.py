"""
PBVS 抓取流程自动化测试（无 GUI）

验证内容：
- PBVS 从观察位姿接近小球上方
- PBVS 下降到预抓取位置
- PBVS 精确接近抓取位置（夹爪中心对准球心）
- 最终夹爪中心-球心偏差 < 2mm
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "models", "scene.xml")


def _load_model():
    import mujoco
    if not os.path.exists(MODEL_PATH):
        pytest.skip(f"Model file not found: {MODEL_PATH}")
    return mujoco.MjModel.from_xml_path(MODEL_PATH)


class TestPBVSGraspFlow:
    """PBVS 抓取流程测试"""

    def test_pbvs_approach_ball(self):
        """完整流程：从观察位姿到抓取位置"""
        import mujoco
        from core.visual_servo import PBVSController
        from core.controller import SimulationController
        from utils import config

        model = _load_model()
        data = mujoco.MjData(model)

        arm_joint_names = config.robot.joint_names
        ik_ee_body_name = config.robot.ik_ee_body_name

        controller = SimulationController(model, data)
        controller.connect()

        # 设置观察位姿
        observe_qpos = np.array(config.robot.observe_qpos)
        for i, name in enumerate(arm_joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            adr = model.jnt_qposadr[jid]
            data.qpos[adr] = observe_qpos[i]
        mujoco.mj_forward(model, data)

        # 获取小球真实位置
        ball_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.grasp_demo.ball_body_name)
        ball_pos = data.xpos[ball_id].copy()

        # 创建 PBVS
        pbvs = PBVSController(
            model=model,
            data=mujoco.MjData(model),
            joint_names=arm_joint_names,
            ee_body_name=ik_ee_body_name,
            Kp=3.0,
            Ko=1.5,
            lambda_damping=0.05,
            pos_tol=1e-3,
            ori_tol=0.017,
        )

        link7_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        link8_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link8")

        def get_gripper_center():
            return (data.xpos[link7_id] + data.xpos[link8_id]) / 2.0

        def get_gripper_to_link6_offset():
            link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
            return data.xpos[link6_id].copy() - get_gripper_center()

        # 步骤 1: 接近小球上方 15cm
        approach_pos = ball_pos.copy()
        approach_pos[2] += 0.15

        # 使用更大的控制周期（0.02s = 10 步仿真）
        ctrl_dt = 0.02
        steps_per_ctrl = int(ctrl_dt / model.opt.timestep)
        duration = 4.0
        max_steps = int(duration / ctrl_dt)

        for step in range(max_steps):
            q = controller.get_joint_positions()
            for i, adr in enumerate(pbvs.joint_qpos_adrs):
                pbvs.data.qpos[adr] = q[i]
            mujoco.mj_forward(model, pbvs.data)

            # 计算当前偏移，将夹爪中心目标转换为 link6 目标
            offset = get_gripper_to_link6_offset()
            link6_target = approach_pos + offset

            # 直接检查夹爪中心是否到位（避免目标漂移导致不收敛）
            gripper_err = np.linalg.norm(get_gripper_center() - approach_pos)
            if gripper_err < 5e-3:
                break

            qvel = pbvs.compute_joint_velocity(link6_target, qpos=q)
            q_new = q + qvel * ctrl_dt
            # 直接设置 qpos 绕过位置控制延迟
            for i, adr in enumerate(pbvs.joint_qpos_adrs):
                data.qpos[adr] = q_new[i]
            mujoco.mj_forward(model, data)

        gripper_err_approach = np.linalg.norm(get_gripper_center() - approach_pos)
        print(f"\n  接近位置偏差: {gripper_err_approach:.4f}m")
        assert gripper_err_approach < 5e-3, f"接近位置偏差过大: {gripper_err_approach:.4f}m"

        # 步骤 2: 下降到预抓取位置（球上方 3cm）
        pre_grasp_pos = ball_pos.copy()
        pre_grasp_pos[2] += 0.03

        for step in range(max_steps):
            q = controller.get_joint_positions()
            for i, adr in enumerate(pbvs.joint_qpos_adrs):
                pbvs.data.qpos[adr] = q[i]
            mujoco.mj_forward(model, pbvs.data)

            offset = get_gripper_to_link6_offset()
            link6_target = pre_grasp_pos + offset

            gripper_err = np.linalg.norm(get_gripper_center() - pre_grasp_pos)
            if gripper_err < 3e-3:
                break

            qvel = pbvs.compute_joint_velocity(link6_target, qpos=q)
            q_new = q + qvel * ctrl_dt
            for i, adr in enumerate(pbvs.joint_qpos_adrs):
                data.qpos[adr] = q_new[i]
            mujoco.mj_forward(model, data)

        gripper_err_pre = np.linalg.norm(get_gripper_center() - pre_grasp_pos)
        print(f"  预抓取位置偏差: {gripper_err_pre:.4f}m")
        assert gripper_err_pre < 3e-3, f"预抓取位置偏差过大: {gripper_err_pre:.4f}m"

        # 步骤 3: 精确接近抓取位置（夹爪中心对准球心）
        grasp_pos = ball_pos.copy()

        for step in range(max_steps):
            q = controller.get_joint_positions()
            for i, adr in enumerate(pbvs.joint_qpos_adrs):
                pbvs.data.qpos[adr] = q[i]
            mujoco.mj_forward(model, pbvs.data)

            offset = get_gripper_to_link6_offset()
            link6_target = grasp_pos + offset

            gripper_err = np.linalg.norm(get_gripper_center() - grasp_pos)
            if gripper_err < 2e-3:
                break

            qvel = pbvs.compute_joint_velocity(link6_target, qpos=q)
            q_new = q + qvel * ctrl_dt
            for i, adr in enumerate(pbvs.joint_qpos_adrs):
                data.qpos[adr] = q_new[i]
            mujoco.mj_forward(model, data)

        final_gripper_center = get_gripper_center()
        gripper_err_grasp = np.linalg.norm(final_gripper_center - grasp_pos)
        print(f"  抓取位置偏差: {gripper_err_grasp:.4f}m")
        print(f"  夹爪中心: {final_gripper_center}")
        print(f"  小球位置: {grasp_pos}")

        assert gripper_err_grasp < 2e-3, f"抓取位置偏差过大: {gripper_err_grasp:.4f}m"
