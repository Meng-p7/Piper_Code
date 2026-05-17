from __future__ import annotations

import numpy as np
import mujoco
from utils.logger import get_logger

logger = get_logger(__name__)


class PBVSController:
    """PBVS（基于位置的视觉伺服）控制器

    在世界坐标系下计算末端位姿误差，通过阻尼伪逆映射为关节速度。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        joint_names: list[str],
        ee_body_name: str,
        Kp: float = 2.0,
        Ko: float = 1.0,
        lambda_damping: float = 0.05,
        pos_tol: float = 1e-3,
        ori_tol: float = 0.017,
        max_iter: int = 5000,
    ) -> None:
        """
        Args:
            model: MuJoCo 模型
            data: MuJoCo 数据（用于计算雅可比，会复制内部状态避免副作用）
            joint_names: 机械臂关节名称列表
            ee_body_name: 末端执行器 body 名称
            Kp: 位置增益
            Ko: 姿态增益
            lambda_damping: 阻尼系数（避免奇异）
            pos_tol: 位置收敛阈值（m）
            ori_tol: 姿态收敛阈值（rad）
            max_iter: 最大迭代步数
        """
        self.model = model
        self.data = data
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)

        self.joint_qpos_adrs = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_qpos_adrs.append(model.jnt_qposadr[jid])

        self.num_joints = len(self.joint_qpos_adrs)
        self.Kp = Kp
        self.Ko = Ko
        self.lambda_damping = lambda_damping
        self.pos_tol = pos_tol
        self.ori_tol = ori_tol
        self.max_iter = max_iter

    def compute_error(
        self, target_pos: np.ndarray, target_ori: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """计算当前末端与目标位姿的误差

        Returns:
            pos_error: 位置误差（世界坐标系，3维）
            ori_error: 姿态误差（轴角形式，3维）
        """
        current_pos = self.data.xpos[self.ee_body_id].copy()
        current_mat = self.data.xmat[self.ee_body_id].copy().reshape(3, 3)

        pos_error = target_pos - current_pos

        if target_ori is None:
            target_ori = np.eye(3)

        # 姿态误差：轴角表示（旋转矩阵差分的对数映射）
        R_err = target_ori @ current_mat.T
        ori_error = self._rotation_matrix_to_axis_angle(R_err)

        return pos_error, ori_error

    def compute_joint_velocity(
        self,
        target_pos: np.ndarray,
        target_ori: np.ndarray | None = None,
        qpos: np.ndarray | None = None,
    ) -> np.ndarray:
        """计算关节速度命令

        控制律: Δq = pinv(J) · [Kp·e_p, Ko·e_o]^T
        使用阻尼最小二乘避免奇异: J_damped = J^T (J J^T + λ²I)^-1

        Args:
            target_pos: 目标位置
            target_ori: 目标旋转矩阵（可选）
            qpos: 当前关节角度（可选，默认用 data.qpos）

        Returns:
            qvel: 关节速度命令
        """
        if qpos is not None:
            for i, adr in enumerate(self.joint_qpos_adrs):
                self.data.qpos[adr] = qpos[i]
            mujoco.mj_forward(self.model, self.data)

        # 计算误差
        pos_error, ori_error = self.compute_error(target_pos, target_ori)

        # 构建任务空间速度指令
        vel_desired = np.zeros(6)
        vel_desired[:3] = self.Kp * pos_error
        vel_desired[3:] = self.Ko * ori_error

        # 计算几何雅可比（世界坐标系，相对于末端执行器）
        jac = np.zeros((6, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac[:3], jac[3:], self.ee_body_id)

        # 提取机械臂对应的列
        jac_arm = np.zeros((6, self.num_joints))
        for i, adr in enumerate(self.joint_qpos_adrs):
            jac_arm[:, i] = jac[:, adr]

        # 阻尼最小二乘求逆
        qvel = self._damped_pinv(jac_arm, vel_desired)

        return qvel

    def is_converged(
        self, target_pos: np.ndarray, target_ori: np.ndarray | None = None
    ) -> bool:
        """判断是否收敛

        位置误差 < pos_tol 且 姿态误差 < ori_tol
        """
        pos_error, ori_error = self.compute_error(target_pos, target_ori)
        pos_norm = np.linalg.norm(pos_error)
        ori_norm = np.linalg.norm(ori_error)

        return pos_norm < self.pos_tol and ori_norm < self.ori_tol

    def run_pbvs_loop(
        self,
        controller,
        target_pos: np.ndarray,
        target_ori: np.ndarray | None = None,
        duration: float | None = None,
        dt: float | None = None,
        on_converged: callable | None = None,
        steps_per_ctrl: int = 1,
    ) -> dict:
        """执行 PBVS 闭环控制循环

        Args:
            controller: 控制器实例（需实现 get_joint_positions, send_joint_command, step）
            target_pos: 目标位置
            target_ori: 目标姿态（可选）
            duration: 最大运行时长（秒），默认用 max_iter * dt
            dt: 控制周期（秒），默认 0.002
            on_converged: 收敛后回调函数，签名为 on_converged(controller) -> None
            steps_per_ctrl: 每个控制命令执行多少步仿真（用于位置控制模式）

        Returns:
            result: 包含收敛状态、误差历史等信息的字典
        """
        if dt is None:
            dt = 0.002
        if duration is None:
            duration = self.max_iter * dt

        max_steps = int(duration / dt)
        pos_errors = []
        ori_errors = []
        converged = False
        converged_step = -1

        for step in range(max_steps):
            q = controller.get_joint_positions()

            # 将当前关节角同步到内部 data（用于计算雅可比和误差）
            for i, adr in enumerate(self.joint_qpos_adrs):
                self.data.qpos[adr] = q[i]
            mujoco.mj_forward(self.model, self.data)

            pos_error, ori_error = self.compute_error(target_pos, target_ori)
            pos_errors.append(np.linalg.norm(pos_error))
            ori_errors.append(np.linalg.norm(ori_error))

            if self.is_converged(target_pos, target_ori):
                converged = True
                converged_step = step
                logger.info(
                    f"PBVS 收敛于第 {step} 步，"
                    f"位置误差={pos_errors[-1]:.6f}m，姿态误差={ori_errors[-1]:.6f}rad"
                )
                if on_converged is not None:
                    on_converged(controller)
                break

            qvel = self.compute_joint_velocity(target_pos, target_ori, qpos=q)

            # 直接积分更新关节位置（避免位置控制延迟）
            q_new = q + qvel * dt
            controller.send_joint_command(q_new)

            for _ in range(steps_per_ctrl):
                controller.step()

        if not converged:
            logger.warning(
                f"PBVS 未收敛，最终位置误差={pos_errors[-1]:.6f}m，"
                f"姿态误差={ori_errors[-1]:.6f}rad"
            )

        return {
            "converged": converged,
            "converged_step": converged_step,
            "pos_errors": np.array(pos_errors),
            "ori_errors": np.array(ori_errors),
            "final_pos_error": pos_errors[-1] if pos_errors else float("inf"),
            "final_ori_error": ori_errors[-1] if ori_errors else float("inf"),
        }

    @staticmethod
    def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
        """旋转矩阵转轴角（3维向量，方向为旋转轴，模长为旋转角）"""
        trace = np.trace(R)
        angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))

        if angle < 1e-6:
            return np.zeros(3)

        rx = R[2, 1] - R[1, 2]
        ry = R[0, 2] - R[2, 0]
        rz = R[1, 0] - R[0, 1]
        axis = np.array([rx, ry, rz])

        sin_angle = np.linalg.norm(axis)
        if sin_angle < 1e-6:
            return np.zeros(3)

        axis = axis / sin_angle
        return axis * angle

    def _damped_pinv(self, J: np.ndarray, vel: np.ndarray) -> np.ndarray:
        """阻尼最小二乘伪逆

        qvel = J^T (J J^T + λ²I)^-1 vel
        """
        JT = J.T
        JJt = J @ JT
        damping = self.lambda_damping ** 2 * np.eye(JJt.shape[0])
        return JT @ np.linalg.solve(JJt + damping, vel)
