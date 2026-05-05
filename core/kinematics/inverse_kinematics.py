from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
import mujoco


class InverseKinematics:
    def __init__(self, model: mujoco.MjModel, ee_body_name: str = "link6",
                 joint_names: list[str] | None = None,
                 gripper_bodies: list[str] | None = None) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        self.gripper_ids = []
        if gripper_bodies:
            for name in gripper_bodies:
                self.gripper_ids.append(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        
        if joint_names is None:
            joint_names = [f"joint{i}" for i in range(1, 7)]
        
        self.joint_ids = []
        self.joint_qpos_adrs = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_ids.append(jid)
            self.joint_qpos_adrs.append(model.jnt_qposadr[jid])
        
        self.joint_limits = self._get_joint_limits()
        self.num_joints = len(self.joint_ids)
        
        self._init_valid_qpos()
    
    def _init_valid_qpos(self) -> None:
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr:adr+7] = [1, 0, 0, 0, 0, 0, 0]
            elif self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_BALL:
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr:adr+4] = [1, 0, 0, 0]
    
    def _get_joint_limits(self) -> list[tuple[float, float]]:
        limits = []
        for jid in self.joint_ids:
            if self.model.jnt_limited[jid]:
                limits.append((self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]))
            else:
                limits.append((-3.14, 3.14))
        return limits
    
    def _random_init(self) -> np.ndarray:
        """在关节限位内生成随机初始值"""
        q = np.zeros(self.num_joints)
        for i, (low, high) in enumerate(self.joint_limits):
            q[i] = np.random.uniform(low, high)
        return q
    
    def solve_position(self, target_pos: np.ndarray, q_init: np.ndarray | None = None,
                       q_full: np.ndarray | None = None, orientation_weight: float = 0.0,
                       max_retries: int = 3) -> tuple[np.ndarray, bool]:
        if q_init is None:
            q_init = np.zeros(self.num_joints)
        
        if q_full is not None:
            self.data.qpos[:] = q_full.copy()
            self._ensure_valid_quaternions()
        
        def cost_function(q):
            for i, adr in enumerate(self.joint_qpos_adrs):
                self.data.qpos[adr] = q[i]
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.xpos[self.ee_id]
            pos_error = np.sum((current_pos - target_pos) ** 2)
            
            if orientation_weight > 0:
                current_mat = self.data.xmat[self.ee_id].reshape(3, 3)
                target_mat = np.eye(3)
                rot_error = np.sum((current_mat - target_mat) ** 2)
                return pos_error + orientation_weight * rot_error
            
            return pos_error
        
        best_result = None
        best_fun = float('inf')
        
        for attempt in range(max_retries + 1):
            if attempt == 0:
                q0 = q_init.copy()
            else:
                q0 = self._random_init()
            
            result = minimize(
                cost_function,
                q0,
                method="L-BFGS-B",
                bounds=self.joint_limits,
                options={"maxiter": 1000, "ftol": 1e-8}
            )
            
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
            
            if result.fun < 1e-4:
                return result.x, True
        
        return best_result.x, False
    
    def solve_pose(self, target_pos: np.ndarray, target_orientation: np.ndarray | None = None,
                   q_init: np.ndarray | None = None, q_full: np.ndarray | None = None,
                   max_retries: int = 3) -> tuple[np.ndarray, bool]:
        if q_init is None:
            q_init = np.zeros(self.num_joints)
        
        if target_orientation is None:
            target_orientation = np.eye(3)
        
        if q_full is not None:
            self.data.qpos[:] = q_full.copy()
            self._ensure_valid_quaternions()
        
        def cost_function(q):
            for i, adr in enumerate(self.joint_qpos_adrs):
                self.data.qpos[adr] = q[i]
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.xpos[self.ee_id]
            pos_error = np.sum((current_pos - target_pos) ** 2)
            
            current_mat = self.data.xmat[self.ee_id].reshape(3, 3)
            rot_error = np.sum((current_mat - target_orientation) ** 2)
            
            return pos_error + 0.5 * rot_error
        
        best_result = None
        best_fun = float('inf')
        
        for attempt in range(max_retries + 1):
            if attempt == 0:
                q0 = q_init.copy()
            else:
                q0 = self._random_init()
            
            result = minimize(
                cost_function,
                q0,
                method="L-BFGS-B",
                bounds=self.joint_limits,
                options={"maxiter": 1000, "ftol": 1e-8}
            )
            
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
            
            if result.fun < 1e-3:
                return result.x, True
        
        return best_result.x, False
    
    def solve_gripper_position(self, target_pos: np.ndarray, q_init: np.ndarray | None = None,
                               q_full: np.ndarray | None = None,
                               max_retries: int = 3) -> tuple[np.ndarray, bool]:
        if not self.gripper_ids:
            return self.solve_position(target_pos, q_init, q_full, max_retries=max_retries)
        if q_init is None:
            q_init = np.zeros(self.num_joints)
        if q_full is not None:
            self.data.qpos[:] = q_full.copy()
            self._ensure_valid_quaternions()
        def cost_function(q):
            for i, adr in enumerate(self.joint_qpos_adrs):
                self.data.qpos[adr] = q[i]
            mujoco.mj_forward(self.model, self.data)
            gripper_center = np.zeros(3)
            for gid in self.gripper_ids:
                gripper_center += self.data.xpos[gid]
            gripper_center /= len(self.gripper_ids)
            return np.sum((gripper_center - target_pos) ** 2)
        
        best_result = None
        best_fun = float('inf')
        
        for attempt in range(max_retries + 1):
            if attempt == 0:
                q0 = q_init.copy()
            else:
                q0 = self._random_init()
            
            result = minimize(
                cost_function,
                q0,
                method="L-BFGS-B",
                bounds=self.joint_limits,
                options={"maxiter": 2000, "ftol": 1e-12}
            )
            
            if result.fun < best_fun:
                best_fun = result.fun
                best_result = result
            
            if result.fun < 1e-6:
                return result.x, True
        
        return best_result.x, False

    def _ensure_valid_quaternions(self) -> None:
        for jid in self.joint_ids:
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                adr = self.model.jnt_qposadr[jid]
                quat = self.data.qpos[adr:adr+4]
                if np.linalg.norm(quat) < 0.5:
                    self.data.qpos[adr:adr+4] = [1, 0, 0, 0]
            elif self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_BALL:
                adr = self.model.jnt_qposadr[jid]
                quat = self.data.qpos[adr:adr+4]
                if np.linalg.norm(quat) < 0.5:
                    self.data.qpos[adr:adr+4] = [1, 0, 0, 0]
