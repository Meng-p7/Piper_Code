import numpy as np
from scipy.optimize import minimize
import mujoco


class InverseKinematics:
    def __init__(self, model, ee_body_name="link6", joint_names=None):
        self.model = model
        self.data = mujoco.MjData(model)
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        
        if joint_names is None:
            joint_names = [f"joint{i}" for i in range(1, 7)]
        
        self.joint_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_ids.append(jid)
        
        self.joint_limits = self._get_joint_limits()
        self.num_joints = len(self.joint_ids)
        
        self._init_valid_qpos()
    
    def _init_valid_qpos(self):
        for jid in range(self.model.njnt):
            if self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr:adr+7] = [1, 0, 0, 0, 0, 0, 0]
            elif self.model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_BALL:
                adr = self.model.jnt_qposadr[jid]
                self.data.qpos[adr:adr+4] = [1, 0, 0, 0]
    
    def _get_joint_limits(self):
        limits = []
        for jid in self.joint_ids:
            if self.model.jnt_limited[jid]:
                limits.append((self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]))
            else:
                limits.append((-3.14, 3.14))
        return limits
    
    def solve_position(self, target_pos, q_init=None, q_full=None, orientation_weight=0.0):
        if q_init is None:
            q_init = np.zeros(self.num_joints)
        
        if q_full is not None:
            self.data.qpos[:] = q_full.copy()
            self._ensure_valid_quaternions()
        
        def cost_function(q):
            self.data.qpos[self.joint_ids] = q
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.xpos[self.ee_id]
            pos_error = np.sum((current_pos - target_pos) ** 2)
            
            if orientation_weight > 0:
                current_mat = self.data.xmat[self.ee_id].reshape(3, 3)
                target_mat = np.eye(3)
                rot_error = np.sum((current_mat - target_mat) ** 2)
                return pos_error + orientation_weight * rot_error
            
            return pos_error
        
        result = minimize(
            cost_function,
            q_init,
            method="L-BFGS-B",
            bounds=self.joint_limits,
            options={"maxiter": 1000, "ftol": 1e-8}
        )
        
        success = result.fun < 1e-4
        return result.x, success
    
    def solve_pose(self, target_pos, target_orientation=None, q_init=None, q_full=None):
        if q_init is None:
            q_init = np.zeros(self.num_joints)
        
        if target_orientation is None:
            target_orientation = np.eye(3)
        
        if q_full is not None:
            self.data.qpos[:] = q_full.copy()
            self._ensure_valid_quaternions()
        
        def cost_function(q):
            self.data.qpos[self.joint_ids] = q
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.xpos[self.ee_id]
            pos_error = np.sum((current_pos - target_pos) ** 2)
            
            current_mat = self.data.xmat[self.ee_id].reshape(3, 3)
            rot_error = np.sum((current_mat - target_orientation) ** 2)
            
            return pos_error + 0.5 * rot_error
        
        result = minimize(
            cost_function,
            q_init,
            method="L-BFGS-B",
            bounds=self.joint_limits,
            options={"maxiter": 1000, "ftol": 1e-8}
        )
        
        success = result.fun < 1e-3
        return result.x, success
    
    def _ensure_valid_quaternions(self):
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
