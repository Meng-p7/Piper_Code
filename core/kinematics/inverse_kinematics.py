import numpy as np
from scipy.optimize import minimize
import mujoco


class InverseKinematics:
    """逆运动学：根据目标位姿计算关节角度"""
    
    def __init__(self, model, ee_body_name="link7", joint_names=None):
        """
        初始化逆运动学
        
        Args:
            model: MuJoCo 模型
            ee_body_name: 末端执行器 body 名称
            joint_names: 关节名称列表，默认为 joint1-joint6
        """
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
    
    def _get_joint_limits(self):
        """获取关节限位"""
        limits = []
        for jid in self.joint_ids:
            idx = jid - self.model.jnt_adr[0]
            if self.model.jnt_limited[jid]:
                limits.append((self.model.jnt_range[jid, 0], self.model.jnt_range[jid, 1]))
            else:
                limits.append((-np.pi, np.pi))
        return limits
    
    def solve_position(self, target_pos, q_init=None, orientation_weight=0.0):
        """
        求解位置逆运动学
        
        Args:
            target_pos: 目标位置 [x, y, z]
            q_init: 初始关节角度，默认为零位
            orientation_weight: 姿态权重，0 表示只考虑位置
            
        Returns:
            q_solution: 求解得到的关节角度
            success: 是否求解成功
        """
        if q_init is None:
            q_init = np.zeros(self.num_joints)
        
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
        
        bounds = self.joint_limits[:self.num_joints]
        
        result = minimize(
            cost_function,
            q_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8}
        )
        
        success = result.fun < 1e-4
        return result.x, success
    
    def solve_pose(self, target_pos, target_orientation=None, q_init=None):
        """
        求解位姿逆运动学（位置+姿态）
        
        Args:
            target_pos: 目标位置 [x, y, z]
            target_orientation: 目标旋转矩阵 [3x3]，默认为单位矩阵
            q_init: 初始关节角度
            
        Returns:
            q_solution: 求解得到的关节角度
            success: 是否求解成功
        """
        if q_init is None:
            q_init = np.zeros(self.num_joints)
        
        if target_orientation is None:
            target_orientation = np.eye(3)
        
        def cost_function(q):
            self.data.qpos[self.joint_ids] = q
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.xpos[self.ee_id]
            pos_error = np.sum((current_pos - target_pos) ** 2)
            
            current_mat = self.data.xmat[self.ee_id].reshape(3, 3)
            rot_error = np.sum((current_mat - target_orientation) ** 2)
            
            return pos_error + 0.5 * rot_error
        
        bounds = self.joint_limits[:self.num_joints]
        
        result = minimize(
            cost_function,
            q_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8}
        )
        
        success = result.fun < 1e-3
        return result.x, success
    
    def solve_with_null_space(self, target_pos, q_init=None, null_space_gain=0.1):
        """
        带零空间优化的逆运动学求解
        
        Args:
            target_pos: 目标位置 [x, y, z]
            q_init: 初始关节角度
            null_space_gain: 零空间增益，用于关节限位优化
            
        Returns:
            q_solution: 求解得到的关节角度
            success: 是否求解成功
        """
        if q_init is None:
            q_init = np.zeros(self.num_joints)
        
        q_mid = np.array([(l + u) / 2 for l, u in self.joint_limits[:self.num_joints]])
        
        def cost_function(q):
            self.data.qpos[self.joint_ids] = q
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.xpos[self.ee_id]
            pos_error = np.sum((current_pos - target_pos) ** 2)
            
            null_space_error = null_space_gain * np.sum((q - q_mid) ** 2)
            
            return pos_error + null_space_error
        
        bounds = self.joint_limits[:self.num_joints]
        
        result = minimize(
            cost_function,
            q_init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-8}
        )
        
        success = result.fun < 1e-4
        return result.x, success
