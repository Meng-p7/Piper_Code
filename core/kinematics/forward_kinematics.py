import numpy as np
import mujoco


class ForwardKinematics:
    """正运动学：根据关节角度计算末端位姿"""
    
    def __init__(self, model, ee_body_name="link7"):
        """
        初始化正运动学
        
        Args:
            model: MuJoCo 模型
            ee_body_name: 末端执行器 body 名称
        """
        self.model = model
        self.data = mujoco.MjData(model)
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
        self.ee_body_name = ee_body_name
    
    def compute(self, qpos):
        """
        计算末端位姿
        
        Args:
            qpos: 关节角度数组
            
        Returns:
            position: 末端位置 [x, y, z]
            orientation: 末端旋转矩阵 [3x3]
        """
        self.data.qpos[:len(qpos)] = qpos
        mujoco.mj_forward(self.model, self.data)
        
        position = self.data.xpos[self.ee_id].copy()
        orientation = self.data.xmat[self.ee_id].copy().reshape(3, 3)
        
        return position, orientation
    
    def get_ee_pose(self, data=None):
        """
        获取当前末端位姿
        
        Args:
            data: MuJoCo 数据对象，若为 None 则使用内部 data
            
        Returns:
            position: 末端位置 [x, y, z]
            orientation: 末端旋转矩阵 [3x3]
        """
        if data is None:
            data = self.data
            
        position = data.xpos[self.ee_id].copy()
        orientation = data.xmat[self.ee_id].copy().reshape(3, 3)
        
        return position, orientation
