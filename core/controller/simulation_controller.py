import numpy as np
import mujoco
from .base_controller import BaseController


class SimulationController(BaseController):
    """MuJoCo 仿真环境控制器"""
    
    def __init__(self, model, data, joint_names=None, gripper_joint_names=None):
        """
        初始化仿真控制器
        
        Args:
            model: MuJoCo 模型
            data: MuJoCo 数据
            joint_names: 关节名称列表
            gripper_joint_names: 夹爪关节名称列表
        """
        super().__init__()
        self.model = model
        self.data = data
        
        if joint_names is None:
            joint_names = [f"joint{i}" for i in range(1, 7)]
        
        self.joint_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_ids.append(jid)
        
        self.gripper_joint_ids = []
        if gripper_joint_names is not None:
            for name in gripper_joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.gripper_joint_ids.append(jid)
        
        self.num_joints = len(self.joint_ids)
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7")
    
    def connect(self):
        """连接仿真环境"""
        self.is_connected = True
        print("Simulation controller connected")
    
    def disconnect(self):
        """断开仿真环境"""
        self.is_connected = False
        print("Simulation controller disconnected")
    
    def get_joint_positions(self):
        """获取当前关节角度"""
        return self.data.qpos[self.joint_ids].copy()
    
    def get_joint_velocities(self):
        """获取当前关节速度"""
        return self.data.qvel[self.joint_ids].copy()
    
    def get_ee_pose(self):
        """获取末端执行器位姿"""
        position = self.data.xpos[self.ee_body_id].copy()
        orientation = self.data.xmat[self.ee_body_id].copy().reshape(3, 3)
        return position, orientation
    
    def send_joint_command(self, q_target):
        """发送关节位置命令"""
        for i, jid in enumerate(self.joint_ids):
            self.data.ctrl[jid] = q_target[i]
    
    def send_gripper_command(self, position):
        """
        发送夹爪位置命令
        
        Args:
            position: 0 表示闭合，1 表示打开
        """
        gripper_range = 0.04
        gripper_pos = position * gripper_range
        
        for jid in self.gripper_joint_ids:
            self.data.ctrl[jid] = gripper_pos
    
    def step(self):
        """执行一步仿真"""
        mujoco.mj_step(self.model, self.data)
    
    def get_camera_image(self, camera_name="camera", width=640, height=480):
        """
        获取相机图像
        
        Args:
            camera_name: 相机名称
            width: 图像宽度
            height: 图像高度
            
        Returns:
            image: RGB 图像数组
        """
        from mujoco import MjvOption, MjrContext, mjtCamera
        
        camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        
        image = np.zeros((height, width, 3), dtype=np.uint8)
        
        return image
    
    def render(self, viewer):
        """渲染仿真画面"""
        viewer.sync()
