from __future__ import annotations

import numpy as np
import mujoco
from .base_controller import BaseController
from utils.config_loader import config
from utils.logger import get_logger

logger = get_logger(__name__)


class SimulationController(BaseController):
    """MuJoCo 仿真环境控制器"""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData,
                 joint_names: list[str] | None = None,
                 gripper_joint_names: list[str] | None = None,
                 ee_body_name: str | None = None,
                 gripper_range: float | None = None) -> None:
        """
        初始化仿真控制器
        
        Args:
            model: MuJoCo 模型
            data: MuJoCo 数据
            joint_names: 关节名称列表，默认从 config.yaml 读取
            gripper_joint_names: 夹爪关节名称列表，默认从 config.yaml 读取
            ee_body_name: 末端执行器 body 名称，默认从 config.yaml 读取
            gripper_range: 夹爪行程范围 (m)，默认从 config.yaml 读取
        """
        super().__init__()
        self.model = model
        self.data = data
        
        if joint_names is None:
            joint_names = config.robot.joint_names
        
        if gripper_joint_names is None:
            try:
                gripper_joint_names = config.robot.gripper_joint_names
            except AttributeError:
                gripper_joint_names = ["joint7", "joint8"]
        
        if ee_body_name is None:
            try:
                ee_body_name = config.robot.ee_body_name
            except AttributeError:
                ee_body_name = "link7"
        
        if gripper_range is None:
            try:
                gripper_range = config.robot.gripper.range
            except AttributeError:
                gripper_range = 0.035
        
        self.gripper_range = gripper_range
        
        self.joint_ids = []
        self.joint_qpos_adrs = []
        self.arm_actuator_ids = []
        for name in joint_names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.joint_ids.append(jid)
            self.joint_qpos_adrs.append(model.jnt_qposadr[jid])
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            self.arm_actuator_ids.append(aid)
        
        self.gripper_joint_ids = []
        self.gripper_actuator_ids = []
        if gripper_joint_names is not None:
            for name in gripper_joint_names:
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.gripper_joint_ids.append(jid)
            gripper_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
            self.gripper_actuator_ids.append(gripper_actuator_id)
        
        self.num_joints = len(self.joint_ids)
        self.ee_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body_name)
    
    def connect(self) -> None:
        """连接仿真环境"""
        self.is_connected = True
        logger.info("Simulation controller connected")
    
    def disconnect(self) -> None:
        """断开仿真环境"""
        self.is_connected = False
        logger.info("Simulation controller disconnected")
    
    def get_joint_positions(self) -> np.ndarray:
        """获取当前关节角度"""
        qpos = np.zeros(len(self.joint_qpos_adrs))
        for i, adr in enumerate(self.joint_qpos_adrs):
            qpos[i] = self.data.qpos[adr]
        return qpos
    
    def get_joint_velocities(self) -> np.ndarray:
        """获取当前关节速度"""
        qvel = np.zeros(len(self.joint_qpos_adrs))
        for i, adr in enumerate(self.joint_qpos_adrs):
            qvel[i] = self.data.qvel[adr]
        return qvel
    
    def get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """获取末端执行器位姿"""
        position = self.data.xpos[self.ee_body_id].copy()
        orientation = self.data.xmat[self.ee_body_id].copy().reshape(3, 3)
        return position, orientation
    
    def send_joint_command(self, q_target: np.ndarray) -> None:
        """发送关节位置命令"""
        for i, aid in enumerate(self.arm_actuator_ids):
            self.data.ctrl[aid] = q_target[i]
    
    def send_joint_velocity(self, qvel: np.ndarray) -> None:
        """
        发送关节速度命令
        
        在位置控制模式下，通过 q_target = q_current + qvel * dt 近似实现速度控制。
        
        Args:
            qvel: 关节速度数组
        """
        q_current = self.get_joint_positions()
        dt = self.model.opt.timestep
        q_target = q_current + qvel * dt
        self.send_joint_command(q_target)
    
    def send_gripper_command(self, position: float) -> None:
        """
        发送夹爪位置命令
        
        Args:
            position: 0 表示闭合，1 表示打开
        """
        gripper_pos = position * self.gripper_range
        
        for aid in self.gripper_actuator_ids:
            self.data.ctrl[aid] = gripper_pos
    
    def step(self) -> None:
        """执行一步仿真"""
        mujoco.mj_step(self.model, self.data)
    
    def render(self, viewer) -> None:
        """渲染仿真画面"""
        viewer.sync()
