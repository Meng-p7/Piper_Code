import os
import numpy as np
import json
from datetime import datetime


class DataRecorder:
    """数据采集与记录模块"""
    
    def __init__(self, save_dir="./data", frequency=50):
        """
        初始化数据记录器
        
        Args:
            save_dir: 数据保存目录
            frequency: 采集频率 (Hz)
        """
        self.save_dir = save_dir
        self.frequency = frequency
        self.is_recording = False
        self.current_session = None
        self.data_buffer = {
            "timestamps": [],
            "joint_positions": [],
            "joint_velocities": [],
            "ee_positions": [],
            "ee_orientations": [],
            "gripper_positions": [],
            "images": [],
            "target_positions": []
        }
        
        os.makedirs(save_dir, exist_ok=True)
    
    def start_session(self, task_name="default"):
        """
        开始新的采集会话
        
        Args:
            task_name: 任务名称
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_name = f"{task_name}_{timestamp}"
        session_dir = os.path.join(self.save_dir, session_name)
        
        os.makedirs(session_dir, exist_ok=True)
        
        self.current_session = {
            "name": session_name,
            "dir": session_dir,
            "task_name": task_name,
            "start_time": timestamp
        }
        
        self.data_buffer = {
            "timestamps": [],
            "joint_positions": [],
            "joint_velocities": [],
            "ee_positions": [],
            "ee_orientations": [],
            "gripper_positions": [],
            "images": [],
            "target_positions": []
        }
        
        self.is_recording = True
        print(f"开始数据采集会话: {session_name}")
    
    def record_step(self, timestamp, qpos, qvel, ee_pos, ee_rot, gripper_pos, 
                    image=None, target_pos=None):
        """
        记录一步数据
        
        Args:
            timestamp: 时间戳
            qpos: 关节角度
            qvel: 关节速度
            ee_pos: 末端位置
            ee_rot: 末端旋转矩阵
            gripper_pos: 夹爪位置
            image: 相机图像 (可选)
            target_pos: 目标位置 (可选)
        """
        if not self.is_recording:
            return
        
        self.data_buffer["timestamps"].append(timestamp)
        self.data_buffer["joint_positions"].append(qpos.tolist())
        self.data_buffer["joint_velocities"].append(qvel.tolist())
        self.data_buffer["ee_positions"].append(ee_pos.tolist())
        self.data_buffer["ee_orientations"].append(ee_rot.tolist())
        self.data_buffer["gripper_positions"].append(gripper_pos)
        
        if image is not None:
            self.data_buffer["images"].append(image)
        
        if target_pos is not None:
            self.data_buffer["target_positions"].append(target_pos.tolist())
    
    def save_session(self):
        """保存当前会话数据"""
        if not self.is_recording or self.current_session is None:
            print("没有活跃的数据采集会话")
            return
        
        session_dir = self.current_session["dir"]
        
        data_dict = {
            "timestamps": self.data_buffer["timestamps"],
            "joint_positions": self.data_buffer["joint_positions"],
            "joint_velocities": self.data_buffer["joint_velocities"],
            "ee_positions": self.data_buffer["ee_positions"],
            "ee_orientations": self.data_buffer["ee_orientations"],
            "gripper_positions": self.data_buffer["gripper_positions"],
            "target_positions": self.data_buffer["target_positions"]
        }
        
        data_file = os.path.join(session_dir, "data.json")
        with open(data_file, "w") as f:
            json.dump(data_dict, f, indent=2)
        
        for i, image in enumerate(self.data_buffer["images"]):
            import cv2
            image_path = os.path.join(session_dir, f"image_{i:05d}.png")
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        metadata = {
            "session_name": self.current_session["name"],
            "task_name": self.current_session["task_name"],
            "start_time": self.current_session["start_time"],
            "num_samples": len(self.data_buffer["timestamps"]),
            "frequency": self.frequency
        }
        
        metadata_file = os.path.join(session_dir, "metadata.json")
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"数据已保存到: {session_dir}")
        print(f"  样本数: {metadata['num_samples']}")
        
        self.is_recording = False
    
    def load_session(self, session_dir):
        """
        加载已保存的会话数据
        
        Args:
            session_dir: 会话目录路径
            
        Returns:
            data: 加载的数据字典
        """
        data_file = os.path.join(session_dir, "data.json")
        
        with open(data_file, "r") as f:
            data = json.load(f)
        
        print(f"已从 {session_dir} 加载数据")
        print(f"  样本数: {len(data['timestamps'])}")
        
        return data
    
    def stop_session(self):
        """停止当前采集会话并保存"""
        if self.is_recording:
            self.save_session()
        else:
            print("没有活跃的数据采集会话")
