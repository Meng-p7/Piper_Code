"""
配置加载模块单元测试

测试内容：
- Config 单例加载成功
- 关键 key 存在且类型正确
- reload 不抛异常
"""

import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestConfig:
    """配置加载测试"""

    def test_config_loads(self):
        """Config() 单例加载成功"""
        from utils import config, load_config
        assert config.simulation is not None
        assert config.robot is not None

        # 也可以通过函数加载
        raw = load_config()
        assert isinstance(raw, dict)
        assert "simulation" in raw

    def test_config_keys_exist(self):
        """验证所有必要 key 存在"""
        from utils import config
        # simulation
        assert hasattr(config.simulation, 'timestep')
        # robot
        assert hasattr(config.robot, 'joint_names')
        assert hasattr(config.robot, 'ee_body_name')
        assert hasattr(config.robot, 'ik_ee_body_name')
        assert hasattr(config.robot, 'gripper')
        assert hasattr(config.robot.gripper, 'open_ctrl')
        assert hasattr(config.robot.gripper, 'close_ctrl')
        assert hasattr(config.robot.gripper, 'range')
        assert hasattr(config.robot, 'home_qpos')
        assert hasattr(config.robot, 'observe_qpos')
        # vision
        assert hasattr(config.vision, 'camera_name')
        assert hasattr(config.vision, 'image_width')
        assert hasattr(config.vision, 'image_height')
        # grasp_demo
        assert hasattr(config.grasp_demo, 'ball_radius')
        assert hasattr(config.grasp_demo, 'approach_height')
        assert hasattr(config.grasp_demo, 'place_position')

    def test_config_value_types(self):
        """验证关键值类型正确"""
        from utils import config
        assert isinstance(config.simulation.timestep, float)
        assert isinstance(config.robot.joint_names, list)
        assert len(config.robot.joint_names) == 6
        assert isinstance(config.robot.gripper.open_ctrl, float)
        assert isinstance(config.robot.home_qpos, list)
        assert len(config.robot.home_qpos) == 6
        assert isinstance(config.vision.image_width, int)
        assert isinstance(config.vision.image_height, int)
        assert isinstance(config.grasp_demo.ball_radius, float)
        assert isinstance(config.grasp_demo.place_position, list)
        assert len(config.grasp_demo.place_position) == 2

    def test_config_singleton(self):
        """Config 是单例"""
        from utils.config_loader import Config
        c1 = Config()
        c2 = Config()
        assert c1 is c2

    def test_config_reload(self):
        """reload 不会抛异常"""
        from utils import config
        config.reload()
        assert config.simulation is not None

    def test_config_get_method(self):
        """config.get() 方法可用"""
        from utils import config
        val = config.get("simulation")
        assert val is not None
        assert hasattr(val, 'timestep')
