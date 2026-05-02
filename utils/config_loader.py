"""
PiperSim 配置加载模块

从 config/config.yaml 加载参数，提供统一的配置访问接口。
替换各模块中的硬编码参数。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import yaml


def _get_project_root() -> str:
    """获取项目根目录（PiperSim/）"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path: Optional[str] = None) -> dict[str, Any]:
    """
    加载 YAML 配置文件

    Args:
        config_path: 配置文件路径，默认为 <项目根>/config/config.yaml

    Returns:
        config: 配置字典
    """
    if config_path is None:
        config_path = os.path.join(_get_project_root(), "config", "config.yaml")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


class _ConfigDict(dict):
    """支持属性访问的字典包装器"""

    def __getattr__(self, key: str) -> Any:
        if key in self:
            value = self[key]
            if isinstance(value, dict):
                return _ConfigDict(value)
            return value
        raise AttributeError(f"Config has no key: {key}")


class Config:
    """
    配置单例类，惰性加载 config.yaml，提供属性风格访问。

    用法:
        from utils import config

        config.simulation.timestep        # 0.002
        config.robot.joint_names          # ["joint1", ..., "joint6"]
        config.robot.gripper.open_ctrl    # 0.035
    """

    _instance: Optional["Config"] = None
    _data: Optional[_ConfigDict] = None

    def __new__(cls) -> "Config":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self) -> None:
        if self._data is None:
            raw = load_config()
            self._data = _ConfigDict(raw)

    def __getattr__(self, key: str) -> Any:
        if self._data is not None and key in self._data:
            value = self._data[key]
            if isinstance(value, dict):
                return _ConfigDict(value)
            return value
        raise AttributeError(f"Config has no key: {key}")

    def __getitem__(self, key: str) -> Any:
        if self._data is not None:
            return self._data[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        if self._data is not None:
            return self._data.get(key, default)
        return default

    def reload(self) -> None:
        """重新加载配置文件"""
        raw = load_config()
        self._data = _ConfigDict(raw)

    @property
    def raw(self) -> dict[str, Any]:
        """返回原始配置字典"""
        return dict(self._data) if self._data is not None else {}


# 全局单例实例
config = Config()
