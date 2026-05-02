"""
PiperSim 结构化日志模块

替换全局 print() 为 Python logging，支持级别控制和文件输出。
用法:
    from utils.logger import get_logger
    logger = get_logger(__name__)
    logger.info("关节角度: %s", qpos)
"""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """
    获取模块级 logger

    Args:
        name: 通常传 __name__

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(
            fmt="[%(levelname)-7s] %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger


def set_level(level: int | str) -> None:
    """
    设置全局日志级别

    Args:
        level: logging.DEBUG / INFO / WARNING / ERROR 或字符串
    """
    logging.getLogger().setLevel(level)


def enable_debug() -> None:
    """开启 DEBUG 级别日志"""
    set_level(logging.DEBUG)
