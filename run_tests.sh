#!/bin/bash
# PiperSim 测试运行脚本（自动禁用 ROS2 pytest 插件）
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python3 -m pytest tests/ "$@"
