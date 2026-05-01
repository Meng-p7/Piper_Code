# Piper 机械臂 MuJoCo 仿真项目

基于 MuJoCo 物理引擎的 Piper 机械臂仿真与控制系统，支持仿真/真机解耦架构，预留手眼标定、运动规划、视觉伺服接口。

## 项目架构

```
Piper_Code/
├── config/                      # 配置文件
│   └── config.yaml              # 系统参数配置
├── models/                      # MuJoCo 模型文件
│   ├── piper.xml                # Piper 机械臂模型
│   └── scene.xml                # 完整场景模型（含小球）
├── core/                        # 核心模块
│   ├── kinematics/              # 运动学模块
│   │   ├── forward_kinematics.py    # 正运动学
│   │   └── inverse_kinematics.py    # 逆运动学（基于优化）
│   ├── trajectory/              # 轨迹规划模块
│   │   └── trajectory_planner.py    # 多种插值方法
│   ├── controller/              # 控制器模块
│   │   ├── base_controller.py       # 控制器基类
│   │   ├── simulation_controller.py # MuJoCo 仿真控制器
│   │   └── real_robot_interface.py  # 真机接口（预留）
│   ├── vision/                  # 视觉模块
│   │   ├── camera.py                # MuJoCo 相机
│   │   └── object_detector.py       # 基于颜色的目标检测
│   ├── calibration/             # 标定模块
│   │   └── hand_eye_calibration.py  # 手眼标定
│   └── data_collection/         # 数据采集模块
│       └── data_recorder.py         # 数据记录与回放
├── utils/                       # 工具函数
├── examples/                    # 示例程序
│   └── grasp_ball_demo.py       # 视觉抓取小球演示
├── tests/                       # 测试代码
├── data/                        # 数据存储目录
├── run_piper.py                 # 基础运行脚本
├── requirements.txt             # 依赖列表
└── README.md                    # 项目文档
```

## 环境要求

### 需要安装的依赖包

```bash
pip install mujoco>=2.3.0
pip install numpy>=1.24.0
pip install scipy>=1.10.0
pip install opencv-python>=4.8.0
pip install matplotlib>=3.7.0
pip install PyYAML>=6.0
```

## 快速开始

### 1. 基础运动控制

```bash
python run_piper.py
```

机械臂将平滑移动到目标位置 `[0.3, 0.2, 0.15]`。

### 2. 视觉抓取小球演示

```bash
python examples/grasp_ball_demo.py
```

演示流程：
1. 机械臂移动到初始位姿
2. 通过相机视觉检测红色小球
3. 移动到小球上方
4. 下降并执行抓取
5. 提升物体
6. 移动到放置位置并释放
7. 返回初始位姿

## 核心模块说明

### 运动学模块 (core/kinematics)

- **ForwardKinematics**: 正运动学，根据关节角度计算末端位姿
- **InverseKinematics**: 逆运动学，支持位置求解、位姿求解、零空间优化

```python
from core.kinematics import ForwardKinematics, InverseKinematics

fk = ForwardKinematics(model, ee_body_name="link7")
ik = InverseKinematics(model, ee_body_name="link7")

# 正运动学
position, orientation = fk.compute(qpos)

# 逆运动学
q_target, success = ik.solve_position([0.3, 0.2, 0.15])
```

### 轨迹规划模块 (core/trajectory)

支持多种插值方法：
- 线性插值
- 三次多项式插值
- 五次多项式插值（位置、速度、加速度连续）
- 梯形速度规划
- 笛卡尔空间直线插值

```python
from core.trajectory import TrajectoryPlanner

planner = TrajectoryPlanner(max_velocity=0.5, max_acceleration=0.3)

# 五次多项式插值
trajectory = planner.quintic_interpolation(q_start, q_end, num_steps=100)

# 梯形速度规划
trajectory, velocities = planner.trapezoidal_velocity(q_start, q_end)
```

### 控制器模块 (core/controller)

仿真/真机解耦架构：
- **BaseController**: 抽象基类，定义通用接口
- **SimulationController**: MuJoCo 仿真控制器
- **RealRobotInterface**: 真机接口预留（需根据实际通信协议实现）

```python
from core.controller import SimulationController

controller = SimulationController(model, data)
controller.connect()

# 获取状态
qpos = controller.get_joint_positions()
ee_pos, ee_rot = controller.get_ee_pose()

# 发送命令
controller.send_joint_command(q_target)
controller.open_gripper()
controller.close_gripper()
```

### 视觉模块 (core/vision)

- **Camera**: MuJoCo 相机，支持 RGB、深度图像获取
- **ObjectDetector**: 基于 HSV 颜色空间的目标检测

```python
from core.vision import Camera, ObjectDetector

camera = Camera(model, data, camera_name="camera")
detector = ObjectDetector()

# 获取图像
image = camera.get_image()
depth = camera.get_depth()

# 检测小球
center, radius = detector.detect_ball_position(image, color_name="red")
```

### 手眼标定模块 (core/calibration)

支持 Eye-in-Hand 和 Eye-to-Hand 标定：

```python
from core.calibration import HandEyeCalibration

calibration = HandEyeCalibration(method="tsai")

# 采集标定数据
calibration.add_sample(robot_pose, camera_pose)

# 执行标定
T_cam_ee, error = calibration.calibrate()

# 坐标变换
point_ee = calibration.transform_point(point_cam)
```

### 数据采集模块 (core/data_collection)

```python
from core.data_collection import DataRecorder

recorder = DataRecorder(save_dir="./data", frequency=50)

# 开始采集
recorder.start_session(task_name="grasp")

# 记录数据
recorder.record_step(timestamp, qpos, qvel, ee_pos, ee_rot, gripper_pos, image)

# 保存数据
recorder.save_session()
```

## 仿真/真机切换

项目采用解耦设计，切换仿真/真机只需更换控制器：

```python
# 仿真模式
from core.controller import SimulationController
controller = SimulationController(model, data)

# 真机模式（需实现具体通信协议）
from core.controller import RealRobotInterface
controller = RealRobotInterface(ip_address="192.168.1.100", port=8080)
```

## 扩展开发

### 添加新的运动规划算法

在 `core/trajectory/` 下创建新文件，继承或扩展 `TrajectoryPlanner`。

### 添加深度学习视觉检测

替换 `core/vision/object_detector.py` 中的检测逻辑，接入 YOLO 等模型。

### 接入真机通信

在 `core/controller/real_robot_interface.py` 中实现具体通信协议：
- TCP/IP 通信
- 串口通信
- ROS 节点通信

## 注意事项

1. 本项目为纯仿真实现，未安装任何额外依赖
2. 真机接口已预留框架，需根据实际硬件实现通信逻辑
3. 手眼标定模块提供算法框架，实际使用需配合标定板
4. 视觉检测基于颜色分割，复杂场景建议接入深度学习模型

## License

MIT License
