# Piper 机械臂 MuJoCo 仿真项目

基于 MuJoCo 物理引擎的 Piper 机械臂仿真与控制系统，支持仿真/真机解耦架构，预留手眼标定、运动规划、视觉伺服接口。

## 项目架构

```
Piper_Code/
├── config/                          # 配置文件
│   └── config.yaml                  # 系统参数配置
├── models/                          # MuJoCo 模型文件
│   ├── agilex_piper/                # Piper 官方模型（含 piper.xml、网格文件）
│   ├── scene.xml                    # 完整场景模型（含小球目标）
│   └── test_scene.xml               # 测试场景
├── core/                            # 核心模块
│   ├── kinematics/                  # 运动学模块
│   │   ├── forward_kinematics.py    # 正运动学（MuJoCo mj_forward）
│   │   └── inverse_kinematics.py    # 逆运动学（L-BFGS-B 优化求解）
│   ├── trajectory/                  # 轨迹规划模块
│   │   └── trajectory_planner.py    # 线性/三次/五次/梯形/笛卡尔插值
│   ├── controller/                  # 控制器模块
│   │   ├── base_controller.py       # 控制器抽象基类
│   │   ├── simulation_controller.py # MuJoCo 仿真控制器
│   │   └── real_robot_interface.py  # 真机接口（预留）
│   ├── vision/                      # 视觉模块
│   │   ├── camera.py                # MuJoCo 相机（RGB/深度/坐标转换）
│   │   └── object_detector.py       # HSV 颜色目标检测
│   ├── calibration/                 # 标定模块
│   │   └── hand_eye_calibration.py  # 手眼标定（Tsai-Lenz，框架代码）
│   └── data_collection/             # 数据采集模块
│       └── data_recorder.py         # 数据记录与回放
├── utils/                           # 工具函数（预留）
├── examples/                        # 示例程序
│   └── grasp_ball_demo.py           # 视觉抓取小球演示
├── tests/                           # 测试代码
│   └── run_piper.py                 # 基础运动测试
├── data/                            # 数据存储目录
├── requirements.txt                 # 依赖列表
└── README.md
```

## 环境要求

```bash
pip install mujoco>=2.3.0 numpy>=1.24.0 scipy>=1.10.0 opencv-python>=4.8.0 matplotlib>=3.7.0 PyYAML>=6.0
```

## 快速开始

### 1. 基础运动控制

```bash
python tests/run_piper.py
```

机械臂将平滑移动到目标位置 `[0.3, 0.2, 0.15]`。

### 2. 视觉抓取小球演示

```bash
python examples/grasp_ball_demo.py
```

演示流程：
1. 机械臂移动到初始位姿（抬头）
2. 移动到低头观察位姿
3. 通过相机视觉检测红色小球（当前直接读取真实位置）
4. 移动到小球正上方 15cm
5. 下降到抓取高度（球半径处）
6. 闭合夹爪执行抓取
7. 提升物体 10cm
8. 移动到放置位置并释放
9. 返回初始位姿

## 核心模块说明

### 运动学模块 (core/kinematics)

- **ForwardKinematics**: 正运动学，根据关节角度计算末端位姿
- **InverseKinematics**: 逆运动学，基于 L-BFGS-B 优化求解，支持：
  - `solve_position()` — 求解末端（link6）位置
  - `solve_pose()` — 求解末端位姿（位置+姿态）
  - `solve_gripper_position()` — 直接求解夹爪中心位置（link7/link8 中点）

```python
from core.kinematics import ForwardKinematics, InverseKinematics

fk = ForwardKinematics(model, ee_body_name="link6")
ik = InverseKinematics(model, ee_body_name="link6", joint_names=[f"joint{i}" for i in range(1, 7)],
                        gripper_bodies=["link7", "link8"])

# 正运动学
position, orientation = fk.compute(qpos)

# 逆运动学 — 求解夹爪中心到达目标位置
q_target, success = ik.solve_gripper_position([0.35, 0, 0.02], q_init=q_current, q_full=q_full)
```

### 轨迹规划模块 (core/trajectory)

支持多种插值方法：
- **线性插值** `linear_interpolation()`
- **三次多项式插值** `cubic_interpolation()` — 速度连续
- **五次多项式插值** `quintic_interpolation()` — 速度和加速度连续
- **梯形速度规划** `trapezoidal_velocity()`
- **笛卡尔空间直线插值** `cartesian_linear()`
- **接近轨迹** `generate_approach_trajectory()` — 先上方再下降

### 控制器模块 (core/controller)

仿真/真机解耦架构：
- **BaseController**: 抽象基类，定义 `connect/disconnect/get_joint_positions/send_joint_command` 等通用接口
- **SimulationController**: MuJoCo 仿真控制器实现
- **RealRobotInterface**: 真机接口预留（需根据实际通信协议实现）

### 视觉模块 (core/vision)

- **Camera**: MuJoCo 渲染相机
  - `get_image()` — 获取 RGB 图像
  - `get_depth()` — 获取深度图像
  - `pixel_to_world(u, v, depth)` — 像素坐标+深度 → 世界坐标
  - `get_camera_params()` — 获取内参 fx, fy, cx, cy
  - `get_camera_pose()` — 获取相机世界位姿

- **ObjectDetector**: 基于 HSV 颜色空间的目标检测
  - `detect_ball_position(image, color_name)` — 检测色球中心和半径
  - 支持红、绿、蓝三种颜色，红色使用双区间 HSV 阈值

### 手眼标定模块 (core/calibration)

当前为框架代码，支持 Tsai-Lenz 算法的基本数据结构：
- `add_sample(robot_pose, camera_pose)` — 添加标定样本
- `calibrate()` — 执行标定（旋转求解部分待完善）

### 数据采集模块 (core/data_collection)

- `DataRecorder` — 以指定频率记录关节状态、末端位姿、图像等数据

---

## 当前状态与已知问题

### 已完成

- [x] Piper 机械臂 URDF/MJCF 模型导入
- [x] 正/逆运动学模块（含夹爪中心 IK 求解）
- [x] 多种轨迹插值算法
- [x] 仿真/真机解耦控制器架构
- [x] MuJoCo 相机 RGB/深度图像获取
- [x] 像素坐标到世界坐标的转换
- [x] HSV 颜色目标检测
- [x] 抓取演示框架（完整流程可运行）

### 已知问题（抓取小球尚未成功）

1. **抓取精度不足** — 机械臂运动到小球附近，但闭合夹爪未能夹住小球
   - 可能原因 a: IK 求解的夹爪中心位置与实际小球仍有偏差（优化器陷入局部最优）
   - 可能原因 b: 夹爪闭合控制参数（`gripper_open_ctrl=0.035`、闭合到 `ball_radius - 0.001`）需要调优
   - 可能原因 c: 物理仿真中接触力、摩擦参数需要调整
   - 待验证: 观察仿真中夹爪中心实际轨迹 vs 小球位置的偏差

2. **视觉检测链路未闭环** — 当前抓取流程直接读取小球真实位置（`get_ball_position()`），相机检测到的世界坐标精度未经验证
   - `pixel_to_world()` 已修正投影公式，但精度尚未端到端验证

3. **scene.xml 模型参数** — 小球摩擦参数 `friction="1.0 0.05 0.01"` 和接触求解器参数可能需要根据实际抓取效果调整

### 下一步修复方向

1. **抓取调试**: 增加夹爪中心与小球的实时距离监控，打印每步的误差
2. **IK 精度**: 验证 `solve_gripper_position()` 是否确实收敛到足够精度（cost < 1e-6）
3. **夹爪参数**: 根据仿真实际效果调整闭合力、闭合速度
4. **仿真物理参数**: 调整小球质量、摩擦、接触刚度等

---

## 开发路线图

### 阶段一：抓取调试完善（近期）

**目标**: 成功抓取小球

**工作内容**:
1. 添加抓取过程的详细日志（夹爪中心 vs 小球位置的实时偏差）
2. 调整 IK 收敛阈值和初始值策略
3. 调整夹爪控制参数和仿真物理参数
4. 将 `grasp_ball_demo.py` 中视觉检测链路接入真实检测（替换 `get_ball_position()` 为 `detect_ball_from_camera()`）

### 阶段二：手眼标定（中期）

**目标**: 建立相机坐标系与机器人基坐标系的精确映射关系

#### 前期准备

1. **标定板模型**
   - 在 MuJoCo 场景中添加棋盘格或 ArUco 标定板
   - 创建 `models/calibration_board.xml`，放置在工作台上
   - 实现标定板角点/ArUco 检测（复用 OpenCV）

2. **多点采集流程**
   - 机械臂移动到多个不同姿态（至少 10-20 组）
   - 每个姿态记录：末端位姿 `T_base_ee`（来自正运动学）+ 标定板在相机中的位姿 `T_cam_board`
   - 数据存入 `data/calibration/` 目录

3. **标定算法完善** — `core/calibration/hand_eye_calibration.py`
   - 当前 `_solve_rotation()` 方法未实现，需要用 Tsai-Lenz 或 Daniilidis 方法求解 `AX = XB` 中的旋转部分
   - 添加 Eye-in-Hand 和 Eye-to-Hand 两种模式
   - 添加标定精度评估（重投影误差）

4. **坐标变换验证**
   - 标定完成后，用已知位置的物体验证像素坐标 → 世界坐标的精度
   - 对比标定前后的 `pixel_to_world()` 精度

#### 代码架构规划

```
core/calibration/
├── __init__.py
├── hand_eye_calibration.py          # 手眼标定算法（Tsai/Daniilidis）
├── calibration_target.py            # 标定目标（棋盘格/ArUco/CharuCo）
├── calibration_data.py              # 标定数据管理（采集、存储、加载）
└── calibration_utils.py             # 工具函数（误差计算、可视化、结果保存）

examples/
├── grasp_ball_demo.py               # 抓取演示
├── calibration_demo.py              # 手眼标定演示（自动采集+标定+验证）
└── calibration_visualize.py         # 标定结果可视化

models/
├── calibration_board.xml            # 棋盘格标定板模型
└── calibration_scene.xml            # 标定场景（含标定板）
```

#### 关键流程

```
1. 加载标定场景 → 2. 机械臂移动到 N 个不同位姿
→ 3. 每个位姿记录 T_base_ee + 检测 T_cam_board
→ 4. 运行 Tsai-Lenz 算法求解 T_ee_cam（相机相对于末端的变换）
→ 5. 用 T_base_ee × T_ee_cam 得到 T_base_cam（相机在基坐标系中的位姿）
→ 6. 验证: 已知世界坐标点 ↔ 相机检测坐标 的偏差
```

### 阶段三：视觉伺服（中长期）

**目标**: 实时视觉反馈驱动机械臂运动，实现动态抓取

#### 前期准备

1. **手眼标定完成** — 视觉伺服的坐标系映射基础
2. **实时相机性能** — 确保 MuJoCo 渲染频率 >= 30fps
3. **雅可比矩阵计算** — 实现图像雅可比（Interaction Matrix / Image Jacobian）

#### 代码架构规划

```
core/visual_servo/
├── __init__.py
├── visual_servo_base.py             # 视觉伺服基类
├── position_based_vs.py             # PBVS（基于位置的视觉伺服）
├── image_based_vs.py                # IBVS（基于图像的视觉伺服）
├── jacobian_estimator.py            # 图像雅可比矩阵估计
└── feature_tracker.py               # 特征点跟踪（光流/模板匹配）

examples/
├── grasp_ball_demo.py
├── calibration_demo.py
├── pbvs_demo.py                     # PBVS 抓取演示
└── ibvs_demo.py                     # IBVS 抓取演示
```

#### 两种视觉伺服方案

**PBVS（基于位置的视觉伺服）** — 推荐优先实现
```
相机检测目标 → 目标在世界坐标中的位置（需手眼标定）
→ 计算位置误差 e = P_target - P_current
→ 通过伪逆雅可比映射到关节速度 Δq = J⁻¹ · e
→ 实时发送关节命令
```
- 优点: 直觉、易实现，复用手眼标定结果
- 缺点: 依赖标定精度

**IBVS（基于图像的视觉伺服）**
```
相机检测目标特征（像素坐标） → 设定期望特征 s*
→ 计算图像误差 e = s - s*
→ 通过图像雅可比映射 Δq = L⁺⁻¹ · e
→ 实时发送关节命令
```
- 优点: 不依赖精确标定，对相机标定误差鲁棒
- 缺点: 需要计算图像雅可比矩阵，可能有局部极小

#### 关键流程（PBVS 为例）

```
循环:
  1. Camera.get_image() + ObjectDetector.detect_ball_position()
  2. Camera.pixel_to_world() → 得到目标世界坐标
  3. 读取当前末端位姿 → 计算位置误差
  4. 雅可比矩阵映射误差到关节空间
  5. 发送关节增量命令
  6. 误差 < 阈值 → 抓取
```

---

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

## 注意事项

1. 本项目基于 MuJoCo 物理仿真，视觉检测依赖 MuJoCo 内置渲染器
2. 真机接口已预留框架（`RealRobotInterface`），需根据实际硬件实现通信逻辑
3. `models/agilex_piper/` 为官方提供的 Piper 模型文件，请勿手动修改
4. 抓取仿真中小球为自由体（`freejoint`），受重力自然下落
5. 手眼标定和视觉伺服为下一阶段开发目标，当前模块为框架代码

## License

MIT License
