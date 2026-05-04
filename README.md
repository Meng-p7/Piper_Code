# PiperSim — Piper 机械臂 MuJoCo 仿真平台

基于 MuJoCo 物理引擎的 Piper 机械臂仿真与控制系统，支持仿真/真机解耦架构，预留手眼标定、运动规划、视觉伺服接口。

## 项目背景

Piper 是一款 6 自由度 SCARA 构型桌面机械臂（末端带 1 自由度平行夹爪），由 **松灵机器人（AgileX）** 出品，常用于教育、科研和轻量级自动化场景。

本项目旨在提供一个 **开箱即用的 Piper 机械臂 MuJoCo 仿真环境**，帮助开发者、机器人爱好者和学生在没有真实硬件的情况下也能学习和研究机械臂控制。项目包含完整的运动学求解、轨迹规划、仿真控制器、视觉检测和抓取流程演示，并预留了真机接口，方便后续迁移到真实硬件上运行。

无论你是机器人初学者还是经验丰富的开发者，都可以通过本项目的示例代码快速上手机械臂仿真与控制。

> **GitHub 地址**: [https://github.com/Meng-p7/PiperSim.git](https://github.com/Meng-p7/PiperSim.git)

---

## 项目架构

```
PiperSim/
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
│   │   └── hand_eye_calibration.py  # 手眼标定（Tsai-Lenz 轴角+SVD，eye_in_hand/eye_to_hand 双模式）
│   └── data_collection/             # 数据采集模块
│       └── data_recorder.py         # 数据记录与回放
├── utils/                           # 工具函数
│   ├── config_loader.py              # YAML 配置加载（单例模式）
│   └── logger.py                     # 结构化日志（logging 模块封装）
├── tests/                           # 测试与示例代码
│   ├── test_kinematics.py            # FK/IK 往返一致性测试
│   ├── test_trajectory.py            # 轨迹规划器/SLERP 正交性测试
│   ├── test_config.py                # 配置加载测试
│   └── test_hand_eye.py              # 手眼标定合成数据验证
├── demos/                           # 演示脚本
│   ├── run_piper.py                  # 基础运动测试
│   ├── grasp_ball_demo.py            # 视觉抓取小球演示（13 步 Pick-and-Place）
│   └── calibration_demo.py           # 完整仿真手眼标定演示
├── data/                            # 数据存储目录
├── requirements.txt                 # 依赖列表
└── README.md
```

## 环境搭建（面向新手）

以下步骤从零开始，手把手教你搭建运行本项目所需的环境。

> **前置要求**: 电脑已安装 Python 3.9~3.11（推荐 3.10）和 Git。如果没有 Git，也可以直接下载 ZIP 压缩包。

---

### 第一步：安装 Miniconda / Anaconda（推荐）

Conda 是一个流行的 Python 环境管理工具，可以让你在同一台电脑上管理多个互不干扰的 Python 环境。

- **Miniconda**（轻量推荐）: 从 [https://docs.anaconda.com/miniconda/](https://docs.anaconda.com/miniconda/) 下载安装
- **Anaconda**（包含更多数据科学包）: 从 [https://www.anaconda.com/download](https://www.anaconda.com/download) 下载安装

安装完成后，打开终端（Windows 请打开 "Anaconda Prompt" 或 "Miniforge Prompt"），验证安装：

```bash
conda --version
```

如果能显示版本号（如 `conda 24.x.x`），说明安装成功。

---

### 第二步：克隆项目

打开终端，进入你想存放项目的目录，然后克隆仓库：

```bash
git clone https://github.com/Meng-p7/PiperSim.git
cd PiperSim
```

> 💡 如果没有安装 Git，也可以去 [https://github.com/Meng-p7/PiperSim](https://github.com/Meng-p7/PiperSim) 点击绿色的 "Code" → "Download ZIP"，解压后进入 `PiperSim` 文件夹，在该目录下打开终端。

---

### 第三步：创建 Conda 环境

为项目创建一个独立的 Python 环境，避免与系统中其他项目的依赖冲突：

```bash
conda create -n piper python=3.10 -y
```

激活环境：

```bash
conda activate piper
```

激活后，终端前面会出现 `(piper)` 字样，表示你已进入该环境。

---

### 第四步：安装 MuJoCo

MuJoCo 是项目依赖的核心物理引擎，负责渲染和物理仿真。

**方法一：通过 pip 安装（推荐，Python 3.10+ 适用）**

```bash
pip install mujoco
```

验证安装：

```bash
python -c "import mujoco; print(mujoco.__version__)"
```

如果能正常输出版本号（如 `3.x.x`），说明安装成功。

如果遇到错误，请参考 [MuJoCo 官方安装指南](https://mujoco.readthedocs.io/en/stable/programming/index.html)。

> **注意**: MuJoCo 需要 OpenGL 支持。如果在远程服务器（无物理显示器）或 WSL 上运行，可能需要安装虚拟显存驱动。常见 Linux 发行版可尝试：
> ```bash
> sudo apt install libgl1-mesa-glx libglib2.0-0  # Ubuntu/Debian
> ```

---

### 第五步：安装其他 Python 依赖

本项目还需要 NumPy、SciPy、OpenCV 等科学计算和计算机视觉库。项目根目录已提供 `requirements.txt`，一键安装：

```bash
pip install -r requirements.txt
```

如果希望单独安装，也可以手动执行以下命令：

```bash
pip install numpy>=1.24.0 scipy>=1.10.0 opencv-python>=4.8.0 matplotlib>=3.7.0 PyYAML>=6.0
```

---

### ✅ 环境验证

运行以下命令，验证所有依赖是否正确安装：

```python
python -c "
import mujoco
import numpy as np
import cv2
import scipy
import matplotlib
import yaml
print('✅ 所有依赖安装成功！')
print(f'  MuJoCo:    {mujoco.__version__}')
print(f'  NumPy:     {np.__version__}')
print(f'  OpenCV:    {cv2.__version__}')
print(f'  SciPy:     {scipy.__version__}')
print(f'  Matplotlib: {matplotlib.__version__}')
"
```

如果看到 ✅ 绿色标记，恭喜你！环境搭建完成，可以开始体验了 🎉

---

## 快速开始

确保你已经 **激活 conda 环境**（终端前面有 `(piper)` 字样），并且位于项目根目录 `PiperSim` 下。

### 1. 基础运动控制

运行基础运动测试脚本，观察机械臂的运动效果：

```bash
python demos/run_piper.py
```

执行后，MuJoCo 仿真窗口将会打开，机械臂将平滑移动到目标位置 `[0.3, 0.2, 0.15]`。

### 2. 视觉抓取小球演示（Pick-and-Place 完整流程）

运行完整的抓取演示，观看机械臂自动识别并抓取小球的完整流程：

```bash
python demos/grasp_ball_demo.py
```

演示流程（13步）：
1. 移动到初始位姿（抬头）
2. 低头观察位姿
3. 通过相机检测小球位置（支持视觉检测+fallback）
4. 快速移动到小球上方 15cm
5. 慢降到球上方 3cm
6. 极慢接近球心
7. 闭合夹爪抓取（温和夹持，5mm穿透）
8. 微提 2cm 测试握力
9. 正式提升 8cm
10. 缓慢升到安全高度
11. 极慢横移到放置位置上方
12. 笛卡尔直线下降至放置位置
13. 缓慢释放物体，返回初始位姿

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

基于 Tsai-Lenz 轴角+SVD 方法，求解 AX=XB 方程：
- 支持 `eye_in_hand`（相机装末端）和 `eye_to_hand`（相机固定）两种模式
- 旋转求解精度 ~1e-16（合成数据验证通过）
- 合成数据验证脚本: `python -c "..."`（见验证方法）

### 数据采集模块 (core/data_collection)

- `DataRecorder` — 以指定频率记录关节状态、末端位姿、图像等数据

---

## 当前状态与已知问题

### 已完成

- [x] Piper 机械臂 URDF/MJCF 模型导入
- [x] 正/逆运动学模块（含夹爪中心 IK 求解）
- [x] 多种轨迹插值算法（含 SLERP 旋转插值）
- [x] 仿真/真机解耦控制器架构
- [x] MuJoCo 相机 RGB/深度图像获取
- [x] 像素坐标到世界坐标的转换
- [x] HSV 颜色目标检测
- [x] 抓取演示框架（完整 13 步 Pick-and-Place 可运行）
- [x] **阶段一：完整抓取调通（2026-05-02）**
- [x] **笛卡尔 SLERP 旋转插值修复**
- [x] **config.yaml 接入（config_loader 单例）**
- [x] **grasp_ball_demo 迁移到 SimulationController**
- [x] **手眼标定 Tsai-Lenz 旋转求解实现（误差 ~1e-16）**
- [x] **10 文件类型注解全覆盖**
- [x] **4 个测试文件（kinematics/trajectory/config/hand_eye）**
- [x] **核心模块结构化日志（logger.py）**

### 待完成

- [ ] 标定板 MuJoCo 模型 + 自动采集流程
- [ ] 视觉伺服（PBVS/IBVS）

---

## 开发路线图

| 参数 | 值 | 说明 |
|------|-----|------|
| `gripper_close_ctrl` | `0.015` | 指间距 30mm，球直径 40mm，温和夹持 5mm/侧穿透 |
| `ball.friction` | `3.0 2.5 0.05` | 高滑动/扭转摩擦 |
| `ball.solref` | `0.015 0.7` | 中等刚度，球可变形但不至于太软 |
| `ball.solimp` | `0.95 0.99 0.002` | 允许适度穿透 |
| `ball.condim` | `4` | 增强接触稳定性 |
| 微提 | 500步 (0.020 m/s) | 极慢测试握力 |
| 提升 | 300步 (0.133 m/s) | |
| 升安全高度 | 300步 (0.217 m/s) | |
| 横移 | 500步 (0.180 m/s) | |
| 下降 | 15段×40步 (0.108 m/s) | 笛卡尔分段直线下降 |
| 释放 | 200步×5迭代 | 2秒缓慢打开 |

### 阶段一踩坑记录

| 问题 | 根因 | 修复 |
|------|------|------|
| 夹不到球 | 夹爪中心定位到球顶(+radius) | 定位到球心 |
| 下降撞地反弹 | 下降太快 0.75 m/s | 三段减速至 0.06 m/s |
| 上升时球掉落 | 球太硬，夹爪没碰到球 | 球变软 + 夹更紧 |
| 水平移动时球掉落 | 横移 0.91 m/s 太快 | 降速 + 先升后横移 |
| 上升打滑 | 球太软接触力不足 | 中等刚度 + 更高摩擦 + 微提测试 |
| 下降时球掉落 | 步骤9/10太快，球提前松动 | 统一降速 + 横移后握力重稳定 |
| 下降沿弯曲路径搓掉球 | 关节空间插值≠笛卡尔直线 | 自定义笛卡尔分段直线下降 |

---

## 🤖 AI 参考手册（给 AI 阅读的项目关键信息）

### piper.xml 关节与夹爪结构

```xml
Piper 机械臂 — 6自由度SCARA式 + 1自由度平行夹爪

关节 (所有 axis="0 0 1", 即绕Z轴旋转):
  joint1~joint6: 臂关节, 所有为铰链(hinge), 运动学链 SCARA 构型
  joint7: link7指, slide, axis 0 0 -1, range 0~0.035
  joint8: link8指, slide, axis 0 0 1, range -0.035~0
  约束: joint8 = -joint7 (equality) → 对称运动

夹爪:
  gripper_open_ctrl  = 0.035 → 指间距 0.070m (全开)
  gripper_close_ctrl = 0.015 → 指间距 0.030m (温和夹持)
  ball diameter = 0.040m → 穿透 0.005m/侧 (配合软接触)
  两指碰撞体: 薄box (0.0025m厚), 位于body原点下方
```

### MuJoCo 接触参数参考

```python
球 geom 参数 (scene.xml):
  friction="2.5 2.0 0.03"   # [滑动, 扭转, 滚动] 摩擦
  solref="0.015 0.7"        # [时间常数, 阻尼比] — 时间常数越小越硬
  solimp="0.95 0.99 0.002"  # [dmin, dmax, width] — 约束阻抗曲线
  condim=4                   # 接触约束维度

经验: 球太硬(如solref=0.005) → 手指被弹开无法接触
     球太软(如solref=0.03)  → 法向力不足, 球会滑落
     适中(0.015)效果好
```

### 坐标变换

```python
camera: pos="0.6 0 0.5"  quat="0.236 0 -0.972 0"
  → 大致指向-X方向, 看向机械臂工作空间
  → pixel_to_world: world = cam_pos + R_cam @ [x_cam, y_cam, z_cam]
  → MuJoCo freejoint qpos格式: [x, y, z, qw, qx, qy, qz]

夹爪中心 = (data.xpos[link7] + data.xpos[link8]) / 2  # link7/link8 body原点中点
IK solver: solve_gripper_position() 直接求解夹爪中心到达目标位置
```

### 常见错误与修复

| 问题 | 根因 | 修复 |
|------|------|------|
| 夹不到球 | 夹爪中心定位到球顶(+radius) | 直接定位到球心 |
| 球从侧面滑落 | 关节空间插值≠笛卡尔直线 | 改用笛卡尔分段IK直线下降 |
| 球在运动中松动 | 前面步骤速度太快累积松动 | 全程速度均匀 + 握力重稳定 |
| IK/Controller 索引错乱 | 用 joint ID 索引 qpos/ctrl | 用 jnt_qposadr / actuator ID |

---

## 开发路线图

### 阶段一：抓取调试完善 ✅ 已完成（2026-05-02）

**结果**: 完整 13 步 Pick-and-Place 流程成功跑通，球被稳定抓取、提升、横移、下降、释放。

### 阶段二：手眼标定（下一步）

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

tests/
└── calibration_visualize.py         # 标定结果可视化

demos/
├── grasp_ball_demo.py               # 抓取演示
└── calibration_demo.py              # 手眼标定演示（自动采集+标定+验证）

models/
├── calibration_board.xml            # 棋盘格标定板模型
└── calibration_scene.xml            # 标定场景（含标定板）
```

#### 关键流程

```
1. 加载标定场景 → 2. 机械臂移动到 N 个不同位姿 (15-20组)
→ 3. 每个位姿记录 T_base_ee + 检测 T_cam_board
→ 4. 运行 Tsai-Lenz 算法求解 T_ee_cam（相机相对于末端的变换）
→ 5. 用 T_base_ee × T_ee_cam 得到 T_base_cam（相机在基坐标系中的位姿）
→ 6. 验证: 已知世界坐标点 ↔ 相机检测坐标 的偏差 < 2mm
```

#### 验收标准

- 重投影误差 < 2 像素
- 标定后的世界坐标误差 < 2mm

### 阶段三：视觉伺服（中长期规划）

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

demos/
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

### 架构优化（穿插进行）

- **config.yaml 接入** — 创建 `utils/config_loader.py`，将硬编码参数统一从配置文件读取
- **统一 Controller 接口** — `grasp_ball_demo.py` 迁移使用 `SimulationController`
- **日志系统** — 将 `print()` 替换为结构化日志记录
- **Diagnostics** — 增加夹爪中心-球心距离的实时监控

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
