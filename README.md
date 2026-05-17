# PiperSim — Piper 机械臂 MuJoCo 仿真平台

基于 MuJoCo 物理引擎的 Piper 机械臂仿真与控制系统，支持仿真/真机解耦架构，含手眼标定、运动规划、视觉伺服接口。

Piper 是一款 6 自由度桌面机械臂（末端带 1 自由度平行夹爪），由松灵机器人（AgileX）出品。本项目为没有真实硬件的开发者、学生提供开箱即用的仿真环境，涵盖运动学、轨迹规划、视觉检测、手眼标定、抓取演示等完整链路，并预留真机接口，可直接迁移到实物运行。

> **GitHub**: [https://github.com/Meng-p7/PiperSim.git](https://github.com/Meng-p7/PiperSim.git)

## 项目架构

```
PiperSim/
├── config/              # YAML 参数配置
├── models/              # MuJoCo 模型（Piper URDF/MJCF + 场景）
├── core/
│   ├── kinematics/      # 正/逆运动学
│   ├── trajectory/      # 轨迹规划
│   ├── controller/      # 仿真/真机解耦控制器
│   ├── vision/          # 相机与目标检测
│   ├── calibration/     # 手眼标定
│   └── data_collection/ # 数据记录
├── utils/               # 配置加载、日志
├── tests/               # 单元测试
├── demos/               # 演示脚本
└── data/                # 数据存储
```

## 环境搭建

```bash
# 1. 克隆项目
git clone https://github.com/Meng-p7/PiperSim.git
cd PiperSim

# 2. 创建环境（推荐 Python 3.10）
conda create -n piper python=3.10 -y
conda activate piper

# 3. 安装依赖
pip install -r requirements.txt
```

## 快速开始

```bash
# 基础运动测试 — 观察机械臂平滑移动到目标位姿
python demos/run_piper.py

# 视觉抓取演示（13 步 Pick-and-Place 完整流程）
python demos/grasp_ball_demo.py

# 手眼标定演示（自动采集标定数据并计算相机-末端变换矩阵）
python demos/calibration_demo.py
```

## 核心模块

### 运动学 — `core/kinematics`

基于 MuJoCo 的正逆运动学求解。

- **ForwardKinematics** — 调用 `mj_forward`，根据 6 个关节角计算末端位置和旋转矩阵
- **InverseKinematics** — 基于 `scipy.optimize.minimize`（L-BFGS-B），在关节限位内求解：
  - `solve_position(target_pos)` — 求解 link6 末端到达目标位置
  - `solve_pose(target_pos, target_ori)` — 同时求解位置和姿态
  - `solve_gripper_position(target_pos)` — 直接以夹爪中心（link7/link8 中点）为目标求解，避免手调偏移

```python
fk = ForwardKinematics(model, ee_body_name="link6")
ik = InverseKinematics(model, ee_body_name="link6", gripper_bodies=["link7", "link8"])

position, orientation = fk.compute(qpos)
q_target, success = ik.solve_gripper_position([0.35, 0, 0.02], q_init=q_current)
```

### 轨迹规划 — `core/trajectory`

支持 6 种插值方法，覆盖关节空间和笛卡尔空间：

| 方法 | 特点 | 适用场景 |
|------|------|----------|
| `linear_interpolation` | 线性等距 | 快速粗略移动 |
| `cubic_interpolation` | 速度连续 | 一般运动 |
| `quintic_interpolation` | 速度+加速度连续 | 平滑运动 |
| `trapezoidal_velocity` | 梯形速度曲线 | 有速度约束的运动 |
| `cartesian_linear` | 位置线性 + 旋转 SLERP | 笛卡尔直线运动 |
| `generate_approach_trajectory` | 先上方再下降 | 抓取接近轨迹 |

### 控制器 — `core/controller`

仿真/真机解耦架构，通过 `BaseController` 抽象基类统一接口：

- **BaseController** — 定义 `connect/disconnect/get_joint_positions/send_joint_command/step` 等通用接口
- **SimulationController** — MuJoCo 仿真实现，管理关节/夹爪 actuator 映射，封装 `mj_step` 调用
- **RealRobotInterface** — 真机接口预留，需根据实际通信协议（TCP/串口/ROS）实现

切换仿真/真机只需更换控制器实例，上层代码无需修改：

```python
# 仿真
controller = SimulationController(model, data)

# 真机（预留）
controller = RealRobotInterface(ip="192.168.1.100")
```

### 视觉 — `core/vision`

- **Camera** — MuJoCo 渲染相机
  - `get_image()` — 获取 RGB 图像（MuJoCo Renderer）
  - `get_depth()` — 获取深度图像
  - `pixel_to_world(u, v, depth)` — 像素坐标 + 深度 → 世界坐标
  - `get_camera_params()` — 内参 fx, fy, cx, cy
  - `get_camera_pose()` — 相机世界位姿

- **ObjectDetector** — 基于 HSV 颜色空间的目标检测（OpenCV）
  - `detect_ball_position(image, color_name)` — 检测色球中心和半径
  - 支持红、绿、蓝三种颜色，红色使用双区间 HSV 阈值避免色相断裂

```python
camera = Camera(model, data, camera_name="camera", width=640, height=480)
detector = ObjectDetector()

image = camera.get_image()
(u, v), radius = detector.detect_ball_position(image, "red")
world_pos = camera.pixel_to_world(u, v, depth=0.5)
```

### 手眼标定 — `core/calibration`

基于 Park 方法求解 AX=XB 方程，实现相机与机械臂的精确标定：

- 支持 **eye_in_hand**（相机装末端）和 **eye_to_hand**（相机固定）两种模式
- 标定精度：旋转误差 < 0.01°，平移误差 < 0.1mm
- 批量验证成功率 100%（5/5 独立标定）
- 配套工具：种子生成器（邻域扩展策略，245 个种子）、批量精度验证脚本

### 数据采集 — `core/data_collection`

- **DataRecorder** — 以指定频率记录关节状态、末端位姿、图像等数据，支持回放

## 测试

```bash
# 运行全部测试（35 个）
python -m pytest tests/ -v -o "addopts="

# 运行单个模块测试
python -m pytest tests/test_kinematics.py -v -o "addopts="
```

覆盖：运动学往返一致性、轨迹规划端点/正交性、配置加载、手眼标定合成数据验证。

## 开发状态

- ✅ 正/逆运动学（含夹爪中心 IK）
- ✅ 多种轨迹插值（含 SLERP 旋转插值）
- ✅ 仿真/真机解耦控制器架构
- ✅ MuJoCo 相机 RGB/深度图像 + 像素→世界坐标转换
- ✅ 抓取演示（完整 13 步 Pick-and-Place）
- ✅ 手眼标定（Park 方法 + 种子生成 + 批量验证）
- 🔄 视觉伺服（PBVS/IBVS）— 详见 [plan.md](plan.md)

## License

MIT License
