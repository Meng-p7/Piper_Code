## Plan: Piper 机械臂项目 — 抓取修复 + 手眼标定 + 视觉伺服

TL;DR：当前代码存在 9 个已确认 Bug（抓取位置错误、IK/Controller 索引脆弱、手眼标定空壳等），分三阶段修复：第一阶段调通抓取，第二阶段完成手眼标定，第三阶段实现视觉伺服。

---

### 代码审查发现的 9 个问题

| # | 严重程度 | 位置 | 问题 | 状态 |
|---|---------|------|------|------|
| 1 | 🔴 致命 | `grasp_ball_demo.py` | **抓取位置错误** — 夹爪中心定位到球顶(+radius)而非球心 | ✅ 已修复 |
| 2 | 🔴 致命 | `grasp_ball_demo.py` | **夹爪闭合理不清** — 闭合目标值、步数、物理步数与接触力建立不匹配 | ✅ 已修复 |
| 3 | 🟡 严重 | `scene.xml` | **物理参数不利抓取** — 球体切向摩擦仅 0.05，手指碰撞体太薄 | ✅ 已修复 |
| 4 | 🟡 严重 | `run_piper.py` | **关节数量错误** — 7 个 arm joint（实际 6 个），joint7 是夹爪手指 | ✅ 已修复 |
| 5 | 🟡 严重 | `simulation_controller.py` | `get_camera_image()` 是空壳 | 🔵 预留（真机接口） |
| 6 | 🟡 严重 | `hand_eye_calibration.py` | `_solve_rotation()` 是空壳 | 🔵 阶段二实现 |
| 7 | 🟠 中等 | IK/Controller | 直接用 joint ID 索引 `qpos`/`ctrl`，靠巧合工作 | ✅ 已修复 |
| 8 | 🟠 中等 | 全局 | `config.yaml` 完全未被加载使用 | 🔵 后续优化 |
| 9 | 🟠 中等 | `grasp_ball_demo.py` | 绕过了 `SimulationController` 直接操作 `data` | 🔵 后续重构 |

---

### 🎯 三阶段规划

```
阶段一: 抓取调通           阶段二: 手眼标定          阶段三: 视觉伺服
[██████████] ✅ 已完成    [░░░░░░░░] 3-5天        [░░░░░░░░] 1-2周
```

---

#### 阶段一：抓取调通 🔧 ✅ 已完成

| 步骤 | 内容 | 状态 |
|------|------|------|
| **1** | 修复抓取位置：`grasp_pos` 不加 `ball_radius`，定位到球心 | ✅ |
| **1.5** | 三阶段减速下降：15cm→3cm→球心，最终接近速度 0.06m/s（比原来慢 12.5x） | ✅ |
| **2** | 修复夹爪闭合参数：闭合到 0.0（全闭靠接触力）、每步 5 次仿真迭代 + 80 步稳定 | ✅ |
| **3** | 调优物理参数：球摩擦 0.05→0.6、加 mass=0.03、软接触 solref="0.005 0.5" | ✅ |
| **4** | 修复 `run_piper.py`：关节数 7→6、ee_body link7→link6、MODEL_PATH、用 Controller 接口 | ✅ |
| **5** | 增加诊断日志：IK 目标 vs 实际误差、夹爪中心-球心偏差、闭合距离 | ✅ |
| **6** | 接入视觉检测链路（`detect_ball_from_camera()` 替换 `get_ball_position()`，含 fallback） | ✅ |
| **6.5** | 修复 IK/Controller 的 qpos/ctrl 索引（用 jnt_qposadr/actuator ID） | ✅ |

**阶段一成果汇总（最终版）**：
- IK 定位精度: 3μm
- 完整的 Pick-and-Place 流程：观察→接近→抓取→提升→横移→下降→释放 ✅
- 抓取策略: 夹爪中心定位球心 + 软接触变形 + 高摩擦 + 温和夹持
- 运动策略: 全程低速均匀（0.02~0.22 m/s），无速度突变
- 防滑体系: 高摩擦球(friction=3.0/2.5) + 软接触(solref=0.015) + 握力重稳定
- 下降方式: 笛卡尔分段直线(custom) — 杜绝关节插值导致的侧向偏移搓球
- 视觉检测已接入，含 fallback 机制
- Controller/IK 索引用 jnt_qposadr/actuator ID 修正
- 修改文件: `grasp_ball_demo.py`, `inverse_kinematics.py`, `simulation_controller.py`, `run_piper.py`, `scene.xml`

**抓取流程（13步）**：
```
1. 移动到初始位姿      6. 闭合夹爪抓取        11. 笛卡尔直线下降
2. 低头观察位姿        7. 微提2cm测试握力     12. 缓慢释放
3. 快速到球上方15cm    8. 正式提升8cm         13. 返回初始位姿
4. 慢降到球上方3cm     9. 缓慢升到安全高度
5. 极慢接近球心       10. 极慢横移到放置上方
```

#### 阶段二：手眼标定 📐

| 步骤 | 内容 | 依赖 | 关键文件 |
|------|------|------|----------|
| **7** | 实现 `_solve_rotation()`：Tsai-Lenz 轴角+SVD 方法 | 阶段一完成 | `hand_eye_calibration.py` |
| **8** | 修复 A/B 矩阵构建的符号和方向 | 步骤7 | `hand_eye_calibration.py` |
| **9** | 创建标定板 MuJoCo 模型 | - *(并行)* | `models/calibration_board.xml` |
| **10** | 创建自动标定采集流程（15-20 姿态） | 步骤7-9 | `examples/calibration_demo.py` |
| **11** | 标定验证 + 结果持久化 | 步骤10 | `hand_eye_calibration.py` |

#### 阶段三：视觉伺服 🎯

| 步骤 | 内容 | 依赖 | 关键文件 |
|------|------|------|----------|
| **12** | 创建 PBVS 模块（基于位置的视觉伺服） | 阶段二完成 | `core/visual_servo/position_based_vs.py` |
| **13** | 雅可比矩阵计算（MuJoCo `mj_jac` 或数值差分） | 步骤12 | `jacobian_estimator.py` |
| **14** | 创建 IBVS 模块（基于图像的视觉伺服，可选） | 步骤12 | `core/visual_servo/image_based_vs.py` |
| **15** | 视觉伺服演示脚本 | 步骤12-13 | `examples/pbvs_demo.py` |

#### 架构优化（穿插） 🏗️

| 步骤 | 内容 | 时机 |
|------|------|------|
| **16** | `grasp_ball_demo.py` 统一使用 `SimulationController` | 阶段一内 |
| **17** | 修复 IK/Controller 用 `jnt_qposadr`/`actuator_actadr` 索引 | 阶段一内 |
| **18** | 创建 `utils/config_loader.py`，加载 `config.yaml` | 任何时机 |
| **19** | 将 `print()` 替换为结构化日志 | 任何时机 |

---

### 🎯 核心发现：为什么抓小球会失败

经过对 `piper.xml` 夹爪结构的分析：

- link7 和 link8 是两个对称的手指，通过 equality 约束联动（joint8 = -joint7）
- 夹爪中心 = link7 和 link8 位置的中点，位于 link6 下方 0.135m
- 当前代码把夹爪中心定位到 `球心 + 0.02m`（球半径），导致手指在**球的上方空气**中闭合

**修复方案**：夹爪中心直接定位到球心（不加 radius），两指从左右对称闭合，物理接触力自然夹住球的赤道面。

---

### 关键逻辑验证

#### 夹爪的工作空间理解

```
piper.xml 夹爪结构:
  link6 (末端)
    ├── link7: joint7 (slide, axis 0 0 -1, range 0~0.035)
    │    位置: pos="0 0 0.13503" (link6 下方 0.135m)
    │    关节值=0 时手指在默认位置
    │    关节值=0.035 时手指向下移动 0.035m
    └── link8: joint8 (slide, axis 0 0 1, range -0.035~0)
         位置: pos="0 0 0.13503" (与 link7 同起点)
         约束: joint8 = -joint7 (equality)
         关节值=-0.035 时手指向上移动 0.035m

夹爪中心 (link7+link8 中点):
  - 始终位于 link6 下方 0.135m 处 (手指对称运动)
  - 两指间距 = 2 × gripper_ctrl

抓取小球 (radius=0.02):
  - 夹爪中心应定位到球心 → 两指从 ±0.019 处包住球赤道
  - 闭合目标 ≈ 0.0 (靠接触力自然停止)
```

#### 坐标变换验证

```
pixel_to_world(u, v, depth):
  camera 位姿: pos="0.6 0 0.5" quat="0.236 0 -0.972 0"
  → 相机光轴大致指向 -X 方向 (看向机械臂)
  → 公式 world = cam_pos + R_cam @ [x_cam, y_cam, z_cam] 正确
```

---

### 决策

- 抓取策略：夹爪中心定位到球心（非球顶），利用物理接触力自然闭合
- 标定方法：使用 Tsai-Lenz（Axis-Angle 版本），Eye-in-Hand 模式
- 视觉伺服：优先 PBVS（依赖标定结果），再考虑 IBVS
- 物理参数：增加球体摩擦和质量以提升抓取稳定性

### 排除范围

- 真机对接（RealRobotInterface 中的 TODO）不在本次规划内
- 多物体识别、复杂场景不在本次规划内
- 避障路径规划不在本次规划内
