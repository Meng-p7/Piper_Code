## Plan: 基于 FOV 边界扫描 + IK 自动生成种子的手眼标定方案

### 问题分析

当前方案的瓶颈在于：**手动定义的 SEED_CONFIGS 覆盖空间有限，加噪声后部分配置落入 FOV 边界导致检测失败。** 成功率 58.7% 的根本原因是种子多样性不足。

### 新策略：缩小标定板 + FOV 边界采样 + IK 求解

**核心思路**：将标定板缩小到原来的一半 → 扩大有效观测区域 → 在区域内均匀采样相机位姿 → 用逆运动学求解关节角 → 自动生成大量有效种子。

### 现有资源

| 组件 | 状态 | 位置 |
|------|------|------|
| 逆运动学求解器 | ✅ 已有 | `core/kinematics/inverse_kinematics.py` |
| 正运动学 | ✅ 已有 | `core/kinematics/forward_kinematics.py` |
| 轨迹规划器 | ✅ 已有 | `core/trajectory/trajectory_planner.py` |
| 相机模块 | ✅ 已有 | `core/vision/camera.py` |
| 手眼标定算法 | ✅ 已有 | `core/calibration/hand_eye_calibration.py` |

### 关键参数

**相机**（piper.xml L230）：
- `fovy=60°`，分辨率 640×480
- `aspect=4/3` → HFOV = 2×atan(tan(30°)×4/3) ≈ **75.2°**，VFOV = **60°**
- 安装在 link6 上：`pos="0.05 0 0.04"` `quat="0 1 0 0"`（绕 X 轴旋转 180°）

**当前标定板**（calibration_scene.xml L22-89）：
- 9×7 方格（8×6 内角点），方格边长 30mm
- 物理尺寸：270×210mm
- 中心位置：(0.5, 0, 0.001)

**有效观测距离**：
- 当前标定板最小距离（完全可见）：≈180mm
- 新标定板（15mm 方格，135×105mm）最小距离：≈90mm
- 最大距离（检测需要足够像素）：约 300-400mm

---

### 实施步骤

#### 第一步：缩小标定板
- **文件**：`models/calibration_scene.xml`
- **改动**：方格边长从 30mm 改为 15mm
  - 格子 `size` 从 `0.015 0.015 0.001` 改为 `0.0075 0.0075 0.001`
  - 格子间距从 30mm 改为 15mm
  - 所有 `pos` 坐标除以 2
- **同步更新**：`demos/calibration_demo.py` 中 `SQUARE_SIZE = 0.015`

#### 第二步：FOV 边界扫描预计算
- **新增文件**：`demos/calibration_seed_generator.py`（独立预计算脚本，不修改现有模块）
- **功能**：
  1. 计算相机在不同距离和角度下，标定板是否完全在 FOV 内
  2. 在有效观测区域内均匀采样相机位姿点
  3. 每个点生成朝向标定板中心的相机旋转矩阵
  4. 通过 `T_cam_link6 = T_base_cam⁻¹ @ T_base_link6` 计算 link6 目标位姿
  5. 调用已有的 `InverseKinematics.solve_pose()` 求解关节角
  6. 验证正运动学确认末端位姿正确
  7. 输出有效关节角列表

**FOV 边界判断**：
```
对每个候选相机位置 cam_pos：
  T_cam_board = inv(T_world_cam) @ T_world_board
  board_corners_cam = T_cam_board @ board_4角

  检查条件：
  1. 所有角点 z_cam > 0（在相机前方）
  2. 所有角点在归一化图像坐标内：
     u = fx * x/z + cx,  0 < u < 640
     v = fy * y/z + cy,  0 < v < 480
  3. 留 5% 边距（不贴边）
  4. 棋盘格面积 > 阈值（保证检测精度）
```

**相机位姿采样空间**：
- 相机位置：标定板中心前方 (0.12 ~ 0.35m) 的球面/柱面区域
- 横向偏移：±0.15m
- 纵向偏移：±0.10m
- 俯仰角：向下 10°~50°（保证相机朝向标定板）
- 偏航角：±30°

**IK 求解**：
- 相机位姿 → link6 位姿：`T_base_link6 = T_base_cam @ T_cam_link6`
  - `T_cam_link6` 是相机到 link6 的固定变换（由安装参数决定）
- 调用 `InverseKinematics.solve_pose(target_pos, target_orientation, q_init)`
- 验证解的有效性（正运动学误差 < 阈值）

#### 第三步：重写 calibration_demo.py 采集逻辑
- **删除**：SEED_CONFIGS 手动列表、random_joint_config()、加权采样
- **新增**：从预计算文件加载有效关节角列表
- **采样策略**：
  1. 从有效列表中随机选择（无噪声）
  2. 若检测失败，尝试同一 IK 解的微扰版本（小噪声 ±0.01rad）
  3. 保留局部修复重试机制（纯净解 → 小噪声 → 跳过）
- **目标**：每个 IK 解都经过 FOV 验证，理论成功率接近 100%

#### 第四步：标定板格子尺寸适配
- `SQUARE_SIZE = 0.015`（15mm）
- `build_object_points()` 使用新的 SQUARE_SIZE
- `findChessboardCorners` 仍然检测 8×6 内角点（不变）

#### 第五步：验证与调优
- 运行完整标定流程，验证检测成功率和标定精度
- 调整 FOV 边距和采样密度
- 确保旋转误差 < 1°，平移误差 < 2mm

---

### 技术细节

**相机到 link6 的变换**（从 piper.xml 提取）：
```
wrist_cam_body pos="0.05 0 0.04" quat="0 1 0 0"

T_link6_cam = [R_x(180°) | (0.05, 0, 0.04); 0 0 0 1]
R_x(180°) = [[1,0,0], [0,-1,0], [0,0,-1]]

T_cam_link6 = inv(T_link6_cam)
  R_cam_link6 = R_x(180°) = [[1,0,0], [0,-1,0], [0,0,-1]]
  t_cam_link6 = -R_x(180°) @ (0.05, 0, 0.04) = (-0.05, 0, 0.04)
```

**IK 使用方式**：
```python
from core.kinematics import InverseKinematics

ik = InverseKinematics(model, ee_body_name="link6")
q_sol, success = ik.solve_pose(
    target_pos=link6_target_pos,
    target_orientation=link6_target_rot,
    q_init=home_qpos
)
```

**预计算流程**：
1. 加载 MuJoCo 模型
2. 初始化 IK 求解器
3. 在相机位姿空间中均匀采样 N 个点（N=500~1000）
4. 对每个点：检查 FOV 约束 → 计算 link6 位姿 → IK 求解 → 验证
5. 保存所有有效关节角到 `.npy` 文件
6. calibration_demo.py 直接加载使用

---

### 预期收益
- **成功率**：从 58.7% 提升到 **90%+**（因为每个种子都经过 FOV 验证）
- **空间多样性**：相机位姿在整个有效区域内均匀分布，远优于手动种子
- **标定精度**：由于样本空间覆盖更好，标定精度应进一步提升
- **维护性**：不再需要手动调参，更换标定板尺寸后重新运行预计算即可

### 风险
- **IK 求解失败**：某些相机位姿可能超出机械臂工作空间 → 过滤掉
- **小标定板检测**：15mm 方格在远距离可能检测困难 → 限制最大距离 + 测试验证
- **IK 解的多解性**：同一末端位姿可能有多个关节角解 → 这反而是好事，增加了多样性
