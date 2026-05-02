## Plan: 手眼标定收尾 + 视觉伺服

**TL;DR**: 手眼标定核心算法已完成（Tsai-Lenz 轴角+SVD，合成数据误差 ~1e-16），剩余标定板模型、采集流程、验证脚本。之后进入阶段三视觉伺服。

---

### 已完成（不再列出细节）

| 阶段 | 内容 |
|------|------|
| 阶段一 | 抓取调通：13步 Pick-and-Place 全流程 |
| 阶段二 核心 | `_solve_rotation()` Tsai-Lenz 实现、A/B 矩阵修复、`eye_mode` 双模式 |
| 架构 | config_loader、Controller 统一接口、SLERP 修复、`get_camera_image` 删除 |
| 代码质量 | 10 文件类型注解、4 测试文件 (kinematics/trajectory/config/hand_eye) |

---

### 阶段二剩余：标定基础设施

| 步骤 | 内容 | 关键文件 |
|------|------|----------|
| **9** | 创建标定板 MuJoCo 模型（棋盘格/ArUco） | `models/calibration_board.xml` |
| **10** | 创建自动标定采集脚本（15-20 姿态 → 记录 T_base_ee + T_cam_board） | `examples/calibration_demo.py` |
| **11** | 标定验证：像素坐标→世界坐标精度对比 | `examples/calibration_demo.py` |

---

### 阶段三：视觉伺服

| 步骤 | 内容 | 关键文件 |
|------|------|----------|
| **12** | PBVS 模块：相机检测目标 → 世界坐标 → 位置误差 → 关节增量 | `core/visual_servo/position_based_vs.py` |
| **13** | 雅可比矩阵（MuJoCo `mj_jac` 或数值差分） | `core/visual_servo/jacobian_estimator.py` |
| **14** | IBVS 模块（可选，基于图像雅可比） | `core/visual_servo/image_based_vs.py` |
| **15** | 视觉伺服演示脚本 | `examples/pbvs_demo.py` |

---

### 架构优化剩余

| 步骤 | 内容 |
|------|------|
| **19** | `print()` → 结构化日志 | ✅ 已完成（核心模块；demo 脚本保留用户可见输出） |

---

### 决策

- 标定方法：Tsai-Lenz 轴角+SVD ✅ 已验证（误差 ~1e-16）
- 视觉伺服：优先 PBVS（依赖手眼标定结果）
- 标定板：棋盘格或 ArUco，放在 `scene.xml` 工作台上

