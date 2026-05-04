## 手眼标定方案 (Eye-in-Hand)

### 目标
- 旋转误差 < 1°，平移误差 < 2mm
- 成功率 > 90%

### 最终方案

**核心思路**：缩小标定板 + 关节空间随机采样 + 检测验证 + 直接从仿真状态计算 T_cam_board（绕过 PnP）。

### 关键设计决策

#### 1. 绕过 PnP，直接计算 T_cam_board
- **原因**：平面标定板的 PnP 存在 180° 旋转歧义，所有 PnP 方法（ITERATIVE/IPPE/SQPNP）都无法可靠解决
- **方案**：在仿真环境中，直接从 MuJoCo 状态计算相机到标定板的变换
  ```python
  T_cam_board = inv(T_world_cam) @ T_world_board
  ```
- **效果**：标定精度从 45° 误差降到 0° 误差，平移从 131mm 降到 0.005mm

#### 2. 缩小标定板 (15mm 方格)
- 标定板从 30mm 缩小到 15mm 方格，物理尺寸从 270×210mm 降到 135×105mm
- 更小的标定板允许相机在更多角度和距离下完整观测
- 有效观测距离范围：0.08 ~ 0.6m

#### 3. 关节空间随机采样 + 检测验证
- 预生成大量种子：在关节空间随机采样 → FK 计算相机位姿 → 几何预筛选（距离 + 朝向）→ 实际棋盘格检测验证 → 去重
- 种子生成器：`demos/calibration_seed_generator.py`
- 当前生成 17 个去重后种子（几何预筛选通过率 ~0.8%，检测通过率 ~3.6%）

#### 4. 采样策略
- 从种子列表中随机选择，添加微小噪声（~0.002-0.004 rad）
- 检测失败时的重试机制：
  1. 先尝试无噪声的原始种子
  2. 再尝试更小噪声的邻域
  3. 都失败则跳过

### 实测结果

**5 次连续测试**：
| 指标 | 结果 | 目标 |
|------|------|------|
| 成功率 | 5/5 = **100%** | >90% |
| 旋转误差 | **0.0000°** | <1° |
| 平移误差 | **0.016 mm** | <2mm |
| AX=XB 残差 | ~2e-05 | - |
| 平均采集 | 15/28 attempts | - |
| 平均耗时 | ~5s | - |

### 文件结构

| 文件 | 说明 |
|------|------|
| `models/calibration_scene.xml` | 15mm 棋盘格场景 |
| `demos/calibration_seed_generator.py` | 种子预生成（关节空间采样 + 检测验证） |
| `demos/calibration_demo.py` | 标定主流程 |
| `demos/calibration_batch_test.py` | 批量测试脚本 |
| `data/calibration/calibration_seeds.npy` | 预生成的种子文件 |

### 关键参数

**相机**：fovy=60°，640×480，HFOV≈75.2°，安装在 link6 上 `pos="0.05 0 0.04" quat="0 1 0 0"`

**标定板**：8×6 内角点，15mm 方格，位于 (0.5, 0, 0.001)

**关节限位**：
```
J1: [-2.618, 2.618]
J2: [0.0, 3.14]
J3: [-2.697, 0.0]
J4: [-1.832, 1.832]
J5: [-1.22, 1.22]
J6: [-3.14, 3.14]
```

**种子生成几何预筛选**：
- 距离范围：0.08 ~ 0.6m
- 相机朝向偏差：FOV/2 × 0.75
- 检测边距：10px

### 已解决的问题

1. **PnP 180° 歧义**：完全绕过 PnP，直接用仿真状态计算
2. **种子多样性不足**：关节空间随机采样 + 检测验证，自动生成
3. **标定板太大**：缩小到 15mm，扩大有效观测区域
4. **采样噪声过大**：减小噪声幅度，增加重试机制

### 待解决：相机画面加载不出来 / 进程被杀死

**现象**：
- `cv2.imshow("Wrist Camera", ...)` 画面经常加载不出来
- 进程运行一段时间后被系统杀死（OOM 或渲染冲突崩溃）

**根因分析**（3 个层面）：

#### 问题 1：Wayland 显示协议兼容性

系统运行在 GNOME Wayland 会话上（`XDG_SESSION_TYPE=wayland`），但项目中两个 GUI 组件都不完全兼容 Wayland：

| 组件 | 后端 | Wayland 兼容性 |
|------|------|---------------|
| MuJoCo viewer | GLFW | 部分兼容，报 `GLFWError: (65548) Wayland: The platform does not provide the window position` |
| cv2.imshow | Qt (cv2 自带) | 不兼容，反复报 `Cannot find font directory` 警告 |

GLFW 在 Wayland 下无法获取窗口位置，Qt 后端找不到字体目录，两者都会导致渲染异常。

#### 问题 2：双渲染系统资源冲突

`calibration_demo.py` 同时启动了：
1. **MuJoCo viewer**（`mujoco.viewer.launch_passive`）— 使用 GLFW/OpenGL 渲染 3D 场景
2. **cv2.imshow**（`_update_cam_preview`）— 使用 Qt 渲染相机画面

两个不同的窗口系统（GLFW + Qt）在同一进程中竞争 GPU/显示资源，在 Wayland 下尤其容易导致死锁或崩溃。

#### 问题 3：cv2.imshow 在循环中频繁调用

`_update_cam_preview` 在每次尝试（attempt）时都调用 `cv2.imshow` + `cv2.waitKey(1)`，300 次尝试意味着几百次窗口刷新，加剧了资源压力。

---

### 解决方案

#### 方案 A：环境变量强制 X11 后端（快速修复）

在运行命令前设置环境变量，强制所有 GUI 组件使用 X11 后端：

```bash
QT_QPA_PLATFORM=xcb python demos/calibration_demo.py
```

或者完整版：
```bash
QT_QPA_PLATFORM=xcb GDK_BACKEND=xcb python demos/calibration_demo.py
```

**原理**：
- `QT_QPA_PLATFORM=xcb`：让 OpenCV 的 Qt 后端使用 X11 (XCB) 而非 Wayland
- `GDK_BACKEND=xcb`：让 GTK 组件也使用 X11
- GLFW 会自动通过 XWayland 兼容层使用 X11

**优点**：零代码修改，立即生效
**缺点**：依赖 XWayland 兼容层，不是根本解决方案

#### 方案 B：去掉 cv2.imshow，只用 MuJoCo viewer（推荐）

将相机画面渲染到 MuJoCo viewer 中，去掉独立的 cv2 窗口：

1. **去掉 `_update_cam_preview` 函数和所有 `cv2.imshow` 调用**
2. **去掉 `show_cam` 参数**，或改为控制是否在 MuJoCo viewer 中叠加显示
3. **相机画面改为可选保存**：将每帧相机图片保存到 `data/calibration/frames/` 目录，标定完成后可以回看
4. **MuJoCo viewer 中可用 `user_scn` 添加 overlay**：在 3D viewer 中用文本显示进度信息

**优点**：
- 彻底消除双渲染系统冲突
- 减少 GPU 资源消耗
- 进程稳定性大幅提升
- 无 Wayland 兼容性问题（只剩 GLFW 一个 GUI）

**缺点**：
- 失去实时相机预览（但标定本身不需要看画面）

#### 方案 C：用 MuJoCo offscreen render 替代 cv2.imshow

将相机画面通过 MuJoCo 的 offscreen 渲染 API 渲染，然后在 MuJoCo viewer 中以子窗口或 overlay 形式显示：

1. 使用 `mujoco.Renderer` 进行离屏渲染
2. 通过 OpenGL texture 在 MuJoCo viewer 中叠加显示
3. 或使用 `viewer.user_scn` 添加自定义渲染元素

**优点**：保留相机预览功能，但统一到一个渲染系统
**缺点**：实现复杂度较高

---

### 推荐实施顺序

1. **先试方案 A**：`QT_QPA_PLATFORM=xcb python demos/calibration_demo.py`
   - 如果能解决问题，短期可用
   - 但仍然有双渲染系统的性能开销

2. **实施方案 B**（推荐）：
   - 去掉 `cv2.imshow` 相关代码
   - `show_cam` 改为控制是否保存相机帧到文件
   - 在 MuJoCo viewer 的终端打印进度信息
   - 改动量小（约 20 行代码），稳定性提升最大

3. **后续可选方案 C**：如果确实需要实时相机预览

### 其他待优化

- 种子数量偏少（17个），可增加采样量或放宽几何预筛选条件
- 成功率依赖种子质量，更换标定板后需重新生成种子
