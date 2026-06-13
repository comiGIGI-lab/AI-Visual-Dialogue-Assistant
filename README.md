# 基于 Orbbec 3D 视觉与姿态识别的办公久坐放松 AI 视觉对话助手

> **当前版本说明**：本项目目前仍处于**动作模仿挑战游戏原型**阶段（基于 PySide6 GUI + Orbbec 3D 深度相机 + YOLO + MediaPipe）。后续计划改造为面向久坐办公人群的 AI 视觉放松助手，通过交互式姿态引导帮助用户活动身体、缓解疲劳。

---

## 目录

- [功能概览](#功能概览)
- [硬件要求](#硬件要求)
- [推荐启动入口](#推荐启动入口)
- [环境安装](#环境安装)
- [模型文件](#模型文件)
- [目录结构](#目录结构)
- [操作说明](#操作说明)
- [常见问题排查](#常见问题排查)
- [命令行参数](#命令行参数)

---

## 功能概览

| 功能 | 说明 | 状态 |
|------|------|------|
| **动作模仿游戏** | PySide6 GUI，系统出题 → 用户做动作 → 做对得分 | ✅ 已实现 |
| **三种难度** | 练习（5s 出题 / 20% 反转）、普通（3s / 40%）、困难（2s / 50%） | ✅ 已实现 |
| **7 种动作识别** | 举手（左/右/双）、跳跃、抬腿（左/右）、弯腰、蹲下 | ✅ 已实现 |
| **连击系统** | 连续答对 combo 加分 + 飘字特效 + 音效反馈 | ✅ 已实现 |
| **排行榜** | 本地 JSON 持久化，Top 20 | ✅ 已实现 |
| **3D 骨架** | Open3D 实时人体骨骼 3D 可视化（可选） | ✅ 已实现 |
| **多人姿态** | YOLO + 多人 IoU 追踪 + 逐人动作识别 | ✅ 已实现 |
| **活体检测** | 深度信息区分真人/照片 | ✅ 已实现 |
| **普通摄像头支持** | 无需 Orbbec 相机即可运行 | ❌ 暂未实现（见下方说明） |
| **办公放松助手** | 久坐检测 + 智能活动建议 + 统计面板 | 🔜 计划中 |

---

## 硬件要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Windows 10 / 11（64 位） |
| **深度相机** | Orbbec 3D 深度相机（Gemini 系列），USB 连接 |
| **GPU（推荐）** | 支持 DirectX 12 的 GPU + onnxruntime-directml |
| **CPU（备选）** | 纯 CPU 推理亦可，帧率约 8-15 FPS |

> ⚠️ **关于普通摄像头**：当前版本**必须连接 Orbbec 3D 相机**才能运行。普通 USB 摄像头降级能力尚未实现（见常见问题第 4 条）。如果你没有 Orbbec 相机，目前只能查看 UI 界面，无法进入游戏。

---

## 推荐启动入口

```bash
# 主入口（PySide6 GUI 游戏）
python run_game_frontend.py
```

其他可用入口（OpenCV 窗口模式，较早期版本）：

| 文件 | 说明 |
|------|------|
| `game_demo_p1.py` | OpenCV 菜单式单人游戏 |
| `game_demo_yolov11.py` | YOLOv11 变体游戏 |
| `pose_detction_3d_v26.py` | 单人姿态 3D 分析 |
| `multiperson_v26_p2.py` | 多人姿态 Plan A（逐人 ROI） |
| `multiperson_v26_p3.py` | 多人姿态 Plan B（全图 MediaPipe） |

---

## 环境安装

### 1. 安装 Python

推荐 Python 3.10 ~ 3.12（必须 64 位）。

### 2. 创建虚拟环境（推荐）

```bash
# conda
conda create -n orbbec python=3.12
conda activate orbbec

# 或 venv
python -m venv venv
venv\Scripts\activate
```

### 3. 安装依赖

```bash
# 安装核心依赖
pip install -r requirements.txt

# （可选）GPU 加速
pip install onnxruntime-directml

# （可选）3D 骨架可视化
pip install open3d
```

### 4. 运行环境检查

```bash
python scripts/check_environment.py
```

按输出提示修复 FAIL 项后再启动游戏。

### 5. 验证安装

```bash
python scripts/check_environment.py
```

---

## 模型文件

### 放置位置

模型文件应放在 `models/` 目录下：

```
models/
├── yolo26n.onnx                 # YOLOv26-nano 人体检测（640×640）
├── yolo11n.onnx                 # YOLOv11-nano 人体检测（备选）
└── pose_landmarker_lite.task    # MediaPipe 姿态估计（33 关键点）
```

### 下载地址

| 模型 | 说明 | 来源 |
|------|------|------|
| `yolo26n.onnx` | YOLOv26 nano ONNX | 从 YOLO 官方 ONNX 导出 |
| `yolo11n.onnx` | YOLOv11 nano ONNX | 从 Ultralytics 导出 `yolo export model=yolo11n.pt format=onnx` |
| `pose_landmarker_lite.task` | MediaPipe Pose | [Google MediaPipe 官方](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker) |

### 其他资源文件

| 文件 | 用途 | 必需 |
|------|------|------|
| `coco.names` | YOLO 80 类 COCO 标签（项目根目录） | ✅ 是 |
| `鸽子舞.mp3` | 游戏背景音乐（项目根目录） | ❌ 否 |

---

## 目录结构

```
项目根目录/
├── run_game_frontend.py           # 🚀 推荐启动入口（PySide6 GUI）
├── requirements.txt               # Python 依赖清单
├── README.md                      # 本文件
├── coco.names                     # YOLO 类别标签
├── 鸽子舞.mp3                     # 背景音乐
├── utils.py                       # 相机帧格式转换工具
│
├── game_frontend/                 # PySide6 前端
│   ├── __init__.py
│   ├── main_window.py             # 主窗口 + 页面路由 + 状态机
│   ├── backend_thread.py          # AI 管线（相机+YOLO+MediaPipe+游戏逻辑）
│   ├── game_widgets.py            # 游戏控件（视频/HUD/倒计时/飘字）
│   ├── menu_widgets.py            # 菜单控件（主菜单/难度/设置）
│   ├── leaderboard_dialog.py      # 排行榜对话框
│   └── styles.py                  # 全局 QSS 暗色主题
│
├── scripts/                       # 工具脚本
│   └── check_environment.py       # 环境检查工具
│
├── models/                        # AI 模型文件
│   ├── yolo26n.onnx
│   ├── yolo11n.onnx
│   └── pose_landmarker_lite.task
│
├── game_demo_p1.py                # [旧版] OpenCV 单人游戏
├── game_demo_yolov11.py           # [旧版] YOLOv11 变体游戏
├── pose_detction_3d_v26.py        # [旧版] 单人姿态分析
├── multiperson_v26_p2.py          # [旧版] 多人姿态 Plan A
├── multiperson_v26_p3.py          # [旧版] 多人姿态 Plan B
├── pose_detection_3d_v26_antispoof.py  # [旧版] 活体检测
├── summary.py / sum1.py           # [旧版] 综合启动器
└── 居家安全监控.py                 # [旧版] 家居监控
```

---

## 操作说明

### 游戏流程

```
主菜单 → 选择难度 → Loading（相机+模型加载）→ 举起双手开始 → 
3秒倒计时 → 30秒游戏 → Game Over → 保存排行榜
```

### 快捷操作

| 按键 | 功能 |
|------|------|
| `F11` | 切换全屏 / 窗口 |
| `R` | 游戏中重新开始 |
| `ESC` / `Q` | 返回菜单 / 退出 |

---

## 常见问题排查

### 1. `pyorbbecsdk` 导入失败

```
ImportError: DLL load failed while importing pyorbbecsdk
```

> **注意**：Orbbec SDK v2 的 pip 包名是 **`pyorbbecsdk2`**，但安装后 Python 代码中仍然使用 `import pyorbbecsdk`（这是 pyorbbecsdk2 包提供的模块名）。

**原因**：Orbbec SDK 运行时 DLL 未安装或 PATH 未配置。

**解决**：
1. `pip install pyorbbecsdk2` 安装 Python 包
2. 从 [Orbbec SDK](https://github.com/orbbec/pyorbbecsdk) 下载并安装 C++ SDK 运行时
2. 确保 Orbbec SDK 的 `bin/` 目录在系统 PATH 中
3. 重启终端后再试

### 2. 模型文件找不到

```
[错误] 未找到 MediaPipe 模型: models/pose_landmarker_lite.task
```

**原因**：`models/` 目录下缺少对应模型文件。

**解决**：
1. 确认 `models/` 目录存在且包含所需文件
2. 运行 `python scripts/check_environment.py` 确认缺失项

### 3. 摄像头无画面 / 相机启动失败

```
[错误] 相机启动失败
```

**原因**：
- Orbbec 3D 相机未通过 USB 连接
- 相机被其他程序占用（如 Orbbec View 同时打开）
- Windows 未识别设备

**解决**：
1. 检查 USB 连接和相机指示灯
2. 关闭其他可能占用相机的程序
3. 在设备管理器中确认"Orbbec"设备正常识别
4. 目前**不支持普通 USB 摄像头**，必须有 Orbbec 设备

### 4. 没有 Orbbec 相机，能否运行？

**目前不能**。项目所有管线（YOLO + MediaPipe + 游戏逻辑）都依赖 Orbbec 相机的彩色+深度帧。普通摄像头的降级方案正在计划中，后续版本会通过 `cv2.VideoCapture(0)` 提供 fallback 通道（届时深度相关功能如 3D 骨架、活体检测将不可用，但核心游戏可玩）。

### 5. ONNX Runtime 只能用 CPU 推理

```
[信息] 未检测到GPU提供程序，使用 CPU 推理
```

**原因**：未安装 GPU 加速包，或 GPU 驱动不兼容。

**解决**：
```bash
# Windows
pip install onnxruntime-directml

# 确认 GPU 驱动为最新
# 确认 Windows 已安装 DirectX 12
```

### 6. `protobuf` 版本冲突

```
TypeError: Descriptors cannot be created directly...
```

**原因**：`mediapipe==0.10.9` 要求 `protobuf < 5`。

**解决**：
```bash
pip install protobuf==4.25.9
```

### 7. 中文显示为方块或乱码

**原因**：系统缺少中文字体。

**解决**：Windows 系统默认已安装微软雅黑（`C:\Windows\Fonts\msyh.ttc`）。如果被删除，请重新安装。

### 8. 启动游戏后画面卡在 Loading

**原因**：相机预热需要时间（约 3-5 秒获取 10 帧稳定曝光）。

**解决**：耐心等待。如果超过 10 秒仍未进入，检查相机连接。

---

## 命令行参数

### `run_game_frontend.py`

无命令行参数，所有设置在 GUI 中完成。

### 旧版独立脚本（参考）

```bash
python game_demo_p1.py --mode normal           # 普通难度
python game_demo_p1.py --mode hard --no-sound  # 困难模式 + 静音
python game_demo_p1.py --device cpu            # 强制 CPU 推理
```
