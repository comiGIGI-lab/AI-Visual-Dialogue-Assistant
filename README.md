# OfficeFit AI 视觉对话放松助手

> 🎥 **Demo 视频**：[待补充]
>
> ⚠️ 最终提交前会更新为可访问的视频链接，展示完整游戏流程与姿态识别效果。

---

## 项目简介

**OfficeFit AI 视觉对话放松助手** 是一款面向办公室久坐人群的智能视觉交互应用。项目基于 **Orbbec 3D 深度相机**、**YOLO 目标检测**、**MediaPipe 姿态估计**和 **PySide6 GUI**，通过实时人体姿态识别引导用户完成身体活动动作，帮助缓解久坐带来的健康风险。

**当前版本** 已实现动作模仿互动游戏原型：系统随机出题，用户按屏幕提示做出对应动作，AI 实时判定正确性并计分。后续将扩展 **Orbbec 主相机 + 笔记本摄像头降级兼容**、**语音对话交互**、**久坐智能提醒** 和 **跟练放松模式**，最终打造完整的办公健康助手。

---

## 当前已实现功能

| 模块 | 功能 | 状态 |
|------|------|:--:|
| **图形界面** | PySide6 暗色主题 GUI，含主菜单、难度选择、游戏 HUD、排行榜对话框 | ✅ |
| **人体检测** | YOLOv26-nano ONNX 推理，支持 GPU（DirectML/CUDA）和 CPU | ✅ |
| **姿态估计** | MediaPipe Pose Landmarker（33 关键点），支持多姿态 | ✅ |
| **动作识别** | 7 种动作：举起双手/左手/右手、跳跃、抬左腿/右腿、蹲下 | ✅ |
| **互动游戏** | 正向/反向出题、三档难度、连击加分、音效反馈、飘字特效 | ✅ |
| **排行榜** | 本地 JSON 持久化，Top 20，支持按难度筛选 | ✅ |
| **多人姿态** | IoU 跨帧追踪 + 逐人独立动作识别（Plan A / Plan B） | ✅ |
| **3D 骨架** | Open3D 实时 3D 骨骼可视化（可选，游戏中按 J 开关） | ✅ |
| **活体检测** | 基于深度信息的真人/照片判别 | ✅ |
| **环境检查** | `scripts/check_environment.py` 一键检测所有依赖和硬件 | ✅ |
| **模型路径兼容** | 同时支持 `models/` 目录和项目根目录放置模型文件 | ✅ |
| **普通摄像头检测** | 环境检查脚本已支持 USB 摄像头检测，为主程序降级做准备 | ✅ |
| **Orbbec 部署文档** | README 含完整硬件连接、SDK 安装和常见问题排查 | ✅ |

---

## 项目运行方式

### 环境准备

```bash
# 1. 创建并激活 conda 环境（推荐 Python 3.10~3.12）
conda create -n officefit python=3.10
conda activate officefit

# 2. 安装依赖
pip install -r requirements.txt

# （可选）GPU 加速
pip install onnxruntime-directml

# （可选）3D 骨架可视化
pip install open3d
```

### 环境检查

```bash
python scripts/check_environment.py
```

该脚本会逐项检查 Python 版本、所有依赖库导入、模型文件完整性、Orbbec 相机连接状态和 USB 摄像头可用性，并在末尾输出 PASS / WARNING / FAIL 总结。

### 启动游戏

```bash
python run_game_frontend.py
```

游戏流程：

```
主菜单 → 选择难度 → 相机+模型加载 → 举起双手开始 → 3 秒倒计时 → 30 秒游戏 → 排行榜
```

### 快捷操作

| 按键 | 功能 |
|------|------|
| `F11` | 切换全屏 / 窗口 |
| `R` | 游戏中重新开始 |
| `ESC` / `Q` | 返回菜单 / 退出 |

---

## 硬件说明

| 项目 | 说明 |
|------|------|
| **主视觉硬件** | Orbbec 3D 深度相机（Gemini 系列），通过 USB 连接 |
| **操作系统** | Windows 10 / 11（64 位） |
| **GPU（推荐）** | 支持 DirectX 12 的 GPU，搭配 `onnxruntime-directml` |
| **CPU（备选）** | 纯 CPU 推理亦可，帧率约 8–15 FPS |
| **降级兼容** | 笔记本自带 USB 摄像头 — 环境检查脚本已可检测，主程序降级通道后续接入 |

> ⚠️ **当前版本需要 Orbbec 3D 相机**。普通摄像头降级兼容正在开发中，届时深度相关特性（3D 骨架、活体检测）在降级模式下不可用，但核心游戏功能将完整保留。

---

## 模型文件说明

以下文件需放置在 `models/` 目录下（环境检查脚本同时兼容项目根目录直接放置）：

| 文件 | 用途 | 大小 |
|------|------|------|
| `yolo26n.onnx` | YOLOv26-nano 人体检测模型（640×640） | ~10 MB |
| `yolo11n.onnx` | YOLOv11-nano 人体检测模型（备选） | ~10 MB |
| `pose_landmarker_lite.task` | MediaPipe 姿态估计模型（33 关键点） | ~5.6 MB |
| `coco.names` | COCO 80 类标签文件 | <1 KB |

---

## 目录结构

```
项目根目录/
├── run_game_frontend.py           # 主入口（PySide6 GUI）
├── requirements.txt               # Python 依赖清单
├── README.md                      # 本文件
├── coco.names                     # YOLO 类别标签
├── utils.py                       # 相机帧格式转换
│
├── game_frontend/                 # PySide6 前端模块
│   ├── main_window.py             # 主窗口 + 页面路由 + 状态机
│   ├── backend_thread.py          # AI 管线（相机 + YOLO + MediaPipe + 游戏）
│   ├── game_widgets.py            # 视频控件 / Loading / 倒计时 / HUD / 飘字
│   ├── menu_widgets.py            # 菜单 / 难度 / 设置页面
│   ├── leaderboard_dialog.py      # 排行榜对话框
│   └── styles.py                  # 全局暗色 QSS 主题
│
├── scripts/                       # 工具脚本
│   └── check_environment.py       # 部署环境一键检查
│
├── models/                        # AI 模型文件
│   ├── yolo26n.onnx
│   ├── yolo11n.onnx
│   └── pose_landmarker_lite.task
│
└── game_demo_p1.py 等             # 早期 OpenCV 原型（保留参考）
```

---

## 环境说明

| 项目 | 说明 |
|------|------|
| **Python 版本** | 推荐 **Python 3.10**，兼容 3.11~3.12；**不推荐 Python 3.13**（部分依赖未充分测试） |
| **架构** | 必须 64 位 Python |
| **Orbbec SDK 包名** | pip 包名是 `pyorbbecsdk2`，安装后 Python 代码使用 `import pyorbbecsdk` |
| **protobuf 兼容** | mediapipe 0.10.x 要求 protobuf < 5，冲突时执行 `pip install protobuf==4.25.9` |

---

## 常见问题排查

### 1. `pyorbbecsdk` 导入失败

> pip 包名是 **`pyorbbecsdk2`**，代码中仍然 `import pyorbbecsdk`。

```bash
pip install pyorbbecsdk2
```

若仍失败，请从 [Orbbec SDK GitHub](https://github.com/orbbec/pyorbbecsdk) 安装 C++ 运行时并将 `bin/` 加入系统 PATH。

### 2. 模型文件找不到

运行 `python scripts/check_environment.py` 确认缺失文件，将对应模型放入 `models/` 或项目根目录。

### 3. 相机启动失败 / 无画面

- 确认 Orbbec 相机已通过 USB 连接，设备管理器中可见
- 关闭其他占用相机的程序（如 Orbbec Viewer）
- 当前版本不支持普通 USB 摄像头作为主相机（降级通道开发中）

### 4. ONNX Runtime 仅 CPU 推理

```bash
pip install onnxruntime-directml
```

### 5. `protobuf` 版本冲突

```bash
pip install protobuf==4.25.9
```

---

## 项目进展

| 阶段 | 内容 | 状态 |
|------|------|:--:|
| Phase 1 | 动作模仿互动游戏原型（PySide6 GUI + YOLO + MediaPipe） | ✅ 已完成 |
| Phase 2 | 部署文件补齐 + 环境检查脚本 + README 文档 | ✅ 已完成 |
| Phase 3 | 笔记本摄像头降级兼容（`cv2.VideoCapture` fallback） | 🔜 计划中 |
| Phase 4 | 语音对话交互（语音出题 + 语音反馈） | 🔜 计划中 |
| Phase 5 | 久坐智能提醒 + 跟练放松模式 | 🔜 计划中 |
| Phase 6 | 运动统计面板 + 历史记录可视化 | 🔜 计划中 |

当前项目正处于从"互动游戏原型"向"AI 视觉对话办公放松助手"的演进阶段。

---

## 注意事项

- ❌ **不要**将 API key、账号密码、访问令牌提交到本仓库
- ❌ **不要**将 Orbbec SDK 源码、固件包、驱动程序提交到本仓库
- ❌ **不要**将 `.onnx`、`.pt` 以外的二进制依赖（DLL、.exe、.whl）提交到本仓库
- ✅ 模型文件（`.onnx`、`.task`）属于项目必需的资源文件，可以提交
- ✅ 建议使用 `.gitignore` 排除 `__pycache__/`、`.vscode/` 等 IDE 和缓存目录
