# OfficeFit AI 视觉对话放松助手

> 🎥 **Demo 视频**：[待补充]
>
> ⚠️ 最终提交前会更新为可访问的视频链接，展示完整游戏流程与姿态识别效果。

---

## 项目简介

**OfficeFit AI 视觉对话放松助手** 是一款面向办公室久坐人群的智能视觉交互应用。项目基于 **Orbbec 3D 深度相机**、**YOLO 目标检测**、**MediaPipe 姿态估计**和 **PySide6 GUI**，通过实时人体姿态识别引导用户完成身体活动动作，帮助缓解久坐带来的健康风险。

**当前版本** 已实现动作模仿互动游戏原型：系统随机出题，用户按屏幕提示做出对应动作，AI 实时判定正确性并计分。后续将扩展 **Orbbec 主相机 + 笔记本摄像头降级兼容**、**语音对话交互**、**久坐智能提醒** 和 **跟练放松模式**，最终打造完整的办公健康助手。

---

## 代码来源与复用说明

本项目部分历史代码片段复用自本人参与的深圳大学追光空间站创新短课小组自研项目。

1. 本人为该课程/小组项目原创成员之一，相关代码由小组成员共同开发；
2. 该课程项目过去未上传至任何公开或私有 Git 仓库，属于小组内部留存的课程作业代码；
3. 本次比赛在该历史原型基础上进行二次开发，主要完成了部署文档、环境检查、OfficeFit 场景化改造、相机兼容方案规划等工作；
4. 不存在抄袭第三方开源项目或盗用他人代码的行为，代码知识产权归原小组成员共同所有，符合本次比赛原创性要求。

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
| `F11` | 切换全屏 / 窗口（默认窗口模式） |
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
| **降级兼容** | 笔记本自带 USB 摄像头 — 已完整支持 RGB 兼容模式 |

### 运行模式

| 模式 | 条件 | 支持功能 |
|------|------|----------|
| **Orbbec 3D 模式** | Orbbec 相机已连接 + pyorbbecsdk 已安装 | RGB + Depth、YOLO、MediaPipe、3D 骨架、活体检测、完整游戏/放松流程 |
| **USB 摄像头兼容模式** | 无 Orbbec 时自动启用 | RGB 图像、YOLO、MediaPipe、互动游戏/放松流程（无深度、3D 骨架、活体检测） |

> 程序会自动检测：优先使用 Orbbec，不可用时降级到 USB 摄像头。也可以通过环境变量强制指定模式（见下方）。

### 强制指定相机模式

Windows PowerShell:
```powershell
# 强制使用普通 USB 摄像头（测试兼容模式）
$env:OFFICEFIT_CAMERA="webcam"; python run_game_frontend.py

# 强制使用 Orbbec（失败则报错退出）
$env:OFFICEFIT_CAMERA="orbbec"; python run_game_frontend.py
```

---

## 模型文件说明

以下文件需放置在 `models/` 目录下（环境检查脚本同时兼容项目根目录直接放置）：

| 文件 | 用途 | 大小 |
|------|------|------|
| `yolo26n.onnx` | YOLOv26-nano 人体检测模型（640×640） | ~10 MB |
| `yolo11n.onnx` | YOLOv11-nano 人体检测模型（备选） | ~10 MB |
| `pose_landmarker_lite.task` | MediaPipe 姿态估计模型（33 关键点） | ~5.6 MB |
| `pose_landmarker_full.task` | 可选，推荐平衡精度 | ~ |
| `pose_landmarker_heavy.task` | 可选，高精度但更慢 | ~ |

通过环境变量切换 Pose 模型：
```powershell
$env:OFFICEFIT_POSE_MODEL="models/pose_landmarker_full.task"
```
不设置时默认使用 lite 版本。可用级别：lite / full / heavy / custom。
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

## 本地语音交互 (MVP)

### 支持的关键词

| 关键词 | 功能 | 回复示例 |
|--------|------|----------|
| 开始放松 | 启动放松训练 | 检测到人 → "好的，我们开始本次放松"；未检测到 → "请站到画面中央" |
| 暂停 | 暂停当前训练 | "已暂停，准备好后可以说继续" |
| 继续 | 恢复暂停的训练 | "好的，我们继续当前放松训练" |
| 换一个动作 | 跳过当前动作 | "好的，切换到下一个动作" |
| 结束 | 结束训练 | "本次放松结束，建议活动肩颈并喝水" |
| 上半身 | 切换上半身模式 | "已切换为上半身放松模式" |
| 全身 | 切换全身模式 | "已切换为全身活力模式" |

### 技术实现

- **语音识别**：`speech_recognition` + `pyaudio`，本地 Google Speech 引擎
- **对话管理**：纯规则匹配，不调用云端 API
- **UI 面板**：左侧「AI 办公守护助手」面板，含状态区、最近输入、AI 建议、推荐动作
- **快捷输入**：面板下方「快捷输入」按钮是无麦克风环境下的备用输入，点击等价于一次语音输入
- **降级兼容**：麦克风不可用时快捷输入始终可用

### 放松模式

| 模式 | 适用场景 | 动作范围 |
|------|----------|----------|
| **上半身放松** (默认) | 坐在桌前、webcam demo | 举手（左/右/双）、双手打开扩胸、左右侧拉伸、颈部转动（引导） |
| **全身互动** | 站立、空间足够 | 以上 + 蹲下、抬腿（左/右）、跳跃 |

在难度选择页可以切换放松类型并查看动作说明。

### 动作说明

点击"查看动作说明"可打开弹窗，展示每个动作的文字说明和示意图（占位）。将图片放入 `assets/exercise_guides/` 目录后，弹窗自动显示对应图片。用户可自行替换。

### 工作流

```
选择放松模式 → Loading → 相机就绪
  → 「说"开始放松"或点击模拟按钮」  ← 新增 ready_waiting 阶段
  → 用户确认 → 举手倒计时 (3s) → 放松训练 (30s) → 完成
```

### 外部 AI 支持 (可选)

| 环境变量 | 说明 |
|----------|------|
| `OFFICEFIT_AI_PROVIDER=none` | 默认，仅使用本地规则 |
| `OFFICEFIT_AI_PROVIDER=openai` | 启用 OpenAI API（需设置 OPENAI_API_KEY） |

启用外部 AI 后：
- 语音识别可使用 Whisper API 提高准确率
- 回复可使用 GPT-4o-mini 根据用户状态生成自然语言

```powershell
$env:OFFICEFIT_AI_PROVIDER="openai"
$env:OPENAI_API_KEY="sk-..."   # 替换为你的 key
python run_game_frontend.py
```

### Qwen / DashScope API 测试

```powershell
$env:DASHSCOPE_API_KEY="sk-你的key"
python scripts/test_qwen_api.py

# 可选: 切换模型
$env:OFFICEFIT_QWEN_MODEL="qwen-turbo"
```

脚本仅做连通性验证，不接入主程序。支持 qwen-plus / qwen-turbo / qwen-max。

> 不要将 API key 写入代码或提交到仓库。

### 成本控制策略

| 数据 | 策略 |
|------|------|
| 原始视频 | 不上传，本地处理 |
| 原始音频 | 不上传（默认）；可选上传短音频到 Whisper API |
| 高频反馈 | 本地规则完成 |
| 外部 AI 失败 | 自动回退本地规则，UI 显示回退信息 |
| AI 回复上下文 | 仅上传结构化摘要（相机模式/深度/动作/语音文本），不上传视频帧 |

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

- 程序会自动尝试 Orbbec → USB 摄像头降级
- 如果两种相机都不可用，确认至少有一个摄像头连接
- 关闭其他占用相机的程序（如 Orbbec Viewer、其他视频应用）
- 使用 `python scripts/check_environment.py` 检查相机状态
- 没有 Orbbec 时，程序自动使用 RGB 兼容模式（深度/3D 骨架/活体检测不可用）

### 4. ONNX Runtime 仅 CPU 推理

```bash
pip install onnxruntime-directml
```

### 5. 麦克风不可用

1. 运行 `python scripts/check_environment.py` 查看麦克风设备列表和编号
2. 手动指定设备编号：
   ```powershell
   $env:OFFICEFIT_MIC_INDEX="0"
   python run_game_frontend.py
   ```
3. 如果仍不可用，使用模拟语音按钮进行演示

> 默认启动只打印当前使用的麦克风，避免刷屏。
> 如需排查麦克风设备，打印完整设备列表（speech_recognition + PyAudio）：
>
> ```powershell
> $env:OFFICEFIT_VOICE_DEBUG="1"
> python run_game_frontend.py
> ```
>
> 或直接运行环境检查脚本（始终打印完整列表）：
>
> ```powershell
> python scripts/check_environment.py
> ```

### 6. `protobuf` 版本冲突

```bash
pip install protobuf==4.25.9
```

---

## 项目进展

| 阶段 | 内容 | 状态 |
|------|------|:--:|
| Phase 1 | 动作模仿互动游戏原型（PySide6 GUI + YOLO + MediaPipe） | ✅ 已完成 |
| Phase 2 | 部署文件补齐 + 环境检查脚本 + README 文档 | ✅ 已完成 |
| Phase 3 | 笔记本摄像头降级兼容（`cv2.VideoCapture` fallback） | ✅ 已完成 |
| Phase 4 | 语音对话交互（本地语音指令 MVP + 上半身模式 + 工作流改进） | ✅ 已完成 |
| Phase 5 | 久坐智能提醒 + 跟练放松模式 | 🔜 计划中 |
| Phase 5.1 | 颈部转动动作（轻量引导，不计分）| ✅ 已完成 |
| Phase 5.2 | 健康定时提醒（久坐 35min + 喝水 60min）| ✅ 已完成 |
| Phase 5.3 | AI 意图识别 + 动作推荐模块 | ✅ 已完成 |
| Phase 6 | 运动统计面板 + 历史记录可视化 | 🔜 计划中 |

### 本地 AI 意图识别

- 支持 18 种久坐症状意图标签（颈部/肩膀/上背/腰部/臀部/体态）
- 复合症状识别（"脖子肩膀腰都酸"）
- 黑名单过滤（手指/手腕/眼部等精细不适不映射到骨架动作）
- 推荐动作全部限制在 MediaPipe 可检测的大肢体动作范围
- 测试: `python scripts/test_intent_recognizer.py`
- 云端大模型仅作为可选增强，默认本地规则可运行

当前项目正处于从"互动游戏原型"向"AI 视觉对话办公放松助手"的演进阶段。

---

## 项目进展

| 阶段 | 内容 | 状态 |
|------|------|:--:|
| Phase 1 | 动作模仿互动游戏原型（PySide6 GUI + YOLO + MediaPipe） | ✅ 已完成 |
| Phase 2 | 部署文件补齐 + 环境检查脚本 + README 文档 | ✅ 已完成 |
| Phase 2.1 | UI 文案已从动作游戏适配为 OfficeFit 办公放松助手原型 | ✅ 已完成 |
| Phase 3 | 笔记本摄像头降级兼容（`cv2.VideoCapture` fallback） | 🔜 计划中 |
| Phase 4 | 语音对话交互（语音出题 + 语音反馈） | 🔜 计划中 |
| Phase 5 | 久坐智能提醒 + 跟练放松模式 | 🔜 计划中 |
| Phase 6 | 运动统计面板 + 历史记录可视化 | 🔜 计划中 |

---

## 命令行参数
## 注意事项

- ❌ **不要**将 API key、账号密码、访问令牌提交到本仓库
- ❌ **不要**将 Orbbec SDK 源码、固件包、驱动程序提交到本仓库
- ❌ **不要**将 `.onnx`、`.pt` 以外的二进制依赖（DLL、.exe、.whl）提交到本仓库
- ✅ 模型文件（`.onnx`、`.task`）属于项目必需的资源文件，可以提交
- ✅ 建议使用 `.gitignore` 排除 `__pycache__/`、`.vscode/` 等 IDE 和缓存目录
