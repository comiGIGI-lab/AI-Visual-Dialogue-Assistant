# ******************************************************************************
#  Copyright (c) 2024 Orbbec 3D Technology, Inc
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
# ******************************************************************************
#
# ============================================================================
# 人体姿态3D分析系统
# ============================================================================
# 功能：
#   1. 基于YOLOv5检测人体目标
#   2. 基于MediaPipe Pose检测33个人体关键点（2D）
#   3. 利用Orbbec深度相机数据计算3D坐标
#   4. 绘制骨架、深度标注
#   5. 动作识别：举手、弯腰、站立
#
# 前置条件：
#   - pip install mediapipe opencv-python numpy onnxruntime pyorbbecsdk
#   - models/yolov5s.onnx (YOLOv5模型)
#   - coco.names (类别标签文件)
#   - Orbbec 3D相机已连接
# ============================================================================

import cv2
import time
import argparse
import numpy as np
import onnxruntime as ort
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False
from pyorbbecsdk import *

import sys
import os
import collections
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import frame_to_bgr_image


def get_onnx_providers(prefer_gpu=True, verbose=True):
    """
    检测并返回最佳的ONNX Runtime执行提供程序

    优先级: CUDA > TensorRT > DirectML > CPU

    参数：
        prefer_gpu: 是否优先使用GPU
        verbose: 是否打印信息

    返回：
        providers: 执行提供程序列表
    """
    available = ort.get_available_providers()
    if verbose:
        print(f"[信息] ONNX Runtime 可用提供程序: {available}")

    if not prefer_gpu:
        if verbose:
            print("[信息] 手动选择 CPU 推理")
        return ['CPUExecutionProvider']

    # 按优先级选择 GPU 提供程序
    gpu_providers = [
        ('CUDAExecutionProvider', 'CUDA'),
        ('TensorrtExecutionProvider', 'TensorRT'),
        ('DmlExecutionProvider', 'DirectML'),
    ]

    for provider, name in gpu_providers:
        if provider in available:
            if verbose:
                print(f"[信息] 使用 {name} GPU 加速推理")
            return [provider, 'CPUExecutionProvider']

    if verbose:
        print("[信息] 未检测到GPU提供程序，使用 CPU 推理")
        print("[提示] 安装GPU加速: pip install onnxruntime-directml")
    return ['CPUExecutionProvider']

# ========== [1] 全局参数设置 ==========

# 相机分辨率设置 (None = 使用默认配置)
COLOR_CAMERA_WIDTH = None
COLOR_CAMERA_HEIGHT = None
DEPTH_CAMERA_WIDTH = None
DEPTH_CAMERA_HEIGHT = None

ESC_KEY = 27  # ESC键退出

# YOLOv5模型参数（这里改了）
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
SCORE_THRESHOLD = 0.5      # 综合得分阈值（类别得分 x 置信度）
NMS_THRESHOLD = 0.45       # 非极大值抑制阈值（去除重叠框）
CONFIDENCE_THRESHOLD = 0.5 # 置信度阈值
MAX_DISPLAY_BOXES = 5      # 最多显示几个检测框

# 深度滤波参数
MIN_DEPTH = 20             # 最小有效深度 20mm
MAX_DEPTH = 10000          # 最大有效深度 10000mm
DEPTH_SAMPLE_SIZE = 5      # 关键点周围采样区域大小（5x5像素）

# 字体与颜色
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
BLACK, RED, WHITE, GREEN = (0, 0, 0), (0, 0, 255), (255, 255, 255), (0, 255, 0)

# 物体检测框颜色循环
PALETTE = [(255, 255, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0),
           (255, 0, 255), (0, 255, 255), (128, 128, 0),
           (128, 0, 128), (0, 128, 128), (128, 128, 128)]

# ========== [2] MediaPipe 初始化 (Tasks API) ==========
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe import Image, ImageFormat
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[错误] 未检测到 mediapipe，请先安装: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

if MEDIAPIPE_AVAILABLE:
    model_path = 'models/pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print(f"[错误] 未找到 MediaPipe 模型: {model_path}")
        print("       请下载模型到 models/ 目录")
        MEDIAPIPE_AVAILABLE = False
    else:
        base_options = BaseOptions(model_asset_path=model_path)
        pose_landmarker_options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(pose_landmarker_options)
        print("[信息] MediaPipe Pose Landmarker 初始化成功 (Lite + VIDEO 模式)")

# ========== [3] 骨架绘制配置 ==========

# MediaPipe 33个关键点的索引定义
# 我们主要使用以下关键点进行骨架绘制和动作识别：
# 11-左肩, 12-右肩, 13-左肘, 14-右肘, 15-左手腕, 16-右手腕
# 23-左髋, 24-右髋, 25-左膝, 26-右膝, 27-左踝, 28-右踝

# 骨架连接关系（用于绘制连线）
# 格式：(起点索引, 终点索引, 颜色名称)
SKELETON_CONNECTIONS = [
    # 左臂（青色）
    (11, 13, 'left'),
    (13, 15, 'left'),
    # 右臂（紫色）
    (12, 14, 'right'),
    (14, 16, 'right'),
    # 躯干（白色）
    (11, 12, 'torso'),   # 左肩 → 右肩
    (11, 23, 'torso'),   # 左肩 → 左髋
    (12, 24, 'torso'),   # 右肩 → 右髋
    (23, 24, 'torso'),   # 左髋 → 右髋
    # 左腿（青色）
    (23, 25, 'left'),
    (25, 27, 'left'),
    # 右腿（紫色）
    (24, 26, 'right'),
    (26, 28, 'right'),
    # 颈部到面部（可选，简化版可省略）
    (11, 0, 'torso'),    # 左肩 → 鼻子（简化颈部）
    (12, 0, 'torso'),    # 右肩 → 鼻子
    # 脊柱（肩中心 → 髋中心，白色粗线）
    (-1, -1, 'spine'),   # 特殊标记：取左右肩中点 → 左右髋中点
]

# 骨架颜色定义
SKELETON_COLORS = {
    'left':  (0, 255, 255),    # 青色 - 左半身
    'right': (255, 0, 255),    # 紫色 - 右半身
    'torso': (255, 255, 255),  # 白色 - 躯干
    'spine': (255, 255, 255),  # 白色 - 脊柱
}

# Open3D 3D骨架颜色（归一化到 [0,1]）
SKELETON_COLORS_3D = {
    'left':  (0/255, 255/255, 255/255),    # 青色
    'right': (255/255, 0/255, 255/255),    # 紫色
    'torso': (255/255, 255/255, 255/255),  # 白色
    'spine': (255/255, 255/255, 255/255),  # 白色 - 脊柱
}

# 关节点半径
JOINT_RADIUS = 3

_profile_count = 0  # 性能分析帧计数器

# ========== [4] 动作识别参数 ==========

# 举手检测阈值（归一化坐标差）
RAISE_THRESHOLD = 0.05       # 约占总高度的5%
# 弯腰检测阈值
BEND_SPINE_THRESHOLD = 0.15  # 脊柱长度小于此值判定为弯腰
BEND_TILT_THRESHOLD = 0.08   # 肩膀倾斜度阈值
# 蹲下：髋部下降超过脊柱的15%
SQUAT_HIP_DROP_RATIO = 0.15
# 跳跃：肩和髋同时上升 > 脊柱5%，然后同时下降 > 脊柱3%
JUMP_UP_RATIO = 0.05
JUMP_DOWN_RATIO = 0.03
# 抬腿：膝高于髋至少脊柱的10%，或膝髋接近水平（大腿抬高到水平位）
LEG_RAISE_KNEE_UP_RATIO = 0.1
# 膝髋高度差在此范围内也视为抬腿（大腿接近水平）
LEG_RAISE_LEVEL_RATIO = 0.05
# 动作确认帧数
ACTION_CONFIRM_FRAMES = 4

# ========== [5] 辅助函数（保留原有功能） ==========

def draw_label(img, label, x, y, color, extra_line=None):
    """
    在图像上绘制标签文字（支持两行）

    参数：
        img: 目标图像
        label: 主标签文字（如 "person:0.92"）
        x, y: 文字位置
        color: 文字颜色
        extra_line: 第二行文字（如深度信息，可选）
    """
    lines = [label] if extra_line is None else [label, extra_line]
    y_offset = 0
    for text in lines:
        # 获取文字尺寸
        ts, bs = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, THICKNESS)
        w, h = ts
        # 防止超出图像底部
        if y + y_offset + h + bs > img.shape[0]:
            break
        # 画黑色背景矩形（让文字更清晰）
        cv2.rectangle(img, (x, y + y_offset), (x + w, y + y_offset + h + bs), BLACK, cv2.FILLED)
        # 在矩形内写文字
        cv2.putText(img, text, (x, y + y_offset + h), FONT_FACE, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)
        y_offset += h + bs


def pre_process(img):
    """
    预处理图像，适配YOLOv5 ONNX模型输入

    步骤：
        1. 缩放为 640x640
        2. BGR → RGB
        3. 归一化到 [0, 1]
        4. 调整维度为 (1, 3, 640, 640)

    返回：
        blob: 预处理后的张量
    """
    blob = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 640, 640)
    return blob


def filter_depth_outliers(depth_values, threshold=0.2):
    """
    过滤深度离群点（去除噪声）

    原理：取中值，保留中值 ±20% 范围内的数据

    参数：
        depth_values: 一维深度值数组
        threshold: 离群阈值（0.2 = 20%）

    返回：
        过滤后的深度值数组
    """
    if depth_values.size == 0:
        return depth_values
    median = np.median(depth_values)
    lower = median * (1 - threshold)
    upper = median * (1 + threshold)
    return depth_values[(depth_values >= lower) & (depth_values <= upper)]


# ========== [6] MediaPipe 姿态检测函数 ==========

def detect_pose(image, bbox, timestamp_ms=0):
    """
    使用 MediaPipe Pose Landmarker (Tasks API, VIDEO 模式) 检测人体关键点

    参数：
        image: 完整的BGR图像
        bbox: YOLO检测到的人体边界框 (x, y, w, h)
        timestamp_ms: 单调递增的时间戳(ms)，用于 VIDEO 模式帧间追踪

    返回：
        landmarks_2d: 33个关键点的 [(x, y, visibility), ...] 列表（像素坐标）
        如果检测失败返回 None
    """
    if not MEDIAPIPE_AVAILABLE:
        return None

    x, y, w, h = bbox

    # 边界框安全检查（确保不超出图像范围）
    img_h, img_w = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return None

    # 裁剪出人体区域
    person_roi = image[y:y+h, x:x+w]

    # 预缩放到 256x256，加快 MediaPipe 推理
    person_roi = cv2.resize(person_roi, (256, 256))

    # MediaPipe 需要 RGB 格式
    roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

    # 转换为 MediaPipe Image 对象
    mp_image = Image(image_format=ImageFormat.SRGB, data=roi_rgb)

    # VIDEO 模式：带时间戳的帧间追踪
    result = pose_landmarker.detect_for_video(mp_image, int(timestamp_ms))

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    # 获取第一组姿态关键点（num_poses=1）
    landmarks = result.pose_landmarks[0]

    # 将归一化坐标（相对 256×256 ROI）转换为全图像素坐标
    landmarks_2d = []
    for landmark in landmarks:
        # landmark.x, landmark.y 是 [0, 1] 范围的归一化坐标
        global_x = int(landmark.x * w) + x
        global_y = int(landmark.y * h) + y
        visibility = landmark.visibility if landmark.visibility else (landmark.presence if landmark.presence else 1.0)
        landmarks_2d.append((global_x, global_y, visibility))

    return landmarks_2d


# ========== [7] 3D坐标计算函数 ==========

def get_keypoint_depth(depth_data, x, y, kernel_size=5):
    """
    获取关键点周围的稳健深度值

    原理：
        1. 在关键点周围取 kernel_size x kernel_size 的区域
        2. 过滤掉 0 值（无效深度）
        3. 取中值，再过滤离群点，再取中值

    参数：
        depth_data: 深度图（2D numpy数组，单位：mm）
        x, y: 关键点像素坐标
        kernel_size: 采样区域大小（奇数）

    返回：
        深度值（mm），如果无效返回 None
    """
    h, w = depth_data.shape
    half = kernel_size // 2

    # 计算采样区域边界（防止越界）
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)

    # 取出区域
    patch = depth_data[y1:y2, x1:x2]

    # 过滤掉 0 值（无效深度）
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None

    # 第一步：取中值
    median = np.median(valid)

    # 第二步：过滤偏离中值 ±20% 的离群点
    lower = median * 0.8
    upper = median * 1.2
    filtered = valid[(valid >= lower) & (valid <= upper)]

    if len(filtered) == 0:
        return median  # 回退到第一次的中值

    # 第三步：取过滤后的中值（更稳健）
    return np.median(filtered)


def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    """
    将 2D 像素坐标 + 深度值 反投影为 3D 相机坐标

    原理公式：
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

    参数：
        u, v: 像素坐标（关键点在图像中的位置）
        depth: 深度值（单位：mm）
        fx, fy: 相机焦距（x方向和y方向）
        cx, cy: 相机光心（主点，图像中心）

    返回：
        (X, Y, Z): 3D相机坐标（单位：mm）
    """
    if depth is None or depth <= 0:
        return None

    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth

    return (X, Y, Z)


# ========== [8] 3D 坐标计算（独立函数，供动作识别复用） ==========

def compute_3d_landmarks(landmarks_2d, depth_data, fx, fy, cx, cy):
    """从 2D 关键点 + 深度图计算 3D 坐标

    返回: list of (X,Y,Z) or None, length 33
    """
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return [None] * 33

    landmarks_3d = []
    for idx, (px, py, vis) in enumerate(landmarks_2d):
        if vis < 0.5:
            landmarks_3d.append(None)
            continue
        depth = get_keypoint_depth(depth_data, px, py, kernel_size=DEPTH_SAMPLE_SIZE)
        if depth is not None:
            landmarks_3d.append(pixel_to_3d(px, py, depth, fx, fy, cx, cy))
        else:
            landmarks_3d.append(None)
    return landmarks_3d


# ========== [9] 骨架可视化函数 ==========

def draw_skeleton(image, landmarks_2d, landmarks_3d, depth_data, fx, fy, cx, cy, action_text=""):
    """
    在图像上绘制骨架、深度标注和动作标签

    参数：
        image: 目标图像（会被直接修改）
        landmarks_2d: 33个关键点的 [(x, y, visibility), ...] 列表
        landmarks_3d: 33个关键点的3D坐标列表（预先计算）
        depth_data: 深度图（2D数组）
        fx, fy, cx, cy: 相机内参
        action_text: 动作识别结果文字

    返回：
        绘制后的图像
    """
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return image

    # 步骤1：绘制关键点深度标注
    for idx, (px, py, vis) in enumerate(landmarks_2d):
        if vis < 0.5:
            cv2.circle(image, (px, py), JOINT_RADIUS, RED, -1)
            continue

        if landmarks_3d[idx] is not None:
            depth = landmarks_3d[idx][2]  # Z = depth in mm
            if idx in [0, 11, 12, 15, 16, 23, 24, 27, 28]:
                depth_text = f"{depth/1000:.1f}m"
                cv2.circle(image, (px, py), JOINT_RADIUS, GREEN, -1)
                cv2.putText(image, depth_text, (px + 5, py + 10),
                           FONT_FACE, 0.35, GREEN, 1, cv2.LINE_AA)
        else:
            cv2.circle(image, (px, py), JOINT_RADIUS, RED, -1)

    # 步骤2：绘制骨架连线
    for start_idx, end_idx, color_name in SKELETON_CONNECTIONS:
        # 脊柱线：肩中心 → 髋中心（特殊处理）
        if start_idx == -1 and end_idx == -1:
            if (landmarks_2d[11][2] > 0.5 and landmarks_2d[12][2] > 0.5 and
                landmarks_2d[23][2] > 0.5 and landmarks_2d[24][2] > 0.5):
                shoulder_cx = (landmarks_2d[11][0] + landmarks_2d[12][0]) // 2
                shoulder_cy = (landmarks_2d[11][1] + landmarks_2d[12][1]) // 2
                hip_cx = (landmarks_2d[23][0] + landmarks_2d[24][0]) // 2
                hip_cy = (landmarks_2d[23][1] + landmarks_2d[24][1]) // 2
                color = SKELETON_COLORS.get(color_name, WHITE)
                cv2.line(image, (shoulder_cx, shoulder_cy), (hip_cx, hip_cy), color, 3)
            continue

        # 检查两个关键点是否都有效
        if (start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d) and
            landmarks_2d[start_idx][2] > 0.5 and landmarks_2d[end_idx][2] > 0.5):

            x1, y1, _ = landmarks_2d[start_idx]
            x2, y2, _ = landmarks_2d[end_idx]
            color = SKELETON_COLORS.get(color_name, WHITE)

            # 画连线
            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    # 步骤3：在画面顶部显示动作识别结果
    if action_text:
        # 画半透明背景条
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 35), BLACK, cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)

        # 写动作文字
        cv2.putText(image, f"ACTION: {action_text}", (10, 25),
                   FONT_FACE, 0.7, GREEN, 2, cv2.LINE_AA)

    return image


# ========== [10] 动作识别函数 ==========

# 站立参考值（用于蹲下检测，随站立帧缓慢更新）
_standing_ref = {'hip_y': None, 'spine': None, 'frames': 0}
# 跳跃追踪：记录肩髋Y的近期移动
_jump_tracker = collections.deque(maxlen=12)  # 每帧存 (shoulder_center_y, hip_center_y)


def recognize_action(landmarks_2d, action_counter):
    """
    根据2D关键点几何关系识别动作（优先级从高到低）

    支持的动作：
        - "RAISING_LEFT_HAND" / "RAISING_RIGHT_HAND" / "RAISING_BOTH_HANDS"
        - "JUMPING"
        - "RAISING_LEFT_LEG" / "RAISING_RIGHT_LEG"
        - "BENDING"
        - "SQUATTING"
        - "STANDING"（默认）
    """
    global _standing_ref, _jump_tracker

    if landmarks_2d is None or len(landmarks_2d) < 33:
        return "STANDING", action_counter

    # 提取关键点
    ls = landmarks_2d[11]; rs = landmarks_2d[12]   # 肩
    le = landmarks_2d[13]; re = landmarks_2d[14]   # 肘
    lw = landmarks_2d[15]; rw = landmarks_2d[16]   # 腕
    lh = landmarks_2d[23]; rh = landmarks_2d[24]   # 髋
    lk = landmarks_2d[25]; rk = landmarks_2d[26]   # 膝
    la = landmarks_2d[27]; ra = landmarks_2d[28]   # 踝

    # 核心点可见度检查
    if (ls[2] < 0.5 or rs[2] < 0.5 or
        lw[2] < 0.5 or rw[2] < 0.5):
        return "STANDING", action_counter

    shoulder_center_y = (ls[1] + rs[1]) / 2
    hip_center_y = (lh[1] + rh[1]) / 2
    spine_2d = abs(hip_center_y - shoulder_center_y)
    adaptive_thr = spine_2d * 0.15

    # ========== [1] 举手检测 ==========
    lw_diff = ls[1] - lw[1]; rw_diff = rs[1] - rw[1]
    le_diff = ls[1] - le[1]; re_diff = rs[1] - re[1]

    left_raised = (lw_diff > 24 or lw_diff > adaptive_thr or
                   le_diff > 24 or le_diff > adaptive_thr)
    right_raised = (rw_diff > 24 or rw_diff > adaptive_thr or
                    re_diff > 24 or re_diff > adaptive_thr)

    if left_raised and right_raised:
        detected_action = "RAISING_BOTH_HANDS"
    elif left_raised:
        detected_action = "RAISING_LEFT_HAND"
    elif right_raised:
        detected_action = "RAISING_RIGHT_HAND"
    else:
        detected_action = "STANDING"

    # ========== [2] 跳跃检测 ==========
    if detected_action == "STANDING":
        _jump_tracker.append((shoulder_center_y, hip_center_y))
        if len(_jump_tracker) >= 8:
            sh_vals = [s for s, _ in _jump_tracker]
            hip_vals = [h for _, h in _jump_tracker]

            # 基线 = 前3帧（跳跃前的站立位置）
            early_sh = sum(sh_vals[:3]) / 3
            early_hip = sum(hip_vals[:3]) / 3

            up_thr = spine_2d * JUMP_UP_RATIO
            down_thr = spine_2d * JUMP_DOWN_RATIO

            min_sh = min(sh_vals)
            min_hip = min(hip_vals)
            min_idx_sh = sh_vals.index(min_sh)
            min_idx_hip = hip_vals.index(min_hip)

            # 上升：最低点显著高于早期基线（Y变小 = 身体上移）
            went_up = (early_sh - min_sh > up_thr and early_hip - min_hip > up_thr)

            # 下落：最低点之后有明显回升（Y变大 = 身体下落）
            after_sh = sh_vals[min_idx_sh:] if min_idx_sh < len(sh_vals) - 1 else [min_sh]
            after_hip = hip_vals[min_idx_hip:] if min_idx_hip < len(hip_vals) - 1 else [min_hip]
            came_down = (max(after_sh) - min_sh > down_thr and
                         max(after_hip) - min_hip > down_thr)

            if went_up and came_down:
                detected_action = "JUMPING"
                _jump_tracker.clear()

    # ========== [3] 抬腿检测 ==========
    if detected_action == "STANDING":
        raise_thr = spine_2d * LEG_RAISE_KNEE_UP_RATIO
        level_thr = spine_2d * LEG_RAISE_LEVEL_RATIO
        left_diff = 0.0
        right_diff = 0.0
        if lh[2] > 0.5 and lk[2] > 0.5:
            left_diff = lh[1] - lk[1]   # 正=膝高于髋, 负=膝低于髋
        if rh[2] > 0.5 and rk[2] > 0.5:
            right_diff = rh[1] - rk[1]

        # 膝高于髋 或 膝髋接近水平（大腿抬高到水平位）
        left_leg = (left_diff > raise_thr) or (abs(left_diff) < level_thr and lh[2] > 0.5 and lk[2] > 0.5)
        right_leg = (right_diff > raise_thr) or (abs(right_diff) < level_thr and rh[2] > 0.5 and rk[2] > 0.5)

        if left_leg and right_leg:
            # 双侧触发时取差值更大的一侧（防止支撑腿代偿误判）
            if left_diff > right_diff:
                detected_action = "RAISING_LEFT_LEG"
            else:
                detected_action = "RAISING_RIGHT_LEG"
        elif left_leg:
            detected_action = "RAISING_LEFT_LEG"
        elif right_leg:
            detected_action = "RAISING_RIGHT_LEG"

    # ========== [4] 弯腰检测 ==========
    if detected_action == "STANDING":
        shoulder_tilt = abs(ls[1] - rs[1])
        if lh[2] > 0.5 and rh[2] > 0.5:
            if spine_2d < BEND_SPINE_THRESHOLD * 480 or shoulder_tilt > BEND_TILT_THRESHOLD * 480:
                detected_action = "BENDING"

    # ========== [5] 蹲下检测 ==========
    if detected_action == "STANDING" and lh[2] > 0.5 and rh[2] > 0.5:
        # 缓慢更新站立参考值
        if _standing_ref['hip_y'] is None:
            _standing_ref['hip_y'] = hip_center_y
            _standing_ref['spine'] = spine_2d
            _standing_ref['frames'] = 1
        else:
            alpha = 0.03
            _standing_ref['hip_y'] = _standing_ref['hip_y'] * (1 - alpha) + hip_center_y * alpha
            _standing_ref['spine'] = _standing_ref['spine'] * (1 - alpha) + spine_2d * alpha
            _standing_ref['frames'] += 1

        if _standing_ref['frames'] > 10:
            ref_hip = _standing_ref['hip_y']
            ref_spine = _standing_ref['spine']
            # 髋部下降超过参考脊柱的 SQUAT_HIP_DROP_RATIO
            if (hip_center_y - ref_hip) > ref_spine * SQUAT_HIP_DROP_RATIO:
                detected_action = "SQUATTING"

    # ========== 动作防抖 ==========
    action_counter[detected_action] = action_counter.get(detected_action, 0) + 1
    for k in list(action_counter):
        if k != detected_action:
            action_counter[k] = 0

    best_action = max(action_counter, key=action_counter.get)
    best_count = action_counter[best_action]

    # 抬腿2帧确认，其他动作4帧防抖
    if best_action in ('RAISING_LEFT_LEG', 'RAISING_RIGHT_LEG'):
        confirm_needed = 2
    else:
        confirm_needed = ACTION_CONFIRM_FRAMES

    if best_count >= confirm_needed:
        return best_action, action_counter
    else:
        return "STANDING", action_counter


# ========== [10] YOLOv26 后处理函数 ==========

def update_3d_skeleton(vis, landmarks_3d, connections, colors):
    """
    更新 Open3D 窗口中的 3D 骨架

    参数：
        vis: Open3D Visualizer 对象
        landmarks_3d: 33个 (X,Y,Z) 或 None 的列表
        connections: 骨架连线定义 [(start, end, color_name), ...]
        colors: 颜色字典 {'left': (R,G,B), ...}
    """
    vis.clear_geometries()

    # 画关节点（相机→Open3D：X左右镜像、Y上下翻转、Z前后镜像）
    for idx, point in enumerate(landmarks_3d):
        if point is None:
            continue
        x, y, z = point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=25)
        sphere.translate((-x, -y, -z))
        sphere.paint_uniform_color([0, 1, 0])  # 绿色
        vis.add_geometry(sphere)

    # 画骨架连线
    for start_idx, end_idx, color_name in connections:
        # 脊柱线：肩中心 → 髋中心（特殊处理）
        if start_idx == -1 and end_idx == -1:
            if (landmarks_3d[11] is not None and landmarks_3d[12] is not None
                    and landmarks_3d[23] is not None and landmarks_3d[24] is not None):
                s_cx = (landmarks_3d[11][0] + landmarks_3d[12][0]) / 2
                s_cy = (landmarks_3d[11][1] + landmarks_3d[12][1]) / 2
                s_cz = (landmarks_3d[11][2] + landmarks_3d[12][2]) / 2
                h_cx = (landmarks_3d[23][0] + landmarks_3d[24][0]) / 2
                h_cy = (landmarks_3d[23][1] + landmarks_3d[24][1]) / 2
                h_cz = (landmarks_3d[23][2] + landmarks_3d[24][2]) / 2
                p1 = (-s_cx, -s_cy, -s_cz)
                p2 = (-h_cx, -h_cy, -h_cz)
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([p1, p2])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                rgb = colors.get(color_name, (1, 1, 1))
                line.paint_uniform_color(rgb)
                vis.add_geometry(line)
            continue

        if (start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d)
                and landmarks_3d[start_idx] is not None
                and landmarks_3d[end_idx] is not None):
            x1, y1, z1 = landmarks_3d[start_idx]
            x2, y2, z2 = landmarks_3d[end_idx]
            p1 = (-x1, -y1, -z1)
            p2 = (-x2, -y2, -z2)
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([p1, p2])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            rgb = colors.get(color_name, (1, 1, 1))
            line.paint_uniform_color(rgb)
            vis.add_geometry(line)

    vis.poll_events()
    vis.update_renderer()


def post_process_with_pose(img, depth_frame, outs, fx, fy, cx, cy, action_counter, class_names=None, vis_3d=None):
    """
    YOLOv26 版本后处理（输出 shape: (1,300,6)，已内置 NMS）

    参数：
        img: 输入图像（BGR）
        depth_frame: Orbbec深度帧
        outs: YOLO模型输出
        fx, fy, cx, cy: 相机内参
        action_counter: 动作计数器

    返回：
        绘制后的图像
        action_counter: 更新后的动作计数器
    """
    # ===== YOLOv26 输出解析 =====
    predictions = np.squeeze(outs[0])  # shape: (300, 6)
    boxes, confidences, class_ids = [], [], []
    img_h, img_w = img.shape[:2]

    # 解析深度数据（与之前完全相同）
    try:
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
    except ValueError:
        print("[警告] 深度数据解析失败")
        return img, action_counter

    depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
    depth_data = depth_data.astype(np.uint16)

    # 遍历 v26 的每个检测结果（已做完 NMS，只需过滤）
    for det in predictions:
        x1, y1, x2, y2, conf, cls_id = det[:6]   # 取出 6 个值
        # 过滤低置信度和非人体目标（0 = person）
        if conf < CONFIDENCE_THRESHOLD or int(cls_id) != 0:
            continue

        # 将坐标映射回原图尺寸（v26 输出的是相对于 640x640 的坐标）
        left = int(x1 * img_w / INPUT_WIDTH)
        top = int(y1 * img_h / INPUT_HEIGHT)
        right = int(x2 * img_w / INPUT_WIDTH)
        bottom = int(y2 * img_h / INPUT_HEIGHT)

        width = right - left
        height = bottom - top
        boxes.append([left, top, width, height])
        confidences.append(float(conf))
        class_ids.append(int(cls_id))

    # 如果没有任何 person 检测，直接返回原图
    if len(boxes) == 0:
        return img, action_counter

    # ===== 姿态检测 + 3D坐标 + 动作识别（与原来完全一致） =====
    result_img = img.copy()

    # 获取所有 person 的索引
    person_indices = [i for i, cls in enumerate(class_ids) if cls == 0]

    if person_indices and MEDIAPIPE_AVAILABLE:
        # 选择面积最大的人体框
        largest_idx = max(person_indices, key=lambda i: boxes[i][2] * boxes[i][3])
        left, top, width, height = boxes[largest_idx]

        # 稍微扩大边界框，给 MediaPipe 更多上下文
        margin = int(0.1 * max(width, height))
        bbox = (left - margin, top - margin, width + 2*margin, height + 2*margin)

        # 运行 MediaPipe 姿态检测
        t_mp = time.time()
        landmarks_2d = detect_pose(img, bbox, int(time.time() * 1000))
        t_mp = (time.time() - t_mp) * 1000

        if landmarks_2d:
            # 先算 3D 坐标
            landmarks_3d = compute_3d_landmarks(landmarks_2d, depth_data, fx, fy, cx, cy)
            # 再用 2D 坐标做动作识别
            action_text, action_counter = recognize_action(landmarks_2d, action_counter)
            # 最后画骨架
            result_img = draw_skeleton(
                result_img, landmarks_2d, landmarks_3d, depth_data,
                fx, fy, cx, cy, action_text
            )

            # 更新 Open3D 3D 骨架窗口
            if vis_3d is not None and landmarks_3d:
                t_o3d = time.time()
                update_3d_skeleton(vis_3d, landmarks_3d,
                                   SKELETON_CONNECTIONS, SKELETON_COLORS_3D)
                t_o3d = (time.time() - t_o3d) * 1000
            else:
                t_o3d = 0

            # 每 30 帧打印一次详细耗时
            global _profile_count
            _profile_count += 1
            if _profile_count % 30 == 0:
                print(f"[计时] MP:{t_mp:.0f}ms  O3D:{t_o3d:.0f}ms  "
                      f"Total:{t_mp + t_o3d:.0f}ms")

    # ===== 只绘制被追踪的那个 person 框（面积最大的人） =====
    if person_indices and MEDIAPIPE_AVAILABLE:
        left, top, width, height = boxes[largest_idx]
        right = min(left + width, depth_data.shape[1])
        bottom = min(top + height, depth_data.shape[0])

        depth_roi = depth_data[top:bottom, left:right]
        depth_values = depth_roi.flatten()
        valid_depths = depth_values[depth_values > 0]
        filtered_depths = filter_depth_outliers(valid_depths)

        if filtered_depths.size > 0:
            depth_at_center = int(np.median(filtered_depths))
            depth_label = f"depth:{depth_at_center}mm"
        else:
            depth_label = "depth:N/A"

        box_color = GREEN
        cv2.rectangle(result_img, (left, top), (left + width, top + height), box_color, 2)

        label = f"person:{confidences[largest_idx]:.2f}"
        draw_label(result_img, label, left + 2, top + height - 5, box_color, depth_label)

    return result_img, action_counter

# ========== [11] 相机配置（保留原有功能） ==========

def get_sw_align_config(pipeline, color_req_width=None, color_req_height=None,
                        depth_req_width=None, depth_req_height=None):
    """
    配置Orbbec相机的软件对齐（D2C）

    参数：
        pipeline: Orbbec管道对象
        color_req_width, color_req_height: 请求的颜色相机分辨率
        depth_req_width, depth_req_height: 请求的深度相机分辨率

    返回：
        Config对象，或None（失败时）
    """
    cw = color_req_width if color_req_width is not None else COLOR_CAMERA_WIDTH
    ch = color_req_height if color_req_height is not None else COLOR_CAMERA_HEIGHT
    dw = depth_req_width if depth_req_width is not None else DEPTH_CAMERA_WIDTH
    dh = depth_req_height if depth_req_height is not None else DEPTH_CAMERA_HEIGHT

    config = Config()
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        # 选择颜色流配置
        color_profile = None
        if cw and ch:
            for cp in color_profiles:
                if cp.get_format() == OBFormat.RGB and cp.get_width() == cw and cp.get_height() == ch:
                    color_profile = cp
                    print(f"[配置] 使用指定彩色分辨率: {cw}x{ch}")
                    break
            if color_profile is None:
                print(f"[配置] 未找到 {cw}x{ch} 彩色配置，使用默认")
        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"[配置] 默认彩色配置: {color_profile.get_width()}x{color_profile.get_height()}")
        config.enable_stream(color_profile)

        # 选择深度流配置
        depth_profile = None
        if dw and dh:
            for dp in depth_profiles:
                if dp.get_width() == dw and dp.get_height() == dh:
                    depth_profile = dp
                    print(f"[配置] 使用指定深度分辨率: {dw}x{dh}")
                    break
            if depth_profile is None:
                print(f"[配置] 未找到 {dw}x{dh} 深度配置，使用默认")
        if depth_profile is None:
            depth_profile = depth_profiles.get_default_video_stream_profile()
            print(f"[配置] 默认深度配置: {depth_profile.get_width()}x{depth_profile.get_height()}")
        config.enable_stream(depth_profile)

    except Exception as e:
        print(f"[错误] 相机配置失败: {e}")
        return None

    return config


# ========== [12] 主程序入口 ==========

if __name__ == '__main__':
    # ===== 解析命令行参数 =====
    parser = argparse.ArgumentParser(description='人体姿态3D分析系统')
    parser.add_argument('--color_width', type=int, default=None, help="彩色相机宽度")
    parser.add_argument('--color_height', type=int, default=None, help="彩色相机高度")
    parser.add_argument('--depth_width', type=int, default=None, help="深度相机宽度")
    parser.add_argument('--depth_height', type=int, default=None, help="深度相机高度")
    parser.add_argument('--no-pose', action='store_true', help="禁用MediaPipe姿态检测")
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'gpu', 'cpu'],
                        help="推理设备选择: auto(自动检测) / gpu(强制GPU) / cpu(强制CPU)")
    args = parser.parse_args()

    # ===== 加载YOLO类别标签 =====
    try:
        with open('coco.names', 'rt') as f:
            classes = f.read().strip().split('\n')
        print(f"[信息] 加载了 {len(classes)} 个类别")
    except FileNotFoundError:
        print("[错误] 未找到 coco.names 文件，请确保文件在当前目录")
        classes = []

    # ===== 初始化ONNX Runtime（YOLO） =====
    try:
        prefer_gpu = args.device != 'cpu'
        providers = get_onnx_providers(prefer_gpu=prefer_gpu)
        sess_options = ort.SessionOptions()
        # GPU 优化：启用图优化和内存限制
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession('models/yolo26n.onnx',
                                            sess_options=sess_options,
                                            providers=providers)
        input_name = ort_session.get_inputs()[0].name
        actual_provider = ort_session.get_providers()[0]
        print(f"[信息] YOLO26n ONNX 模型加载成功 (设备: {actual_provider})")
    except Exception as e:
        print(f"[错误] YOLO模型加载失败: {e}")
        print("[提示] 如GPU加载失败，尝试: pip install onnxruntime-directml 或使用 --device cpu")
        exit(1)

    # ===== 检查MediaPipe =====
    if args.no_pose:
        MEDIAPIPE_AVAILABLE = False
        print("[信息] 已禁用MediaPipe姿态检测（--no-pose）")
    elif not MEDIAPIPE_AVAILABLE:
        print("[警告] MediaPipe未安装，姿态检测功能不可用")
        print("       安装命令: pip install mediapipe")

    # ===== 启动Orbbec相机 =====
    pipeline = Pipeline()
    config = get_sw_align_config(pipeline, args.color_width, args.color_height,
                                  args.depth_width, args.depth_height)
    if config is None:
        print("[错误] 未找到合适的流配置")
        exit(1)

    pipeline.start(config)
    print("[信息] 相机已启动")

    # 创建D2C对齐过滤器（深度对齐到彩色图）
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    # ===== 获取相机内参（用于3D坐标计算） =====
    camera_param = pipeline.get_camera_param()
    fx = camera_param.rgb_intrinsic.fx
    fy = camera_param.rgb_intrinsic.fy
    cx = camera_param.rgb_intrinsic.cx
    cy = camera_param.rgb_intrinsic.cy
    print(f"[信息] 相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # ===== Open3D 3D 骨架窗口（按 J 切换，默认关闭） =====
    vis_3d = None
    _o3d_enabled = False

    def create_o3d_window():
        """创建 Open3D 3D 骨架窗口"""
        v = o3d.visualization.Visualizer()
        v.create_window(window_name="3D Skeleton", width=800, height=600)
        vc = v.get_view_control()
        vc.set_front([0, 0, -1])
        vc.set_up([0, 1, 0])
        ro = v.get_render_option()
        ro.line_width = 5.0
        ro.point_size = 8.0
        return v

    # ===== 初始化变量 =====
    prev_time = time.time()
    action_counter = {}  # 动作计数器（用于防抖）
    frame_count = 0      # 帧计数

    print("\n" + "="*50)
    print("人体姿态3D分析系统已启动")
    print("按 ESC 或 Q 退出  |  按 J 切换 3D 骨架")
    print("="*50 + "\n")

    # 全屏窗口
    cv2.namedWindow('Human Pose 3D Analysis', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Human Pose 3D Analysis', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # ===== 主循环 =====
    while True:
        # ---- 获取帧 ----
        frames = pipeline.wait_for_frames(1000)
        if not frames:
            continue

        frames = align_filter.process(frames)
        if not frames:
            continue
        frames = frames.as_frame_set()

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        # ---- 彩色图转OpenCV格式 ----
        img_bgr = frame_to_bgr_image(color_frame)
        if img_bgr is None:
            continue

        # ---- YOLO推理 ----
        t_pre0 = time.time()
        input_tensor = pre_process(img_bgr)
        t_pre = (time.time() - t_pre0) * 1000
        t0 = time.time()
        outputs = ort_session.run(None, {input_name: input_tensor})
        t_yolo = (time.time() - t0) * 1000

        # ---- 综合后处理（YOLO + 姿态 + 3D + 动作识别）----
        t1 = time.time()
        result, action_counter = post_process_with_pose(
            img_bgr, depth_frame, outputs,
            fx, fy, cx, cy, action_counter,
            vis_3d=vis_3d
        )
        t_post = (time.time() - t1) * 1000

        # ---- 计算帧率 ----
        curr_time = time.time()
        frame_time_ms = (curr_time - prev_time) * 1000
        fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0
        prev_time = curr_time
        frame_count += 1

        # ---- 显示信息 右上角 ----
        img_w = result.shape[1]
        cv2.putText(result, f"FPS:{fps:.1f}", (img_w - 160, 20),
                   FONT_FACE, 0.5, RED, 1, cv2.LINE_AA)
        cv2.putText(result, f"Pre:{t_pre:.0f}ms YOLO:{t_yolo:.0f}ms", (img_w - 190, 40),
                   FONT_FACE, 0.45, RED, 1, cv2.LINE_AA)
        cv2.putText(result, f"Post:{t_post:.0f}ms", (img_w - 190, 60),
                   FONT_FACE, 0.5, RED, 1, cv2.LINE_AA)
        o3d_status = "ON" if _o3d_enabled else "OFF"
        cv2.putText(result, f"O3D:{o3d_status} (J)", (img_w - 160, 80),
                   FONT_FACE, 0.4, GREEN if _o3d_enabled else (100, 100, 100), 1, cv2.LINE_AA)

        # ---- 显示结果 ----
        cv2.imshow('Human Pose 3D Analysis', result)

        # 按键处理
        key = cv2.waitKey(1)
        if key in (ESC_KEY, ord('q'), ord('Q')):
            print("\n[信息] 用户退出")
            break
        elif key in (ord('j'), ord('J')):
            if not O3D_AVAILABLE:
                print("[信息] Open3D 未安装，无法开启3D骨架")
            elif _o3d_enabled:
                vis_3d.destroy_window()
                vis_3d = None
                _o3d_enabled = False
                print("[信息] Open3D 3D 骨架已关闭")
            else:
                vis_3d = create_o3d_window()
                _o3d_enabled = True
                print("[信息] Open3D 3D 骨架已开启")

    # ===== 清理 =====
    if vis_3d is not None:
        vis_3d.destroy_window()
    cv2.destroyAllWindows()
    pipeline.stop()

    # 释放MediaPipe资源
    if MEDIAPIPE_AVAILABLE:
        pose_landmarker.close()

    print("[信息] 程序已安全退出")
