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
# 人体活体检测系统（基于YOLOv26 + 深度信息）
# ============================================================================
# 功能：
#   1. 基于YOLOv26检测人体目标
#   2. 基于MediaPipe Pose检测33个人体关键点（2D）
#   3. 利用Orbbec深度相机D2C对齐后的深度数据计算3D坐标
#   4. 绘制骨架、深度标注、动作识别
#   5. 【新增】基于深度信息判断真人 vs 照片/纸片（活体检测）
#
# 活体检测原理：
#   - 真人是3D物体：身体有前后凹凸（鼻子凸出、手臂伸展、躯干弯曲）
#     → 深度值分布范围大，标准差大
#   - 照片/纸片是平面：表面各点到相机的距离几乎相同
#     → 深度值非常集中，范围小，标准差小
#
# 前置条件：
#   - pip install mediapipe opencv-python numpy onnxruntime pyorbbecsdk
#   - models/yolo26n.onnx
#   - coco.names
#   - Orbbec 3D相机已连接
# ============================================================================

import cv2
import time
import argparse
import numpy as np
import onnxruntime as ort
from pyorbbecsdk import *

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import frame_to_bgr_image

# ========== [1] 全局参数设置 ==========

# 相机分辨率设置 (None = 使用默认配置)
COLOR_CAMERA_WIDTH = None
COLOR_CAMERA_HEIGHT = None
DEPTH_CAMERA_WIDTH = None
DEPTH_CAMERA_HEIGHT = None

ESC_KEY = 27  # ESC键退出

# YOLOv26模型参数
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5
MAX_DISPLAY_BOXES = 5

# 深度滤波参数
MIN_DEPTH = 20             # 最小有效深度 20mm
MAX_DEPTH = 10000          # 最大有效深度 10000mm
DEPTH_SAMPLE_SIZE = 5      # 关键点周围采样区域大小（5x5像素）

# ========== [1.1] 活体检测参数 ==========
# 深度范围阈值：真人人体框内 最大深度 - 最小深度 通常 > 此值（单位mm）
SPOOF_DEPTH_RANGE_THRESHOLD = 35
# 深度标准差阈值：真人人体框内深度的标准差通常 > 此值（单位mm）
SPOOF_DEPTH_STD_THRESHOLD = 12
# 有效深度像素占比阈值：照片可能反射率低导致大量无效值，真人通常占比高
SPOOF_VALID_RATIO_THRESHOLD = 0.3
# 活体检测结果平滑帧数（连续N帧结果一致才翻转状态，防止抖动）
LIVENESS_SMOOTH_FRAMES = 5

# 字体与颜色
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
BLACK, RED, WHITE, GREEN = (0, 0, 0), (0, 0, 255), (255, 255, 255), (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)

# 物体检测框颜色循环
PALETTE = [(255, 255, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0),
           (255, 0, 255), (0, 255, 255), (128, 128, 0),
           (128, 0, 128), (0, 128, 128), (128, 128, 128)]

# ========== [2] MediaPipe 初始化 ==========
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[错误] 未检测到 mediapipe，请先安装: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

if MEDIAPIPE_AVAILABLE:
    mp_pose = mp.solutions.pose
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    pose_detector = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    print("[信息] MediaPipe Pose 初始化成功")

# ========== [3] 骨架绘制配置 ==========

SKELETON_CONNECTIONS = [
    (11, 13, 'left'),
    (13, 15, 'left'),
    (12, 14, 'right'),
    (14, 16, 'right'),
    (11, 12, 'torso'),
    (11, 23, 'torso'),
    (12, 24, 'torso'),
    (23, 24, 'torso'),
    (23, 25, 'left'),
    (25, 27, 'left'),
    (24, 26, 'right'),
    (26, 28, 'right'),
    (11, 0, 'torso'),
    (12, 0, 'torso'),
]

SKELETON_COLORS = {
    'left':  (0, 255, 255),
    'right': (255, 0, 255),
    'torso': (255, 255, 255),
}

JOINT_RADIUS = 3

# ========== [4] 动作识别参数 ==========

RAISE_THRESHOLD = 0.05
BEND_SPINE_THRESHOLD = 0.15
BEND_TILT_THRESHOLD = 0.08
ACTION_CONFIRM_FRAMES = 3

# ========== [5] 辅助函数 ==========

def draw_label(img, label, x, y, color, extra_line=None):
    lines = [label] if extra_line is None else [label, extra_line]
    y_offset = 0
    for text in lines:
        ts, bs = cv2.getTextSize(text, FONT_FACE, FONT_SCALE, THICKNESS)
        w, h = ts
        if y + y_offset + h + bs > img.shape[0]:
            break
        cv2.rectangle(img, (x, y + y_offset), (x + w, y + y_offset + h + bs), BLACK, cv2.FILLED)
        cv2.putText(img, text, (x, y + y_offset + h), FONT_FACE, FONT_SCALE, color, THICKNESS, cv2.LINE_AA)
        y_offset += h + bs


def pre_process(img):
    blob = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob


def filter_depth_outliers(depth_values, threshold=0.2):
    if depth_values.size == 0:
        return depth_values
    median = np.median(depth_values)
    lower = median * (1 - threshold)
    upper = median * (1 + threshold)
    return depth_values[(depth_values >= lower) & (depth_values <= upper)]


# ========== [6] 活体检测核心函数 ==========

def analyze_depth_liveness(depth_roi):
    """
    分析人体框内的深度分布，判断是真人还是照片/纸片。

    原理：
        真人是3D物体：
          - 鼻子比脸颊凸出约 20-40mm
          - 手臂比躯干前后差约 50-150mm
          - 肩膀到胸腔有自然弧度
          → 深度范围通常 > 50mm，标准差 > 15mm

        照片/平板是2D平面：
          - 所有像素到相机距离几乎相同
          - 即便有轻微弯曲，深度变化 < 10mm
          → 深度范围 < 30mm，标准差 < 10mm

    参数：
        depth_roi: 人体框内的深度图（2D numpy数组，单位mm，已过滤无效值）

    返回：
        dict: {
            'is_real': bool,           # True=真人, False=照片
            'confidence': float,       # 判断置信度 0~1
            'depth_range': float,      # 深度范围 mm
            'depth_std': float,        # 深度标准差 mm
            'valid_ratio': float,      # 有效深度像素占比
            'reason': str,             # 判断依据文字说明
        }
    """
    total_pixels = depth_roi.size
    if total_pixels == 0:
        return {
            'is_real': False,
            'confidence': 0.0,
            'depth_range': 0,
            'depth_std': 0,
            'valid_ratio': 0,
            'reason': 'no depth data',
        }

    # 提取有效深度值（> 0）
    valid_depths = depth_roi[depth_roi > 0].astype(np.float32)
    valid_ratio = valid_depths.size / total_pixels

    if valid_depths.size < 20:
        return {
            'is_real': False,
            'confidence': 0.3,
            'depth_range': 0,
            'depth_std': 0,
            'valid_ratio': valid_ratio,
            'reason': 'too few valid pixels',
        }

    # 过滤离群点后再统计
    filtered = filter_depth_outliers(valid_depths, threshold=0.2)
    if filtered.size < 10:
        filtered = valid_depths  # 回退

    depth_range = float(np.max(filtered) - np.min(filtered))
    depth_std = float(np.std(filtered))
    depth_median = float(np.median(filtered))

    # ---- 综合判断 ----
    # 计分制：每个维度独立打分，加权汇总
    score = 0.0
    reasons = []

    # 维度1：深度范围（权重 0.4）
    if depth_range >= SPOOF_DEPTH_RANGE_THRESHOLD * 2:
        score += 0.4
        reasons.append(f'range={depth_range:.0f}mm(>={SPOOF_DEPTH_RANGE_THRESHOLD*2})')
    elif depth_range >= SPOOF_DEPTH_RANGE_THRESHOLD:
        ratio = (depth_range - SPOOF_DEPTH_RANGE_THRESHOLD) / SPOOF_DEPTH_RANGE_THRESHOLD
        score += 0.2 + 0.2 * ratio
        reasons.append(f'range={depth_range:.0f}mm(moderate)')
    else:
        reasons.append(f'range={depth_range:.0f}mm(flat)')

    # 维度2：深度标准差（权重 0.4）
    if depth_std >= SPOOF_DEPTH_STD_THRESHOLD * 2:
        score += 0.4
        reasons.append(f'std={depth_std:.0f}mm(high)')
    elif depth_std >= SPOOF_DEPTH_STD_THRESHOLD:
        ratio = (depth_std - SPOOF_DEPTH_STD_THRESHOLD) / SPOOF_DEPTH_STD_THRESHOLD
        score += 0.2 + 0.2 * ratio
        reasons.append(f'std={depth_std:.0f}mm(moderate)')
    else:
        reasons.append(f'std={depth_std:.0f}mm(low)')

    # 维度3：有效像素占比（权重 0.2）
    # 真人通常有较高的有效深度占比；照片如果表面反光可能导致大量无效值
    if valid_ratio >= SPOOF_VALID_RATIO_THRESHOLD:
        score += 0.2
    else:
        # 低有效率可能是照片反光，但也可能是距离太远，给低分
        score += 0.2 * (valid_ratio / SPOOF_VALID_RATIO_THRESHOLD)
        reasons.append(f'valid={valid_ratio:.0%}(low)')

    is_real = score >= 0.5
    confidence = min(1.0, score / 0.8) if is_real else min(1.0, (1.0 - score) / 0.5)

    return {
        'is_real': is_real,
        'confidence': confidence,
        'depth_range': depth_range,
        'depth_std': depth_std,
        'valid_ratio': valid_ratio,
        'reason': ' | '.join(reasons),
    }


def smooth_liveness(current_is_real, history):
    """
    活体检测结果时间域平滑，防止逐帧抖动。

    参数：
        current_is_real: 当前帧的判断结果 bool
        history: 历史结果队列 list[bool]

    返回：
        smoothed_is_real: 平滑后的结果 bool
        history: 更新后的历史队列
    """
    history.append(current_is_real)
    if len(history) > LIVENESS_SMOOTH_FRAMES:
        history.pop(0)

    true_count = sum(history)
    # 多数投票：超过半数帧为真才判定为真人
    smoothed_is_real = true_count > len(history) / 2
    return smoothed_is_real, history


# ========== [7] MediaPipe 姿态检测函数 ==========

def detect_pose(image, bbox):
    if not MEDIAPIPE_AVAILABLE:
        return None

    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    x = max(0, x)
    y = max(0, y)
    w = min(w, img_w - x)
    h = min(h, img_h - y)

    if w <= 0 or h <= 0:
        return None

    person_roi = image[y:y+h, x:x+w]
    roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
    results = pose_detector.process(roi_rgb)

    if not results.pose_landmarks:
        return None

    landmarks_2d = []
    for landmark in results.pose_landmarks.landmark:
        global_x = int(landmark.x * w) + x
        global_y = int(landmark.y * h) + y
        visibility = landmark.visibility
        landmarks_2d.append((global_x, global_y, visibility))

    return landmarks_2d


# ========== [8] 3D坐标计算函数 ==========

def get_keypoint_depth(depth_data, x, y, kernel_size=5):
    h, w = depth_data.shape
    half = kernel_size // 2
    y1 = max(0, y - half)
    y2 = min(h, y + half + 1)
    x1 = max(0, x - half)
    x2 = min(w, x + half + 1)

    patch = depth_data[y1:y2, x1:x2]
    valid = patch[patch > 0]
    if len(valid) == 0:
        return None

    median = np.median(valid)
    lower = median * 0.8
    upper = median * 1.2
    filtered = valid[(valid >= lower) & (valid <= upper)]

    if len(filtered) == 0:
        return median
    return np.median(filtered)


def pixel_to_3d(u, v, depth, fx, fy, cx, cy):
    if depth is None or depth <= 0:
        return None
    X = (u - cx) * depth / fx
    Y = (v - cy) * depth / fy
    Z = depth
    return (X, Y, Z)


# ========== [9] 骨架可视化函数 ==========

def draw_skeleton(image, landmarks_2d, depth_data, fx, fy, cx, cy, action_text=""):
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return image, None

    landmarks_3d = []

    for idx, (px, py, vis) in enumerate(landmarks_2d):
        if vis < 0.5:
            landmarks_3d.append(None)
            continue

        depth = get_keypoint_depth(depth_data, px, py, kernel_size=DEPTH_SAMPLE_SIZE)

        if depth is not None:
            point_3d = pixel_to_3d(px, py, depth, fx, fy, cx, cy)
            landmarks_3d.append(point_3d)

            if idx in [0, 11, 12, 15, 16, 23, 24, 27, 28]:
                depth_text = f"{depth/1000:.1f}m"
                cv2.circle(image, (px, py), JOINT_RADIUS, GREEN, -1)
                cv2.putText(image, depth_text, (px + 5, py + 10),
                           FONT_FACE, 0.35, GREEN, 1, cv2.LINE_AA)
        else:
            landmarks_3d.append(None)
            cv2.circle(image, (px, py), JOINT_RADIUS, RED, -1)

    for start_idx, end_idx, color_name in SKELETON_CONNECTIONS:
        if (start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d) and
            landmarks_2d[start_idx][2] > 0.5 and landmarks_2d[end_idx][2] > 0.5):
            x1, y1, _ = landmarks_2d[start_idx]
            x2, y2, _ = landmarks_2d[end_idx]
            color = SKELETON_COLORS.get(color_name, WHITE)
            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    if action_text:
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (image.shape[1], 35), BLACK, cv2.FILLED)
        cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
        cv2.putText(image, f"ACTION: {action_text}", (10, 25),
                   FONT_FACE, 0.7, GREEN, 2, cv2.LINE_AA)

    return image, landmarks_3d


# ========== [10] 动作识别函数 ==========

def recognize_action(landmarks_2d, action_counter):
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return "STANDING", action_counter

    left_shoulder  = landmarks_2d[11]
    right_shoulder = landmarks_2d[12]
    left_wrist  = landmarks_2d[15]
    right_wrist = landmarks_2d[16]
    left_hip  = landmarks_2d[23]
    right_hip = landmarks_2d[24]

    if (left_shoulder[2] < 0.5 or right_shoulder[2] < 0.5 or
        left_wrist[2] < 0.5 or right_wrist[2] < 0.5):
        return "STANDING", action_counter

    left_raise_diff = left_shoulder[1] - left_wrist[1]
    right_raise_diff = right_shoulder[1] - right_wrist[1]

    left_raised = left_raise_diff > (RAISE_THRESHOLD * 480)
    right_raised = right_raise_diff > (RAISE_THRESHOLD * 480)

    detected_action = "STANDING"

    if left_raised and right_raised:
        detected_action = "RAISING_BOTH_HANDS"
    elif left_raised:
        detected_action = "RAISING_LEFT_HAND"
    elif right_raised:
        detected_action = "RAISING_RIGHT_HAND"

    if detected_action == "STANDING":
        shoulder_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        hip_center_y = (left_hip[1] + right_hip[1]) / 2
        spine_length = abs(hip_center_y - shoulder_center_y)
        shoulder_tilt = abs(left_shoulder[1] - right_shoulder[1])

        if left_hip[2] > 0.5 and right_hip[2] > 0.5:
            if spine_length < BEND_SPINE_THRESHOLD * 480 or shoulder_tilt > BEND_TILT_THRESHOLD * 480:
                detected_action = "BENDING"

    if detected_action not in action_counter:
        action_counter = {k: 0 for k in action_counter}
        action_counter[detected_action] = 1
    else:
        action_counter[detected_action] += 1

    best_action = max(action_counter, key=action_counter.get)
    best_count = action_counter[best_action]

    if best_count >= ACTION_CONFIRM_FRAMES:
        return best_action, action_counter
    else:
        return "STANDING", action_counter


# ========== [11] YOLOv26 后处理 + 活体检测 ==========

def post_process_with_pose(img, depth_frame, outs, fx, fy, cx, cy, action_counter, liveness_history):
    """
    YOLOv26 后处理（输出 shape: (1,300,6)，已内置 NMS）
    新增：活体检测，基于深度分布判断真人 vs 照片
    """
    predictions = np.squeeze(outs[0])  # shape: (300, 6)
    boxes, confidences, class_ids = [], [], []
    img_h, img_w = img.shape[:2]

    # 解析深度数据
    try:
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
    except ValueError:
        print("[警告] 深度数据解析失败")
        return img, action_counter, liveness_history

    depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
    depth_data = depth_data.astype(np.uint16)

    # 遍历 v26 的每个检测结果（已做完 NMS，只需过滤）
    for det in predictions:
        x1, y1, x2, y2, conf, cls_id = det[:6]
        if conf < CONFIDENCE_THRESHOLD or int(cls_id) != 0:
            continue

        left = int(x1 * img_w / INPUT_WIDTH)
        top = int(y1 * img_h / INPUT_HEIGHT)
        right = int(x2 * img_w / INPUT_WIDTH)
        bottom = int(y2 * img_h / INPUT_HEIGHT)

        width = right - left
        height = bottom - top
        boxes.append([left, top, width, height])
        confidences.append(float(conf))
        class_ids.append(int(cls_id))

    if len(boxes) == 0:
        return img, action_counter, liveness_history

    # ===== 姿态检测 + 3D坐标 + 动作识别 =====
    result_img = img.copy()

    person_indices = [i for i, cls in enumerate(class_ids) if cls == 0]

    if person_indices and MEDIAPIPE_AVAILABLE:
        largest_idx = max(person_indices, key=lambda i: boxes[i][2] * boxes[i][3])
        left, top, width, height = boxes[largest_idx]

        margin = int(0.1 * max(width, height))
        bbox = (left - margin, top - margin, width + 2*margin, height + 2*margin)

        landmarks_2d = detect_pose(img, bbox)

        if landmarks_2d:
            action_text, action_counter = recognize_action(landmarks_2d, action_counter)
            result_img, landmarks_3d = draw_skeleton(
                result_img, landmarks_2d, depth_data,
                fx, fy, cx, cy, action_text
            )

    # ===== 绘制检测框 + 活体检测 =====
    for i in range(len(boxes)):
        left, top, width, height = boxes[i]
        right_bound = min(left + width, depth_data.shape[1])
        bottom_bound = min(top + height, depth_data.shape[0])

        # 提取人体框内的深度ROI
        depth_roi = depth_data[top:bottom_bound, left:right_bound]
        depth_values = depth_roi.flatten()
        valid_depths = depth_values[depth_values > 0]
        filtered_depths = filter_depth_outliers(valid_depths)

        if filtered_depths.size > 0:
            depth_at_center = int(np.median(filtered_depths))
            depth_label = f"depth:{depth_at_center}mm"
        else:
            depth_label = "depth:N/A"

        # ---- 活体检测 ----
        liveness_result = analyze_depth_liveness(depth_roi)
        smoothed_real, liveness_history = smooth_liveness(
            liveness_result['is_real'], liveness_history
        )

        # 根据活体检测结果选择框颜色和标签
        if smoothed_real:
            liveness_label = "REAL PERSON"
            box_color = GREEN
        else:
            liveness_label = "PHOTO/SPOOF"
            box_color = RED

        # 绘制检测框
        cv2.rectangle(result_img, (left, top), (left + width, top + height), box_color, 3)

        # 第一行：类别 + 置信度
        label = f"{classes[class_ids[i]]}:{confidences[i]:.2f}"
        # 第二行：活体检测结果 + 深度统计
        stats_text = (f"{liveness_label} "
                      f"range:{liveness_result['depth_range']:.0f}mm "
                      f"std:{liveness_result['depth_std']:.0f}mm")

        label_x = max(2, min(left + 2, img.shape[1] - 200 - 2))
        label_y = max(2 + 35, min(top + height - 5, img.shape[0] - 2))
        draw_label(result_img, label, label_x, label_y - 35, box_color, stats_text)

        # 在框内左上角绘制活体状态指示灯
        indicator_color = GREEN if smoothed_real else RED
        cv2.circle(result_img, (left + 15, top + 15), 10, indicator_color, -1)
        cv2.putText(result_img, "LIVE" if smoothed_real else "FAKE",
                   (left + 28, top + 20), FONT_FACE, 0.5, indicator_color, 2, cv2.LINE_AA)

    return result_img, action_counter, liveness_history


# ========== [12] 相机配置 ==========

def get_sw_align_config(pipeline, color_req_width=None, color_req_height=None,
                        depth_req_width=None, depth_req_height=None):
    cw = color_req_width if color_req_width is not None else COLOR_CAMERA_WIDTH
    ch = color_req_height if color_req_height is not None else COLOR_CAMERA_HEIGHT
    dw = depth_req_width if depth_req_width is not None else DEPTH_CAMERA_WIDTH
    dh = depth_req_height if depth_req_height is not None else DEPTH_CAMERA_HEIGHT

    config = Config()
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

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


# ========== [13] 主程序入口 ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='人体活体检测系统（基于深度信息）')
    parser.add_argument('--color_width', type=int, default=None, help="彩色相机宽度")
    parser.add_argument('--color_height', type=int, default=None, help="彩色相机高度")
    parser.add_argument('--depth_width', type=int, default=None, help="深度相机宽度")
    parser.add_argument('--depth_height', type=int, default=None, help="深度相机高度")
    parser.add_argument('--no-pose', action='store_true', help="禁用MediaPipe姿态检测")
    parser.add_argument('--range_threshold', type=float, default=SPOOF_DEPTH_RANGE_THRESHOLD,
                        help=f"深度范围阈值(mm)，默认{SPOOF_DEPTH_RANGE_THRESHOLD}")
    parser.add_argument('--std_threshold', type=float, default=SPOOF_DEPTH_STD_THRESHOLD,
                        help=f"深度标准差阈值(mm)，默认{SPOOF_DEPTH_STD_THRESHOLD}")
    args = parser.parse_args()

    # 支持命令行覆盖阈值
    SPOOF_DEPTH_RANGE_THRESHOLD = args.range_threshold
    SPOOF_DEPTH_STD_THRESHOLD = args.std_threshold

    # ===== 加载YOLO类别标签 =====
    try:
        with open('coco.names', 'rt') as f:
            classes = f.read().strip().split('\n')
        print(f"[信息] 加载了 {len(classes)} 个类别")
    except FileNotFoundError:
        print("[错误] 未找到 coco.names 文件，请确保文件在当前目录")
        classes = []

    # ===== 初始化ONNX Runtime（YOLOv26） =====
    try:
        ort_session = ort.InferenceSession('models/yolo26n.onnx')
        input_name = ort_session.get_inputs()[0].name
        print("[信息] YOLO26n ONNX 模型加载成功")
    except Exception as e:
        print(f"[错误] YOLO模型加载失败: {e}")
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

    # ===== 获取相机内参 =====
    camera_param = pipeline.get_camera_param()
    fx = camera_param.rgb_intrinsic.fx
    fy = camera_param.rgb_intrinsic.fy
    cx = camera_param.rgb_intrinsic.cx
    cy = camera_param.rgb_intrinsic.cy
    print(f"[信息] 相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # ===== 初始化变量 =====
    prev_time = time.time()
    action_counter = {}
    liveness_history = []  # 活体检测结果历史（用于时间域平滑）
    frame_count = 0

    print("\n" + "="*60)
    print("人体活体检测系统已启动（基于YOLOv26 + D2C深度信息）")
    print(f"活体检测参数: range_threshold={SPOOF_DEPTH_RANGE_THRESHOLD}mm, "
          f"std_threshold={SPOOF_DEPTH_STD_THRESHOLD}mm")
    print("按 ESC 或 Q 退出")
    print("="*60 + "\n")

    # ===== 主循环 =====
    while True:
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

        img_bgr = frame_to_bgr_image(color_frame)
        if img_bgr is None:
            continue

        # YOLO推理
        input_tensor = pre_process(img_bgr)
        start_infer = time.time()
        outputs = ort_session.run(None, {input_name: input_tensor})
        inference_time_ms = (time.time() - start_infer) * 1000

        # 综合后处理（YOLO + 姿态 + 3D + 动作识别 + 活体检测）
        result, action_counter, liveness_history = post_process_with_pose(
            img_bgr, depth_frame, outputs,
            fx, fy, cx, cy, action_counter, liveness_history
        )

        # 计算帧率
        curr_time = time.time()
        frame_time_ms = (curr_time - prev_time) * 1000
        fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0
        prev_time = curr_time
        frame_count += 1

        # 显示信息
        cv2.putText(result, f"YOLO:{inference_time_ms:.1f}ms", (10, 20),
                   FONT_FACE, 0.5, RED, 1, cv2.LINE_AA)
        cv2.putText(result, f"FPS:{fps:.1f}", (10, 40),
                   FONT_FACE, 0.5, RED, 1, cv2.LINE_AA)
        mp_status = "ON" if MEDIAPIPE_AVAILABLE else "OFF"
        cv2.putText(result, f"Pose:{mp_status}", (10, 60),
                   FONT_FACE, 0.5, GREEN if MEDIAPIPE_AVAILABLE else RED, 1, cv2.LINE_AA)
        cv2.putText(result, f"Liveness:ON", (10, 80),
                   FONT_FACE, 0.5, GREEN, 1, cv2.LINE_AA)

        cv2.imshow('Human Liveness Detection (Depth-based)', result)

        key = cv2.waitKey(1)
        if key in (ESC_KEY, ord('q'), ord('Q')):
            print("\n[信息] 用户退出")
            break

    # ===== 清理 =====
    cv2.destroyAllWindows()
    pipeline.stop()

    if MEDIAPIPE_AVAILABLE:
        pose_detector.close()

    print("[信息] 程序已安全退出")
