# ============================================================================
# 公共场所安全监控与互动小游戏 — 统一启动器
# ============================================================================
# 融合两个功能模块:
#   1. sum1.py      — 多人安全监控 (YOLOv26 + 活体检测 + 摔倒报警)
#   2. game_demo_yolov11.py — 动作模仿游戏 (YOLOv11 + 计分 + 排行榜)
#
# 使用方式:
#   python summary.py
#
# 功能:
#   - 启动后显示主菜单，包含两个功能按钮
#   - 点击按钮进入对应功能，ESC 返回菜单
#   - 各功能独立运行，显示效果与单独运行原文件完全一致
# ============================================================================

import cv2
import time
import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage, ImageDraw, ImageFont
from pyorbbecsdk import *
import sys
import os
import json
import random
import collections
import datetime
import threading
import winsound

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from utils import frame_to_bgr_image

# Open3D (可选)
try:
    import open3d as o3d
    O3D_AVAILABLE = True
except ImportError:
    O3D_AVAILABLE = False

# MediaPipe (可选)
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe import Image as MPImage, ImageFormat
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[警告] 未检测到 mediapipe，请先安装: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

# pygame 背景音乐 (可选)
try:
    import pygame
    pygame.mixer.init()
    _BGM_PATH = os.path.join(os.path.dirname(__file__), '鸽子舞.mp3')
    if os.path.exists(_BGM_PATH):
        pygame.mixer.music.load(_BGM_PATH)
        BGM_AVAILABLE = True
        print("[信息] 背景音乐加载成功: 鸽子舞.mp3")
    else:
        BGM_AVAILABLE = False
        print("[警告] 未找到背景音乐文件: 鸽子舞.mp3")
except ImportError:
    BGM_AVAILABLE = False
    print("[警告] 未安装 pygame，背景音乐不可用")


# ############################################################################
#
#  [A] 共享工具函数与常量
#
# ############################################################################

ESC_KEY = 27

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
BLACK = (0, 0, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)
ORANGE = (0, 165, 255)

INPUT_WIDTH, INPUT_HEIGHT = 640, 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5

MIN_DEPTH = 20
MAX_DEPTH = 10000
DEPTH_SAMPLE_SIZE = 5

# ---------- PIL 中文字体渲染 ----------

_FONT_CACHE = {}


def _get_font(size):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def put_chinese_text(img, text, position, font_size, color, anchor='lt'):
    """用PIL在OpenCV图像上绘制中文文本。anchor: lt=左上, mt=中上, rt=右上"""
    if not text:
        return
    pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = position
    if anchor == 'mt':
        x = x - tw // 2
    elif anchor == 'rt':
        x = x - tw
    draw.text((x, y), text, font=font, fill=color[::-1])
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img[:] = result


def put_chinese_text_copy(img, text, position, font_size, color, anchor='lt'):
    """返回新图像，不修改原图"""
    if not text:
        return img
    pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y = position
    if anchor == 'mt':
        x = x - tw // 2
    elif anchor == 'rt':
        x = x - tw
    draw.text((x, y), text, font=font, fill=color[::-1])
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


# ---------- 通用辅助函数 ----------

def get_onnx_providers(prefer_gpu=True, verbose=True):
    available = ort.get_available_providers()
    if verbose:
        print(f"[信息] ONNX Runtime 可用提供程序: {available}")
    if not prefer_gpu:
        if verbose:
            print("[信息] 手动选择 CPU 推理")
        return ['CPUExecutionProvider']
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


def draw_label(img, label, x, y, color, extra_line=None):
    lines = [label] if extra_line is None else [label, extra_line]
    y_offset = 0
    for text in lines:
        ts, bs = cv2.getTextSize(text, FONT_FACE, 0.5, 1)
        w, h = ts
        if y + y_offset + h + bs > img.shape[0]:
            break
        cv2.rectangle(img, (x, y + y_offset), (x + w, y + y_offset + h + bs), BLACK, cv2.FILLED)
        cv2.putText(img, text, (x, y + y_offset + h), FONT_FACE, 0.5, color, 1, cv2.LINE_AA)
        y_offset += h + bs


def draw_label_cn(img, label, x, y, color, font_size=16, extra_line=None):
    """支持中文的标签绘制"""
    lines = [label] if extra_line is None else [label, extra_line]
    font = _get_font(font_size)
    y_offset = 0
    for text in lines:
        pil_tmp = PILImage.new('RGB', (1, 1))
        draw_tmp = ImageDraw.Draw(pil_tmp)
        bbox = draw_tmp.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if y + y_offset + th + 4 > img.shape[0]:
            break
        cv2.rectangle(img, (x, y + y_offset), (x + tw + 4, y + y_offset + th + 4), BLACK, cv2.FILLED)
        put_chinese_text(img, text, (x + 2, y + y_offset + 2), font_size, color)
        y_offset += th + 4


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


# ---------- 相机配置 ----------

def get_sw_align_config(pipeline, color_req_width=None, color_req_height=None,
                        depth_req_width=None, depth_req_height=None):
    config = Config()
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)

        color_profile = None
        if color_req_width and color_req_height:
            for cp in color_profiles:
                if cp.get_format() == OBFormat.RGB and cp.get_width() == color_req_width and cp.get_height() == color_req_height:
                    color_profile = cp
                    print(f"[配置] 使用指定彩色分辨率: {color_req_width}x{color_req_height}")
                    break
            if color_profile is None:
                print(f"[配置] 未找到 {color_req_width}x{color_req_height} 彩色配置，使用默认")
        if color_profile is None:
            color_profile = color_profiles.get_default_video_stream_profile()
            print(f"[配置] 默认彩色配置: {color_profile.get_width()}x{color_profile.get_height()}")
        config.enable_stream(color_profile)

        depth_profile = None
        if depth_req_width and depth_req_height:
            for dp in depth_profiles:
                if dp.get_width() == depth_req_width and dp.get_height() == depth_req_height:
                    depth_profile = dp
                    print(f"[配置] 使用指定深度分辨率: {depth_req_width}x{depth_req_height}")
                    break
            if depth_profile is None:
                print(f"[配置] 未找到 {depth_req_width}x{depth_req_height} 深度配置，使用默认")
        if depth_profile is None:
            depth_profile = depth_profiles.get_default_video_stream_profile()
            print(f"[配置] 默认深度配置: {depth_profile.get_width()}x{depth_profile.get_height()}")
        config.enable_stream(depth_profile)

    except Exception as e:
        print(f"[错误] 相机配置失败: {e}")
        return None
    return config


# ---------- 3D 坐标计算 (共享) ----------

SKELETON_CONNECTIONS = [
    (11, 13, 'left'), (13, 15, 'left'),
    (12, 14, 'right'), (14, 16, 'right'),
    (11, 12, 'torso'), (11, 23, 'torso'), (12, 24, 'torso'), (23, 24, 'torso'),
    (23, 25, 'left'), (25, 27, 'left'),
    (24, 26, 'right'), (26, 28, 'right'),
    (11, 0, 'torso'), (12, 0, 'torso'),
    (-1, -1, 'spine'),
]

JOINT_RADIUS = 3


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
    lower, upper = median * 0.8, median * 1.2
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


def compute_3d_landmarks(landmarks_2d, depth_data, fx, fy, cx, cy):
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


# ############################################################################
#
#  [B] 功能一: 多人安全监控 (源自 sum1.py)
#
# ############################################################################

# ---------- sum1 专用常量 ----------

ACTION_CN = {
    'STANDING': '站立',
    'RAISING_BOTH_HANDS': '举起双手',
    'RAISING_LEFT_HAND': '举起左手',
    'RAISING_RIGHT_HAND': '举起右手',
    'JUMPING': '跳跃',
    'RAISING_LEFT_LEG': '抬起左腿',
    'RAISING_RIGHT_LEG': '抬起右腿',
    'BENDING': '弯腰',
    'SQUATTING': '蹲下',
}

PALETTE = [(255, 255, 255), (0, 255, 0), (0, 0, 255), (255, 255, 0),
           (255, 0, 255), (0, 255, 255), (128, 128, 0),
           (128, 0, 128), (0, 128, 128), (128, 128, 128)]

MAX_DISPLAY_BOXES = 10

# 活体检测参数
SPOOF_DEPTH_RANGE_THRESHOLD = 35
SPOOF_DEPTH_STD_THRESHOLD = 12
SPOOF_VALID_RATIO_THRESHOLD = 0.3
LIVENESS_SMOOTH_FRAMES = 5

# 摔倒检测参数
FALL_TORSO_ANGLE_THRESHOLD = 55
FALL_ASPECT_RATIO_THRESHOLD = 1.6
FALL_HEAD_DROP_RATIO = 0.65
FALL_CONFIRM_FRAMES = 8
FALL_ALARM_COOLDOWN = 10

# 多人参数
MAX_PERSONS_MONITOR = 3

PERSON_COLORS = [
    ((255, 255, 0), (0, 255, 255), (255, 255, 255)),
    ((255, 0, 255), (128, 0, 255), (200, 200, 200)),
    ((0, 255, 0), (0, 200, 0), (180, 255, 180)),
    ((255, 128, 0), (255, 200, 0), (200, 200, 200)),
    ((255, 0, 0), (200, 0, 0), (200, 200, 200)),
    ((0, 128, 255), (0, 0, 255), (200, 200, 200)),
]

# 动作识别参数 (sum1)
S1_RAISE_THRESHOLD = 0.05
S1_BEND_SPINE_THRESHOLD = 0.22
S1_BEND_TILT_THRESHOLD = 0.06
S1_SQUAT_HIP_DROP_RATIO = 0.15
S1_JUMP_UP_RATIO = 0.05
S1_JUMP_DOWN_RATIO = 0.03
S1_LEG_RAISE_KNEE_UP_RATIO = 0.1
S1_LEG_RAISE_LEVEL_RATIO = 0.05
S1_ACTION_CONFIRM_FRAMES = 4


# ---------- sum1: 人员追踪器 ----------

class PersonTracker:
    """基于IoU的轻量级跨帧人员追踪"""

    def __init__(self, max_disappeared=10, min_iou=0.1):
        self.next_id = 0
        self.tracked = {}
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou
        self._person_states_ref = None  # 由 run_monitor 设置

    def set_person_states_ref(self, ref):
        self._person_states_ref = ref

    def update(self, boxes):
        if len(boxes) == 0:
            result = {}
            for pid in list(self.tracked):
                self.tracked[pid]['disappeared'] += 1
                if self.tracked[pid]['disappeared'] > self.max_disappeared:
                    del self.tracked[pid]
            return result

        if len(self.tracked) == 0:
            result = {}
            for box in boxes:
                pid = self.next_id
                self.next_id += 1
                self.tracked[pid] = {'bbox': box, 'disappeared': 0}
                result[pid] = box
            return result

        tracked_ids = list(self.tracked.keys())
        n_tracked = len(tracked_ids)
        n_boxes = len(boxes)

        iou_matrix = np.zeros((n_tracked, n_boxes))
        for i, pid in enumerate(tracked_ids):
            for j, box in enumerate(boxes):
                iou_matrix[i, j] = self._iou(self.tracked[pid]['bbox'], box)

        matches_list = []
        for i in range(n_tracked):
            for j in range(n_boxes):
                if iou_matrix[i, j] > self.min_iou:
                    matches_list.append((iou_matrix[i, j], i, j))
        matches_list.sort(reverse=True)

        matched = {}
        used_tracks = set()
        used_boxes = set()

        for iou_val, i, j in matches_list:
            if i not in used_tracks and j not in used_boxes:
                pid = tracked_ids[i]
                matched[pid] = boxes[j]
                self.tracked[pid]['bbox'] = boxes[j]
                self.tracked[pid]['disappeared'] = 0
                used_tracks.add(i)
                used_boxes.add(j)

        for i, pid in enumerate(tracked_ids):
            if i not in used_tracks:
                self.tracked[pid]['disappeared'] += 1
                if self.tracked[pid]['disappeared'] > self.max_disappeared:
                    del self.tracked[pid]
                    if self._person_states_ref is not None and pid in self._person_states_ref:
                        del self._person_states_ref[pid]

        for j, box in enumerate(boxes):
            if j not in used_boxes:
                pid = self.next_id
                self.next_id += 1
                self.tracked[pid] = {'bbox': box, 'disappeared': 0}
                matched[pid] = box

        return matched

    @staticmethod
    def _iou(boxA, boxB):
        xA, yA, wA, hA = boxA
        xB, yB, wB, hB = boxB
        x1, y1 = max(xA, xB), max(yA, yB)
        x2, y2 = min(xA + wA, xB + wB), min(yA + hA, yB + hB)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        areaA, areaB = wA * hA, wB * hB
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0


# ---------- sum1: 多人颜色工具 ----------

def get_person_colors(person_id):
    left, right, spine = PERSON_COLORS[person_id % len(PERSON_COLORS)]
    return left, right, spine, spine


# ---------- sum1: MediaPipe 全图多姿态检测 ----------

def detect_pose_all(pose_landmarker, image, timestamp_ms=0):
    if pose_landmarker is None:
        return []
    img_h, img_w = image.shape[:2]
    max_dim = 640
    scale = min(max_dim / img_w, max_dim / img_h)
    if scale < 1.0:
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(image, (new_w, new_h))
    else:
        resized = image
        new_w, new_h = img_w, img_h
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    mp_image = ImageFormat and MPImage(image_format=ImageFormat.SRGB, data=rgb)
    result = pose_landmarker.detect_for_video(mp_image, int(timestamp_ms))
    if not result.pose_landmarks:
        return []
    all_landmarks = []
    for landmarks in result.pose_landmarks:
        landmarks_2d = []
        for lm in landmarks:
            px = int(lm.x * img_w)
            py = int(lm.y * img_h)
            visibility = lm.visibility if lm.visibility else (lm.presence if lm.presence else 1.0)
            landmarks_2d.append((px, py, visibility))
        all_landmarks.append(landmarks_2d)
    return all_landmarks


def match_poses_to_boxes(poses_landmarks, person_boxes):
    matched_pairs = []
    used_boxes = set()
    for landmarks in poses_landmarks:
        valid_pts = []
        for idx in [11, 12, 23, 24]:
            if idx < len(landmarks) and landmarks[idx][2] > 0.5:
                valid_pts.append((landmarks[idx][0], landmarks[idx][1]))
        if len(valid_pts) < 2:
            valid_pts = [(lm[0], lm[1]) for lm in landmarks if lm[2] > 0.5]
            if len(valid_pts) < 3:
                continue
        cx = int(sum(p[0] for p in valid_pts) / len(valid_pts))
        cy = int(sum(p[1] for p in valid_pts) / len(valid_pts))
        best_j = -1
        best_score = 0.0
        for j, (bx, by, bw, bh) in enumerate(person_boxes):
            if j in used_boxes:
                continue
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                dx = min(cx - bx, bx + bw - cx)
                dy = min(cy - by, by + bh - cy)
                score = (dx * dy) / (bw * bh)
                if score > best_score:
                    best_score = score
                    best_j = j
        if best_j >= 0:
            matched_pairs.append((landmarks, best_j))
            used_boxes.add(best_j)
        else:
            matched_pairs.append((landmarks, -1))
    return matched_pairs


# ---------- sum1: 活体检测 ----------

def analyze_depth_liveness(depth_roi):
    total_pixels = depth_roi.size
    if total_pixels == 0:
        return {
            'is_real': False, 'confidence': 0.0,
            'depth_range': 0, 'depth_std': 0, 'valid_ratio': 0,
            'reason': 'no depth data',
        }
    valid_depths = depth_roi[depth_roi > 0].astype(np.float32)
    valid_ratio = valid_depths.size / total_pixels
    if valid_depths.size < 10:
        return {
            'is_real': True, 'confidence': 0.1,
            'depth_range': 0, 'depth_std': 0, 'valid_ratio': valid_ratio,
            'reason': f'too few pixels({valid_depths.size})',
        }
    median_depth = np.median(valid_depths)
    depth_lo = median_depth * 0.8
    depth_hi = median_depth * 1.2
    person_depths = valid_depths[(valid_depths >= depth_lo) & (valid_depths <= depth_hi)]
    if person_depths.size < 10:
        person_depths = valid_depths
    filtered = filter_depth_outliers(person_depths, threshold=0.2)
    if filtered.size < 5:
        filtered = person_depths
    depth_range = float(np.max(filtered) - np.min(filtered))
    depth_std = float(np.std(filtered))

    score = 0.0
    reasons = []
    if depth_range >= SPOOF_DEPTH_RANGE_THRESHOLD * 2:
        score += 0.4
        reasons.append(f'range={depth_range:.0f}mm(high)')
    elif depth_range >= SPOOF_DEPTH_RANGE_THRESHOLD:
        ratio = (depth_range - SPOOF_DEPTH_RANGE_THRESHOLD) / SPOOF_DEPTH_RANGE_THRESHOLD
        score += 0.2 + 0.2 * ratio
        reasons.append(f'range={depth_range:.0f}mm(moderate)')
    else:
        reasons.append(f'range={depth_range:.0f}mm(flat)')
    if depth_std >= SPOOF_DEPTH_STD_THRESHOLD * 2:
        score += 0.4
        reasons.append(f'std={depth_std:.0f}mm(high)')
    elif depth_std >= SPOOF_DEPTH_STD_THRESHOLD:
        ratio = (depth_std - SPOOF_DEPTH_STD_THRESHOLD) / SPOOF_DEPTH_STD_THRESHOLD
        score += 0.2 + 0.2 * ratio
        reasons.append(f'std={depth_std:.0f}mm(moderate)')
    else:
        reasons.append(f'std={depth_std:.0f}mm(low)')
    if valid_ratio >= SPOOF_VALID_RATIO_THRESHOLD:
        score += 0.2
    elif depth_range < SPOOF_DEPTH_RANGE_THRESHOLD:
        score += 0.2 * (valid_ratio / SPOOF_VALID_RATIO_THRESHOLD) * 0.5
        reasons.append(f'valid={valid_ratio:.0%}(low)')
    else:
        score += 0.15

    is_real = score >= 0.5
    confidence = min(1.0, score / 0.8) if is_real else min(1.0, (1.0 - score) / 0.5)
    return {
        'is_real': is_real, 'confidence': confidence,
        'depth_range': depth_range, 'depth_std': depth_std,
        'valid_ratio': valid_ratio, 'reason': ' | '.join(reasons),
    }


def smooth_liveness(current_is_real, history):
    history.append(current_is_real)
    if len(history) > LIVENESS_SMOOTH_FRAMES:
        history.pop(0)
    true_count = sum(history)
    smoothed_is_real = true_count > len(history) / 2
    return smoothed_is_real, history


# ---------- sum1: 摔倒检测 ----------

def detect_fall(landmarks_2d, bbox):
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return False, ""
    ls = landmarks_2d[11]; rs = landmarks_2d[12]
    lh = landmarks_2d[23]; rh = landmarks_2d[24]
    nose = landmarks_2d[0]
    lk = landmarks_2d[25]; rk = landmarks_2d[26]
    if ls[2] < 0.3 or rs[2] < 0.3 or lh[2] < 0.3 or rh[2] < 0.3:
        return False, ""
    shoulder_cx = (ls[0] + rs[0]) / 2
    shoulder_cy = (ls[1] + rs[1]) / 2
    hip_cx = (lh[0] + rh[0]) / 2
    hip_cy = (lh[1] + rh[1]) / 2
    dx = hip_cx - shoulder_cx
    dy = hip_cy - shoulder_cy
    torso_angle = abs(np.degrees(np.arctan2(abs(dx), abs(dy))))
    left, top, width, height = bbox
    aspect_ratio = width / max(height, 1)
    head_low = False
    if nose[2] > 0.3:
        head_relative_y = (nose[1] - top) / max(height, 1)
        head_low = head_relative_y > FALL_HEAD_DROP_RATIO
    shoulder_y = (ls[1] + rs[1]) / 2
    hip_y = (lh[1] + rh[1]) / 2
    body_flat = abs(shoulder_y - hip_y) < height * 0.2

    def _hip_angle(a, b, c):
        ba = (a[0] - b[0], a[1] - b[1])
        bc = (c[0] - b[0], c[1] - b[1])
        dot = ba[0] * bc[0] + ba[1] * bc[1]
        mag = ((ba[0]**2 + ba[1]**2) ** 0.5) * ((bc[0]**2 + bc[1]**2) ** 0.5)
        if mag < 1:
            return 180.0
        return abs(np.degrees(np.arccos(max(-1, min(1, dot / mag)))))

    is_bending = False
    if lk[2] > 0.3 and rk[2] > 0.3:
        left_hip_angle = _hip_angle(ls, lh, lk)
        right_hip_angle = _hip_angle(rs, rh, rk)
        if min(left_hip_angle, right_hip_angle) < 160:
            left_knee_angle = _hip_angle(lh, lk, landmarks_2d[27]) if landmarks_2d[27][2] > 0.3 else 180
            right_knee_angle = _hip_angle(rh, rk, landmarks_2d[28]) if landmarks_2d[28][2] > 0.3 else 180
            if left_knee_angle > 150 or right_knee_angle > 150:
                is_bending = True

    if is_bending:
        return False, ""

    is_falling = False
    reasons = []
    if torso_angle > FALL_TORSO_ANGLE_THRESHOLD:
        if aspect_ratio > FALL_ASPECT_RATIO_THRESHOLD:
            is_falling = True
            reasons.append(f"tors{torso_angle:.0f}+ar{aspect_ratio:.1f}")
        elif head_low:
            is_falling = True
            reasons.append(f"tors{torso_angle:.0f}+head_low")
        elif body_flat:
            is_falling = True
            reasons.append(f"tors{torso_angle:.0f}+flat")
    if not is_falling and aspect_ratio > 1.8 and head_low:
        is_falling = True
        reasons.append(f"ar{aspect_ratio:.1f}+head_low")
    return is_falling, " | ".join(reasons)


def check_fall_alarm(fall_states, person_id, is_falling):
    if person_id not in fall_states:
        fall_states[person_id] = {'fall_frames': 0, 'last_alarm': 0}
    state = fall_states[person_id]
    curr_time = time.time()
    if is_falling:
        state['fall_frames'] += 1
    else:
        state['fall_frames'] = 0
    if state['fall_frames'] >= FALL_CONFIRM_FRAMES:
        if curr_time - state['last_alarm'] > FALL_ALARM_COOLDOWN:
            state['last_alarm'] = curr_time
            return True, True
        return False, True
    return False, False


def trigger_fall_alarm(person_id):
    def _alarm():
        for _ in range(3):
            winsound.Beep(1000, 300)
            time.sleep(0.1)
    threading.Thread(target=_alarm, daemon=True).start()
    print(f"[报警] 人员{person_id} 检测到摔倒！")


# ---------- sum1: 骨架绘制 ----------

def draw_skeleton_multi(image, landmarks_2d, landmarks_3d, depth_data,
                        fx, fy, cx, cy, person_id):
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return image
    left_color, right_color, torso_color, spine_color = get_person_colors(person_id)
    for idx, (px, py, vis) in enumerate(landmarks_2d):
        if vis < 0.5:
            cv2.circle(image, (px, py), JOINT_RADIUS, RED, -1)
            continue
        if landmarks_3d[idx] is not None:
            depth = landmarks_3d[idx][2]
            if idx in [0, 11, 12, 15, 16, 23, 24, 27, 28]:
                depth_text = f"{depth/1000:.1f}m"
                cv2.circle(image, (px, py), JOINT_RADIUS, GREEN, -1)
                cv2.putText(image, depth_text, (px + 5, py + 10),
                            FONT_FACE, 0.35, GREEN, 1, cv2.LINE_AA)
        else:
            cv2.circle(image, (px, py), JOINT_RADIUS, RED, -1)
    color_map = {'left': left_color, 'right': right_color,
                 'torso': torso_color, 'spine': spine_color}
    for start_idx, end_idx, part_name in SKELETON_CONNECTIONS:
        if start_idx == -1 and end_idx == -1:
            if (landmarks_2d[11][2] > 0.5 and landmarks_2d[12][2] > 0.5 and
                    landmarks_2d[23][2] > 0.5 and landmarks_2d[24][2] > 0.5):
                shoulder_cx = (landmarks_2d[11][0] + landmarks_2d[12][0]) // 2
                shoulder_cy = (landmarks_2d[11][1] + landmarks_2d[12][1]) // 2
                hip_cx = (landmarks_2d[23][0] + landmarks_2d[24][0]) // 2
                hip_cy = (landmarks_2d[23][1] + landmarks_2d[24][1]) // 2
                cv2.line(image, (shoulder_cx, shoulder_cy),
                         (hip_cx, hip_cy), spine_color, 3)
            continue
        if (start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d) and
                landmarks_2d[start_idx][2] > 0.5 and landmarks_2d[end_idx][2] > 0.5):
            x1, y1, _ = landmarks_2d[start_idx]
            x2, y2, _ = landmarks_2d[end_idx]
            cv2.line(image, (x1, y1), (x2, y2), color_map.get(part_name, WHITE), 2)
    return image


# ---------- sum1: 动作识别 ----------

def recognize_action_monitor(landmarks_2d, person_id, person_states):
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return "STANDING"
    if person_id not in person_states:
        person_states[person_id] = {
            'standing_ref': {'hip_y': None, 'spine': None, 'frames': 0},
            'jump_tracker': collections.deque(maxlen=12),
            'action_counter': {}
        }
    state = person_states[person_id]
    standing_ref = state['standing_ref']
    jump_tracker = state['jump_tracker']
    action_counter = state['action_counter']

    ls = landmarks_2d[11]; rs = landmarks_2d[12]
    le = landmarks_2d[13]; re = landmarks_2d[14]
    lw = landmarks_2d[15]; rw = landmarks_2d[16]
    lh = landmarks_2d[23]; rh = landmarks_2d[24]
    lk = landmarks_2d[25]; rk = landmarks_2d[26]
    if ls[2] < 0.5 or rs[2] < 0.5 or lw[2] < 0.5 or rw[2] < 0.5:
        return "STANDING"

    shoulder_center_y = (ls[1] + rs[1]) / 2
    hip_center_y = (lh[1] + rh[1]) / 2
    spine_2d = abs(hip_center_y - shoulder_center_y)
    adaptive_thr = spine_2d * 0.15

    # [1] 举手
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

    # [2] 跳跃
    if detected_action == "STANDING":
        jump_tracker.append((shoulder_center_y, hip_center_y))
        if len(jump_tracker) >= 8:
            sh_vals = [s for s, _ in jump_tracker]
            hip_vals = [h for _, h in jump_tracker]
            early_sh = sum(sh_vals[:3]) / 3
            early_hip = sum(hip_vals[:3]) / 3
            up_thr = spine_2d * S1_JUMP_UP_RATIO
            down_thr = spine_2d * S1_JUMP_DOWN_RATIO
            min_sh = min(sh_vals)
            min_hip = min(hip_vals)
            min_idx_sh = sh_vals.index(min_sh)
            min_idx_hip = hip_vals.index(min_hip)
            went_up = (early_sh - min_sh > up_thr and early_hip - min_hip > up_thr)
            after_sh = sh_vals[min_idx_sh:] if min_idx_sh < len(sh_vals) - 1 else [min_sh]
            after_hip = hip_vals[min_idx_hip:] if min_idx_hip < len(hip_vals) - 1 else [min_hip]
            came_down = (max(after_sh) - min_sh > down_thr and
                         max(after_hip) - min_hip > down_thr)
            if went_up and came_down:
                detected_action = "JUMPING"
                jump_tracker.clear()

    # [3] 抬腿
    if detected_action == "STANDING":
        raise_thr = spine_2d * S1_LEG_RAISE_KNEE_UP_RATIO
        level_thr = spine_2d * S1_LEG_RAISE_LEVEL_RATIO
        left_diff = 0.0; right_diff = 0.0
        if lh[2] > 0.5 and lk[2] > 0.5:
            left_diff = lh[1] - lk[1]
        if rh[2] > 0.5 and rk[2] > 0.5:
            right_diff = rh[1] - rk[1]
        left_leg = (left_diff > raise_thr) or (abs(left_diff) < level_thr and lh[2] > 0.5 and lk[2] > 0.5)
        right_leg = (right_diff > raise_thr) or (abs(right_diff) < level_thr and rh[2] > 0.5 and rk[2] > 0.5)
        if left_leg and right_leg:
            if left_diff > right_diff:
                detected_action = "RAISING_LEFT_LEG"
            else:
                detected_action = "RAISING_RIGHT_LEG"
        elif left_leg:
            detected_action = "RAISING_LEFT_LEG"
        elif right_leg:
            detected_action = "RAISING_RIGHT_LEG"

    # [4] 弯腰
    if detected_action == "STANDING":
        def _angle(a, b, c):
            ba = (a[0] - b[0], a[1] - b[1])
            bc = (c[0] - b[0], c[1] - b[1])
            dot = ba[0] * bc[0] + ba[1] * bc[1]
            mag = ((ba[0]**2 + ba[1]**2) ** 0.5) * ((bc[0]**2 + bc[1]**2) ** 0.5)
            if mag < 1:
                return 180.0
            return abs(np.degrees(np.arccos(max(-1, min(1, dot / mag)))))
        hip_angle = 180.0
        if lh[2] > 0.3 and lk[2] > 0.3:
            hip_angle = min(hip_angle, _angle(ls, lh, lk))
        if rh[2] > 0.3 and rk[2] > 0.3:
            hip_angle = min(hip_angle, _angle(rs, rh, rk))
        if hip_angle < 160:
            detected_action = "BENDING"
        elif lh[2] > 0.5 and rh[2] > 0.5:
            shoulder_tilt = abs(ls[1] - rs[1])
            if spine_2d < S1_BEND_SPINE_THRESHOLD * 480 or shoulder_tilt > S1_BEND_TILT_THRESHOLD * 480:
                detected_action = "BENDING"

    # [5] 蹲下
    if detected_action == "STANDING" and lh[2] > 0.5 and rh[2] > 0.5:
        if standing_ref['hip_y'] is None:
            standing_ref['hip_y'] = hip_center_y
            standing_ref['spine'] = spine_2d
            standing_ref['frames'] = 1
        else:
            alpha = 0.03
            standing_ref['hip_y'] = standing_ref['hip_y'] * (1 - alpha) + hip_center_y * alpha
            standing_ref['spine'] = standing_ref['spine'] * (1 - alpha) + spine_2d * alpha
            standing_ref['frames'] += 1
        if standing_ref['frames'] > 10:
            ref_hip = standing_ref['hip_y']
            ref_spine = standing_ref['spine']
            if (hip_center_y - ref_hip) > ref_spine * S1_SQUAT_HIP_DROP_RATIO:
                detected_action = "SQUATTING"

    # 防抖
    action_counter[detected_action] = action_counter.get(detected_action, 0) + 1
    for k in list(action_counter):
        if k != detected_action:
            action_counter[k] = 0
    best_action = max(action_counter, key=action_counter.get)
    best_count = action_counter[best_action]
    if best_count >= S1_ACTION_CONFIRM_FRAMES:
        return best_action
    else:
        return "STANDING"


def cleanup_person_states(person_states, active_ids):
    for pid in list(person_states):
        if pid not in active_ids:
            del person_states[pid]


# ---------- sum1: Open3D 3D骨架 ----------

def update_3d_skeleton_multi(vis, all_landmarks_3d):
    vis.clear_geometries()
    for pid, landmarks_3d in all_landmarks_3d.items():
        left_color, right_color, torso_color_3d, spine_color_3d = get_person_colors(pid)
        left_3d = (left_color[2]/255, left_color[1]/255, left_color[0]/255)
        right_3d = (right_color[2]/255, right_color[1]/255, right_color[0]/255)
        torso_3d = (torso_color_3d[2]/255, torso_color_3d[1]/255, torso_color_3d[0]/255)
        spine_3d = (spine_color_3d[2]/255, spine_color_3d[1]/255, spine_color_3d[0]/255)
        color_map = {'left': left_3d, 'right': right_3d,
                     'torso': torso_3d, 'spine': spine_3d}
        for idx, point in enumerate(landmarks_3d):
            if point is None:
                continue
            x, y, z = point
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=20)
            sphere.translate((-x, -y, -z))
            sphere.paint_uniform_color([0, 1, 0])
            vis.add_geometry(sphere)
        for start_idx, end_idx, part_name in SKELETON_CONNECTIONS:
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
                    line.paint_uniform_color(spine_3d)
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
                line.paint_uniform_color(color_map.get(part_name, (1, 1, 1)))
                vis.add_geometry(line)
    vis.poll_events()
    vis.update_renderer()


# ---------- sum1: 主后处理函数 ----------

def post_process_with_pose(img, depth_frame, outs, fx, fy, cx, cy,
                           person_tracker, liveness_histories, fall_states,
                           person_states, vis_3d=None):
    predictions = np.squeeze(outs[0])
    boxes, confidences, class_ids = [], [], []
    img_h, img_w = img.shape[:2]

    try:
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
    except ValueError:
        print("[警告] 深度数据解析失败")
        return img
    depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
    depth_data = depth_data.astype(np.uint16)

    # v26 YOLO output: (300, 6) -> x1, y1, x2, y2, conf, cls_id
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

    result_img = img.copy()
    person_boxes = [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                    for i, cls_id in enumerate(class_ids) if cls_id == 0]
    if len(person_boxes) == 0:
        cleanup_person_states(person_states, set())
        return result_img

    # 全图 MediaPipe
    all_poses = detect_pose_all(monitor_pose_landmarker, img, int(time.time() * 1000))
    matched_pairs = match_poses_to_boxes(all_poses, person_boxes)
    box_pose_map = {}
    for landmarks, box_idx in matched_pairs:
        if box_idx >= 0 and box_idx not in box_pose_map:
            box_pose_map[box_idx] = landmarks

    sorted_box_indices = sorted(box_pose_map.keys(),
                                key=lambda j: person_boxes[j][2] * person_boxes[j][3],
                                reverse=True)
    if len(sorted_box_indices) > MAX_PERSONS_MONITOR:
        sorted_box_indices = sorted_box_indices[:MAX_PERSONS_MONITOR]

    track_boxes = [person_boxes[j] for j in sorted_box_indices]
    tracked = person_tracker.update(track_boxes)

    box_to_landmarks = {}
    for j in sorted_box_indices:
        box_to_landmarks[person_boxes[j]] = box_pose_map[j]
    pid_to_landmarks = {}
    for pid, tbbox in tracked.items():
        if tbbox in box_to_landmarks:
            pid_to_landmarks[pid] = box_to_landmarks[tbbox]

    active_ids = set(person_tracker.tracked.keys())
    cleanup_person_states(person_states, active_ids)

    all_landmarks_3d = {}
    pid_actions = {}
    for pid, landmarks_2d in pid_to_landmarks.items():
        landmarks_3d = compute_3d_landmarks(landmarks_2d, depth_data, fx, fy, cx, cy)
        action_text = recognize_action_monitor(landmarks_2d, pid, person_states)
        pid_actions[pid] = action_text
        result_img = draw_skeleton_multi(
            result_img, landmarks_2d, landmarks_3d, depth_data,
            fx, fy, cx, cy, pid
        )
        all_landmarks_3d[pid] = landmarks_3d

    if vis_3d is not None and all_landmarks_3d:
        update_3d_skeleton_multi(vis_3d, all_landmarks_3d)

    bbox_to_pid = {bbox: pid for pid, bbox in tracked.items()}

    for i in range(len(boxes)):
        if class_ids[i] != 0:
            continue
        left, top, width, height = boxes[i]
        right = min(left + width, depth_data.shape[1])
        bottom = min(top + height, depth_data.shape[0])
        bbox_key = (left, top, width, height)

        if bbox_key in bbox_to_pid:
            pid = bbox_to_pid[bbox_key]
            box_color, _, _, _ = get_person_colors(pid)
            action_text = pid_actions.get(pid, "")
            action_cn = ACTION_CN.get(action_text, action_text)

            # 活体检测
            depth_roi = depth_data[top:bottom, left:right]
            liveness_result = analyze_depth_liveness(depth_roi)
            if pid not in liveness_histories:
                liveness_histories[pid] = []
            smoothed_real, liveness_histories[pid] = smooth_liveness(
                liveness_result['is_real'], liveness_histories[pid]
            )
            liveness_text = "真人" if smoothed_real else "照片"
            liveness_color = GREEN if smoothed_real else RED

            # 摔倒检测
            landmarks_2d = pid_to_landmarks.get(pid)
            is_falling, fall_reason = detect_fall(landmarks_2d, (left, top, width, height))
            should_alarm, is_confirmed_fall = check_fall_alarm(fall_states, pid, is_falling)
            if should_alarm:
                trigger_fall_alarm(pid)

            label = f"人员{pid}: {action_cn} | {liveness_text}"
            if is_confirmed_fall:
                label += " | 摔倒!"

            font = _get_font(22)
            pil_tmp = PILImage.new('RGB', (1, 1))
            draw_tmp = ImageDraw.Draw(pil_tmp)
            bbox_t = draw_tmp.textbbox((0, 0), label, font=font)
            tw, th = bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]
            lx = max(0, left + width // 2 - tw // 2)
            lx = min(lx, img.shape[1] - tw - 4)
            ly = max(th + 4, top - 8)
            overlay = result_img.copy()
            cv2.rectangle(overlay, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 4),
                          BLACK, cv2.FILLED)
            cv2.addWeighted(overlay, 0.7, result_img, 0.3, 0, result_img)
            label_color = RED if is_confirmed_fall else box_color
            put_chinese_text(result_img, label, (lx, ly - th), 22, label_color)

            # 活体状态指示灯
            cv2.circle(result_img, (left + 15, top + 15), 10, liveness_color, -1)
            put_chinese_text(result_img, "活" if smoothed_real else "假",
                             (left + 28, top + 10), 16, liveness_color)

            # 摔倒警告闪烁
            if is_confirmed_fall:
                if int(time.time() * 4) % 2 == 0:
                    cv2.rectangle(result_img, (left, top), (left + width, top + height), RED, 4)
                    put_chinese_text(result_img, "摔倒报警!",
                                     (left, top + height + 5), 24, RED, anchor='mt')
        else:
            box_color = PALETTE[class_ids[i] % len(PALETTE)]

        cv2.rectangle(result_img, (left, top), (left + width, top + height), box_color, 2)

        if bbox_key not in bbox_to_pid:
            depth_roi = depth_data[top:bottom, left:right]
            depth_values = depth_roi.flatten()
            valid_depths = depth_values[depth_values > 0]
            filtered_depths = filter_depth_outliers(valid_depths)
            if filtered_depths.size > 0:
                depth_at_center = int(np.median(filtered_depths))
                depth_label = f"深度:{depth_at_center}mm"
            else:
                depth_label = "深度:无数据"
            # 需要加载 classes
            cls_name = "obj"
            try:
                cls_name = _monitor_classes[class_ids[i]] if class_ids[i] < len(_monitor_classes) else "obj"
            except Exception:
                pass
            label = f"{cls_name}:{confidences[i]:.2f}"
            label_x = max(2, min(left + 2, img.shape[1] - 100 - 2))
            label_y = max(2 + 35, min(top + height - 5, img.shape[0] - 2))
            draw_label_cn(result_img, label, label_x, label_y - 35, box_color, font_size=14, extra_line=depth_label)

    put_chinese_text(result_img, f"检测人数: {len(all_landmarks_3d)}  |  Plan B 模式",
                     (10, 60), 18, GREEN)
    return result_img


# ---------- sum1: run_monitor() 主函数 ----------

# 全局: MediaPipe pose_landmarker 用于 sum1 (多姿态)
monitor_pose_landmarker = None
_monitor_classes = []


def run_monitor():
    """运行安全监控功能。ESC/Q 返回启动器菜单。"""
    global monitor_pose_landmarker, _monitor_classes

    print("\n" + "=" * 60)
    print("  进入: 多人安全监控")
    print("=" * 60)

    # 加载 coco.names
    coco_path = os.path.join(os.path.dirname(__file__), 'coco.names')
    try:
        with open(coco_path, 'rt', encoding='utf-8') as f:
            _monitor_classes = f.read().strip().split('\n')
        print(f"[信息] 加载了 {len(_monitor_classes)} 个类别")
    except FileNotFoundError:
        print("[错误] 未找到 coco.names 文件")
        _monitor_classes = []

    # 加载 YOLO26n 模型
    yolo_path = os.path.join(os.path.dirname(__file__), 'models', 'yolo26n.onnx')
    try:
        providers = get_onnx_providers(prefer_gpu=True)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(yolo_path,
                                            sess_options=sess_options,
                                            providers=providers)
        input_name = ort_session.get_inputs()[0].name
        actual_provider = ort_session.get_providers()[0]
        print(f"[信息] YOLO26n ONNX 模型加载成功 (设备: {actual_provider})")
    except Exception as e:
        print(f"[错误] YOLO模型加载失败: {e}")
        return

    # 初始化 MediaPipe (多姿态)
    mp_available = False
    if MEDIAPIPE_AVAILABLE:
        mp_model_path = os.path.join(os.path.dirname(__file__), 'models', 'pose_landmarker_lite.task')
        if os.path.exists(mp_model_path):
            base_options_mp = BaseOptions(model_asset_path=mp_model_path)
            pose_options = vision.PoseLandmarkerOptions(
                base_options=base_options_mp,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=MAX_PERSONS_MONITOR,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_segmentation_masks=False
            )
            monitor_pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            mp_available = True
            print(f"[信息] MediaPipe Pose Landmarker 初始化成功 (num_poses={MAX_PERSONS_MONITOR})")
        else:
            print(f"[错误] 未找到 MediaPipe 模型: {mp_model_path}")
    else:
        print("[警告] MediaPipe未安装，姿态检测功能不可用")

    # 相机
    pipeline = Pipeline()
    config = get_sw_align_config(pipeline)
    if config is None:
        print("[错误] 未找到合适的流配置")
        if monitor_pose_landmarker is not None:
            monitor_pose_landmarker.close()
            monitor_pose_landmarker = None
        return
    pipeline.start(config)
    print("[信息] 相机已启动")
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    camera_param = pipeline.get_camera_param()
    fx = camera_param.rgb_intrinsic.fx
    fy = camera_param.rgb_intrinsic.fy
    cx = camera_param.rgb_intrinsic.cx
    cy = camera_param.rgb_intrinsic.cy
    print(f"[信息] 相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Open3D
    vis_3d = None
    _o3d_enabled = False

    def create_o3d_window():
        v = o3d.visualization.Visualizer()
        v.create_window(window_name="3D Skeleton (Multi-Person)", width=800, height=600)
        vc = v.get_view_control()
        vc.set_front([0, 0, -1])
        vc.set_up([0, 1, 0])
        ro = v.get_render_option()
        ro.line_width = 5.0
        ro.point_size = 8.0
        return v

    person_tracker = PersonTracker(max_disappeared=10, min_iou=0.1)
    person_states = {}
    person_tracker.set_person_states_ref(person_states)
    liveness_histories = {}
    fall_states = {}
    prev_time = time.time()
    frame_count = 0
    last_outputs = None
    INFERENCE_SKIP = 4
    profile_count = 0

    WINDOW_NAME = '多人姿态+活体+摔倒检测'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print(f"\n安全监控已启动 (Max={MAX_PERSONS_MONITOR}, Skip={INFERENCE_SKIP})")
    print("按 ESC 或 Q 返回菜单  |  按 J 切换 3D 骨架\n")

    running = True
    while running:
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

        # YOLO推理（跳帧）
        frame_count += 1
        t_pre0 = time.time()
        if frame_count % (INFERENCE_SKIP + 1) == 1 or last_outputs is None:
            input_tensor = pre_process(img_bgr)
            t_pre = (time.time() - t_pre0) * 1000
            t0 = time.time()
            last_outputs = ort_session.run(None, {input_name: input_tensor})
            t_yolo = (time.time() - t0) * 1000
        else:
            t_pre = 0
            t_yolo = 0
        outputs = last_outputs

        t1 = time.time()
        result = post_process_with_pose(
            img_bgr, depth_frame, outputs,
            fx, fy, cx, cy, person_tracker, liveness_histories,
            fall_states, person_states, vis_3d=vis_3d
        )
        t_post = (time.time() - t1) * 1000

        curr_time = time.time()
        frame_time_ms = (curr_time - prev_time) * 1000
        fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0
        prev_time = curr_time

        img_w = result.shape[1]
        put_chinese_text(result, f"帧率:{fps:.1f}", (img_w - 180, 18), 16, RED)
        put_chinese_text(result, f"预处理:{t_pre:.0f}ms  YOLO:{t_yolo:.0f}ms", (img_w - 260, 38), 14, RED)
        put_chinese_text(result, f"后处理:{t_post:.0f}ms", (img_w - 180, 56), 14, RED)
        o3d_status = "开启" if _o3d_enabled else "关闭"
        o3d_color = GREEN if _o3d_enabled else (100, 100, 100)
        put_chinese_text(result, f"3D骨架:{o3d_status} (J)", (img_w - 180, 74), 14, o3d_color)

        cv2.imshow(WINDOW_NAME, result)

        key = cv2.waitKey(1)
        if key in (ESC_KEY, ord('q'), ord('Q')):
            print("\n[信息] 安全监控退出，返回菜单")
            running = False
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

    # 清理
    if vis_3d is not None:
        vis_3d.destroy_window()
    cv2.destroyWindow(WINDOW_NAME)
    pipeline.stop()
    if monitor_pose_landmarker is not None:
        monitor_pose_landmarker.close()
        monitor_pose_landmarker = None
    print("[信息] 安全监控资源已释放")


# ############################################################################
#
#  [C] 功能二: 动作模仿游戏 (源自 game_demo_yolov11.py)
#
# ############################################################################

# ---------- game: 游戏参数 ----------

GAME_ROUND_SECONDS = 40
DIFFICULTY_CONFIG = {
    '练习': {'interval': 6.5, 'reverse_prob': 0.2},
    '普通': {'interval': 3.9, 'reverse_prob': 0.4},
    '困难': {'interval': 2.6, 'reverse_prob': 0.5},
}
SCORE_FORWARD = 10
SCORE_REVERSE = 20
SCORE_COMBO_BONUS = 5

MENU_MAIN = 0
MENU_DIFFICULTY = 1
MENU_SETTINGS = 2
MENU_LEADERBOARD = 3
MENU_PLAYING = 4
MENU_NAME_INPUT = 5
LEADERBOARD_FILE = os.path.join(os.path.dirname(__file__), 'game_leaderboard_v11.json')

# 动作映射表
FORWARD_MAP = {
    'BOTH_HANDS': ('举起双手!', 'RAISING_BOTH_HANDS'),
    'LEFT_HAND': ('举起左手!', 'RAISING_LEFT_HAND'),
    'RIGHT_HAND': ('举起右手!', 'RAISING_RIGHT_HAND'),
    'SQUAT': ('蹲下!', 'SQUATTING'),
    'LEFT_LEG': ('抬起左腿!', 'RAISING_LEFT_LEG'),
    'RIGHT_LEG': ('抬起右腿!', 'RAISING_RIGHT_LEG'),
}

REVERSE_MAP = {
    'LEFT_HAND': ('反向:举起左手!', 'RAISING_RIGHT_HAND'),
    'RIGHT_HAND': ('反向:举起右手!', 'RAISING_LEFT_HAND'),
    'LEFT_LEG': ('反向:抬起左腿!', 'RAISING_RIGHT_LEG'),
    'RIGHT_LEG': ('反向:抬起右腿!', 'RAISING_LEFT_LEG'),
    'STAND': ('反向:站立!', 'SQUATTING'),
}

GREEN_FOR_FORWARD = (0, 220, 50)
RED_FOR_REVERSE = (50, 50, 255)

REVERSE_TO_FORWARD_ACTION = {
    'RAISING_RIGHT_HAND': 'RAISING_LEFT_HAND',
    'RAISING_LEFT_HAND': 'RAISING_RIGHT_HAND',
    'RAISING_RIGHT_LEG': 'RAISING_LEFT_LEG',
    'RAISING_LEFT_LEG': 'RAISING_RIGHT_LEG',
    'SQUATTING': 'STANDING',
}
SCORE_PENALTY = 15

ACTION_CHINESE_MAP = {
    'STANDING': '站立',
    'RAISING_BOTH_HANDS': '举起双手',
    'RAISING_LEFT_HAND': '举起左手',
    'RAISING_RIGHT_HAND': '举起右手',
    'SQUATTING': '蹲下',
    'RAISING_LEFT_LEG': '抬起左腿',
    'RAISING_RIGHT_LEG': '抬起右腿',
    'JUMPING': '跳跃',
    'BENDING': '弯腰',
}

# 骨架配置 (game)
SKELETON_COLORS = {
    'left': (255, 255, 0), 'right': (0, 255, 255),
    'torso': (255, 255, 255), 'spine': (255, 255, 255),
}
SKELETON_COLORS_3D = {
    'left': (0/255, 255/255, 255/255), 'right': (255/255, 0/255, 255/255),
    'torso': (255/255, 255/255, 255/255), 'spine': (255/255, 255/255, 255/255),
}

# 动作识别参数 (game)
G_RAISE_THRESHOLD = 0.04
G_BEND_SPINE_THRESHOLD = 0.12
G_BEND_TILT_THRESHOLD = 0.06
G_SQUAT_HIP_DROP_RATIO = 0.05
G_JUMP_UP_RATIO = 0.04
G_JUMP_DOWN_RATIO = 0.02
G_LEG_RAISE_KNEE_UP_RATIO = 0.04
G_LEG_RAISE_LEVEL_RATIO = 0.04
G_ACTION_CONFIRM_FRAMES = 2


# ---------- game: 动作识别 (game版，参数不同) ----------

def recognize_action_game(landmarks_2d, action_counter, standing_ref, jump_tracker):
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return "STANDING", action_counter

    ls = landmarks_2d[11]; rs = landmarks_2d[12]
    le = landmarks_2d[13]; re = landmarks_2d[14]
    lw = landmarks_2d[15]; rw = landmarks_2d[16]
    lh = landmarks_2d[23]; rh = landmarks_2d[24]
    lk = landmarks_2d[25]; rk = landmarks_2d[26]

    if ls[2] < 0.3 or rs[2] < 0.3 or lw[2] < 0.3 or rw[2] < 0.3:
        return "STANDING", action_counter

    shoulder_center_y = (ls[1] + rs[1]) / 2
    hip_center_y = (lh[1] + rh[1]) / 2
    spine_2d = abs(hip_center_y - shoulder_center_y)
    adaptive_thr = spine_2d * 0.12

    # [1] 举手
    lw_diff = ls[1] - lw[1]; rw_diff = rs[1] - rw[1]
    le_diff = ls[1] - le[1]; re_diff = rs[1] - re[1]
    left_raised = (lw_diff > 18 or lw_diff > adaptive_thr or
                   le_diff > 18 or le_diff > adaptive_thr)
    right_raised = (rw_diff > 18 or rw_diff > adaptive_thr or
                    re_diff > 18 or re_diff > adaptive_thr)
    if left_raised and right_raised:
        detected_action = "RAISING_BOTH_HANDS"
    elif left_raised:
        detected_action = "RAISING_LEFT_HAND"
    elif right_raised:
        detected_action = "RAISING_RIGHT_HAND"
    else:
        detected_action = "STANDING"

    # [2] 跳跃
    if detected_action == "STANDING":
        jump_tracker.append((shoulder_center_y, hip_center_y))
        if len(jump_tracker) >= 8:
            sh_vals = [s for s, _ in jump_tracker]
            hip_vals = [h for _, h in jump_tracker]
            early_sh = sum(sh_vals[:3]) / 3
            early_hip = sum(hip_vals[:3]) / 3
            up_thr = spine_2d * G_JUMP_UP_RATIO
            down_thr = spine_2d * G_JUMP_DOWN_RATIO
            min_sh = min(sh_vals)
            min_hip = min(hip_vals)
            min_idx_sh = sh_vals.index(min_sh)
            min_idx_hip = hip_vals.index(min_hip)
            went_up = (early_sh - min_sh > up_thr and early_hip - min_hip > up_thr)
            after_sh = sh_vals[min_idx_sh:] if min_idx_sh < len(sh_vals) - 1 else [min_sh]
            after_hip = hip_vals[min_idx_hip:] if min_idx_hip < len(hip_vals) - 1 else [min_hip]
            came_down = (max(after_sh) - min_sh > down_thr and
                         max(after_hip) - min_hip > down_thr)
            if went_up and came_down:
                detected_action = "JUMPING"
                jump_tracker.clear()

    # [3] 抬腿
    if detected_action == "STANDING":
        raise_thr = spine_2d * G_LEG_RAISE_KNEE_UP_RATIO
        level_thr = spine_2d * G_LEG_RAISE_LEVEL_RATIO
        left_diff = 0.0; right_diff = 0.0
        if lh[2] > 0.3 and lk[2] > 0.3:
            left_diff = lh[1] - lk[1]
        if rh[2] > 0.3 and rk[2] > 0.3:
            right_diff = rh[1] - rk[1]
        left_leg = (left_diff > raise_thr) or (abs(left_diff) < level_thr and lh[2] > 0.3 and lk[2] > 0.3)
        right_leg = (right_diff > raise_thr) or (abs(right_diff) < level_thr and rh[2] > 0.3 and rk[2] > 0.3)
        if left_leg and right_leg:
            if left_diff > right_diff:
                detected_action = "RAISING_LEFT_LEG"
            else:
                detected_action = "RAISING_RIGHT_LEG"
        elif left_leg:
            detected_action = "RAISING_LEFT_LEG"
        elif right_leg:
            detected_action = "RAISING_RIGHT_LEG"

    # [4] 弯腰
    if detected_action == "STANDING":
        shoulder_tilt = abs(ls[1] - rs[1])
        if lh[2] > 0.3 and rh[2] > 0.3:
            if spine_2d < G_BEND_SPINE_THRESHOLD * 480 or shoulder_tilt > G_BEND_TILT_THRESHOLD * 480:
                detected_action = "BENDING"

    # [5] 蹲下
    if detected_action == "STANDING" and lh[2] > 0.3 and rh[2] > 0.3:
        if standing_ref['hip_y'] is None:
            standing_ref['hip_y'] = hip_center_y
            standing_ref['spine'] = spine_2d
            standing_ref['frames'] = 1
        else:
            alpha = 0.03
            standing_ref['hip_y'] = standing_ref['hip_y'] * (1 - alpha) + hip_center_y * alpha
            standing_ref['spine'] = standing_ref['spine'] * (1 - alpha) + spine_2d * alpha
            standing_ref['frames'] += 1
        if standing_ref['frames'] > 10:
            ref_hip = standing_ref['hip_y']
            ref_spine = standing_ref['spine']
            if (hip_center_y - ref_hip) > ref_spine * G_SQUAT_HIP_DROP_RATIO:
                detected_action = "SQUATTING"

    # 防抖
    action_counter[detected_action] = action_counter.get(detected_action, 0) + 1
    for k in list(action_counter):
        if k != detected_action:
            action_counter[k] = 0
    best_action = max(action_counter, key=action_counter.get)
    best_count = action_counter[best_action]
    if best_count >= G_ACTION_CONFIRM_FRAMES:
        return best_action, action_counter
    else:
        return "STANDING", action_counter


# ---------- game: GameState 类 ----------

class GameState:
    def __init__(self, mode='普通', player_name='玩家'):
        self.mode = mode
        cfg = DIFFICULTY_CONFIG[mode]
        self.prompt_interval = cfg['interval']
        self.reverse_prob = cfg['reverse_prob']
        self.player_name = player_name
        self.score = 0
        self.combo = 0
        self.high_score = 0
        self.time_left = GAME_ROUND_SECONDS
        self.game_over = False
        self.prompt_name = None
        self.display_text = ""
        self.expected_action = None
        self.prompt_type = 'forward'
        self.prompt_color = GREEN_FOR_FORWARD
        self.prompt_timer = 0.0
        self._correct_frames = 0
        self._correct_needed = 1
        self._last_wrong = None
        self._saved = False
        self._last_prompts = collections.deque(maxlen=3)
        self.sound_enabled = True
        self._feedbacks = []
        self._pick_new_prompt()

    def _pick_new_prompt(self):
        pool = list(FORWARD_MAP.keys()) + ['STAND']
        candidates = [a for a in pool if a not in self._last_prompts]
        if not candidates:
            candidates = pool
        action_name = random.choice(candidates)
        if action_name == 'STAND':
            is_reverse = True
        else:
            is_reverse = random.random() < self.reverse_prob
        if is_reverse and action_name in REVERSE_MAP:
            display, expected = REVERSE_MAP[action_name]
            self.prompt_type = 'reverse'
            self.prompt_color = RED_FOR_REVERSE
            self.prompt_name = action_name
            self.display_text = display
            self.expected_action = expected
        else:
            display, expected = FORWARD_MAP[action_name]
            self.prompt_type = 'forward'
            self.prompt_color = GREEN_FOR_FORWARD
            self.prompt_name = action_name
            self.display_text = display
            self.expected_action = expected
        self._last_prompts.append(self.prompt_name)
        self._correct_frames = 0
        self._last_wrong = None
        self.prompt_timer = 0.0

    def update(self, dt):
        if self.game_over:
            return
        self.time_left -= dt
        self.prompt_timer += dt
        if self.time_left <= 0:
            self.time_left = 0
            self.game_over = True
            return
        for fb in self._feedbacks[:]:
            fb['life'] -= dt
            if fb['life'] <= 0:
                self._feedbacks.remove(fb)
        if self.prompt_timer >= self.prompt_interval:
            self.combo = 0
            self._pick_new_prompt()

    def check_answer(self, user_action):
        if self.game_over:
            return False
        if user_action == "STANDING":
            self._correct_frames = 0
            return False
        if user_action == self.expected_action:
            self._correct_frames += 1
            if self._correct_frames >= self._correct_needed:
                return True
        else:
            self._correct_frames = 0
        return False

    def on_correct(self):
        self.combo += 1
        points = SCORE_REVERSE if self.prompt_type == 'reverse' else SCORE_FORWARD
        combo_bonus = min(self.combo * 10, 50)
        points += combo_bonus
        self.score += points
        if self.score > self.high_score:
            self.high_score = self.score
        lines = [f"+{points}"]
        if self.combo >= 2:
            lines.append(f"连击 x{self.combo}!")
        self._feedbacks.append({
            'text': '  '.join(lines),
            'color': YELLOW if self.combo < 2 else (0, 255, 0) if self.combo < 4 else (0, 165, 255),
            'life': 1.5, 'max_life': 1.5,
        })
        if self.sound_enabled:
            def _beeps():
                if self.combo >= 3:
                    winsound.Beep(1200, 80)
                winsound.Beep(880, 100)
                winsound.Beep(1100, 120)
            threading.Thread(target=_beeps, daemon=True).start()
        self._pick_new_prompt()

    def on_wrong(self, action_text=None):
        if action_text == "STANDING" or action_text is None:
            self._last_wrong = None
            return
        if action_text == self._last_wrong:
            return
        self._last_wrong = action_text
        self.combo = 0
        if self.mode == '困难' and self.prompt_type == 'reverse':
            forward_action = REVERSE_TO_FORWARD_ACTION.get(self.expected_action)
            if forward_action and action_text == forward_action:
                self.score = max(0, self.score - SCORE_PENALTY)
                self._feedbacks.append({
                    'text': f'-{SCORE_PENALTY} 做反了!',
                    'color': RED, 'life': 1.5, 'max_life': 1.5,
                })
                if self.sound_enabled:
                    threading.Thread(target=lambda: winsound.Beep(200, 200), daemon=True).start()
                return
        if self.sound_enabled:
            threading.Thread(target=lambda: winsound.Beep(300, 150), daemon=True).start()

    def reset(self, mode=None, player_name=None):
        if mode:
            self.mode = mode
            cfg = DIFFICULTY_CONFIG[mode]
            self.prompt_interval = cfg['interval']
            self.reverse_prob = cfg['reverse_prob']
        if player_name:
            self.player_name = player_name
        self.score = 0
        self.combo = 0
        self.time_left = GAME_ROUND_SECONDS
        self.game_over = False
        self._last_prompts.clear()
        self._correct_frames = 0
        self._feedbacks.clear()
        self._saved = False
        self._pick_new_prompt()


# ---------- game: 排行榜 ----------

def load_leaderboard():
    try:
        with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_leaderboard(entries):
    with open(LEADERBOARD_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


def add_to_leaderboard(player_name, score, difficulty):
    entries = load_leaderboard()
    entries.append({
        'name': player_name,
        'score': score,
        'difficulty': difficulty,
        'date': datetime.date.today().isoformat(),
    })
    entries.sort(key=lambda x: x['score'], reverse=True)
    entries = entries[:50]
    save_leaderboard(entries)


# ---------- game: 菜单按钮 ----------

class MenuButton:
    def __init__(self, text, x, y, w, h, action=None, color=WHITE, hover_color=GREEN):
        self.text = text
        self.rect = (x, y, w, h)
        self.action = action
        self.color = color
        self.hover_color = hover_color
        self.hovered = False

    def contains(self, mx, my):
        x, y, w, h = self.rect
        return x <= mx <= x + w and y <= my <= y + h

    def draw(self, img, font_size=28):
        x, y, w, h = self.rect
        color = self.hover_color if self.hovered else self.color
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        put_chinese_text(img, self.text, (x + w // 2, y + h // 2 - font_size // 2), font_size, color, anchor='mt')


# ---------- game: 菜单系统 ----------

class GameMenu:
    def __init__(self):
        self.state = MENU_MAIN
        self.difficulty = '普通'
        self.sound_enabled = True
        self.player_name = ''
        self.buttons = []
        self.name_input_active = False
        self.name_input_text = ''
        self.leaderboard_scroll = 0
        self.build_buttons(1280, 720)
        self.mx, self.my = -1, -1
        self.clicked = False

    def set_state(self, state, iw=1280, ih=720):
        self.state = state
        self.build_buttons(iw, ih)

    def build_buttons(self, iw, ih):
        bw, bh = 300, 60
        cx = iw // 2 - bw // 2
        self.buttons = []
        if self.state == MENU_MAIN:
            items = [
                ('开始游戏', 'start'),
                ('难度选择', 'difficulty'),
                ('设置', 'settings'),
                ('排行榜', 'leaderboard'),
                ('退出游戏', 'exit'),
            ]
        elif self.state == MENU_DIFFICULTY:
            items = [
                ('练习', 'diff_练习'),
                ('普通', 'diff_普通'),
                ('困难', 'diff_困难'),
                ('返回', 'back'),
            ]
        elif self.state == MENU_SETTINGS:
            sound_label = '音效: 开启' if self.sound_enabled else '音效: 关闭'
            items = [
                (sound_label, 'toggle_sound'),
                ('返回', 'back'),
            ]
        elif self.state == MENU_LEADERBOARD:
            btn = MenuButton('返回', cx, ih - 80, bw, bh, action='back')
            self.buttons.append(btn)
            return
        elif self.state == MENU_NAME_INPUT:
            items = [
                ('确认', 'confirm_name'),
                ('返回', 'back'),
            ]
            start_y = ih * 2 // 3
            for i, (text, action) in enumerate(items):
                btn = MenuButton(text, cx, start_y + i * (bh + 12), bw, bh, action=action)
                self.buttons.append(btn)
            return
        else:
            return
        start_y = ih // 2 - (len(items) * (bh + 12)) // 2
        for i, (text, action) in enumerate(items):
            btn = MenuButton(text, cx, start_y + i * (bh + 12), bw, bh, action=action)
            self.buttons.append(btn)

    def handle_key(self, key):
        if self.state != MENU_NAME_INPUT:
            return
        if key == 8:  # Backspace
            self.name_input_text = self.name_input_text[:-1]
        elif key == 13:  # Enter
            if self.name_input_text.strip():
                self.player_name = self.name_input_text.strip()
                self.state = MENU_PLAYING
        elif 32 <= key <= 126:
            if len(self.name_input_text) < 12:
                self.name_input_text += chr(key)

    def handle_click(self):
        for btn in self.buttons:
            if btn.hovered and btn.action:
                action = btn.action
                if action == 'start':
                    self.name_input_text = self.player_name
                    self.state = MENU_NAME_INPUT
                elif action == 'difficulty':
                    self.state = MENU_DIFFICULTY
                elif action == 'settings':
                    self.state = MENU_SETTINGS
                elif action == 'leaderboard':
                    self.state = MENU_LEADERBOARD
                    self.leaderboard_scroll = 0
                elif action == 'exit':
                    return 'quit'
                elif action == 'back':
                    self.state = MENU_MAIN
                elif action == 'confirm_name':
                    if self.name_input_text.strip():
                        self.player_name = self.name_input_text.strip()
                        self.state = MENU_PLAYING
                elif action == 'toggle_sound':
                    self.sound_enabled = not self.sound_enabled
                    self.state = MENU_MAIN
                elif action.startswith('diff_'):
                    self.difficulty = action.replace('diff_', '')
                    self.state = MENU_MAIN
                if self.state != MENU_PLAYING:
                    self.build_buttons(1280, 720)
        return None

    def update_hover(self, mx, my):
        self.mx, self.my = mx, my
        for btn in self.buttons:
            btn.hovered = btn.contains(mx, my)

    def draw(self, img):
        ih, iw = img.shape[:2]
        img[:] = (20, 20, 25)
        title = '动作模仿游戏'
        put_chinese_text(img, title, (iw // 2, ih // 4 - 40), 48, WHITE, anchor='mt')

        if self.state == MENU_NAME_INPUT:
            put_chinese_text(img, '请输入你的名字', (iw // 2, ih // 3 - 30), 32, YELLOW, anchor='mt')
            box_w, box_h = 400, 50
            box_x = iw // 2 - box_w // 2
            box_y = ih // 3 + 20
            cv2.rectangle(img, (box_x, box_y), (box_x + box_w, box_y + box_h), WHITE, 2)
            display_name = self.name_input_text
            if int(time.time() * 2) % 2 == 0:
                display_name += '|'
            put_chinese_text(img, display_name, (box_x + 10, box_y + 10), 28, WHITE)
            put_chinese_text(img, '输入名字后点击确认，或按回车键', (iw // 2, box_y + box_h + 20), 18, (150, 150, 150), anchor='mt')
            for btn in self.buttons:
                btn.draw(img)
        else:
            for btn in self.buttons:
                btn.draw(img)
            if self.state == MENU_MAIN:
                info = f'当前难度: {self.difficulty}  |  音效: {"开启" if self.sound_enabled else "关闭"}'
                if self.player_name:
                    info += f'  |  玩家: {self.player_name}'
                put_chinese_text(img, info, (iw // 2, ih - 60), 20, (150, 150, 150), anchor='mt')
            elif self.state == MENU_LEADERBOARD:
                entries = load_leaderboard()
                y_start = 100
                line_h = 30
                visible_count = (ih - 200) // line_h
                if entries:
                    put_chinese_text(img, '排名  玩家  分数  难度  日期', (iw // 2, y_start - 35), 22, YELLOW, anchor='mt')
                    max_scroll = max(0, len(entries) - visible_count)
                    self.leaderboard_scroll = min(self.leaderboard_scroll, max_scroll)
                    start_idx = self.leaderboard_scroll
                    end_idx = min(start_idx + visible_count, len(entries))
                    for j in range(start_idx, end_idx):
                        e = entries[j]
                        y = y_start + (j - start_idx) * line_h
                        name = e.get('name', '未知')
                        line = f"{j+1}.  {name}  {e['score']}  {e['difficulty']}  {e.get('date','')}"
                        put_chinese_text(img, line, (iw // 2, y), 20, WHITE, anchor='mt')
                    if len(entries) > visible_count:
                        hint = f"显示 {start_idx+1}-{end_idx} / 共 {len(entries)} 条  |  滚轮翻页"
                        put_chinese_text(img, hint, (iw // 2, ih - 120), 16, (120, 120, 120), anchor='mt')
                else:
                    put_chinese_text(img, '暂无记录', (iw // 2, y_start + 20), 24, (120, 120, 120), anchor='mt')


# ---------- game: Game UI 绘制 ----------

def draw_game_ui(img, game):
    if game.game_over:
        draw_game_over(img, game)
        return
    ih, iw = img.shape[:2]

    # 顶部出题提示
    if game.display_text:
        font_size = 42
        font = _get_font(font_size)
        pil_tmp = PILImage.new('RGB', (1, 1))
        draw_tmp = ImageDraw.Draw(pil_tmp)
        bbox = draw_tmp.textbbox((0, 0), game.display_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (iw - tw) // 2
        ty = 8
        overlay = img.copy()
        bar_h = th + 24
        cv2.rectangle(overlay, (tx - 20, 0), (tx + tw + 20, bar_h), BLACK, cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        put_chinese_text(img, game.display_text, (tx, ty), font_size, game.prompt_color)

    # 右上角计分板
    score_lines = [
        (f"分数: {game.score}", GREEN),
        (f"连击: x{game.combo}", CYAN) if game.combo > 0 else None,
        (f"时间: {game.time_left:.0f}秒", GREEN if game.time_left > 10 else (YELLOW if game.time_left > 5 else RED)),
    ]
    y_off = 30
    font_sz = 22
    font = _get_font(font_sz)
    for item in score_lines:
        if item is None:
            continue
        line, color = item
        pil_tmp = PILImage.new('RGB', (1, 1))
        draw_tmp = ImageDraw.Draw(pil_tmp)
        bbox = draw_tmp.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        rx = iw - tw - 15
        put_chinese_text(img, line, (rx, y_off), font_sz, color)
        y_off += 30

    # 左上角玩家名字
    if game.player_name:
        put_chinese_text(img, f"玩家: {game.player_name}", (15, 30), 22, WHITE)

    # 得分飘字
    for fb in game._feedbacks:
        alpha = fb['life'] / fb['max_life']
        font_size = int(48 * (0.7 + 0.3 * alpha))
        color = tuple(int(c * alpha) for c in fb['color'])
        font = _get_font(font_size)
        pil_tmp = PILImage.new('RGB', (1, 1))
        draw_tmp = ImageDraw.Draw(pil_tmp)
        bbox = draw_tmp.textbbox((0, 0), fb['text'], font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (iw - tw) // 2
        ty = ih // 2 - th // 2 - 40
        ty -= int((1 - alpha) * 40)
        put_chinese_text(img, fb['text'], (tx, ty), font_size, color)

    # 底部进度条
    progress = game.time_left / GAME_ROUND_SECONDS
    bar_w = int((iw - 60) * progress)
    bar_y = ih - 30
    cv2.rectangle(img, (30, bar_y), (iw - 30, bar_y + 12), (60, 60, 60), cv2.FILLED)
    cv2.rectangle(img, (30, bar_y), (30 + bar_w, bar_y + 12),
                  GREEN if progress > 0.3 else YELLOW if progress > 0.1 else RED, cv2.FILLED)
    mode_text = f"模式: {game.mode}"
    put_chinese_text(img, mode_text, (30, bar_y - 8), 18, WHITE)
    put_chinese_text(img, "[R] 重新开始  [1/2/3] 难度",
                     (iw - 280, bar_y - 8), 16, (150, 150, 150))


def draw_game_over(img, game):
    img[:] = (20, 20, 20)
    ih, iw = img.shape[:2]
    cx, cy = iw // 2, ih // 2
    lines = [
        ("游戏结束", 2.0, 4, RED),
        ("", 0, 0, WHITE),
        (f"玩家: {game.player_name}", 1.0, 3, CYAN),
        (f"最终分数: {game.score}", 1.2, 3, WHITE),
        (f"最高分数: {game.high_score}", 0.8, 2, YELLOW),
        (f"难度: {game.mode}", 0.7, 2, (180, 180, 180)),
        ("", 0, 0, WHITE),
        ("按 R 键重新开始", 1.0, 3, GREEN),
        ("按 ESC 键返回菜单", 0.7, 2, (150, 150, 150)),
    ]
    total_h = 0
    spacing = 50
    for text, scale, thick, color in lines:
        if text:
            font = _get_font(int(scale * 24))
            pil_tmp = PILImage.new('RGB', (1, 1))
            draw_tmp = ImageDraw.Draw(pil_tmp)
            bbox = draw_tmp.textbbox((0, 0), text, font=font)
            total_h += (bbox[3] - bbox[1]) + spacing // 2
        else:
            total_h += 15
    start_y = cy - total_h // 2
    current_y = start_y
    for text, scale, thick, color in lines:
        if not text:
            current_y += 15
            continue
        font_size = int(scale * 24)
        font = _get_font(font_size)
        pil_tmp = PILImage.new('RGB', (1, 1))
        draw_tmp = ImageDraw.Draw(pil_tmp)
        bbox = draw_tmp.textbbox((0, 0), text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = cx - tw // 2
        put_chinese_text(img, text, (tx, current_y), font_size, color)
        current_y += th + spacing


# ---------- game: 骨架绘制 ----------

def draw_skeleton(image, landmarks_2d, landmarks_3d, depth_data, fx, fy, cx, cy):
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return image
    for idx, (px, py, vis) in enumerate(landmarks_2d):
        if vis < 0.5:
            cv2.circle(image, (px, py), JOINT_RADIUS, RED, -1)
            continue
        if landmarks_3d[idx] is not None:
            cv2.circle(image, (px, py), JOINT_RADIUS, GREEN, -1)
        else:
            cv2.circle(image, (px, py), JOINT_RADIUS, RED, -1)
    for start_idx, end_idx, color_name in SKELETON_CONNECTIONS:
        if start_idx == -1 and end_idx == -1:
            if (landmarks_2d[11][2] > 0.5 and landmarks_2d[12][2] > 0.5 and
                    landmarks_2d[23][2] > 0.5 and landmarks_2d[24][2] > 0.5):
                shoulder_cx = (landmarks_2d[11][0] + landmarks_2d[12][0]) // 2
                shoulder_cy = (landmarks_2d[11][1] + landmarks_2d[12][1]) // 2
                hip_cx = (landmarks_2d[23][0] + landmarks_2d[24][0]) // 2
                hip_cy = (landmarks_2d[23][1] + landmarks_2d[24][1]) // 2
                cv2.line(image, (shoulder_cx, shoulder_cy), (hip_cx, hip_cy), WHITE, 3)
            continue
        if (start_idx < len(landmarks_2d) and end_idx < len(landmarks_2d) and
                landmarks_2d[start_idx][2] > 0.5 and landmarks_2d[end_idx][2] > 0.5):
            x1, y1, _ = landmarks_2d[start_idx]
            x2, y2, _ = landmarks_2d[end_idx]
            color = SKELETON_COLORS.get(color_name, WHITE)
            cv2.line(image, (x1, y1), (x2, y2), color, 2)
    return image


# ---------- game: MediaPipe 姿态检测 (单人) ----------

def detect_pose(pose_landmarker, image, bbox, timestamp_ms=0):
    if pose_landmarker is None:
        return None
    x, y, w, h = bbox
    img_h, img_w = image.shape[:2]
    x = max(0, x); y = max(0, y)
    w = min(w, img_w - x); h = min(h, img_h - y)
    if w <= 0 or h <= 0:
        return None
    person_roi = image[y:y+h, x:x+w]
    person_roi = cv2.resize(person_roi, (256, 256))
    roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
    mp_image = MPImage(image_format=ImageFormat.SRGB, data=roi_rgb)
    result = pose_landmarker.detect_for_video(mp_image, int(timestamp_ms))
    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None
    landmarks = result.pose_landmarks[0]
    landmarks_2d = []
    for lm in landmarks:
        global_x = int(lm.x * w) + x
        global_y = int(lm.y * h) + y
        visibility = lm.visibility if lm.visibility else (lm.presence if lm.presence else 1.0)
        landmarks_2d.append((global_x, global_y, visibility))
    return landmarks_2d


# ---------- game: Open3D 3D骨架 (单人) ----------

def update_3d_skeleton(vis, landmarks_3d, connections, colors):
    vis.clear_geometries()
    for idx, point in enumerate(landmarks_3d):
        if point is None:
            continue
        x, y, z = point
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=25)
        sphere.translate((-x, -y, -z))
        sphere.paint_uniform_color([0, 1, 0])
        vis.add_geometry(sphere)
    for start_idx, end_idx, color_name in connections:
        if start_idx == -1 and end_idx == -1:
            if (landmarks_3d[11] is not None and landmarks_3d[12] is not None
                    and landmarks_3d[23] is not None and landmarks_3d[24] is not None):
                s_cx = (landmarks_3d[11][0] + landmarks_3d[12][0]) / 2
                s_cy = (landmarks_3d[11][1] + landmarks_3d[12][1]) / 2
                s_cz = (landmarks_3d[11][2] + landmarks_3d[12][2]) / 2
                h_cx = (landmarks_3d[23][0] + landmarks_3d[24][0]) / 2
                h_cy = (landmarks_3d[23][1] + landmarks_3d[24][1]) / 2
                h_cz = (landmarks_3d[23][2] + landmarks_3d[24][2]) / 2
                p1, p2 = (-s_cx, -s_cy, -s_cz), (-h_cx, -h_cy, -h_cz)
                line = o3d.geometry.LineSet()
                line.points = o3d.utility.Vector3dVector([p1, p2])
                line.lines = o3d.utility.Vector2iVector([[0, 1]])
                line.paint_uniform_color(colors.get(color_name, (1, 1, 1)))
                vis.add_geometry(line)
            continue
        if (start_idx < len(landmarks_3d) and end_idx < len(landmarks_3d)
                and landmarks_3d[start_idx] is not None
                and landmarks_3d[end_idx] is not None):
            x1, y1, z1 = landmarks_3d[start_idx]
            x2, y2, z2 = landmarks_3d[end_idx]
            p1, p2 = (-x1, -y1, -z1), (-x2, -y2, -z2)
            line = o3d.geometry.LineSet()
            line.points = o3d.utility.Vector3dVector([p1, p2])
            line.lines = o3d.utility.Vector2iVector([[0, 1]])
            line.paint_uniform_color(colors.get(color_name, (1, 1, 1)))
            vis.add_geometry(line)
    vis.poll_events()
    vis.update_renderer()


# ---------- game: 人员追踪 (RGB+深度融合) ----------

def _bbox_median_depth(box, depth_data):
    l, t, w, h = box
    dh, dw = depth_data.shape[:2]
    x1 = max(0, int(l)); y1 = max(0, int(t))
    x2 = min(dw, int(l + w)); y2 = min(dh, int(t + h))
    roi = depth_data[y1:y2, x1:x2]
    valid = roi[(roi > 0) & (roi < 10000)]
    if valid.size == 0:
        return None
    return float(np.median(valid))


def _select_person(tracked_state, person_indices, boxes, depth_data=None):
    if not person_indices:
        tracked_state['lost_frames'] += 1
        if tracked_state['lost_frames'] > tracked_state['max_lost']:
            tracked_state['center'] = None
            tracked_state['depth'] = None
        return None
    prev_center = tracked_state['center']
    prev_depth = tracked_state['depth']
    if prev_center is None:
        idx = max(person_indices, key=lambda i: boxes[i][2] * boxes[i][3])
    else:
        def combined_score(i):
            l, t, w, h = boxes[i]
            cx, cy = l + w / 2, t + h / 2
            spatial_dist = ((cx - prev_center[0]) ** 2 + (cy - prev_center[1]) ** 2) ** 0.5
            score = spatial_dist
            if depth_data is not None and prev_depth is not None:
                d = _bbox_median_depth(boxes[i], depth_data)
                if d is not None:
                    depth_diff = abs(d - prev_depth)
                    score += depth_diff * 0.3
            return score
        idx = min(person_indices, key=combined_score)
    l, t, w, h = boxes[idx]
    tracked_state['center'] = (l + w / 2, t + h / 2)
    tracked_state['lost_frames'] = 0
    if depth_data is not None:
        d = _bbox_median_depth(boxes[idx], depth_data)
        if d is not None:
            tracked_state['depth'] = d
    return idx


# ---------- game: YOLOv11 后处理 + 姿态 + 游戏逻辑 ----------

def post_process_yolov11(img, depth_frame, outs, fx, fy, cx, cy,
                         action_counter, standing_ref, jump_tracker,
                         tracked_person, game_pose_landmarker,
                         game, vis_3d=None):
    predictions = np.squeeze(outs[0])
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    boxes, confidences, class_ids = [], [], []
    img_h, img_w = img.shape[:2]

    try:
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
    except ValueError:
        print("[警告] 深度数据解析失败")
        return img, action_counter
    depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
    depth_data = np.where((depth_data > MIN_DEPTH) & (depth_data < MAX_DEPTH), depth_data, 0)
    depth_data = depth_data.astype(np.uint16)

    for det in predictions:
        bbox = det[:4]
        class_scores = det[4:]
        conf = np.max(class_scores)
        cls_id = np.argmax(class_scores)
        if conf < CONFIDENCE_THRESHOLD or int(cls_id) != 0:
            continue
        cx_box, cy_box, w_box, h_box = bbox
        left = int((cx_box - w_box / 2) * img_w / INPUT_WIDTH)
        top = int((cy_box - h_box / 2) * img_h / INPUT_HEIGHT)
        right = int((cx_box + w_box / 2) * img_w / INPUT_WIDTH)
        bottom = int((cy_box + h_box / 2) * img_h / INPUT_HEIGHT)
        width = right - left
        height = bottom - top
        boxes.append([left, top, width, height])
        confidences.append(float(conf))
        class_ids.append(int(cls_id))

    result_img = img.copy()
    if len(boxes) == 0:
        return result_img, action_counter

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(indices) == 0:
        return result_img, action_counter

    person_indices = [i for i in indices if class_ids[i] == 0]
    if hasattr(indices, 'flatten'):
        person_indices = [i for i in indices.flatten() if class_ids[i] == 0]

    if person_indices and game_pose_landmarker is not None:
        selected_idx = _select_person(tracked_person, person_indices, boxes, depth_data)
        if selected_idx is None:
            return result_img, action_counter
        left, top, width, height = boxes[selected_idx]
        margin = int(0.2 * max(width, height))
        bbox = (left - margin, top - margin, width + 2*margin, height + 2*margin)
        landmarks_2d = detect_pose(game_pose_landmarker, img, bbox, int(time.time() * 1000))
        if landmarks_2d:
            landmarks_3d = compute_3d_landmarks(landmarks_2d, depth_data, fx, fy, cx, cy)
            action_text, action_counter = recognize_action_game(
                landmarks_2d, action_counter, standing_ref, jump_tracker)
            if not game.game_over:
                if game.check_answer(action_text):
                    game.on_correct()
                elif action_text != "STANDING":
                    game.on_wrong(action_text)
            action_cn = ACTION_CHINESE_MAP.get(action_text, action_text)
            put_chinese_text(result_img, f"动作: {action_cn}", (10, result_img.shape[0] - 90), 26, GREEN)
            result_img = draw_skeleton(result_img, landmarks_2d, landmarks_3d, depth_data, fx, fy, cx, cy)
            if vis_3d is not None and landmarks_3d:
                update_3d_skeleton(vis_3d, landmarks_3d, SKELETON_CONNECTIONS, SKELETON_COLORS_3D)

    return result_img, action_counter


# ---------- game: run_game() 主函数 ----------

def run_game():
    """运行动作模仿游戏。ESC 从游戏内菜单返回启动器菜单。"""
    print("\n" + "=" * 50)
    print("  进入: 动作模仿游戏")
    print("=" * 50)

    # 加载 coco.names
    coco_path = os.path.join(os.path.dirname(__file__), 'coco.names')
    try:
        with open(coco_path, 'rt', encoding='utf-8') as f:
            game_classes = f.read().strip().split('\n')
        print(f"[信息] 加载了 {len(game_classes)} 个类别")
    except FileNotFoundError:
        print("[错误] 未找到 coco.names 文件")
        game_classes = []

    # 加载 YOLOv11 模型
    yolo_model_path = os.path.join(os.path.dirname(__file__), 'models', 'yolo11n.onnx')
    try:
        providers = get_onnx_providers(prefer_gpu=True)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(yolo_model_path,
                                            sess_options=sess_options,
                                            providers=providers)
        input_name = ort_session.get_inputs()[0].name
        actual_provider = ort_session.get_providers()[0]
        print(f"[信息] YOLOv11n ONNX 模型加载成功 (设备: {actual_provider})")
    except Exception as e:
        print(f"[错误] YOLO模型加载失败: {e}")
        return

    # 初始化 MediaPipe (单姿态)
    game_pose_landmarker = None
    if MEDIAPIPE_AVAILABLE:
        mp_model_path = os.path.join(os.path.dirname(__file__), 'models', 'pose_landmarker_lite.task')
        if os.path.exists(mp_model_path):
            base_options_mp = BaseOptions(model_asset_path=mp_model_path)
            pose_options = vision.PoseLandmarkerOptions(
                base_options=base_options_mp,
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
                min_pose_detection_confidence=0.3,
                min_pose_presence_confidence=0.3,
                min_tracking_confidence=0.3,
                output_segmentation_masks=False
            )
            game_pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
            print("[信息] MediaPipe Pose Landmarker 初始化成功 (Lite + VIDEO 模式)")
        else:
            print(f"[错误] 未找到 MediaPipe 模型: {mp_model_path}")
    else:
        print("[警告] MediaPipe未安装，姿态检测功能不可用")

    # 相机
    pipeline = Pipeline()
    config = get_sw_align_config(pipeline)
    if config is None:
        print("[错误] 未找到合适的流配置")
        if game_pose_landmarker is not None:
            game_pose_landmarker.close()
        return
    pipeline.start(config)
    print("[信息] 相机已启动")
    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    camera_param = pipeline.get_camera_param()
    fx = camera_param.rgb_intrinsic.fx
    fy = camera_param.rgb_intrinsic.fy
    cx = camera_param.rgb_intrinsic.cx
    cy = camera_param.rgb_intrinsic.cy
    print(f"[信息] 相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    # Open3D
    vis_3d = None
    _o3d_enabled = False

    def create_o3d_window():
        v = o3d.visualization.Visualizer()
        v.create_window(window_name="3D 骨架", width=800, height=600)
        vc = v.get_view_control()
        vc.set_front([0, 0, -1])
        vc.set_up([0, 1, 0])
        ro = v.get_render_option()
        ro.line_width = 5.0
        ro.point_size = 8.0
        return v

    # 游戏菜单
    menu = GameMenu()

    # 游戏状态
    game = None
    action_counter = {}
    standing_ref = {'hip_y': None, 'spine': None, 'frames': 0}
    jump_tracker = collections.deque(maxlen=12)
    tracked_person = {
        'center': None,
        'depth': None,
        'lost_frames': 0,
        'max_lost': 15,
    }
    prev_time = time.time()
    frame_count = 0
    last_outputs = None
    INFERENCE_SKIP = 8

    WINDOW_NAME = '动作模仿游戏-yolov11'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 鼠标回调
    def on_mouse(event, x, y, flags, param):
        menu.update_hover(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            menu.clicked = True
        if event == cv2.EVENT_MOUSEWHEEL and menu.state == MENU_LEADERBOARD:
            if flags > 0:
                menu.leaderboard_scroll = max(0, menu.leaderboard_scroll - 1)
            else:
                menu.leaderboard_scroll += 1

    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    print(f"\n动作模仿游戏已启动 (Skip={INFERENCE_SKIP})")
    print("鼠标点击菜单按钮操作")
    print("游戏中: ESC=返回菜单  R=重新开始  J=Open3D开关\n")

    running = True
    while running:
        # ---- 菜单模式 ----
        if menu.state != MENU_PLAYING:
            menu_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            menu_img[:] = (20, 20, 25)
            menu.draw(menu_img)
            if menu.clicked:
                result = menu.handle_click()
                if result == 'quit':
                    print("\n[信息] 退出游戏，返回启动器")
                    running = False
                    menu.clicked = False
                    continue
                menu.clicked = False
            cv2.imshow(WINDOW_NAME, menu_img)
            key = cv2.waitKey(30) & 0xFF
            if key == ESC_KEY:
                print("\n[信息] 退出游戏，返回启动器")
                running = False
                continue
            if menu.state == MENU_NAME_INPUT and key != 255:
                menu.handle_key(key)
            if menu.state == MENU_PLAYING:
                game = GameState(mode=menu.difficulty, player_name=menu.player_name)
                game.sound_enabled = menu.sound_enabled
                action_counter = {}
                standing_ref = {'hip_y': None, 'spine': None, 'frames': 0}
                jump_tracker = collections.deque(maxlen=12)
                tracked_person['center'] = None
                tracked_person['depth'] = None
                tracked_person['lost_frames'] = 0
                prev_time = time.time()
                frame_count = 0
                last_outputs = None
                if BGM_AVAILABLE and menu.sound_enabled:
                    pygame.mixer.music.play(-1)
            continue

        # ---- 游戏模式 ----
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

        # YOLOv11 推理（跳帧）
        frame_count += 1
        if frame_count % (INFERENCE_SKIP + 1) == 1 or last_outputs is None:
            input_tensor = pre_process(img_bgr)
            last_outputs = ort_session.run(None, {input_name: input_tensor})
        outputs = last_outputs

        result, action_counter = post_process_yolov11(
            img_bgr, depth_frame, outputs,
            fx, fy, cx, cy,
            action_counter, standing_ref, jump_tracker,
            tracked_person, game_pose_landmarker,
            game, vis_3d=vis_3d
        )

        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        dt = min(dt, 0.2)

        if not game.game_over:
            game.update(dt)

        if game.game_over and not getattr(game, '_saved', False) and game.score > 0:
            add_to_leaderboard(game.player_name, game.score, game.mode)
            game._saved = True
            if BGM_AVAILABLE:
                pygame.mixer.music.stop()

        draw_game_ui(result, game)
        cv2.imshow(WINDOW_NAME, result)

        key = cv2.waitKey(1) & 0xFF
        if key == ESC_KEY:
            print("\n[信息] 返回游戏菜单")
            if BGM_AVAILABLE:
                pygame.mixer.music.stop()
            menu.set_state(MENU_MAIN, result.shape[1], result.shape[0])
            game = None
        elif key == ord('r') or key == ord('R'):
            game.reset(game.mode)
            action_counter = {}
            print(f"[游戏] 重新开始! 模式: {game.mode}")
        elif key == ord('1'):
            game.reset('练习')
            action_counter = {}
            print("[游戏] 切换到练习模式")
        elif key == ord('2'):
            game.reset('普通')
            action_counter = {}
            print("[游戏] 切换到普通模式")
        elif key == ord('3'):
            game.reset('困难')
            action_counter = {}
            print("[游戏] 切换到困难模式")
        elif key == ord('j') or key == ord('J'):
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

    # 清理
    if vis_3d is not None:
        vis_3d.destroy_window()
    cv2.destroyWindow(WINDOW_NAME)
    pipeline.stop()
    if game_pose_landmarker is not None:
        game_pose_landmarker.close()
    print("[信息] 游戏资源已释放")


# ############################################################################
#
#  [D] 启动器主菜单
#
# ############################################################################

WINDOW_NAME_LAUNCHER = '公共场所安全监控与互动小游戏'


def draw_launcher_menu(img, buttons, mouse_x, mouse_y):
    """绘制启动器菜单"""
    ih, iw = img.shape[:2]
    img[:] = (20, 20, 25)

    # 标题
    title = "公共场所安全监控与互动小游戏"
    title_y = ih // 4 - 25
    put_chinese_text(img, title, (iw // 2, title_y), 50, WHITE, anchor='mt')

    # 按钮
    for btn in buttons:
        x, y, w, h = btn['rect']
        is_hover = (x <= mouse_x <= x + w and y <= mouse_y <= y + h)
        border_color = GREEN if is_hover else WHITE
        text_color = GREEN if is_hover else WHITE
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), border_color, 2)
        put_chinese_text(img, btn['text'], (x + w // 2, y + h // 2 - 16), 34, text_color, anchor='mt')

    # 底部提示
    put_chinese_text(img, "按 ESC 键退出", (iw // 2, ih - 60), 22, (120, 120, 120), anchor='mt')


if __name__ == '__main__':
    # 创建启动器窗口
    cv2.namedWindow(WINDOW_NAME_LAUNCHER, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME_LAUNCHER, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 按钮定义
    btn_w, btn_h = 320, 80
    cx = 640  # 1280 / 2

    buttons = [
        {
            'text': '安全监控',
            'rect': (cx - btn_w // 2, 360, btn_w, btn_h),  # 约1/2高度
            'action': 'monitor',
        },
        {
            'text': '互动游戏',
            'rect': (cx - btn_w // 2, 360 + btn_h + 100, btn_w, btn_h),  # 100px间距
            'action': 'game',
        },
    ]

    mouse_x, mouse_y = -1, -1
    mouse_clicked = False

    def launcher_mouse_cb(event, x, y, flags, param):
        global mouse_x, mouse_y, mouse_clicked
        mouse_x, mouse_y = x, y
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_clicked = True

    cv2.setMouseCallback(WINDOW_NAME_LAUNCHER, launcher_mouse_cb)

    print("\n" + "=" * 60)
    print("  公共场所安全监控与互动小游戏 — 启动器")
    print("  点击按钮选择功能，按 ESC 退出")
    print("=" * 60 + "\n")

    while True:
        menu_img = np.zeros((720, 1280, 3), dtype=np.uint8)
        draw_launcher_menu(menu_img, buttons, mouse_x, mouse_y)

        if mouse_clicked:
            mouse_clicked = False
            for btn in buttons:
                x, y, w, h = btn['rect']
                if x <= mouse_x <= x + w and y <= mouse_y <= y + h:
                    if btn['action'] == 'monitor':
                        cv2.destroyWindow(WINDOW_NAME_LAUNCHER)
                        run_monitor()
                        # 重建启动器窗口
                        cv2.namedWindow(WINDOW_NAME_LAUNCHER, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(WINDOW_NAME_LAUNCHER, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.setMouseCallback(WINDOW_NAME_LAUNCHER, launcher_mouse_cb)
                    elif btn['action'] == 'game':
                        cv2.destroyWindow(WINDOW_NAME_LAUNCHER)
                        run_game()
                        # 重建启动器窗口
                        cv2.namedWindow(WINDOW_NAME_LAUNCHER, cv2.WINDOW_NORMAL)
                        cv2.setWindowProperty(WINDOW_NAME_LAUNCHER, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                        cv2.setMouseCallback(WINDOW_NAME_LAUNCHER, launcher_mouse_cb)
                    break

        cv2.imshow(WINDOW_NAME_LAUNCHER, menu_img)
        key = cv2.waitKey(30)
        if key == ESC_KEY:
            print("\n[信息] 启动器退出")
            break

    cv2.destroyAllWindows()
    print("[信息] 程序已安全退出")
