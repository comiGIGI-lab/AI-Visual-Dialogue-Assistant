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
# 多人人体姿态3D分析系统 — Plan B (Full-Image Single MediaPipe Call + Matching)
# ============================================================================
# 策略：
#   YOLO检测多人 → 全图一次MediaPipe调用(num_poses=N) → IoU匹配姿势到YOLO框
#   → IoU追踪分配稳定ID → 逐人动作识别 → 多人骨架绘制 → 多人Open3D骨架
#
# 对比Plan A:
#   Plan A: 逐人ROI裁剪 + N次MediaPipe调用 → 精度高但慢
#   Plan B: 全图1次MediaPipe调用 → 速度快但小目标精度可能略低
#
# 前置条件：
#   - pip install mediapipe opencv-python numpy onnxruntime pyorbbecsdk
#   - models/yolo26n.onnx
#   - models/pose_landmarker_lite.task
#   - coco.names
#   - Orbbec 3D相机已连接
# ============================================================================

import cv2
import time
import argparse
import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage, ImageDraw, ImageFont
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

# ========== [1] 全局参数设置 ==========

COLOR_CAMERA_WIDTH = None
COLOR_CAMERA_HEIGHT = None
DEPTH_CAMERA_WIDTH = None
DEPTH_CAMERA_HEIGHT = None

ESC_KEY = 27

INPUT_WIDTH, INPUT_HEIGHT = 640, 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5
MAX_DISPLAY_BOXES = 10

MIN_DEPTH = 20
MAX_DEPTH = 10000
DEPTH_SAMPLE_SIZE = 5

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
THICKNESS = 1
BLACK, RED, WHITE, GREEN = (0, 0, 0), (0, 0, 255), (255, 255, 255), (0, 255, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

# 动作中文名称映射
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

# ========== [2] 多人参数 ==========

MAX_PERSONS = 3

PERSON_COLORS = [
    ((255, 255, 0), (0, 255, 255), (255, 255, 255)),    # P0
    ((255, 0, 255), (128, 0, 255), (200, 200, 200)),    # P1
    ((0, 255, 0), (0, 200, 0), (180, 255, 180)),        # P2
    ((255, 128, 0), (255, 200, 0), (200, 200, 200)),    # P3
    ((255, 0, 0), (200, 0, 0), (200, 200, 200)),        # P4
    ((0, 128, 255), (0, 0, 255), (200, 200, 200)),      # P5
]

# ========== [3] MediaPipe 初始化 (Tasks API) ==========
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe import Image, ImageFormat
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[错误] 未检测到 mediapipe，请先安装: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

_mp_initialized = False

def init_mediapipe(num_poses=3):
    """初始化/重新初始化 MediaPipe (Plan B 需要 num_poses > 1)"""
    global _mp_initialized, pose_landmarker
    model_path = 'models/pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print(f"[错误] 未找到 MediaPipe 模型: {model_path}")
        return False
    base_options = BaseOptions(model_asset_path=model_path)
    pose_landmarker_options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=num_poses,                # Plan B: 全图多姿态
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(pose_landmarker_options)
    _mp_initialized = True
    print(f"[信息] MediaPipe Pose Landmarker 初始化成功 (Lite + VIDEO, Plan B, num_poses={num_poses})")
    return True

if MEDIAPIPE_AVAILABLE:
    MEDIAPIPE_AVAILABLE = init_mediapipe(MAX_PERSONS)

# ========== [4] 骨架绘制配置 ==========

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
    (-1, -1, 'spine'),
]

JOINT_RADIUS = 3

_profile_count = 0

# ========== [5] 动作识别参数 ==========

RAISE_THRESHOLD = 0.05
BEND_SPINE_THRESHOLD = 0.15
BEND_TILT_THRESHOLD = 0.08
SQUAT_HIP_DROP_RATIO = 0.15
JUMP_UP_RATIO = 0.05
JUMP_DOWN_RATIO = 0.03
LEG_RAISE_KNEE_UP_RATIO = 0.1
LEG_RAISE_LEVEL_RATIO = 0.05
ACTION_CONFIRM_FRAMES = 4

# ========== [6] 辅助函数 ==========

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


# ========== [7] 人员追踪器 (IoU-based) ==========

class PersonTracker:
    """基于IoU的轻量级跨帧人员追踪"""

    def __init__(self, max_disappeared=10, min_iou=0.1):
        self.next_id = 0
        self.tracked = {}
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou

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
                    if pid in _person_states:
                        del _person_states[pid]

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


# ========== [8] 多人颜色工具 ==========

def get_person_colors(person_id):
    left, right, spine = PERSON_COLORS[person_id % len(PERSON_COLORS)]
    return left, right, spine, spine


# ========== [9] MediaPipe 全图多姿态检测 (Plan B核心) ==========

def detect_pose_all(image, timestamp_ms=0):
    """
    Plan B核心: 全图一次 MediaPipe 调用，返回多人的关键点列表

    参数:
        image: 完整BGR图像
        timestamp_ms: 时间戳(ms)，用于 VIDEO 模式帧间追踪

    返回:
        list of [[(x, y, visibility), ...], ...]  # 每人33个关键点(像素坐标)
        如果检测失败返回空列表
    """
    if not MEDIAPIPE_AVAILABLE:
        return []

    img_h, img_w = image.shape[:2]

    # 若图像过大，等比缩放到最大边 <= 640 以加速
    max_dim = 640
    scale = min(max_dim / img_w, max_dim / img_h)
    if scale < 1.0:
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized = cv2.resize(image, (new_w, new_h))
    else:
        resized = image
        new_w, new_h = img_w, img_h

    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

    result = pose_landmarker.detect_for_video(mp_image, int(timestamp_ms))

    if not result.pose_landmarks:
        return []

    all_landmarks = []
    for landmarks in result.pose_landmarks:
        landmarks_2d = []
        for lm in landmarks:
            # 归一化坐标(0~1) → 原始图像像素坐标
            px = int(lm.x * img_w)
            py = int(lm.y * img_h)
            visibility = lm.visibility if lm.visibility else (lm.presence if lm.presence else 1.0)
            landmarks_2d.append((px, py, visibility))
        all_landmarks.append(landmarks_2d)

    return all_landmarks


def match_poses_to_boxes(poses_landmarks, person_boxes):
    """
    将 MediaPipe 姿态匹配到 YOLO 检测框

    策略: 用姿态中心点(肩髋均值)与检测框的包含关系 + 距离打分

    参数:
        poses_landmarks: detect_pose_all 返回的多人关键点列表
        person_boxes: YOLO检测的 person 框列表 [(x,y,w,h), ...]

    返回:
        list of (pose_landmarks, box_idx)  # 匹配对
        未匹配的姿态单独返回 (pose_landmarks, -1)
    """
    matched_pairs = []
    used_boxes = set()

    for landmarks in poses_landmarks:
        # 计算姿态中心点 (肩 + 髋 中心)
        valid_pts = []
        for idx in [11, 12, 23, 24]:
            if idx < len(landmarks) and landmarks[idx][2] > 0.5:
                valid_pts.append((landmarks[idx][0], landmarks[idx][1]))

        if len(valid_pts) < 2:
            # 不可靠的姿态，尝试用所有可见点
            valid_pts = [(lm[0], lm[1]) for lm in landmarks if lm[2] > 0.5]
            if len(valid_pts) < 3:
                continue

        cx = int(sum(p[0] for p in valid_pts) / len(valid_pts))
        cy = int(sum(p[1] for p in valid_pts) / len(valid_pts))

        # 找包含中心点的最佳检测框
        best_j = -1
        best_score = 0.0

        for j, (bx, by, bw, bh) in enumerate(person_boxes):
            if j in used_boxes:
                continue
            # 中心点在框内
            if bx <= cx <= bx + bw and by <= cy <= by + bh:
                # 中心距离框边缘的最小距离作为得分（越靠近中心越好）
                dx = min(cx - bx, bx + bw - cx)
                dy = min(cy - by, by + bh - cy)
                score = (dx * dy) / (bw * bh)  # 归一化
                if score > best_score:
                    best_score = score
                    best_j = j

        if best_j >= 0:
            matched_pairs.append((landmarks, best_j))
            used_boxes.add(best_j)
        else:
            # 未匹配的姿态也保留（可能是YOLO漏检）
            matched_pairs.append((landmarks, -1))

    return matched_pairs


# ========== [10] 3D坐标计算函数 ==========

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


# ========== [11] 多人骨架绘制 ==========

def draw_skeleton_multi(image, landmarks_2d, landmarks_3d, depth_data,
                        fx, fy, cx, cy, action_text, person_id):
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

    # 人员标签
    if landmarks_2d[0][2] > 0.5:
        head_x, head_y = landmarks_2d[0][0], landmarks_2d[0][1]
    elif landmarks_2d[11][2] > 0.5 and landmarks_2d[12][2] > 0.5:
        head_x = (landmarks_2d[11][0] + landmarks_2d[12][0]) // 2
        head_y = min(landmarks_2d[11][1], landmarks_2d[12][1])
    else:
        return image

    label_y = max(10, head_y - 25)
    action_cn = ACTION_CN.get(action_text, action_text)
    label = f"人员{person_id}: {action_cn}"
    font = _get_font(16)
    pil_tmp = PILImage.new('RGB', (1, 1))
    draw_tmp = ImageDraw.Draw(pil_tmp)
    bbox = draw_tmp.textbbox((0, 0), label, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    lx = max(0, head_x - tw // 2)
    lx = min(lx, image.shape[1] - tw - 2)
    cv2.rectangle(image, (lx, label_y - th - 4), (lx + tw + 4, label_y + 4),
                  BLACK, cv2.FILLED)
    put_chinese_text(image, label, (lx + 2, label_y - th - 2), 16, left_color)

    return image


# ========== [12] 动作识别函数 (per-person state) ==========

_person_states = {}


def _get_person_state(person_id):
    if person_id not in _person_states:
        _person_states[person_id] = {
            'standing_ref': {'hip_y': None, 'spine': None, 'frames': 0},
            'jump_tracker': collections.deque(maxlen=12),
            'action_counter': {}
        }
    return _person_states[person_id]


def recognize_action(landmarks_2d, person_id):
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return "STANDING"

    state = _get_person_state(person_id)
    standing_ref = state['standing_ref']
    jump_tracker = state['jump_tracker']
    action_counter = state['action_counter']

    ls = landmarks_2d[11]; rs = landmarks_2d[12]
    le = landmarks_2d[13]; re = landmarks_2d[14]
    lw = landmarks_2d[15]; rw = landmarks_2d[16]
    lh = landmarks_2d[23]; rh = landmarks_2d[24]
    lk = landmarks_2d[25]; rk = landmarks_2d[26]
    la = landmarks_2d[27]; ra = landmarks_2d[28]

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

            up_thr = spine_2d * JUMP_UP_RATIO
            down_thr = spine_2d * JUMP_DOWN_RATIO

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
        raise_thr = spine_2d * LEG_RAISE_KNEE_UP_RATIO
        level_thr = spine_2d * LEG_RAISE_LEVEL_RATIO
        left_diff = 0.0
        right_diff = 0.0
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
        shoulder_tilt = abs(ls[1] - rs[1])
        if lh[2] > 0.5 and rh[2] > 0.5:
            if spine_2d < BEND_SPINE_THRESHOLD * 480 or shoulder_tilt > BEND_TILT_THRESHOLD * 480:
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
            if (hip_center_y - ref_hip) > ref_spine * SQUAT_HIP_DROP_RATIO:
                detected_action = "SQUATTING"

    # 防抖
    action_counter[detected_action] = action_counter.get(detected_action, 0) + 1
    for k in list(action_counter):
        if k != detected_action:
            action_counter[k] = 0

    best_action = max(action_counter, key=action_counter.get)
    best_count = action_counter[best_action]

    if best_count >= ACTION_CONFIRM_FRAMES:
        return best_action
    else:
        return "STANDING"


def cleanup_person_states(active_ids):
    for pid in list(_person_states):
        if pid not in active_ids:
            del _person_states[pid]


# ========== [13] Open3D 多人3D骨架 ==========

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


# ========== [14] YOLO后处理 + 多人姿态 (Plan B核心流程) ==========

def post_process_with_pose(img, depth_frame, outs, fx, fy, cx, cy,
                           person_tracker, vis_3d=None):
    """
    Plan B流程:
      1. YOLO检测 → 提取person框
      2. 全图一次 MediaPipe → 获取所有姿态
      3. 匹配姿态到YOLO框 → IoU追踪分配稳定ID
      4. 逐人3D计算 + 动作识别 + 骨架绘制
    """
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

    # 提取所有 person 框
    person_boxes = [(boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3])
                    for i, cls_id in enumerate(class_ids) if cls_id == 0]

    if len(person_boxes) == 0:
        cleanup_person_states(set())
        return result_img

    # ===== Plan B核心: 全图一次 MediaPipe 调用 =====
    t_mp = time.time()
    all_poses = detect_pose_all(img, int(time.time() * 1000))
    t_mp = (time.time() - t_mp) * 1000

    # 匹配姿态到YOLO框
    matched_pairs = match_poses_to_boxes(all_poses, person_boxes)

    # 只取已匹配到YOLO框的姿态（跳过未匹配的，防止MediaPipe误检产生"幽灵人"）
    # box_idx -> landmarks（一个框最多一个姿态，选置信度最高的）
    box_pose_map = {}  # box_idx -> landmarks (取最先匹配的)
    for landmarks, box_idx in matched_pairs:
        if box_idx >= 0 and box_idx not in box_pose_map:
            box_pose_map[box_idx] = landmarks

    # 限制人数（优先大面积框）
    sorted_box_indices = sorted(box_pose_map.keys(),
                                key=lambda j: person_boxes[j][2] * person_boxes[j][3],
                                reverse=True)
    if len(sorted_box_indices) > MAX_PERSONS:
        sorted_box_indices = sorted_box_indices[:MAX_PERSONS]

    # IoU追踪: 用匹配到的YOLO框作为输入
    track_boxes = [person_boxes[j] for j in sorted_box_indices]
    tracked = person_tracker.update(track_boxes)

    # 构建 pid -> landmarks 映射（用box值字典反查，避免zip错位）
    box_to_landmarks = {}
    for j in sorted_box_indices:
        box_to_landmarks[person_boxes[j]] = box_pose_map[j]

    pid_to_landmarks = {}
    for pid, tbbox in tracked.items():
        if tbbox in box_to_landmarks:
            pid_to_landmarks[pid] = box_to_landmarks[tbbox]

    # 用跟踪器内部ID（含暂时消失的），而非仅当前帧可见ID
    active_ids = set(person_tracker.tracked.keys())
    cleanup_person_states(active_ids)

    all_landmarks_3d = {}
    pid_actions = {}

    for pid, landmarks_2d in pid_to_landmarks.items():
        landmarks_3d = compute_3d_landmarks(landmarks_2d, depth_data, fx, fy, cx, cy)
        action_text = recognize_action(landmarks_2d, pid)
        pid_actions[pid] = action_text

        result_img = draw_skeleton_multi(
            result_img, landmarks_2d, landmarks_3d, depth_data,
            fx, fy, cx, cy, action_text, pid
        )
        all_landmarks_3d[pid] = landmarks_3d

    global _profile_count
    _profile_count += 1
    if _profile_count % 30 == 0:
        print(f"[计时 Plan B] MP(全图):{t_mp:.0f}ms  "
              f"检测:{len(all_poses)}人  追踪:{len(all_landmarks_3d)}人")

    # Open3D
    if vis_3d is not None and all_landmarks_3d:
        t_o3d = time.time()
        update_3d_skeleton_multi(vis_3d, all_landmarks_3d)
        t_o3d = (time.time() - t_o3d) * 1000
        if _profile_count % 30 == 0:
            print(f"[计时] O3D(多人):{t_o3d:.0f}ms")

    # bbox→pid 反查
    bbox_to_pid = {bbox: pid for pid, bbox in tracked.items()}

    # 绘制所有YOLO检测框（只画person，非人类跳过）
    for i in range(len(boxes)):
        if class_ids[i] != 0:
            continue
        left, top, width, height = boxes[i]
        right = min(left + width, depth_data.shape[1])
        bottom = min(top + height, depth_data.shape[0])
        bbox_key = (left, top, width, height)

        # 判断是否为已追踪的person框
        if bbox_key in bbox_to_pid:
            pid = bbox_to_pid[bbox_key]
            box_color, _, _, _ = get_person_colors(pid)
            action_text = pid_actions.get(pid, "")
            action_cn = ACTION_CN.get(action_text, action_text)
            label = f"人员{pid}: {action_cn}"

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
            put_chinese_text(result_img, label, (lx, ly - th), 22, box_color)
        else:
            box_color = PALETTE[class_ids[i] % len(PALETTE)]

        cv2.rectangle(result_img, (left, top), (left + width, top + height), box_color, 2)

        # 非person框保留原来的小标签
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

            label = f"{classes[class_ids[i]]}:{confidences[i]:.2f}"
            label_x = max(2, min(left + 2, img.shape[1] - 100 - 2))
            label_y = max(2 + 35, min(top + height - 5, img.shape[0] - 2))
            draw_label_cn(result_img, label, label_x, label_y - 35, box_color, font_size=14, extra_line=depth_label)

    put_chinese_text(result_img, f"检测人数: {len(all_landmarks_3d)}  |  Plan B 模式",
                (10, 60), 18, GREEN)

    return result_img


# ========== [15] 相机配置 ==========

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


# ========== [16] 主程序入口 ==========

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='多人人体姿态3D分析系统 (Plan B)')
    parser.add_argument('--color_width', type=int, default=None)
    parser.add_argument('--color_height', type=int, default=None)
    parser.add_argument('--depth_width', type=int, default=None)
    parser.add_argument('--depth_height', type=int, default=None)
    parser.add_argument('--no-pose', action='store_true')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu'])
    parser.add_argument('--max-persons', type=int, default=MAX_PERSONS,
                        help=f"最大处理人数 (默认: {MAX_PERSONS})")
    args = parser.parse_args()

    _max_persons_default = MAX_PERSONS
    MAX_PERSONS = args.max_persons

    # 若命令行指定了不同人数，重新初始化MediaPipe
    if args.max_persons != _max_persons_default:
        if MEDIAPIPE_AVAILABLE:
            pose_landmarker.close()
            MEDIAPIPE_AVAILABLE = init_mediapipe(MAX_PERSONS)

    try:
        with open('coco.names', 'rt') as f:
            classes = f.read().strip().split('\n')
        print(f"[信息] 加载了 {len(classes)} 个类别")
    except FileNotFoundError:
        print("[错误] 未找到 coco.names 文件")
        classes = []

    try:
        prefer_gpu = args.device != 'cpu'
        providers = get_onnx_providers(prefer_gpu=prefer_gpu)
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession('models/yolo26n.onnx',
                                            sess_options=sess_options,
                                            providers=providers)
        input_name = ort_session.get_inputs()[0].name
        actual_provider = ort_session.get_providers()[0]
        print(f"[信息] YOLO26n ONNX 模型加载成功 (设备: {actual_provider})")
    except Exception as e:
        print(f"[错误] YOLO模型加载失败: {e}")
        exit(1)

    if args.no_pose:
        MEDIAPIPE_AVAILABLE = False
        print("[信息] 已禁用MediaPipe姿态检测（--no-pose）")
    elif not MEDIAPIPE_AVAILABLE:
        print("[警告] MediaPipe未安装，姿态检测功能不可用")

    pipeline = Pipeline()
    config = get_sw_align_config(pipeline, args.color_width, args.color_height,
                                  args.depth_width, args.depth_height)
    if config is None:
        print("[错误] 未找到合适的流配置")
        exit(1)

    pipeline.start(config)
    print("[信息] 相机已启动")

    align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

    camera_param = pipeline.get_camera_param()
    fx = camera_param.rgb_intrinsic.fx
    fy = camera_param.rgb_intrinsic.fy
    cx = camera_param.rgb_intrinsic.cx
    cy = camera_param.rgb_intrinsic.cy
    print(f"[信息] 相机内参: fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")

    vis_3d = None
    _o3d_enabled = False

    def create_o3d_window():
        v = o3d.visualization.Visualizer()
        v.create_window(window_name="3D Skeleton (Multi-Person Plan B)", width=800, height=600)
        vc = v.get_view_control()
        vc.set_front([0, 0, -1])
        vc.set_up([0, 1, 0])
        ro = v.get_render_option()
        ro.line_width = 5.0
        ro.point_size = 8.0
        return v

    person_tracker = PersonTracker(max_disappeared=10, min_iou=0.1)
    prev_time = time.time()
    frame_count = 0

    print("\n" + "=" * 50)
    print(f"多人人体姿态3D分析系统已启动 (Plan B, Max={MAX_PERSONS})")
    print("按 ESC 或 Q 退出  |  按 J 切换 3D 骨架")
    print("=" * 50 + "\n")

    cv2.namedWindow('多人姿态3D分析 (Plan B)', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('多人姿态3D分析 (Plan B)',
                          cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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

        t_pre0 = time.time()
        input_tensor = pre_process(img_bgr)
        t_pre = (time.time() - t_pre0) * 1000
        t0 = time.time()
        outputs = ort_session.run(None, {input_name: input_tensor})
        t_yolo = (time.time() - t0) * 1000

        t1 = time.time()
        result = post_process_with_pose(
            img_bgr, depth_frame, outputs,
            fx, fy, cx, cy, person_tracker,
            vis_3d=vis_3d
        )
        t_post = (time.time() - t1) * 1000

        curr_time = time.time()
        frame_time_ms = (curr_time - prev_time) * 1000
        fps = 1000.0 / frame_time_ms if frame_time_ms > 0 else 0
        prev_time = curr_time
        frame_count += 1

        img_w = result.shape[1]
        put_chinese_text(result, f"帧率:{fps:.1f}", (img_w - 180, 18), 16, RED)
        put_chinese_text(result, f"预处理:{t_pre:.0f}ms  YOLO:{t_yolo:.0f}ms", (img_w - 260, 38), 14, RED)
        put_chinese_text(result, f"后处理:{t_post:.0f}ms", (img_w - 180, 56), 14, RED)
        o3d_status = "开启" if _o3d_enabled else "关闭"
        o3d_color = GREEN if _o3d_enabled else (100, 100, 100)
        put_chinese_text(result, f"3D骨架:{o3d_status} (J)", (img_w - 180, 74), 14, o3d_color)

        cv2.imshow('多人姿态3D分析 (Plan B)', result)

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

    if vis_3d is not None:
        vis_3d.destroy_window()
    cv2.destroyAllWindows()
    pipeline.stop()

    if MEDIAPIPE_AVAILABLE:
        pose_landmarker.close()

    print("[信息] 程序已安全退出")
