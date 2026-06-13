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
# 动作模仿挑战游戏 — Game Demo P1 (单人模式)
# ============================================================================
# 系统随机出题（绿色=正向 红色=反向），用户做出对应动作，做对得分。
# 支持三种难度（练习/普通/困难），30s倒计时，计分板，音效反馈。
#
# 前置条件：
#   - pip install mediapipe opencv-python numpy onnxruntime pyorbbecsdk
#   - models/yolo26n.onnx / models/pose_landmarker_lite.task / coco.names
#   - Orbbec 3D相机已连接
# ============================================================================

import cv2
import time
import argparse
import random
import winsound
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

# ========== [1] 全局参数 ==========
COLOR_CAMERA_WIDTH = None
COLOR_CAMERA_HEIGHT = None
DEPTH_CAMERA_WIDTH = None
DEPTH_CAMERA_HEIGHT = None
ESC_KEY = 27

INPUT_WIDTH, INPUT_HEIGHT = 640, 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.5
MAX_DISPLAY_BOXES = 5

MIN_DEPTH, MAX_DEPTH = 20, 10000
DEPTH_SAMPLE_SIZE = 5

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
BLACK, RED, WHITE, GREEN = (0, 0, 0), (0, 0, 255), (255, 255, 255), (0, 255, 0)
YELLOW, CYAN, MAGENTA = (0, 255, 255), (255, 255, 0), (255, 0, 255)
ORANGE = (0, 165, 255)

# ========== [2] 游戏参数 ==========
GAME_ROUND_SECONDS = 30                  # 每轮时长
DIFFICULTY_CONFIG = {
    'practice': {'interval': 5.0,  'reverse_prob': 0.2},
    'normal':   {'interval': 3.0,  'reverse_prob': 0.4},
    'hard':     {'interval': 2.0,  'reverse_prob': 0.5},
}
SCORE_FORWARD = 10
SCORE_REVERSE = 20
SCORE_COMBO_BONUS = 5

# 菜单状态
MENU_MAIN = 0
MENU_DIFFICULTY = 1
MENU_SETTINGS = 2
MENU_LEADERBOARD = 3
MENU_PLAYING = 4
LEADERBOARD_FILE = 'game_leaderboard.json'

# ========== [3] 动作映射表 ==========
# 正向: 指令名 → (显示名, 期望动作标签)
FORWARD_MAP = {
    'BOTH_HANDS': ('举起双手!', 'RAISING_BOTH_HANDS'),
    'LEFT_HAND':  ('举起左手!', 'RAISING_LEFT_HAND'),
    'RIGHT_HAND': ('举起右手!', 'RAISING_RIGHT_HAND'),
    'SQUAT':      ('蹲下!',     'SQUATTING'),
    'LEFT_LEG':   ('抬起左腿!',  'RAISING_LEFT_LEG'),
    'RIGHT_LEG':  ('抬起右腿!',  'RAISING_RIGHT_LEG'),
}

# 反向: 指令名 → (显示名, 期望动作标签)
REVERSE_MAP = {
    'LEFT_HAND':  ('反向:举起左手!', 'RAISING_RIGHT_HAND'),
    'RIGHT_HAND': ('反向:举起右手!', 'RAISING_LEFT_HAND'),
    'LEFT_LEG':   ('反向:抬起左腿!', 'RAISING_RIGHT_LEG'),
    'RIGHT_LEG':  ('反向:抬起右腿!', 'RAISING_LEFT_LEG'),
    'STAND':      ('反向:站立!',     'SQUATTING'),
}

GREEN_FOR_FORWARD = (0, 220, 50)
RED_FOR_REVERSE = (50, 50, 255)

# ========== [4] MediaPipe 初始化 ==========
try:
    import mediapipe as mp
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe import Image as MPImage, ImageFormat
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[错误] 未检测到 mediapipe，请先安装: pip install mediapipe")
    MEDIAPIPE_AVAILABLE = False

if MEDIAPIPE_AVAILABLE:
    model_path = 'models/pose_landmarker_lite.task'
    if not os.path.exists(model_path):
        print(f"[错误] 未找到 MediaPipe 模型: {model_path}")
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

# ========== [5] 骨架配置 ==========
SKELETON_CONNECTIONS = [
    (11, 13, 'left'), (13, 15, 'left'),
    (12, 14, 'right'), (14, 16, 'right'),
    (11, 12, 'torso'), (11, 23, 'torso'), (12, 24, 'torso'), (23, 24, 'torso'),
    (23, 25, 'left'), (25, 27, 'left'),
    (24, 26, 'right'), (26, 28, 'right'),
    (11, 0, 'torso'), (12, 0, 'torso'),
    (-1, -1, 'spine'),
]
SKELETON_COLORS = {
    'left': (255, 255, 0), 'right': (0, 255, 255),
    'torso': (255, 255, 255), 'spine': (255, 255, 255),
}
SKELETON_COLORS_3D = {
    'left': (0/255, 255/255, 255/255), 'right': (255/255, 0/255, 255/255),
    'torso': (255/255, 255/255, 255/255), 'spine': (255/255, 255/255, 255/255),
}
JOINT_RADIUS = 3

# ========== [6] 动作识别参数 ==========
RAISE_THRESHOLD = 0.05
BEND_SPINE_THRESHOLD = 0.15
BEND_TILT_THRESHOLD = 0.08
SQUAT_HIP_DROP_RATIO = 0.07
JUMP_UP_RATIO = 0.05
JUMP_DOWN_RATIO = 0.03
LEG_RAISE_KNEE_UP_RATIO = 0.05
LEG_RAISE_LEVEL_RATIO = 0.05
ACTION_CONFIRM_FRAMES = 4

# ========== [7] 辅助函数 ==========
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
    lower, upper = median * (1 - threshold), median * (1 + threshold)
    return depth_values[(depth_values >= lower) & (depth_values <= upper)]

# 中文字体（OpenCV putText 不支持中文，需用 PIL）
_FONT_CACHE = {}
def _get_font(size):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]

def put_chinese_text(img, text, position, font_size, color, anchor='lt'):
    """用PIL在OpenCV图像上绘制中文文本。anchor: lt=左上, mt=中上"""
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
    draw.text((x, y), text, font=font, fill=color[::-1])  # BGR→RGB
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    img[:] = result

# ========== [8] 动作识别（来自v26，不变） ==========
_standing_ref = {'hip_y': None, 'spine': None, 'frames': 0}
_jump_tracker = collections.deque(maxlen=12)

def recognize_action(landmarks_2d, action_counter):
    global _standing_ref, _jump_tracker
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return "STANDING", action_counter

    ls = landmarks_2d[11]; rs = landmarks_2d[12]
    le = landmarks_2d[13]; re = landmarks_2d[14]
    lw = landmarks_2d[15]; rw = landmarks_2d[16]
    lh = landmarks_2d[23]; rh = landmarks_2d[24]
    lk = landmarks_2d[25]; rk = landmarks_2d[26]
    la = landmarks_2d[27]; ra = landmarks_2d[28]

    if ls[2] < 0.5 or rs[2] < 0.5 or lw[2] < 0.5 or rw[2] < 0.5:
        return "STANDING", action_counter

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
        _jump_tracker.append((shoulder_center_y, hip_center_y))
        if len(_jump_tracker) >= 8:
            sh_vals = [s for s, _ in _jump_tracker]
            hip_vals = [h for _, h in _jump_tracker]
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
                _jump_tracker.clear()

    # [3] 抬腿
    if detected_action == "STANDING":
        raise_thr = spine_2d * LEG_RAISE_KNEE_UP_RATIO
        level_thr = spine_2d * LEG_RAISE_LEVEL_RATIO
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
        shoulder_tilt = abs(ls[1] - rs[1])
        if lh[2] > 0.5 and rh[2] > 0.5:
            if spine_2d < BEND_SPINE_THRESHOLD * 480 or shoulder_tilt > BEND_TILT_THRESHOLD * 480:
                detected_action = "BENDING"

    # [5] 蹲下
    if detected_action == "STANDING" and lh[2] > 0.5 and rh[2] > 0.5:
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
        return best_action, action_counter
    else:
        return "STANDING", action_counter


# ========== [9] 游戏状态管理 ==========
class GameState:
    def __init__(self, mode='practice'):
        self.mode = mode
        cfg = DIFFICULTY_CONFIG[mode]
        self.prompt_interval = cfg['interval']
        self.reverse_prob = cfg['reverse_prob']

        self.score = 0
        self.combo = 0
        self.high_score = 0
        self.time_left = GAME_ROUND_SECONDS
        self.game_over = False

        # 当前出题
        self.prompt_name = None       # 指令名
        self.display_text = ""        # 显示文字
        self.expected_action = None   # 期望动作标签
        self.prompt_type = 'forward'  # 'forward' | 'reverse'
        self.prompt_color = GREEN_FOR_FORWARD
        self.prompt_timer = 0.0       # 距上次出题的时间

        # 动作确认防抖: 用户保持正确动作N帧才算完成
        self._correct_frames = 0
        self._correct_needed = 2      # 需保持约0.07s (30fps)
        self._last_wrong = None       # 错误动作防抖（同一错误不重复触发）
        self._saved = False           # 排行榜是否已保存

        # 出题历史 (避免连续重复)
        self._last_prompts = collections.deque(maxlen=3)

        # 音效开关
        self.sound_enabled = True

        # 得分飘字特效
        self._feedbacks = []  # [{'text': str, 'color': tuple, 'life': float, 'max_life': float}]

        # 生成第一题
        self._pick_new_prompt()

    def _pick_new_prompt(self):
        """随机出题，保证不与最近3次重复"""
        # 出题池: 正向动作 + 反向专属 STAND
        pool = list(FORWARD_MAP.keys()) + ['STAND']
        candidates = [a for a in pool if a not in self._last_prompts]
        if not candidates:
            candidates = pool

        action_name = random.choice(candidates)
        # STAND 只为反向存在（说站立做蹲下）
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
        """每帧调用，更新计时和出题"""
        if self.game_over:
            return

        self.time_left -= dt
        self.prompt_timer += dt

        if self.time_left <= 0:
            self.time_left = 0
            self.game_over = True
            return

        # 更新飘字特效
        for fb in self._feedbacks[:]:
            fb['life'] -= dt
            if fb['life'] <= 0:
                self._feedbacks.remove(fb)

        # 超时未完成 → 换题
        if self.prompt_timer >= self.prompt_interval:
            self.combo = 0
            self._pick_new_prompt()

    def check_answer(self, user_action):
        """检查用户动作是否匹配期望。返回 True/False"""
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
        """做对后：加分、连击、出新题、音效、飘字"""
        self.combo += 1
        points = SCORE_REVERSE if self.prompt_type == 'reverse' else SCORE_FORWARD
        combo_bonus = min(self.combo * 10, 50)
        points += combo_bonus

        self.score += points
        if self.score > self.high_score:
            self.high_score = self.score

        # 飘字特效
        lines = [f"+{points}"]
        if self.combo >= 2:
            lines.append(f"COMBO x{self.combo}!")
        self._feedbacks.append({
            'text': '  '.join(lines),
            'color': YELLOW if self.combo < 2 else (0, 255, 0) if self.combo < 4 else (0, 165, 255),
            'life': 1.5,
            'max_life': 1.5,
        })

        # 音效
        if self.sound_enabled:
            if self.combo >= 3:
                winsound.Beep(1200, 80)
            winsound.Beep(880, 100)
            winsound.Beep(1100, 120)

        self._pick_new_prompt()

    def on_wrong(self, action_text=None):
        """做错：中断连击（带防抖，同一错误动作不重复触发）"""
        if action_text == "STANDING" or action_text is None:
            self._last_wrong = None
            return
        if action_text == self._last_wrong:
            return
        self._last_wrong = action_text
        self.combo = 0
        if self.sound_enabled:
            winsound.Beep(300, 150)

    def reset(self, mode=None):
        """重置游戏"""
        if mode:
            self.mode = mode
            cfg = DIFFICULTY_CONFIG[mode]
            self.prompt_interval = cfg['interval']
            self.reverse_prob = cfg['reverse_prob']
        self.score = 0
        self.combo = 0
        self.time_left = GAME_ROUND_SECONDS
        self.game_over = False
        self._last_prompts.clear()
        self._correct_frames = 0
        self._feedbacks.clear()
        self._saved = False
        self._pick_new_prompt()


# ========== [10] 排行榜 ==========
def load_leaderboard():
    import json
    try:
        with open(LEADERBOARD_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def save_leaderboard(entries):
    import json
    with open(LEADERBOARD_FILE, 'w') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

# ========== [11] 菜单系统 ==========
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
        # 圆角矩形按钮背景
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), cv2.FILLED)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        # 文字
        put_chinese_text(img, self.text, (x + w // 2, y + h // 2 - font_size // 2), font_size, color, anchor='mt')

class Menu:
    def __init__(self):
        self.state = MENU_MAIN
        self.difficulty = 'normal'
        self.sound_enabled = True
        self.buttons = []
        self.build_buttons(1280, 720)
        self.mx, self.my = -1, -1
        self.clicked = False

    def set_state(self, state, iw=1280, ih=720):
        self.state = state
        self.build_buttons(iw, ih)

    def build_buttons(self, iw, ih):
        """根据当前菜单状态和窗口大小构建按钮"""
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
                ('练习 (Practice)', 'diff_practice'),
                ('普通 (Normal)', 'diff_normal'),
                ('困难 (Hard)', 'diff_hard'),
                ('返回', 'back'),
            ]
        elif self.state == MENU_SETTINGS:
            sound_label = '音效: ON' if self.sound_enabled else '音效: OFF'
            items = [
                (sound_label, 'toggle_sound'),
                ('返回', 'back'),
            ]
        elif self.state == MENU_LEADERBOARD:
            items = [
                ('返回', 'back'),
            ]
        else:
            return

        start_y = ih // 2 - (len(items) * (bh + 12)) // 2
        for i, (text, action) in enumerate(items):
            btn = MenuButton(text, cx, start_y + i * (bh + 12), bw, bh, action=action)
            self.buttons.append(btn)

    def handle_click(self):
        for btn in self.buttons:
            if btn.hovered and btn.action:
                action = btn.action
                if action == 'start':
                    self.state = MENU_PLAYING
                elif action == 'difficulty':
                    self.state = MENU_DIFFICULTY
                elif action == 'settings':
                    self.state = MENU_SETTINGS
                elif action == 'leaderboard':
                    self.state = MENU_LEADERBOARD
                elif action == 'exit':
                    return 'quit'
                elif action == 'back':
                    self.state = MENU_MAIN
                elif action == 'toggle_sound':
                    self.sound_enabled = not self.sound_enabled
                    self.state = MENU_MAIN
                elif action.startswith('diff_'):
                    self.difficulty = action.replace('diff_', '')
                    self.state = MENU_MAIN
                # 状态变更后重建按钮
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

        # 标题
        title = '动作模仿挑战'
        put_chinese_text(img, title, (iw // 2, ih // 4 - 40), 48, WHITE, anchor='mt')

        # 按钮
        for btn in self.buttons:
            btn.draw(img)

        # 底部信息
        if self.state == MENU_MAIN:
            info = f'当前难度: {self.difficulty.upper()}  |  音效: {"ON" if self.sound_enabled else "OFF"}'
            put_chinese_text(img, info, (iw // 2, ih - 60), 20, (150, 150, 150), anchor='mt')
        elif self.state == MENU_LEADERBOARD:
            entries = load_leaderboard()
            y = ih // 2 - 40
            if entries:
                put_chinese_text(img, '排名  分数  难度  日期', (iw // 2, y), 22, YELLOW, anchor='mt')
                for j, e in enumerate(entries[:8]):
                    y += 32
                    line = f"{j+1}.  {e['score']}  {e['difficulty']}  {e.get('date','')}"
                    put_chinese_text(img, line, (iw // 2, y), 20, WHITE, anchor='mt')
            else:
                put_chinese_text(img, '暂无记录', (iw // 2, y + 20), 24, (120, 120, 120), anchor='mt')


# ========== [12] Game UI 绘制 ==========
def draw_game_ui(img, game):
    """在画面上叠加游戏UI：提示文字、计分板、倒计时、模式"""

    if game.game_over:
        draw_game_over(img, game)
        return

    ih, iw = img.shape[:2]

    # ---- 顶部居中: 出题提示 ----
    if game.display_text:
        font_size = 42
        font = _get_font(font_size)
        # 用PIL测量文字尺寸
        pil_tmp = PILImage.new('RGB', (1, 1))
        draw_tmp = ImageDraw.Draw(pil_tmp)
        bbox = draw_tmp.textbbox((0, 0), game.display_text, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = (iw - tw) // 2
        ty = 8

        # 半透明背景条
        overlay = img.copy()
        bar_h = th + 24
        cv2.rectangle(overlay, (tx - 20, 0), (tx + tw + 20, bar_h), BLACK, cv2.FILLED)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

        put_chinese_text(img, game.display_text, (tx, ty), font_size, game.prompt_color)

    # ---- 右上角: 计分板 ----
    score_lines = [
        f"Score: {game.score}",
        f"Combo: x{game.combo}" if game.combo > 0 else "",
        f"Time: {game.time_left:.0f}s",
    ]
    y_off = 30
    # 倒计时颜色
    time_color = GREEN if game.time_left > 10 else (YELLOW if game.time_left > 5 else RED)
    colors = [GREEN, CYAN, time_color]
    for i, line in enumerate(score_lines):
        if not line:
            continue
        ts, bs = cv2.getTextSize(line, FONT_FACE, 0.6, 2)
        tw, th = ts
        # 右对齐
        rx = iw - tw - 15
        ry = y_off + i * 30
        cv2.putText(img, line, (rx, ry), FONT_FACE, 0.6, colors[i], 2, cv2.LINE_AA)

    # ---- 屏幕中央: 得分飘字 ----
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
        # 向上飘动
        ty -= int((1 - alpha) * 40)
        put_chinese_text(img, fb['text'], (tx, ty), font_size, color)

    # ---- 底部: 模式 + 进度条 ----
    progress = game.time_left / GAME_ROUND_SECONDS
    bar_w = int((iw - 60) * progress)
    bar_y = ih - 30
    cv2.rectangle(img, (30, bar_y), (iw - 30, bar_y + 12), (60, 60, 60), cv2.FILLED)
    cv2.rectangle(img, (30, bar_y), (30 + bar_w, bar_y + 12),
                 GREEN if progress > 0.3 else YELLOW if progress > 0.1 else RED, cv2.FILLED)

    mode_text = f"Mode: {game.mode.upper()}"
    cv2.putText(img, mode_text, (30, bar_y - 6), FONT_FACE, 0.5, WHITE, 1, cv2.LINE_AA)
    cv2.putText(img, "[R] Restart  [1/2/3] Difficulty",
                (iw - 280, bar_y - 6), FONT_FACE, 0.4, (150, 150, 150), 1, cv2.LINE_AA)


def draw_game_over(img, game):
    """全屏游戏结束画面"""
    img[:] = (20, 20, 20)

    ih, iw = img.shape[:2]
    cx, cy = iw // 2, ih // 2

    lines = [
        ("GAME OVER", 2.0, 4, RED),
        ("", 0, 0, WHITE),
        (f"Final Score: {game.score}", 1.2, 3, WHITE),
        (f"Best Score:  {game.high_score}", 0.8, 2, YELLOW),
        (f"Difficulty: {game.mode.upper()}", 0.7, 2, (180, 180, 180)),
        ("", 0, 0, WHITE),
        ("Press R to Restart", 1.0, 3, GREEN),
    ]

    total_h = 0
    spacing = 50
    for text, scale, thick, color in lines:
        if text:
            ts, _ = cv2.getTextSize(text, FONT_FACE, scale, thick)
            total_h += ts[1] + spacing // 2
        else:
            total_h += 15

    start_y = cy - total_h // 2
    current_y = start_y

    for text, scale, thick, color in lines:
        if not text:
            current_y += 15
            continue
        ts, _ = cv2.getTextSize(text, FONT_FACE, scale, thick)
        tw, th = ts
        tx = cx - tw // 2
        cv2.putText(img, text, (tx, current_y + th), FONT_FACE, scale, color, thick, cv2.LINE_AA)
        current_y += th + spacing


# ========== [11] 骨架绘制（简化版，用于游戏） ==========
def draw_skeleton(image, landmarks_2d, landmarks_3d, depth_data, fx, fy, cx, cy):
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return image

    # 关键点
    for idx, (px, py, vis) in enumerate(landmarks_2d):
        if vis < 0.5:
            cv2.circle(image, (px, py), JOINT_RADIUS, RED, -1)
            continue
        if landmarks_3d[idx] is not None:
            cv2.circle(image, (px, py), JOINT_RADIUS, GREEN, -1)
        else:
            cv2.circle(image, (px, py), JOINT_RADIUS, RED, -1)

    # 骨架连线
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


# ========== [12] 3D坐标 ==========
def get_keypoint_depth(depth_data, x, y, kernel_size=5):
    h, w = depth_data.shape
    half = kernel_size // 2
    y1, y2 = max(0, y - half), min(h, y + half + 1)
    x1, x2 = max(0, x - half), min(w, x + half + 1)
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
    return ((u - cx) * depth / fx, (v - cy) * depth / fy, depth)

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


# ========== [13] MediaPipe 姿态检测 ==========
def detect_pose(image, bbox, timestamp_ms=0):
    if not MEDIAPIPE_AVAILABLE:
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


# ========== [14] Open3D 3D骨架 ==========
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


# ========== [15] YOLO后处理 + 姿态 + 游戏逻辑 ==========
def post_process_with_pose(img, depth_frame, outs, fx, fy, cx, cy,
                           action_counter, game, vis_3d=None):
    predictions = np.squeeze(outs[0])
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

    if len(boxes) == 0:
        return result_img, action_counter

    person_indices = [i for i, cls in enumerate(class_ids) if cls == 0]

    if person_indices and MEDIAPIPE_AVAILABLE:
        largest_idx = max(person_indices, key=lambda i: boxes[i][2] * boxes[i][3])
        left, top, width, height = boxes[largest_idx]
        margin = int(0.1 * max(width, height))
        bbox = (left - margin, top - margin, width + 2*margin, height + 2*margin)

        landmarks_2d = detect_pose(img, bbox, int(time.time() * 1000))

        if landmarks_2d:
            landmarks_3d = compute_3d_landmarks(landmarks_2d, depth_data, fx, fy, cx, cy)
            action_text, action_counter = recognize_action(landmarks_2d, action_counter)

            # ═══════ 游戏逻辑 ═══════
            if not game.game_over:
                if game.check_answer(action_text):
                    game.on_correct()
                elif action_text != "STANDING":
                    game.on_wrong(action_text)

            result_img = draw_skeleton(
                result_img, landmarks_2d, landmarks_3d, depth_data,
                fx, fy, cx, cy
            )

            if vis_3d is not None and landmarks_3d:
                update_3d_skeleton(vis_3d, landmarks_3d,
                                   SKELETON_CONNECTIONS, SKELETON_COLORS_3D)

    return result_img, action_counter


# ========== [16] 相机配置 ==========
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


# ========== [17] 主程序 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='动作模仿挑战游戏 (Game Demo P1)')
    parser.add_argument('--mode', type=str, default='normal',
                        choices=['practice', 'normal', 'hard'],
                        help="游戏难度: practice / normal / hard")
    parser.add_argument('--no-sound', action='store_true', help="禁用音效")
    parser.add_argument('--no-pose', action='store_true', help="禁用MediaPipe姿态检测")
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'gpu', 'cpu'])
    parser.add_argument('--color_width', type=int, default=None)
    parser.add_argument('--color_height', type=int, default=None)
    parser.add_argument('--depth_width', type=int, default=None)
    parser.add_argument('--depth_height', type=int, default=None)
    args = parser.parse_args()

    # 加载类别
    try:
        with open('coco.names', 'rt') as f:
            classes = f.read().strip().split('\n')
        print(f"[信息] 加载了 {len(classes)} 个类别")
    except FileNotFoundError:
        print("[错误] 未找到 coco.names 文件")
        classes = []

    # ONNX Runtime
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
    elif not MEDIAPIPE_AVAILABLE:
        print("[警告] MediaPipe未安装，姿态检测功能不可用")

    # 相机
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

    # Open3D
    vis_3d = None
    _o3d_enabled = False

    def create_o3d_window():
        v = o3d.visualization.Visualizer()
        v.create_window(window_name="3D Skeleton", width=800, height=600)
        vc = v.get_view_control()
        vc.set_front([0, 0, -1])
        vc.set_up([0, 1, 0])
        ro = v.get_render_option()
        ro.line_width = 5.0
        ro.point_size = 8.0
        return v

    # 菜单系统
    menu = Menu()
    menu.difficulty = args.mode
    if args.no_sound:
        menu.sound_enabled = False

    # 游戏状态（延迟创建）
    game = None
    action_counter = {}
    prev_time = time.time()

    print("\n" + "=" * 50)
    print("动作模仿挑战游戏 — Game Demo P1")
    print("鼠标点击菜单按钮操作")
    print("游戏中: ESC/Q=退出  R=重新开始  J=Open3D开关")
    print("=" * 50 + "\n")

    cv2.namedWindow('Game: Action Challenge', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('Game: Action Challenge', cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN)

    # 鼠标回调
    def on_mouse(event, x, y, flags, param):
        menu.update_hover(x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            menu.clicked = True

    cv2.setMouseCallback('Game: Action Challenge', on_mouse)

    while True:
        # ---- 菜单模式 ----
        if menu.state != MENU_PLAYING:
            menu_img = np.zeros((720, 1280, 3), dtype=np.uint8)
            menu_img[:] = (20, 20, 25)
            menu.draw(menu_img)

            if menu.clicked:
                result = menu.handle_click()
                if result == 'quit':
                    print("\n[信息] 用户退出")
                    break
                if menu.state == MENU_PLAYING:
                    game = GameState(mode=menu.difficulty)
                    game.sound_enabled = menu.sound_enabled
                    action_counter = {}
                    prev_time = time.time()
                menu.clicked = False

            cv2.imshow('Game: Action Challenge', menu_img)
            key = cv2.waitKey(30) & 0xFF
            if key in (ESC_KEY, ord('q'), ord('Q')):
                print("\n[信息] 用户退出")
                break
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

        # YOLO
        input_tensor = pre_process(img_bgr)
        outputs = ort_session.run(None, {input_name: input_tensor})

        # 后处理 + 姿态 + 游戏
        result, action_counter = post_process_with_pose(
            img_bgr, depth_frame, outputs,
            fx, fy, cx, cy, action_counter,
            game, vis_3d=vis_3d
        )

        # ---- 游戏时间更新 ----
        curr_time = time.time()
        dt = curr_time - prev_time
        prev_time = curr_time
        dt = min(dt, 0.2)

        if not game.game_over:
            game.update(dt)

        if game.game_over and not getattr(game, '_saved', False) and game.score > 0:
            # 保存分数到排行榜（仅一次）
            import datetime
            entries = load_leaderboard()
            entries.append({
                'score': game.score,
                'difficulty': game.mode.upper(),
                'date': datetime.date.today().isoformat(),
            })
            entries.sort(key=lambda x: x['score'], reverse=True)
            entries = entries[:20]
            save_leaderboard(entries)
            game._saved = True  # 标记已保存，防止重复

        # ---- 绘制游戏UI ----
        draw_game_ui(result, game)

        cv2.imshow('Game: Action Challenge', result)

        # ---- 按键处理 ----
        key = cv2.waitKey(1) & 0xFF
        if key in (ESC_KEY, ord('q'), ord('Q')):
            print("\n[信息] 返回菜单")
            menu.set_state(MENU_MAIN, result.shape[1], result.shape[0])
            game = None
        elif key == ord('r') or key == ord('R'):
            game.reset(game.mode)
            action_counter = {}
            print(f"[游戏] 重新开始! 模式: {game.mode.upper()}")
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
    cv2.destroyAllWindows()
    pipeline.stop()
    if MEDIAPIPE_AVAILABLE:
        pose_landmarker.close()
    print("[信息] 程序已安全退出")
