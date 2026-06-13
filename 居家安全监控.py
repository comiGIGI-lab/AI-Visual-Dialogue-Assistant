# ============================================================================
# 居家安全监控系统 — 跌倒检测
# ============================================================================
# 使用 Orbbec 3D 相机 + YOLOv11 + MediaPipe 进行人体姿态分析，
# 当检测到有人跌倒时发出报警声。
#
# 前置条件：
#   - pip install mediapipe opencv-python numpy onnxruntime pyorbbecsdk
#   - models/yolo11n.onnx / models/pose_landmarker_lite.task / coco.names
#   - Orbbec 3D 相机已连接
# ============================================================================

import cv2
import time
import argparse
import threading
import numpy as np
import onnxruntime as ort
from PIL import Image as PILImage, ImageDraw, ImageFont
from pyorbbecsdk import *

import sys
import os
import collections
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from utils import frame_to_bgr_image

# ========== [1] 全局参数 ==========
ESC_KEY = 27
INPUT_WIDTH, INPUT_HEIGHT = 640, 640
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
BLACK = (0, 0, 0)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)

# ========== [2] 跌倒检测参数 ==========
# 躯干角度阈值：肩-臀连线与水平线的夹角小于此值认为是躺倒
TORSO_ANGLE_THRESHOLD = 35  # 度
# 身体宽高比阈值：bbox宽/高大于此值认为是躺倒
BODY_RATIO_THRESHOLD = 1.2
# 髋部下降速率阈值（像素/秒），突然下降认为是跌倒
HIP_DROP_SPEED_THRESHOLD = 200
# 连续确认帧数：连续多少帧检测到跌倒才报警
FALL_CONFIRM_FRAMES = 8
# 报警冷却时间（秒）：报警后多久不再重复报警
ALARM_COOLDOWN = 10
# 多人跟踪的最大人数
MAX_TRACKED_PERSONS = 5

# ========== [3] 中文字体 ==========
_FONT_CACHE = {}

def _get_font(size):
    if size not in _FONT_CACHE:
        try:
            _FONT_CACHE[size] = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", size)
        except Exception:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]

def put_chinese_text(img, text, position, font_size, color, anchor='lt'):
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

# ========== [4] ONNX 推理辅助 ==========
def get_onnx_providers(prefer_gpu=True):
    available = ort.get_available_providers()
    gpu_providers = [
        ('CUDAExecutionProvider', 'CUDA'),
        ('DmlExecutionProvider', 'DirectML'),
    ]
    for provider, name in gpu_providers:
        if provider in available:
            print(f"[信息] 使用 {name} GPU 加速推理")
            return [provider, 'CPUExecutionProvider']
    print("[信息] 使用 CPU 推理")
    return ['CPUExecutionProvider']

def pre_process(img):
    blob = cv2.resize(img, (INPUT_WIDTH, INPUT_HEIGHT))
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB)
    blob = blob.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]
    return blob

# ========== [5] MediaPipe 初始化 ==========
try:
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    from mediapipe import Image as MPImage, ImageFormat
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    print("[错误] 未安装 mediapipe")
    MEDIAPIPE_AVAILABLE = False

pose_landmarker = None
_mp_timestamp_counter = 0  # 每次调用 detect_for_video 递增

if MEDIAPIPE_AVAILABLE:
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'pose_landmarker_lite.task')
    if os.path.exists(model_path):
        base_options = BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=MAX_TRACKED_PERSONS,
            min_pose_detection_confidence=0.3,
            min_pose_presence_confidence=0.3,
            min_tracking_confidence=0.3,
            output_segmentation_masks=False
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(options)
        print("[信息] MediaPipe 初始化成功")
    else:
        print(f"[错误] 未找到模型: {model_path}")
        MEDIAPIPE_AVAILABLE = False

# ========== [6] 报警系统 ==========
class AlarmSystem:
    def __init__(self):
        self._active = False
        self._last_alarm_time = 0
        self._thread = None

    def trigger(self):
        now = time.time()
        if now - self._last_alarm_time < ALARM_COOLDOWN:
            return
        if self._active:
            return
        self._active = True
        self._last_alarm_time = now
        self._thread = threading.Thread(target=self._beep_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._active = False

    def _beep_loop(self):
        import winsound
        for _ in range(6):
            if not self._active:
                break
            winsound.Beep(1000, 300)
            time.sleep(0.1)
        self._active = False

# ========== [7] 跌倒检测逻辑 ==========
class FallDetector:
    def __init__(self):
        self.person_states = {}  # person_id -> state dict
        self.alarm = AlarmSystem()
        self.alarm_active = False

    def analyze_person(self, landmarks_2d, bbox):
        """分析单个人的跌倒状态。返回 (is_fall, reason)"""
        if landmarks_2d is None or len(landmarks_2d) < 33:
            return False, ""

        ls = landmarks_2d[11]  # 左肩
        rs = landmarks_2d[12]  # 右肩
        lh = landmarks_2d[23]  # 左臀
        rh = landmarks_2d[24]  # 右臀
        lk = landmarks_2d[25]  # 左膝
        rk = landmarks_2d[26]  # 右膝
        la = landmarks_2d[27]  # 左踝
        ra = landmarks_2d[28]  # 右踝
        nose = landmarks_2d[0] # 鼻子

        # 检查关键点可见性
        key_points = [ls, rs, lh, rh]
        if any(p[2] < 0.3 for p in key_points):
            return False, ""

        # --- 判据1: 躯干角度 ---
        shoulder_cx = (ls[0] + rs[0]) / 2
        shoulder_cy = (ls[1] + rs[1]) / 2
        hip_cx = (lh[0] + rh[0]) / 2
        hip_cy = (lh[1] + rh[1]) / 2

        dx = hip_cx - shoulder_cx
        dy = hip_cy - shoulder_cy
        # 与水平线的夹角
        if abs(dx) < 1 and abs(dy) < 1:
            torso_angle = 90
        else:
            torso_angle = abs(np.degrees(np.arctan2(abs(dy), abs(dx))))

        # 躯干接近水平（角度小）→ 可能躺倒
        torso_horizontal = torso_angle < TORSO_ANGLE_THRESHOLD

        # --- 判据2: 身体宽高比 ---
        left, top, w, h = bbox
        if h > 0:
            ratio = w / h
        else:
            ratio = 0
        body_wide = ratio > BODY_RATIO_THRESHOLD

        # --- 判据3: 头部位置 ---
        # 正常站立时头部在上方，跌倒时头部位置降低
        head_low = False
        if nose[2] > 0.3:
            # 头部y坐标大于身体bbox中心 → 头部在下方
            head_low = nose[1] > (top + h * 0.6)

        # --- 综合判断 ---
        fall_signals = sum([torso_horizontal, body_wide, head_low])

        if fall_signals >= 2:
            reason = []
            if torso_horizontal:
                reason.append(f"躯干角度{torso_angle:.0f}度")
            if body_wide:
                reason.append(f"宽高比{ratio:.1f}")
            if head_low:
                reason.append("头部位置低")
            return True, "、".join(reason)

        return False, ""

    def update(self, persons_data, frame_time):
        """更新所有人的状态。persons_data: [(landmarks_2d, bbox), ...]"""
        current_ids = set()
        any_fall = False

        for i, (landmarks, bbox) in enumerate(persons_data):
            pid = i  # 简化：用索引作为ID
            current_ids.add(pid)

            is_fall, reason = self.analyze_person(landmarks, bbox)

            if pid not in self.person_states:
                self.person_states[pid] = {
                    'fall_frames': 0,
                    'is_fall': False,
                    'reason': '',
                    'fall_start_time': 0,
                }

            state = self.person_states[pid]

            if is_fall:
                state['fall_frames'] += 1
                state['reason'] = reason
                if state['fall_frames'] >= FALL_CONFIRM_FRAMES and not state['is_fall']:
                    state['is_fall'] = True
                    state['fall_start_time'] = frame_time
            else:
                state['fall_frames'] = max(0, state['fall_frames'] - 2)
                if state['fall_frames'] == 0:
                    state['is_fall'] = False
                    state['reason'] = ''

            if state['is_fall']:
                any_fall = True

        # 清理消失的人
        for pid in list(self.person_states.keys()):
            if pid not in current_ids:
                del self.person_states[pid]

        # 报警控制
        if any_fall:
            self.alarm.trigger()
            self.alarm_active = True
        else:
            self.alarm.stop()
            self.alarm_active = False

        return any_fall

    def get_fall_info(self, pid):
        if pid in self.person_states:
            return self.person_states[pid]
        return None

# ========== [8] YOLO 后处理 + 姿态检测 ==========
def detect_and_analyze(img, outs, fall_detector, frame_time):
    predictions = np.squeeze(outs[0])
    if predictions.shape[0] < predictions.shape[1]:
        predictions = predictions.T

    img_h, img_w = img.shape[:2]
    boxes, confidences = [], []

    for det in predictions:
        bbox = det[:4]
        class_scores = det[4:]
        conf = np.max(class_scores)
        cls_id = np.argmax(class_scores)

        if conf < CONFIDENCE_THRESHOLD or int(cls_id) != 0:
            continue

        cx, cy, w, h = bbox
        left = int((cx - w / 2) * img_w / INPUT_WIDTH)
        top = int((cy - h / 2) * img_h / INPUT_HEIGHT)
        right = int((cx + w / 2) * img_w / INPUT_WIDTH)
        bottom = int((cy + h / 2) * img_h / INPUT_HEIGHT)
        boxes.append([left, top, right - left, bottom - top])
        confidences.append(float(conf))

    result_img = img.copy()

    if not boxes:
        fall_detector.update([], frame_time)
        return result_img

    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    if len(indices) == 0:
        fall_detector.update([], frame_time)
        return result_img

    if hasattr(indices, 'flatten'):
        indices = indices.flatten()

    # 收集所有人的姿态数据
    persons_data = []
    person_bboxes = []

    for idx in indices:
        left, top, w, h = boxes[idx]
        margin = int(0.05 * max(w, h))
        bbox = (left - margin, top - margin, w + 2 * margin, h + 2 * margin)

        # MediaPipe 姿态检测
        landmarks_2d = None
        if MEDIAPIPE_AVAILABLE and pose_landmarker:
            x, y, bw, bh = bbox
            x = max(0, x); y = max(0, y)
            bw = min(bw, img_w - x); bh = min(bh, img_h - y)
            if bw > 0 and bh > 0:
                person_roi = img[y:y+bh, x:x+bw]
                person_roi = cv2.resize(person_roi, (256, 256))
                roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)
                mp_image = MPImage(image_format=ImageFormat.SRGB, data=roi_rgb)
                global _mp_timestamp_counter
                _mp_timestamp_counter += 1
                result = pose_landmarker.detect_for_video(mp_image, _mp_timestamp_counter)
                if result.pose_landmarks and len(result.pose_landmarks) > 0:
                    lm = result.pose_landmarks[0]
                    landmarks_2d = []
                    for pt in lm:
                        gx = int(pt.x * bw) + x
                        gy = int(pt.y * bh) + y
                        vis = pt.visibility if pt.visibility else 0.5
                        landmarks_2d.append((gx, gy, vis))

        persons_data.append((landmarks_2d, bbox))
        person_bboxes.append(bbox)

    # 跌倒分析
    any_fall = fall_detector.update(persons_data, frame_time)

    # 绘制结果
    for i, ((landmarks, bbox), b) in enumerate(zip(persons_data, person_bboxes)):
        left, top, w, h = b
        info = fall_detector.get_fall_info(i)
        is_fall = info['is_fall'] if info else False

        color = RED if is_fall else GREEN
        label = "跌倒!" if is_fall else "正常"
        label_color = RED if is_fall else GREEN

        # 画bbox
        cv2.rectangle(result_img, (left, top), (left + w, top + h), color, 2)

        # 标签背景（用PIL测量中文宽度）
        font = _get_font(18)
        pil_tmp = PILImage.new('RGB', (1, 1))
        draw_tmp = ImageDraw.Draw(pil_tmp)
        bbox_t = draw_tmp.textbbox((0, 0), label, font=font)
        lw, lh = bbox_t[2] - bbox_t[0], bbox_t[3] - bbox_t[1]
        cv2.rectangle(result_img, (left, top - lh - 8), (left + lw + 10, top), color, cv2.FILLED)
        put_chinese_text(result_img, label, (left + 5, top - lh - 4), 18, BLACK)

        # 画骨架
        if landmarks:
            _draw_skeleton(result_img, landmarks, is_fall)

        # 跌倒原因
        if is_fall and info['reason']:
            put_chinese_text(result_img, info['reason'],
                           (left, top + h + 15), 14, RED)

    return result_img

def _draw_skeleton(image, landmarks, is_fall):
    color = RED if is_fall else GREEN
    connections = [
        (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 12), (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (24, 26), (26, 28),
    ]
    for s, e in connections:
        if (s < len(landmarks) and e < len(landmarks) and
                landmarks[s][2] > 0.3 and landmarks[e][2] > 0.3):
            cv2.line(image, (landmarks[s][0], landmarks[s][1]),
                    (landmarks[e][0], landmarks[e][1]), color, 2)
    for pt in landmarks:
        if pt[2] > 0.3:
            cv2.circle(image, (pt[0], pt[1]), 3, color, -1)

# ========== [9] UI 绘制 ==========
def draw_monitor_ui(img, fall_detector, fps, person_count):
    ih, iw = img.shape[:2]

    # 顶部状态栏
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (iw, 50), BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)

    # 标题
    put_chinese_text(img, "居家安全监控", (15, 12), 24, WHITE)

    # FPS
    put_chinese_text(img, f"FPS: {fps:.0f}", (iw - 150, 15), 18, CYAN)

    # 人数
    put_chinese_text(img, f"检测人数: {person_count}", (iw - 300, 15), 18, GREEN)

    # 时间
    time_str = time.strftime("%Y-%m-%d %H:%M:%S")
    put_chinese_text(img, time_str, (iw // 2 - 80, 15), 18, WHITE)

    # 报警状态
    if fall_detector.alarm_active:
        # 闪烁红色边框
        if int(time.time() * 4) % 2 == 0:
            cv2.rectangle(img, (0, 0), (iw - 1, ih - 1), RED, 6)
        # 报警文字
        put_chinese_text(img, "警告: 检测到跌倒!", (iw // 2, 70), 36, RED, anchor='mt')

        # 显示跌倒详情
        y_off = 110
        for pid, state in fall_detector.person_states.items():
            if state['is_fall']:
                elapsed = time.time() - state['fall_start_time']
                put_chinese_text(img, f"人员{pid+1}: 跌倒已持续 {elapsed:.0f} 秒",
                               (iw // 2, y_off), 22, YELLOW, anchor='mt')
                y_off += 30
    else:
        put_chinese_text(img, "监控正常 - 未检测到异常", (iw // 2, 70), 22, GREEN, anchor='mt')

    # 底部提示
    put_chinese_text(img, "按 ESC 退出  |  按 S 静音/取消静音  |  按 R 清除报警",
                    (15, ih - 30), 16, (150, 150, 150))

# ========== [10] 相机配置 ==========
def get_camera_config(pipeline):
    config = Config()
    try:
        color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        color_profile = color_profiles.get_default_video_stream_profile()
        depth_profile = depth_profiles.get_default_video_stream_profile()
        config.enable_stream(color_profile)
        config.enable_stream(depth_profile)
        print(f"[配置] 彩色: {color_profile.get_width()}x{color_profile.get_height()}")
        print(f"[配置] 深度: {depth_profile.get_width()}x{depth_profile.get_height()}")
    except Exception as e:
        print(f"[错误] 相机配置失败: {e}")
        return None
    return config

# ========== [11] 主程序 ==========
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='居家安全监控 - 跌倒检测')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'gpu', 'cpu'])
    parser.add_argument('--skip', type=int, default=10, help="每隔几帧推理一次（默认10，即每11帧推理1次）")
    parser.add_argument('--threshold', type=int, default=TORSO_ANGLE_THRESHOLD,
                        help="跌倒检测灵敏度（角度，越小越灵敏）")
    args = parser.parse_args()

    # 更新阈值
    TORSO_ANGLE_THRESHOLD_ACTUAL = args.threshold

    # 加载模型
    yolo_path = os.path.join(os.path.dirname(__file__), 'models', 'yolo11n.onnx')
    try:
        providers = get_onnx_providers(prefer_gpu=args.device != 'cpu')
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        ort_session = ort.InferenceSession(yolo_path, sess_options=opts, providers=providers)
        input_name = ort_session.get_inputs()[0].name
        print(f"[信息] YOLOv11n 加载成功 (设备: {ort_session.get_providers()[0]})")
    except Exception as e:
        print(f"[错误] 模型加载失败: {e}")
        exit(1)

    if not MEDIAPIPE_AVAILABLE:
        print("[错误] MediaPipe 不可用，无法进行姿态检测")
        exit(1)

    # 相机
    pipeline = Pipeline()
    config = get_camera_config(pipeline)
    if config is None:
        exit(1)
    pipeline.start(config)
    print("[信息] 相机已启动")

    # 初始化
    fall_detector = FallDetector()
    fall_detector.alarm._active = False  # 确保初始不报警

    frame_count = 0
    last_outputs = None
    fps = 0
    fps_timer = time.time()
    fps_count = 0
    sound_muted = False

    WINDOW_NAME = '居家安全监控'
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("\n" + "=" * 50)
    print("居家安全监控系统已启动")
    print("ESC=退出  S=静音  R=清除报警")
    print("=" * 50 + "\n")

    try:
        while True:
            frames = pipeline.wait_for_frames(1000)
            if not frames:
                continue
            frames = frames.as_frame_set()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            img_bgr = frame_to_bgr_image(color_frame)
            if img_bgr is None:
                continue

            # FPS 计算
            fps_count += 1
            if time.time() - fps_timer >= 1.0:
                fps = fps_count / (time.time() - fps_timer)
                fps_count = 0
                fps_timer = time.time()

            # YOLO 推理（跳帧）
            frame_count += 1
            if frame_count % (args.skip + 1) == 1 or last_outputs is None:
                input_tensor = pre_process(img_bgr)
                last_outputs = ort_session.run(None, {input_name: input_tensor})
            outputs = last_outputs

            frame_time = time.time()

            # 检测 + 跌倒分析
            result = detect_and_analyze(img_bgr, outputs, fall_detector, frame_time)

            # 统计人数
            person_count = len(fall_detector.person_states)

            # 绘制 UI
            draw_monitor_ui(result, fall_detector, fps, person_count)

            cv2.imshow(WINDOW_NAME, result)

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ESC_KEY:
                break
            elif key == ord('s') or key == ord('S'):
                sound_muted = not sound_muted
                if sound_muted:
                    fall_detector.alarm.stop()
                    fall_detector.alarm.trigger = lambda: None  # 禁用报警
                    print("[信息] 已静音")
                else:
                    fall_detector.alarm.trigger = lambda: fall_detector.alarm.__class__.trigger(fall_detector.alarm)
                    print("[信息] 已取消静音")
            elif key == ord('r') or key == ord('R'):
                fall_detector.alarm.stop()
                fall_detector.alarm_active = False
                for state in fall_detector.person_states.values():
                    state['is_fall'] = False
                    state['fall_frames'] = 0
                print("[信息] 已清除报警")

    except KeyboardInterrupt:
        print("\n[信息] 用户中断")
    finally:
        fall_detector.alarm.stop()
        cv2.destroyAllWindows()
        pipeline.stop()
        if pose_landmarker:
            pose_landmarker.close()
        print("[信息] 监控系统已退出")
