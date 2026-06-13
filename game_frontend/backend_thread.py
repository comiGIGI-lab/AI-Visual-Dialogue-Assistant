"""后端工作线程 — 相机采集 + YOLO + MediaPipe + 游戏状态机

Phase 流程:
  loading   → 相机启动 → AI 模型加载
  countdown → 3 秒倒计时
  playing   → 单人游戏 / 多人对战
  game_over → 游戏结束
"""
import time
import threading
import numpy as np
import cv2
import onnxruntime as ort
import sys
import os
import collections

_parent_dir = os.path.dirname(os.path.dirname(__file__))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from PySide6.QtCore import QThread, Signal

# ---------- 线程安全的帧缓冲区 ----------
_frame_lock = threading.Lock()
_frame_buffer = None

# 独立 raw 帧缓冲区：相机直出，不经过 AI 处理
_raw_lock = threading.Lock()
_raw_buffer = None


def write_frame(frame: np.ndarray):
    global _frame_buffer
    with _frame_lock:
        _frame_buffer = frame.copy()


def read_frame():
    with _frame_lock:
        if _frame_buffer is None:
            return None
        return _frame_buffer.copy()


def write_raw_frame(frame: np.ndarray):
    global _raw_buffer
    with _raw_lock:
        _raw_buffer = frame.copy()


def read_raw_frame():
    with _raw_lock:
        if _raw_buffer is None:
            return None
        return _raw_buffer.copy()


def _import_game_utils():
    import game_demo_p1 as g
    return g


# ---------- 多人骨架绘制 ----------

PERSON_COLORS = [
    ((255, 255, 0), (0, 255, 255), (255, 255, 255)),     # P0 黄/青
    ((255, 0, 255), (128, 0, 255), (200, 200, 200)),      # P1 紫
    ((0, 255, 0), (0, 200, 0), (180, 255, 180)),          # P2 绿
]

SKELETON_CONNECTIONS = [
    (11, 13, 'left'), (13, 15, 'left'),
    (12, 14, 'right'), (14, 16, 'right'),
    (11, 12, 'torso'), (11, 23, 'torso'), (12, 24, 'torso'), (23, 24, 'torso'),
    (23, 25, 'left'), (25, 27, 'left'),
    (24, 26, 'right'), (26, 28, 'right'),
    (11, 0, 'torso'), (12, 0, 'torso'),
    (-1, -1, 'spine'),
]


def _get_person_colors(pid):
    left, right, spine = PERSON_COLORS[pid % len(PERSON_COLORS)]
    return left, right, spine


def draw_skeleton_multi(img, landmarks_2d, pid, gray=False):
    """绘制单人骨架，支持灰色（已淘汰）"""
    if landmarks_2d is None or len(landmarks_2d) < 33:
        return
    left_c, right_c, spine_c = _get_person_colors(pid)

    for px, py, vis in landmarks_2d:
        if gray:
            c = (100, 100, 100) if vis > 0.5 else (60, 60, 60)
        else:
            c = (0, 255, 0) if vis > 0.5 else (0, 0, 255)
        cv2.circle(img, (px, py), 3, c, -1)

    for s, e, cn in SKELETON_CONNECTIONS:
        if s == -1 and e == -1:
            if all(landmarks_2d[i][2] > 0.5 for i in [11, 12, 23, 24]):
                scx = (landmarks_2d[11][0] + landmarks_2d[12][0]) // 2
                scy = (landmarks_2d[11][1] + landmarks_2d[12][1]) // 2
                hcx = (landmarks_2d[23][0] + landmarks_2d[24][0]) // 2
                hcy = (landmarks_2d[23][1] + landmarks_2d[24][1]) // 2
                c = (100, 100, 100) if gray else spine_c
                cv2.line(img, (scx, scy), (hcx, hcy), c, 3)
            continue
        if (s < len(landmarks_2d) and e < len(landmarks_2d)
                and landmarks_2d[s][2] > 0.5 and landmarks_2d[e][2] > 0.5):
            if gray:
                c = (100, 100, 100)
            elif cn == 'left':
                c = left_c
            elif cn == 'right':
                c = right_c
            else:
                c = spine_c
            cv2.line(img, (landmarks_2d[s][0], landmarks_2d[s][1]),
                     (landmarks_2d[e][0], landmarks_2d[e][1]), c, 2)


class PersonTracker:
    """基于IoU的轻量级跨帧人员追踪"""

    def __init__(self, max_disappeared=10, min_iou=0.1):
        self.next_id = 0
        self.tracked = {}
        self.max_disappeared = max_disappeared
        self.min_iou = min_iou

    def update(self, boxes):
        if len(boxes) == 0:
            for pid in list(self.tracked):
                self.tracked[pid]['disappeared'] += 1
                if self.tracked[pid]['disappeared'] > self.max_disappeared:
                    del self.tracked[pid]
            return {}

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


class BackendThread(QThread):
    status_updated = Signal(dict)

    def __init__(self, mode='normal', sound_enabled=True, parent=None):
        super().__init__(parent)
        self.mode = mode
        self.sound_enabled = sound_enabled
        self._running = False

    def _emit_loading(self, message, percent):
        self.status_updated.emit({
            'phase': 'loading',
            'loading_message': message,
            'loading_percent': percent,
        })

    def run(self):
        self._running = True

        # ═══════════ Phase: loading — 启动相机 ═══════════
        self._emit_loading('正在启动相机...', 5)

        from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, AlignFilter, OBStreamType

        try:
            pipeline = Pipeline()
            config = Config()
            color_profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
            depth_profiles = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            config.enable_stream(color_profiles.get_default_video_stream_profile())
            config.enable_stream(depth_profiles.get_default_video_stream_profile())
            pipeline.start(config)
            cam_w = color_profiles.get_default_video_stream_profile().get_width()
            cam_h = color_profiles.get_default_video_stream_profile().get_height()
            print(f"[后端] 相机已启动 ({cam_w}x{cam_h})")
        except Exception as e:
            print(f"[错误] 相机启动失败: {e}")
            self._emit_loading(f'相机启动失败: {e}', -1)
            return

        align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
        camera_param = pipeline.get_camera_param()
        fx = camera_param.rgb_intrinsic.fx
        fy = camera_param.rgb_intrinsic.fy
        cx = camera_param.rgb_intrinsic.cx
        cy = camera_param.rgb_intrinsic.cy

        g = _import_game_utils()

        # 预推帧
        self._emit_loading('正在获取画面...', 15)
        for _ in range(10):
            try:
                frames = pipeline.wait_for_frames(200)
                if frames:
                    frames = align_filter.process(frames)
                    if frames:
                        fs = frames.as_frame_set()
                        cf = fs.get_color_frame()
                        if cf:
                            img = g.frame_to_bgr_image(cf)
                            if img is not None:
                                write_frame(img)
            except Exception:
                pass

        # ═══════════ Phase: loading — 加载 MediaPipe ═══════════
        self._emit_loading('正在加载 MediaPipe 姿态模型...', 30)

        try:
            import mediapipe as mp
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe import Image as MPImage, ImageFormat
        except ImportError:
            print("[错误] MediaPipe 未安装")
            self._emit_loading('MediaPipe 未安装', -1)
            return

        mp_model = 'models/pose_landmarker_lite.task'
        if not os.path.exists(mp_model):
            print(f"[错误] 未找到 MediaPipe 模型: {mp_model}")
            self._emit_loading('模型文件缺失', -1)
            return

        base_opt = BaseOptions(model_asset_path=mp_model)
        mp_opts = vision.PoseLandmarkerOptions(
            base_options=base_opt,
            running_mode=vision.RunningMode.VIDEO,
            num_poses=3,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        pose_landmarker = vision.PoseLandmarker.create_from_options(mp_opts)
        print("[后端] MediaPipe 初始化成功 (num_poses=3)")

        # ═══════════ Phase: loading — 加载 ONNX ═══════════
        self._emit_loading('正在加载 YOLO 检测模型...', 55)

        providers = g.get_onnx_providers(prefer_gpu=True)
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            ort_session = ort.InferenceSession('models/yolo26n.onnx',
                                               sess_options=sess_opt,
                                               providers=providers)
            input_name = ort_session.get_inputs()[0].name
            device_name = ort_session.get_providers()[0]
            print(f"[后端] YOLO 模型加载成功 (设备: {device_name})")
            self._emit_loading(f'YOLO 已就绪 ({device_name})', 80)
        except Exception as e:
            print(f"[错误] YOLO 模型加载失败: {e}")
            self._emit_loading(f'模型加载失败: {e}', -1)
            return

        # ═══════════ 初始化游戏组件 ═══════════
        self._emit_loading('正在准备游戏...', 90)

        g._standing_ref = {'hip_y': None, 'spine': None, 'frames': 0}
        g._jump_tracker = collections.deque(maxlen=12)

        from game_demo_p1 import GameState
        game = GameState(mode=self.mode)
        game.sound_enabled = self.sound_enabled
        action_counter = {}
        prev_time = time.time()
        fps_deque = collections.deque(maxlen=60)

        # 多人追踪器
        person_tracker = PersonTracker()

        # 每人独立的动作状态
        _person_states = {}

        def _get_person_state(pid):
            if pid not in _person_states:
                _person_states[pid] = {
                    'standing_ref': {'hip_y': None, 'spine': None, 'frames': 0},
                    'jump_tracker': collections.deque(maxlen=12),
                    'action_counter': {}
                }
            return _person_states[pid]

        # ═══════════ AI 处理：多人版 ═══════════
        def run_ai_pipeline_multi(img_bgr, max_persons=3):
            """YOLO + MediaPipe 多人管线
            单人模式 max_persons=1 只处理最大的人，避免浪费 GPU
            返回: (all_actions, all_landmarks, result_img)
            """
            blob = g.pre_process(img_bgr)
            outputs = ort_session.run(None, {input_name: blob})
            predictions = np.squeeze(outputs[0])

            img_h, img_w = img_bgr.shape[:2]
            boxes = []
            for det in predictions:
                x1, y1, x2, y2, conf, cls_id = det[:6]
                if conf < g.CONFIDENCE_THRESHOLD or int(cls_id) != 0:
                    continue
                left = int(x1 * img_w / g.INPUT_WIDTH)
                top = int(y1 * img_h / g.INPUT_HEIGHT)
                right = int(x2 * img_w / g.INPUT_WIDTH)
                bottom = int(y2 * img_h / g.INPUT_HEIGHT)
                boxes.append([left, top, right - left, bottom - top])

            # 按面积排序，取前 N 个
            boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
            boxes = boxes[:max_persons]

            # 追踪分配稳定ID
            matched = person_tracker.update(boxes)  # {pid: box}

            all_actions = {}
            all_landmarks = {}

            for pid, box in matched.items():
                left, top, w, h = box
                margin = int(0.1 * max(w, h))
                bx, by = max(0, left - margin), max(0, top - margin)
                bw = min(w + 2 * margin, img_w - bx)
                bh = min(h + 2 * margin, img_h - by)

                if bw <= 0 or bh <= 0:
                    continue

                roi = img_bgr[by:by + bh, bx:bx + bw]
                roi = cv2.resize(roi, (256, 256))
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                mp_img = MPImage(image_format=ImageFormat.SRGB, data=roi_rgb)
                mp_result = pose_landmarker.detect_for_video(mp_img, int(time.time() * 1000))

                if not mp_result.pose_landmarks:
                    continue

                landmarks_2d = []
                for lm in mp_result.pose_landmarks[0]:
                    gx = int(lm.x * bw) + bx
                    gy = int(lm.y * bh) + by
                    vis = lm.visibility if lm.visibility else (lm.presence if lm.presence else 1.0)
                    landmarks_2d.append((gx, gy, vis))

                all_landmarks[pid] = landmarks_2d
                state = _get_person_state(pid)
                action_text, state['action_counter'] = g.recognize_action(
                    landmarks_2d, state['action_counter'])
                all_actions[pid] = action_text

            # 清理离开的人
            active_ids = set(matched.keys())
            for pid in list(_person_states):
                if pid not in active_ids and pid not in person_tracker.tracked:
                    del _person_states[pid]

            # 绘制
            result = img_bgr.copy()
            for pid, landmarks_2d in all_landmarks.items():
                draw_skeleton_multi(result, landmarks_2d, pid)
                # 标签
                if landmarks_2d and len(landmarks_2d) >= 33:
                    head_x = int(sum(landmarks_2d[i][0] for i in [0, 11, 12] if landmarks_2d[i][2] > 0.5) / max(1, sum(1 for i in [0, 11, 12] if landmarks_2d[i][2] > 0.5)))
                    head_y = min(lm[1] for lm in landmarks_2d[:11] if lm[2] > 0.5)
                    action = all_actions.get(pid, '?')
                    label = f"P{pid}: {action}"
                    cv2.putText(result, label, (head_x - 40, head_y - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            return all_actions, all_landmarks, result

        # ═══════════ Phase: waiting_for_hands ═══════════
        print(f"[后端] 游戏就绪 难度: {self.mode.upper()}")
        phase = 'waiting_for_hands'
        countdown_start = 0
        hands_window = collections.deque(maxlen=30)
        HANDS_RATIO = 0.6

        # 单人等待举手用 counter
        wait_counter = {}

        # 简化的单人管线（waiting 阶段用）
        def run_single_for_hands(img_bgr):
            blob = g.pre_process(img_bgr)
            outputs = ort_session.run(None, {input_name: blob})
            predictions = np.squeeze(outputs[0])
            img_h, img_w = img_bgr.shape[:2]
            boxes = []
            for det in predictions:
                x1, y1, x2, y2, conf, cls_id = det[:6]
                if conf < g.CONFIDENCE_THRESHOLD or int(cls_id) != 0:
                    continue
                left = int(x1 * img_w / g.INPUT_WIDTH)
                top = int(y1 * img_h / g.INPUT_HEIGHT)
                right = int(x2 * img_w / g.INPUT_WIDTH)
                bottom = int(y2 * img_h / g.INPUT_HEIGHT)
                boxes.append([left, top, right - left, bottom - top])

            action_text = "STANDING"
            landmarks_2d = None

            if boxes:
                largest = max(boxes, key=lambda b: b[2] * b[3])
                left, top, w, h = largest
                margin = int(0.1 * max(w, h))
                bx, by, bw, bh = max(0, left - margin), max(0, top - margin), w + 2 * margin, h + 2 * margin
                bw = min(bw, img_w - bx)
                bh = min(bh, img_h - by)
                if bw > 0 and bh > 0:
                    roi = img_bgr[by:by + bh, bx:bx + bw]
                    roi = cv2.resize(roi, (256, 256))
                    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                    mp_img = MPImage(image_format=ImageFormat.SRGB, data=roi_rgb)
                    mp_result = pose_landmarker.detect_for_video(mp_img, int(time.time() * 1000))
                    if mp_result.pose_landmarks and len(mp_result.pose_landmarks) > 0:
                        landmarks_2d = []
                        for lm in mp_result.pose_landmarks[0]:
                            gx = int(lm.x * bw) + bx
                            gy = int(lm.y * bh) + by
                            vis = lm.visibility if lm.visibility else (lm.presence if lm.presence else 1.0)
                            landmarks_2d.append((gx, gy, vis))

            if landmarks_2d:
                action_text, _ = g.recognize_action(landmarks_2d, wait_counter)

            result = img_bgr.copy()
            if landmarks_2d:
                draw_skeleton_multi(result, landmarks_2d, 0)

            return action_text, result

        while self._running:
            try:
                frames = pipeline.wait_for_frames(100)
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

                img_bgr = g.frame_to_bgr_image(color_frame)
                if img_bgr is None:
                    continue

                if phase in ('waiting_for_hands', 'countdown'):
                    action_text, result = run_single_for_hands(img_bgr)
                    hands_up = (action_text == 'RAISING_BOTH_HANDS')
                    hands_window.append(hands_up)
                    wr = sum(hands_window) / len(hands_window) if hands_window else 0

                    write_frame(result)

                    if phase == 'waiting_for_hands':
                        if wr >= HANDS_RATIO and len(hands_window) >= 5:
                            phase = 'countdown'
                            countdown_start = time.time()
                            print(f"[后端] 检测到举手 (比例={wr:.0%})，开始倒计时")
                        else:
                            self.status_updated.emit({
                                'phase': 'waiting_for_hands',
                                'mode': self.mode,
                            })
                            continue

                    if phase == 'countdown':
                        elapsed = time.time() - countdown_start
                        cd_remain = max(0, 3 - int(elapsed))
                        if cd_remain > 0:
                            self.status_updated.emit({
                                'phase': 'countdown',
                                'countdown': cd_remain,
                                'mode': self.mode,
                            })
                            continue
                        else:
                            phase = 'playing'
                            prev_time = time.time()
                            fps_deque.clear()
                            action_counter = {}
                            print("[后端] 倒计时结束，游戏开始!")
                            continue

                # ═══════════ Phase: playing ═══════════
                curr_time = time.time()
                dt = curr_time - prev_time
                prev_time = curr_time
                if dt > 0 and dt < 1.0:
                    fps_deque.append(1.0 / dt)
                fps = sum(fps_deque) / len(fps_deque) if fps_deque else 0
                dt = min(dt, 0.2)

                all_actions, all_landmarks, result = run_ai_pipeline_multi(img_bgr, max_persons=1)

                # 取第一个人的动作
                pids = sorted(all_actions.keys())
                action_text = all_actions.get(pids[0], 'STANDING') if pids else 'STANDING'
                if all_landmarks:
                    if not game.game_over:
                        if game.check_answer(action_text):
                            game.on_correct()
                        elif action_text != "STANDING":
                            game.on_wrong(action_text)
                if not game.game_over:
                    game.update(dt)

                write_frame(result)

                status = {
                    'phase': 'game_over' if game.game_over else 'playing',
                    'fps': round(fps, 1),
                    'score': game.score,
                    'combo': game.combo,
                    'time_left': game.time_left,
                    'mode': game.mode,
                    'action_text': action_text,
                    'prompt_text': game.display_text,
                    'prompt_color': game.prompt_color,
                    'prompt_type': game.prompt_type,
                    'game_over': game.game_over,
                    'high_score': game.high_score,
                }

                self.status_updated.emit(status)

            except Exception as e:
                print(f"[后端] 循环异常: {e}")
                import traceback
                traceback.print_exc()
                continue

        # ---- 清理 ----
        try:
            pipeline.stop()
            pose_landmarker.close()
        except Exception:
            pass
        print("[后端] 线程已退出")

    def stop(self):
        self._running = False
        self.quit()
        if not self.wait(3000):
            self.terminate()
            self.wait()
