"""后端工作线程 — 相机采集 + YOLO + MediaPipe + 训练状态机

Phase 流程:
  loading   → 相机启动 → AI 模型加载
  countdown → 3 秒倒计时
  playing   → 放松训练进行中
  game_over → 训练完成
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
    ((255, 255, 0), (0, 255, 255), (255, 255, 255)),
    ((255, 0, 255), (128, 0, 255), (200, 200, 200)),
    ((0, 255, 0), (0, 200, 0), (180, 255, 180)),
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

    # 相机断联检测阈值：连续读取失败帧数 ≈ 3 秒（office / training 共用）
    CAMERA_FAIL_THRESHOLD = 30

    def __init__(self, mode='normal', sound_enabled=True, parent=None,
                 workout_mode='upper_body', run_mode='training'):
        super().__init__(parent)
        self.mode = mode
        self.sound_enabled = sound_enabled
        self.workout_mode = workout_mode  # 'upper_body' | 'full_body'
        self.run_mode = run_mode  # 'office' | 'training'
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

        from game_frontend.camera_provider import (
            create_camera_provider, WebcamCameraProvider,
            CameraUnavailableError,
        )

        provider = None
        fallback_reason = None
        error_message = None

        try:
            provider = create_camera_provider(mode='auto')
            provider.start()
        except CameraUnavailableError as e:
            # ── Orbbec 设备不可用 ──
            error_message = str(e)
            print(f"[后端] Orbbec 不可用: {error_message}")

            if provider is not None and provider.mode == 'orbbec':
                # 强制 orbbec 模式 — 不降级, 直接报错
                self._emit_loading(
                    f'未检测到 Orbbec 3D 相机\n\n'
                    f'{error_message}\n\n'
                    f'当前为强制 Orbbec 模式。\n'
                    f'可关闭程序后使用以下命令切换为普通摄像头兼容模式:\n\n'
                    f'PowerShell:\n'
                    f'  $env:OFFICEFIT_CAMERA="webcam"\n'
                    f'  python run_game_frontend.py\n\n'
                    f'或删除环境变量恢复自动模式:\n'
                    f'  Remove-Item Env:OFFICEFIT_CAMERA',
                    -1,
                )
                self.status_updated.emit({
                    'phase': 'error',
                    'camera_source': 'orbbec',
                    'depth_available': False,
                    'camera_mode_text': 'Orbbec 模式 — 设备不可用',
                    'fallback_reason': None,
                    'error_message': error_message,
                })
                return

            # auto 模式 — 降级到 webcam
            fallback_reason = (
                f"未检测到 Orbbec 3D 相机，已切换普通摄像头兼容模式"
            )
            print(f"[后端] {fallback_reason}")

            try:
                provider.stop()
            except Exception:
                pass

            try:
                provider = WebcamCameraProvider(camera_index=0)
                provider.mode = 'auto'
                provider.fallback_reason = fallback_reason
                provider.start()
            except RuntimeError as e2:
                error_message = (
                    f"Orbbec 不可用, webcam 降级也失败: {e2}"
                )
                print(f"[错误] {error_message}")
                self._emit_loading(error_message, -1)
                self.status_updated.emit({
                    'phase': 'error',
                    'camera_source': 'none',
                    'depth_available': False,
                    'camera_mode_text': '无可用相机',
                    'fallback_reason': fallback_reason,
                    'error_message': error_message,
                })
                return
        except RuntimeError as e:
            # ── 其他启动异常 ──
            error_message = str(e)
            print(f"[错误] 相机启动异常: {error_message}")

            if provider is not None and provider.mode == 'orbbec':
                self._emit_loading(
                    f'Orbbec 相机启动失败\n\n{error_message}\n\n'
                    f'当前为强制 Orbbec 模式。\n'
                    f'可关闭程序后使用以下命令切换:\n'
                    f'  $env:OFFICEFIT_CAMERA="webcam"\n'
                    f'  python run_game_frontend.py',
                    -1,
                )
                self.status_updated.emit({
                    'phase': 'error',
                    'camera_source': 'orbbec',
                    'depth_available': False,
                    'camera_mode_text': 'Orbbec 启动失败',
                    'fallback_reason': None,
                    'error_message': error_message,
                })
                return

            # auto 模式下也尝试降级
            fallback_reason = (
                f"Orbbec 启动失败: {error_message}"
                f" → 已降级到 USB 摄像头"
            )
            print(f"[后端] {fallback_reason}")
            try:
                provider.stop()
            except Exception:
                pass
            try:
                provider = WebcamCameraProvider(camera_index=0)
                provider.mode = 'auto'
                provider.fallback_reason = fallback_reason
                provider.start()
            except RuntimeError as e2:
                error_message = f"webcam 降级也失败: {e2}"
                print(f"[错误] {error_message}")
                self._emit_loading(error_message, -1)
                self.status_updated.emit({
                    'phase': 'error',
                    'camera_source': 'none',
                    'depth_available': False,
                    'camera_mode_text': '无可用相机',
                    'fallback_reason': fallback_reason,
                    'error_message': error_message,
                })
                return

        if not provider.is_opened():
            self._emit_loading('相机启动失败: 设备未能打开', -1)
            return

        cam_status = provider.get_status()
        camera_source = cam_status['source']
        depth_available = cam_status['depth_available']
        cam_w, cam_h = cam_status.get('resolution', (0, 0))

        if not fallback_reason:
            fallback_reason = cam_status.get('fallback_reason')

        camera_mode_text = (
            "Orbbec 3D 深度相机模式" if depth_available
            else "普通摄像头兼容模式（深度功能不可用）"
        )
        print(
            f"[后端] 相机已启动 (source={camera_source}, "
            f"{cam_w}x{cam_h}, depth={depth_available})"
        )
        if fallback_reason:
            print(f"[后端] 降级原因: {fallback_reason}")

        # 内参
        fx = fy = cx = cy = None
        if depth_available and cam_status.get('intrinsics'):
            intr = cam_status['intrinsics']
            fx, fy, cx, cy = intr['fx'], intr['fy'], intr['cx'], intr['cy']

        g = _import_game_utils()

        # 预推帧
        self._emit_loading('正在获取画面...', 15)
        warmup_ok = False
        for i in range(15):
            try:
                packet = provider.read(timeout_ms=300)
                if packet and packet.color_bgr is not None:
                    write_frame(packet.color_bgr)
                    warmup_ok = True
                    break
            except Exception:
                pass
            # 防止无限等待: 最多 15 次 × 300ms = 4.5s
        if not warmup_ok:
            self._emit_loading(
                '相机已连接但无法读取画面, 请检查相机是否被其他程序占用',
                -1,
            )
            try:
                provider.stop()
            except Exception:
                pass
            return

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

        # Pose 模型路径：环境变量 > 默认 lite
        mp_model = os.environ.get(
            'OFFICEFIT_POSE_MODEL',
            'models/pose_landmarker_lite.task',
        )
        if not os.path.exists(mp_model):
            print(f"[错误] 未找到 MediaPipe 模型: {mp_model}")
            print("[提示] 可设置环境变量 OFFICEFIT_POSE_MODEL 指定路径")
            print("[提示] 可用模型: lite / full / heavy")
            self._emit_loading(f'模型文件缺失: {mp_model}', -1)
            return
        # 模型级别显示名
        _pose_level = "custom"
        if "lite" in mp_model.lower():
            _pose_level = "lite"
        elif "full" in mp_model.lower():
            _pose_level = "full"
        elif "heavy" in mp_model.lower():
            _pose_level = "heavy"
        print(f"[后端] Pose 模型: {mp_model} ({_pose_level})")
        self._emit_loading(f'Pose 模型 ({_pose_level}): {os.path.basename(mp_model)}', 25)

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

        # ═══════════ 初始化训练组件 ═══════════
        game = None
        prev_time = time.time()
        fps_deque = collections.deque(maxlen=60)
        action_counter = {}
        person_tracker = PersonTracker()
        _person_states = {}
        office_start_time = time.time()

        # ── 相机断联检测计数（office / training 共用） ──
        _frame_fail_count = 0

        if self.run_mode == 'training':
            self._emit_loading('正在准备训练...', 90)
            g._standing_ref = {'hip_y': None, 'spine': None, 'frames': 0}
            g._jump_tracker = collections.deque(maxlen=12)
            from game_demo_p1 import GameState
            game = GameState(mode=self.mode, workout_mode=self.workout_mode)
            game.sound_enabled = self.sound_enabled

            def _get_person_state(pid):
                if pid not in _person_states:
                    _person_states[pid] = {
                        'standing_ref': {'hip_y': None, 'spine': None, 'frames': 0},
                        'jump_tracker': collections.deque(maxlen=12),
                        'action_counter': {}
                    }
                return _person_states[pid]
        else:
            self._emit_loading('进入办公观察模式...', 90)
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

            # 过滤太小的人（忽略背景远处的人）
            min_area = (img_w * img_h) * 0.02  # 至少占画面 2%
            boxes = [b for b in boxes if b[2] * b[3] >= min_area]
            # 按 (面积 × 中心权重) 排序 — 优先画面中央的大目标
            cx, cy = img_w / 2, img_h / 2
            def _score(b):
                area = b[2] * b[3]
                bx_c = b[0] + b[2] / 2
                by_c = b[1] + b[3] / 2
                dist = ((bx_c - cx) ** 2 + (by_c - cy) ** 2) ** 0.5
                center_weight = max(0, 1.0 - dist / max(img_w, img_h))
                return area * (0.6 + 0.4 * center_weight)
            boxes.sort(key=_score, reverse=True)
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
                # 标签：内部名 → 中文显示名，PIL 绘制（cv2.putText 不支持中文）
                if landmarks_2d and len(landmarks_2d) >= 33:
                    head_x = int(sum(landmarks_2d[i][0] for i in [0, 11, 12] if landmarks_2d[i][2] > 0.5) / max(1, sum(1 for i in [0, 11, 12] if landmarks_2d[i][2] > 0.5)))
                    head_y = min(lm[1] for lm in landmarks_2d[:11] if lm[2] > 0.5)
                    action = all_actions.get(pid, '?')
                    display_name = g.get_action_display_name(action) if hasattr(g, 'get_action_display_name') else action
                    # 仅当有有效中文名时绘制（避免 STANDING 等英文出现）
                    if display_name and not any('一' <= c <= '鿿' for c in display_name):
                        pass  # 纯英文内部名，不绘制在视频上
                    else:
                        g.put_chinese_text(
                            result, display_name,
                            (head_x - 40, head_y - 12),
                            18, (255, 255, 255), anchor='mt')

            return all_actions, all_landmarks, result

        # ═══════════ Office 观察模式 ═══════════
        if self.run_mode == 'office':
            print(f"[Training] Entering office observation mode")
            _office_frame = 0
            _last_ai_result = None
            _last_office_error = None  # 避免相同异常反复刷屏
            while self._running:
                try:
                    packet = provider.read(timeout_ms=100)
                    if not packet or packet.color_bgr is None:
                        _frame_fail_count += 1
                        if _frame_fail_count >= self.CAMERA_FAIL_THRESHOLD:
                            self.status_updated.emit({
                                'phase': 'camera_error',
                                'camera_source': camera_source,
                                'depth_available': depth_available,
                                'camera_mode_text': camera_mode_text,
                                'camera_error': True,
                                'camera_error_message':
                                    '摄像头连接中断。',
                            })
                            _frame_fail_count = 0
                        continue
                    _frame_fail_count = 0
                    img_bgr = packet.color_bgr

                    # 每 2 帧跑一次 AI，降低 CPU 占用
                    _office_frame += 1
                    if _office_frame % 2 == 0:
                        _last_ai_result = run_ai_pipeline_multi(img_bgr, max_persons=1)
                        _, _, result = _last_ai_result
                    elif _last_ai_result is not None:
                        _, _, result = _last_ai_result
                    else:
                        result = img_bgr.copy()
                    write_frame(result)

                    office_elapsed = int(time.time() - office_start_time)
                    person_visible = bool(
                        _last_ai_result and _last_ai_result[0]
                    )
                    self.status_updated.emit({
                        'phase': 'office',
                        'camera_source': camera_source,
                        'depth_available': depth_available,
                        'camera_mode_text': camera_mode_text,
                        'fallback_reason': fallback_reason,
                        'error_message': error_message,
                        'fps': 0,
                        'office_elapsed_seconds': office_elapsed,
                        'person_present': person_visible,
                    })
                except Exception as e:
                    # 仅在异常信息变化时打印一次，避免 office 循环反复刷屏
                    msg = str(e)
                    if msg != _last_office_error:
                        print(f"[后端] office 循环异常: {msg}")
                        _last_office_error = msg
                    continue
            # office 循环结束，清理
            try:
                provider.stop()
                pose_landmarker.close()
            except Exception:
                pass
            print("[Training] Office backend thread stopped")
            return

        # ═══════════ Phase: countdown — 直接 3 秒倒计时 ═══════════
        print(f"[Training] Session ready, mode: {self.mode}")
        phase = 'countdown'
        countdown_start = time.time()

        # ── 相机断联检测（阈值使用类常量 CAMERA_FAIL_THRESHOLD） ──
        _frame_fail_count = 0

        while self._running:
            try:
                packet = provider.read(timeout_ms=100)
                if not packet or packet.color_bgr is None:
                    _frame_fail_count += 1
                    if _frame_fail_count >= self.CAMERA_FAIL_THRESHOLD:
                        self.status_updated.emit({
                            'phase': 'camera_error',
                            'camera_source': camera_source,
                            'depth_available': depth_available,
                            'camera_mode_text': camera_mode_text,
                            'camera_error': True,
                            'camera_error_message':
                                '摄像头连接中断。\n'
                                '请检查连接后按 ESC 返回主菜单再重新进入；\n'
                                '或关闭程序后重新启动。\n'
                                '如需临时使用普通摄像头:\n'
                                '$env:OFFICEFIT_CAMERA="webcam"\n'
                                'python run_game_frontend.py',
                            'error_message':
                                f'连续 {_frame_fail_count} 帧读取失败',
                        })
                        _frame_fail_count = 0  # 重置以避免重复弹
                    continue
                _frame_fail_count = 0

                img_bgr = packet.color_bgr

                if phase == 'countdown':
                    write_frame(img_bgr)
                    elapsed = time.time() - countdown_start
                    cd_remain = max(0, 3 - int(elapsed))
                    if cd_remain > 0:
                        self.status_updated.emit({
                            'phase': 'countdown',
                            'countdown': cd_remain,
                            'mode': self.mode,
                            'camera_source': camera_source,
                            'depth_available': depth_available,
                            'camera_mode_text': camera_mode_text,
                            'fallback_reason': fallback_reason,
                            'error_message': error_message,
                        })
                        continue
                    else:
                        phase = 'playing'
                        prev_time = time.time()
                        fps_deque.clear()
                        action_counter = {}
                        print("[Training] Starting relaxation session")
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
                    'workout_mode': self.workout_mode,
                    'action_text': action_text,
                    'prompt_text': game.display_text,
                    'prompt_color': game.prompt_color,
                    'prompt_type': game.prompt_type,
                    'game_over': game.game_over,
                    'high_score': game.high_score,
                    'camera_source': camera_source,
                    'depth_available': depth_available,
                    'camera_mode_text': camera_mode_text,
                    'fallback_reason': fallback_reason,
                    'error_message': error_message,
                }

                self.status_updated.emit(status)

            except Exception as e:
                print(f"[后端] 循环异常: {e}")
                import traceback
                traceback.print_exc()
                continue

        # ---- 清理 ----
        try:
            provider.stop()
            pose_landmarker.close()
        except Exception:
            pass
        print("[Training] Backend thread stopped")

    def stop(self):
        self._running = False
        self.quit()
        if not self.wait(3000):
            self.terminate()
            self.wait()
