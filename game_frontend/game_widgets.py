"""游戏页面控件：视频显示 + Loading + 倒计时 + HUD 叠加层"""
import time
import numpy as np
import cv2
from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QProgressBar,
                                QVBoxLayout, QHBoxLayout, QGraphicsOpacityEffect)
from PySide6.QtCore import Qt, QTimer, QEasingCurve, QVariantAnimation, Signal, Property
from PySide6.QtGui import QImage, QPixmap, QPainter, QFont, QColor, QPen

from game_frontend.backend_thread import read_frame
from game_frontend.styles import get_difficulty_color

# ---------- BGR → RGB 颜色常量 ----------
GREEN_FORWARD = QColor(50, 220, 0)
RED_REVERSE = QColor(255, 50, 50)
TEXT_WHITE = QColor(255, 255, 255)
TEXT_YELLOW = QColor(255, 255, 0)
TEXT_CYAN = QColor(0, 255, 255)
TEXT_ORANGE = QColor(255, 165, 0)
OVERLAY_BG = QColor(0, 0, 0, 180)

def _bgr_to_qcolor(bgr_tuple):
    if bgr_tuple is None:
        return TEXT_WHITE
    b, g, r = bgr_tuple
    return QColor(r, g, b)


class VideoWidget(QWidget):
    """视频显示控件：定时从共享缓冲区拉取帧并用 QPainter 绘制"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self._frame = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll_frame)
        self._timer.start(33)

    def _poll_frame(self):
        frame = read_frame()
        if frame is not None:
            self._frame = frame
            self.update()

    def paintEvent(self, event):
        if self._frame is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        img_rgb = cv2.cvtColor(self._frame, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        x = (self.width() - pixmap.width()) // 2
        y = (self.height() - pixmap.height()) // 2
        painter.drawPixmap(x, y, pixmap)
        self._video_rect = (x, y, pixmap.width(), pixmap.height())


# ---------- Loading 覆盖层 ----------

class LoadingOverlay(QWidget):
    """加载覆盖层：预编程平滑进度条动画（不依赖后端进度）"""

    STAGES = [
        (15, "正在启动 Orbbec 3D 相机..."),
        (30, "请站到画面中央，启动姿态识别模型..."),
        (55, "正在加载检测模型..."),
        (80, "正在准备放松训练..."),
        (90, "即将就绪..."),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setStyleSheet("background-color: #1a1a2e;")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        self._label = QLabel("Loading...")
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            "color: #ffffff; font-size: 28px; font-weight: bold; font-family: 'Microsoft YaHei';")
        layout.addWidget(self._label)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.setFormat("%p%")
        self._progress.setFixedSize(400, 24)
        self._progress.setStyleSheet("""
            QProgressBar {
                background-color: #0f0f1e; border: 2px solid #0f3460;
                border-radius: 8px; height: 24px; text-align: center;
                color: #e0e0e0; font-size: 14px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f3460, stop:0.5 #00ff88, stop:1 #0f3460);
                border-radius: 6px;
            }
        """)
        layout.addWidget(self._progress, alignment=Qt.AlignCenter)

        self._current = 0
        self._fill_anim = None
        self._msg_index = 0
        self._done = False

        # 相机状态信息（loading 期间可动态更新）
        self._camera_status_label = QLabel("")
        self._camera_status_label.setAlignment(Qt.AlignCenter)
        self._camera_status_label.setStyleSheet(
            "color: #8899aa; font-size: 14px; font-family: 'Microsoft YaHei';")
        layout.addWidget(self._camera_status_label)

    def set_camera_status(self, text: str):
        """loading 期间更新相机状态文字"""
        self._camera_status_label.setText(text)

    def show_error(self, title: str, message: str, hint: str = ""):
        """显示启动失败错误信息"""
        if self._fill_anim is not None:
            self._fill_anim.stop()
        self._progress.setValue(0)
        self._progress.setVisible(False)

        lines = []
        if title:
            lines.append(title)
        if message:
            lines.append(message)
        if hint:
            lines.append(hint)
        display = "\n".join(lines) if lines else "启动失败"

        self._label.setText(display)
        self._label.setStyleSheet(
            "color: #ff6666; font-size: 20px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        self._camera_status_label.setText("")
        self._done = True  # 阻止 finish_loading 覆盖错误信息

    def start_fake_loading(self):
        """启动预编程的平滑加载动画：0→90% 在 3.5s 内完成"""
        if self._fill_anim is not None:
            self._fill_anim.stop()
        self._current = 0
        self._msg_index = 0
        self._done = False
        self._progress.setValue(0)
        self._label.setText("Loading...")

        self._fill_anim = QVariantAnimation(self)
        self._fill_anim.setDuration(3500)
        self._fill_anim.setStartValue(0)
        self._fill_anim.setEndValue(90)
        self._fill_anim.setEasingCurve(QEasingCurve.InOutCubic)
        self._fill_anim.valueChanged.connect(self._on_fill_value)
        self._fill_anim.start()

    def finish_loading(self):
        """后端就绪 → 快速填满到 100% → 显示 Done"""
        if self._done:
            return
        self._done = True
        if self._fill_anim is not None:
            self._fill_anim.stop()
        target = max(self._current, 90)
        self._fill_anim = QVariantAnimation(self)
        self._fill_anim.setDuration(500)
        self._fill_anim.setStartValue(target)
        self._fill_anim.setEndValue(100)
        self._fill_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._fill_anim.valueChanged.connect(self._on_finish_value)
        self._fill_anim.finished.connect(self._on_done_label)
        self._fill_anim.start()

    def _on_fill_value(self, val):
        self._current = int(val)
        self._progress.setValue(self._current)
        while (self._msg_index < len(self.STAGES)
               and self._current >= self.STAGES[self._msg_index][0]):
            self._label.setText(self.STAGES[self._msg_index][1])
            self._msg_index += 1

    def _on_finish_value(self, val):
        self._current = int(val)
        self._progress.setValue(self._current)

    def _on_done_label(self):
        self._label.setText("Done")
        QTimer.singleShot(400, self.hide)

    def reset(self):
        """重置到初始状态"""
        if self._fill_anim is not None:
            self._fill_anim.stop()
        self._current = 0
        self._msg_index = 0
        self._done = False
        self._progress.setValue(0)
        self._progress.setVisible(True)
        self._label.setText("Loading...")
        self._label.setStyleSheet(
            "color: #ffffff; font-size: 28px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        self._camera_status_label.setText("")


# ---------- 倒计时覆盖层 ----------

class CountdownOverlay(QWidget):
    """举手等待 + 3 秒倒计时覆盖层（带发光环动画）"""
    finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet("background-color: rgba(0, 0, 0, 170);")

        self._mode = 'waiting'
        self._countdown = 3
        self._font_size = 160
        self._ring_radius = 0.0   # 环形动画半径比例 (0.0~1.0)
        self._anim = None
        self._pulse = 0.0

    def show_waiting(self):
        self._mode = 'waiting'
        self._countdown = 0
        self.setVisible(True)
        self.raise_()
        if self._anim is not None:
            self._anim.stop()
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(1500)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._anim.setLoopCount(-1)
        self._anim.valueChanged.connect(self._on_pulse)
        self._anim.start()

    def start_countdown(self, countdown=3):
        if self._anim is not None:
            self._anim.stop()
        self._mode = 'countdown'
        self._countdown = countdown
        self._font_size = 160
        self._ring_radius = 1.0
        self._animate_number()

    def _on_pulse(self, val):
        self._pulse = val
        self.update()

    def _animate_number(self):
        if self._anim is not None:
            self._anim.stop()
        self._ring_radius = 1.0
        self._anim = QVariantAnimation(self)
        self._anim.setDuration(750)
        self._anim.setStartValue(1.0)
        self._anim.setEndValue(0.15)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self._anim.valueChanged.connect(self._on_ring)
        self._anim.start()
        # 字号缩放
        self._font_size = 160

    def _on_ring(self, val):
        self._ring_radius = val
        self._font_size = int(48 + (160 - 48) * val)
        self.update()

    def update_countdown(self, countdown):
        if countdown != self._countdown and self._mode == 'countdown':
            self._countdown = countdown
            if countdown > 0:
                self._animate_number()
            else:
                self.setVisible(False)
                self._mode = 'hidden'
                self.finished.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w // 2, h // 2

        if self._mode == 'waiting':
            alpha = int(180 + 75 * self._pulse)
            hint_color = QColor(255, 255, 255, alpha)

            # 提示文字
            hint_font = QFont("Microsoft YaHei", 30, QFont.Bold)
            painter.setFont(hint_font)
            painter.setPen(hint_color)
            painter.drawText(0, cy - 50, w, 100, Qt.AlignCenter, "请举起双手开始游戏")
            return

        if self._mode == 'countdown' and self._countdown > 0:
            # 发光环
            ring_max = min(w, h) * 0.35
            r = int(ring_max * self._ring_radius)
            if r > 8:
                if self._countdown >= 3:
                    ring_color = QColor(0, 210, 255, 60)
                elif self._countdown == 2:
                    ring_color = QColor(255, 200, 0, 80)
                else:
                    ring_color = QColor(255, 60, 60, 100)
                painter.setPen(QPen(ring_color, 4))
                painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(cx - r, cy - r, r * 2, r * 2)

            # 提示文字
            hint_font = QFont("Microsoft YaHei", 26, QFont.Bold)
            painter.setFont(hint_font)
            painter.setPen(QColor(255, 255, 255, 180))
            painter.drawText(0, cy - 120, w, 50, Qt.AlignCenter, "请举起双手开始游戏")

            # 倒计时数字 (带发光)
            num_font = QFont("Microsoft YaHei", self._font_size, QFont.Bold)
            painter.setFont(num_font)
            if self._countdown >= 3:
                color = QColor(255, 255, 255)
                glow = QColor(0, 210, 255, 60)
            elif self._countdown == 2:
                color = QColor(255, 200, 0)
                glow = QColor(255, 200, 0, 80)
            else:
                color = QColor(255, 60, 60)
                glow = QColor(255, 60, 60, 100)
            # 发光层
            painter.setPen(QPen(glow, 12))
            painter.drawText(0, cy - self._font_size // 2, w, self._font_size + 20,
                           Qt.AlignCenter, str(self._countdown))
            # 数字层
            painter.setPen(color)
            painter.drawText(0, cy - self._font_size // 2, w, self._font_size + 20,
                           Qt.AlignCenter, str(self._countdown))


# ---------- 飘字特效 ----------

class FloatTextWidget(QWidget):
    animation_done = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._text = ""
        self._color = TEXT_WHITE
        self._opacity = 1.0
        self._scale = 1.0
        self._offset_y = 0
        self._offset_anim = None
        self._fade_anim = None
        self._scale_anim = None

    def show_float_text(self, text, color_rgb=None):
        for anim in (self._offset_anim, self._fade_anim, self._scale_anim):
            if anim is not None:
                anim.stop()
        self._text = text
        self._color = QColor(*color_rgb) if color_rgb else TEXT_YELLOW
        self._opacity = 1.0
        self._scale = 1.8
        self._offset_y = 0
        self.setVisible(True)
        self.raise_()

        self._offset_anim = QVariantAnimation(self)
        self._offset_anim.setDuration(1200)
        self._offset_anim.setStartValue(0)
        self._offset_anim.setEndValue(80)
        self._offset_anim.setEasingCurve(QEasingCurve.OutCubic)
        self._offset_anim.valueChanged.connect(self._on_offset_changed)

        self._scale_anim = QVariantAnimation(self)
        self._scale_anim.setDuration(300)
        self._scale_anim.setStartValue(1.8)
        self._scale_anim.setEndValue(1.0)
        self._scale_anim.setEasingCurve(QEasingCurve.OutBack)
        self._scale_anim.valueChanged.connect(self._on_scale_changed)

        self._fade_anim = QVariantAnimation(self)
        self._fade_anim.setDuration(1200)
        self._fade_anim.setStartValue(1.0)
        self._fade_anim.setEndValue(0.0)
        self._fade_anim.setEasingCurve(QEasingCurve.InCubic)
        self._fade_anim.valueChanged.connect(self._on_fade_changed)
        self._fade_anim.finished.connect(self._on_anim_done)

        self._offset_anim.start()
        self._scale_anim.start()
        self._fade_anim.start()

    def _on_offset_changed(self, val):
        self._offset_y = val
        self.update()

    def _on_scale_changed(self, val):
        self._scale = val
        self.update()

    def _on_fade_changed(self, val):
        self._opacity = val
        self.update()

    def _on_anim_done(self):
        self.setVisible(False)
        self.animation_done.emit()

    def paintEvent(self, event):
        if not self._text:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        font = QFont("Microsoft YaHei", int(48 * self._scale), QFont.Bold)
        painter.setFont(font)
        c = self._color
        alpha = int(255 * self._opacity)
        # 发光层
        painter.setPen(QColor(c.red(), c.green(), c.blue(), alpha // 3))
        rect = self.rect()
        rect.translate(0, -self._offset_y - 2)
        painter.drawText(rect, Qt.AlignCenter, self._text)
        rect.translate(0, 4)
        painter.drawText(rect, Qt.AlignCenter, self._text)
        # 文字层
        rect.translate(0, -2)
        painter.setPen(QColor(c.red(), c.green(), c.blue(), alpha))
        painter.drawText(rect, Qt.AlignCenter, self._text)


# ---------- HUD 控件 ----------

class PromptBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(80)
        self.setObjectName("promptBarWidget")
        self._label = QLabel()
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setFont(QFont("Microsoft YaHei", 28, QFont.Bold))
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

    def update_prompt(self, text, color_bgr, prompt_type):
        self._color = _bgr_to_qcolor(color_bgr)
        if text:
            self._label.setText(text)
            bg = 'rgba(200,40,40,50)' if prompt_type == 'reverse' else 'rgba(0,200,100,50)'
            border = '#ff4466' if prompt_type == 'reverse' else '#00ff88'
            style = (f"color: {self._color.name()}; font-size: 32px; font-weight: bold;"
                     f" background-color: {bg};"
                     f" border: 1px solid {border}; border-radius: 10px;"
                     f" padding: 6px 28px;")
            self._label.setStyleSheet(style)
        else:
            self._label.clear()


class ScorePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(240)
        self.setObjectName("glassPanel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(8)

        self._score_label = QLabel("放松积分  0")
        self._score_label.setObjectName("scoreLabel")
        self._score_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._combo_label = QLabel("")
        self._combo_label.setObjectName("comboLabel")
        self._combo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._time_label = QLabel("剩余  30s")
        self._time_label.setObjectName("timeLabel")
        self._time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._fps_label = QLabel("FPS  --")
        self._fps_label.setObjectName("fpsLabel")
        self._fps_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        layout.addWidget(self._score_label)
        layout.addWidget(self._combo_label)
        layout.addWidget(self._time_label)
        layout.addWidget(self._fps_label)
        layout.addStretch()

    def update_status(self, score, combo, time_left, fps=0):
        self._score_label.setText(f"放松积分  {score}")
        if combo >= 2:
            self._combo_label.setText(f"连续完成  x{combo}")
            self._combo_label.setObjectName("comboLabelHigh" if combo >= 5 else "comboLabel")
            self._combo_label.style().unpolish(self._combo_label)
            self._combo_label.style().polish(self._combo_label)
        else:
            self._combo_label.setText("")

        self._time_label.setText(f"剩余  {int(time_left)}s")
        self._time_label.setObjectName("timeLabelWarn" if time_left <= 5 else "timeLabel")
        self._time_label.style().unpolish(self._time_label)
        self._time_label.style().polish(self._time_label)

        self._fps_label.setText(f"FPS  {int(fps)}")


class TimeBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setObjectName("timeBarWidget")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 6, 24, 4)

        self._bar = QProgressBar()
        self._bar.setRange(0, 1000)
        self._bar.setValue(1000)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(14)

        info_layout = QHBoxLayout()
        info_layout.setContentsMargins(0, 0, 0, 0)
        self._mode_label = QLabel()
        self._mode_label.setObjectName("modeLabel")
        info_layout.addWidget(self._mode_label)
        info_layout.addStretch()
        self._camera_label = QLabel()
        self._camera_label.setObjectName("cameraModeLabel")
        self._camera_label.setStyleSheet(
            "color: #6677aa; font-size: 12px; font-family: 'Microsoft YaHei';")
        info_layout.addWidget(self._camera_label)

        layout.addWidget(self._bar)
        layout.addLayout(info_layout)

    def update_time(self, time_left, total=30.0):
        progress = int(time_left / total * 1000)
        self._bar.setValue(max(0, min(1000, progress)))
        if time_left > 10:
            chunk = "#00ff88"
        elif time_left > 5:
            chunk = "#ffaa00"
        else:
            chunk = "#ff4444"
        self._bar.setStyleSheet(
            f"QProgressBar::chunk {{ background: {chunk}; border-radius: 5px; }}"
            "QProgressBar { background-color: rgba(255,255,255,0.05);"
            " border: 1px solid rgba(255,255,255,0.08); border-radius: 7px; height: 12px; }")

    def update_mode(self, mode):
        color = get_difficulty_color(mode)
        self._mode_label.setText(f"Mode  {mode.upper()}")
        self._mode_label.setStyleSheet(f"color: {color}; font-size: 15px;"
                                        " font-family: 'Microsoft YaHei'; font-weight: bold;")

    def update_camera(self, camera_source, depth_available,
                      camera_mode_text="", fallback_reason=""):
        if camera_mode_text:
            text = camera_mode_text
        elif camera_source == 'orbbec':
            if depth_available:
                text = "Orbbec 3D 深度相机模式"
            else:
                text = "Orbbec 彩色模式"
        elif camera_source == 'webcam':
            text = "普通摄像头兼容模式"
        else:
            text = ""

        if depth_available:
            color = "#00ff88"
        elif camera_source == 'webcam':
            color = "#ffaa00"
        else:
            color = "#6677aa"

        self._camera_label.setText(text)
        self._camera_label.setStyleSheet(
            f"color: {color}; font-size: 12px; font-family: 'Microsoft YaHei';")

        # 如果有降级原因, 在 tooltip 中显示
        if fallback_reason:
            self._camera_label.setToolTip(fallback_reason)



# ---------- 游戏主页面 ----------

class GamePage(QWidget):
    """游戏主页面：视频背景 + Loading/倒计时覆盖层 + 游戏 HUD"""
    float_text_requested = Signal(str, tuple)
    countdown_finished = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #1a1a2e;")
        self.setFocusPolicy(Qt.StrongFocus)
        self._status = {}
        self._phase = 'idle'

        # 视频层
        self._video = VideoWidget(self)

        # 覆盖层
        self._loading_overlay = LoadingOverlay(self)
        self._countdown_overlay = CountdownOverlay(self)
        self._countdown_overlay.finished.connect(self.countdown_finished.emit)

        # HUD
        self._prompt_bar = PromptBar(self)
        self._score_panel = ScorePanel(self)
        self._time_bar = TimeBar(self)

        self._float_text = FloatTextWidget(self)
        self.float_text_requested.connect(self._float_text.show_float_text)

        self._last_combo = 0
        self._last_score = 0

        # 初始状态：显示 loading
        self._loading_overlay.show()
        self._loading_overlay.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        w, h = self.width(), self.height()
        self._video.setGeometry(0, 0, w, h)
        self._loading_overlay.setGeometry(0, 0, w, h)
        self._countdown_overlay.setGeometry(0, 0, w, h)
        self._prompt_bar.setGeometry(0, 0, w, 80)
        self._score_panel.setGeometry(w - 250, 88, 240, 170)
        self._time_bar.setGeometry(0, h - 56, w, 56)
        self._float_text.setGeometry(0, 0, w, h)

    def update_status(self, status: dict):
        phase = status.get('phase', 'idle')
        self._status = status
        self._phase = phase

        # 相机状态信息（跨所有 phase 更新）
        camera_text = status.get('camera_mode_text', '')
        if camera_text and self._loading_overlay.isVisible():
            self._loading_overlay.set_camera_status(camera_text)

        if phase == 'error':
            # 后端启动失败 — 在 loading 覆盖层上显示错误信息
            self._loading_overlay.show()
            self._loading_overlay.raise_()
            err_msg = status.get('error_message', '未知错误')
            fallback = status.get('fallback_reason', '')
            camera = status.get('camera_mode_text', '')
            self._loading_overlay.show_error(
                camera,
                err_msg,
                fallback,
            )
            return

        if phase == 'loading':
            self._loading_overlay.show()
            self._loading_overlay.raise_()
            # loading 阶段也显示相机模式信息（如果已有）
            if status.get('camera_source'):
                self._loading_overlay.set_camera_status(
                    status.get('camera_mode_text', '')
                    + (f"\n{status.get('fallback_reason', '')}"
                       if status.get('fallback_reason') else "")
                )
            return

        if phase != 'loading' and self._loading_overlay.isVisible():
            self._loading_overlay.finish_loading()

        if phase == 'waiting_for_hands':
            if not self._countdown_overlay.isVisible() or self._countdown_overlay._mode != 'waiting':
                self._countdown_overlay.show_waiting()
            self._prompt_bar.update_prompt('', None, 'forward')
            return

        if phase == 'countdown':
            cd = status.get('countdown', 3)
            if self._countdown_overlay._mode != 'countdown':
                self._countdown_overlay.start_countdown(cd)
            else:
                self._countdown_overlay.update_countdown(cd)
            self._prompt_bar.update_prompt('', None, 'forward')
            return

        if phase in ('playing', 'game_over'):
            if self._countdown_overlay.isVisible():
                self._countdown_overlay.hide()
            self._update_single_status(status)

    def _update_single_status(self, status):
        prev_score = self._last_score
        prev_combo = self._last_combo
        self._last_score = status.get('score', 0)
        self._last_combo = status.get('combo', 0)

        self._prompt_bar.update_prompt(
            status.get('prompt_text', ''),
            status.get('prompt_color', None),
            status.get('prompt_type', 'forward'),
        )
        self._score_panel.update_status(
            status.get('score', 0),
            status.get('combo', 0),
            status.get('time_left', 0),
            status.get('fps', 0),
        )
        self._time_bar.update_time(status.get('time_left', 30))
        self._time_bar.update_mode(status.get('mode', 'practice'))
        self._time_bar.update_camera(
            status.get('camera_source', 'unknown'),
            status.get('depth_available', False),
            status.get('camera_mode_text', ''),
            status.get('fallback_reason', ''),
        )

        new_score = status.get('score', 0)
        new_combo = status.get('combo', 0)
        if new_score > prev_score:
            lines = [f"+{new_score - prev_score}"]
            if new_combo >= 2:
                lines.append(f"COMBO x{new_combo}!")
            text = "  ".join(lines)
            if new_combo >= 4:
                color = (255, 165, 0)
            elif new_combo >= 2:
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)
            self.float_text_requested.emit(text, color)

    def reset_loading(self):
        """重置到 loading 状态（游戏重新开始时调用）"""
        self._phase = 'idle'
        self._last_score = 0
        self._last_combo = 0
        self._loading_overlay.reset()
        self._loading_overlay.show()
        self._loading_overlay.raise_()
        self._loading_overlay.start_fake_loading()
        self._countdown_overlay.hide()

    def is_video_ready(self):
        return self._video._frame is not None
