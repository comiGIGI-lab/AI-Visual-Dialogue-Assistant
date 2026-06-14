"""训练页面控件：视频显示 + Loading + 倒计时 + HUD 叠加层"""
import time
import numpy as np
import cv2
from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QProgressBar,
                                QVBoxLayout, QHBoxLayout,
                                QGraphicsOpacityEffect, QTextEdit)
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
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
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
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
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

    def show_dynamic_hint(self, title: str = "", subtitle: str = "",
                          color: str = "white"):
        """显示动态提示（语音状态、识别结果、AI 建议等）"""
        self._mode = 'dynamic_hint'
        self._hint_title = title
        self._hint_subtitle = subtitle
        self._hint_color = color
        self.setVisible(True)
        self.raise_()
        self.update()

    def show_office_idle(self):
        """office 模式专属等待文案（区别于训练用 show_waiting_ready）"""
        self.show_dynamic_hint(
            "办公守护中",
            "如果感到不适，可以说：我肩膀酸 / 我脖子酸 / 腰背僵硬",
            "white",
        )

    def show_waiting_ready(self):
        """相机就绪但尚未请求开始 — 提示用户通过语音/按钮开始"""
        self._mode = 'ready_waiting'
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

        if self._mode == 'dynamic_hint':
            title = getattr(self, '_hint_title', '')
            subtitle = getattr(self, '_hint_subtitle', '')
            color_name = getattr(self, '_hint_color', 'white')
            color_map = {
                "white": QColor(255, 255, 255),
                "cyan": QColor(0, 212, 255),
                "green": QColor(0, 255, 136),
                "yellow": QColor(255, 200, 0),
                "red": QColor(255, 80, 80),
            }
            c = color_map.get(color_name, QColor(255, 255, 255))

            title_font = QFont("Microsoft YaHei", 32, QFont.Bold)
            painter.setFont(title_font)
            painter.setPen(c)
            painter.drawText(0, cy - 60, w, 60, Qt.AlignCenter, title)

            if subtitle:
                sub_font = QFont("Microsoft YaHei", 20, QFont.Bold)
                painter.setFont(sub_font)
                alpha = 200
                painter.setPen(QColor(c.red(), c.green(), c.blue(), alpha))
                painter.drawText(0, cy + 10, w, 80, Qt.AlignCenter, subtitle)
            return

        if self._mode == 'ready_waiting':
            alpha = int(180 + 75 * self._pulse)
            hint_color = QColor(255, 255, 255, alpha)

            hint_font = QFont("Microsoft YaHei", 28, QFont.Bold)
            painter.setFont(hint_font)
            painter.setPen(hint_color)
            painter.drawText(0, cy - 70, w, 50, Qt.AlignCenter,
                           "相机已就绪，请站到画面中央")

            sub_font = QFont("Microsoft YaHei", 20, QFont.Bold)
            painter.setFont(sub_font)
            painter.setPen(QColor(0, 212, 255, alpha))
            painter.drawText(0, cy, w, 50, Qt.AlignCenter,
                           "说「开始放松」或点击左侧模拟按钮")
            return

        if self._mode == 'waiting':
            alpha = int(180 + 75 * self._pulse)
            hint_color = QColor(255, 255, 255, alpha)

            # 提示文字
            hint_font = QFont("Microsoft YaHei", 30, QFont.Bold)
            painter.setFont(hint_font)
            painter.setPen(hint_color)
            painter.drawText(0, cy - 50, w, 100, Qt.AlignCenter, "准备开始训练")
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
            painter.drawText(0, cy - 120, w, 50, Qt.AlignCenter, "已准备开始，请跟随引导进入放松训练")

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

# ── 音量条 ──────────────────────────────────────────────

class VolumeBar(QWidget):
    """小型音量指示条：0.0~1.0 RMS 音量 → 绿色条宽度"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._level = 0.0
        self.setFixedHeight(6)
        self.setMinimumWidth(80)

    def set_level(self, level: float):
        self._level = max(0.0, min(1.0, level))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # 背景
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(30, 30, 50))
        painter.drawRoundedRect(0, 0, w, h, 3, 3)

        # 音量条
        if self._level > 0.01:
            bar_w = int(w * self._level)
            if self._level < 0.3:
                c = QColor(0, 200, 100)    # 低音量绿
            elif self._level < 0.6:
                c = QColor(0, 255, 136)    # 中音量亮绿
            else:
                c = QColor(255, 200, 0)    # 高音量黄
            painter.setBrush(c)
            painter.drawRoundedRect(0, 0, bar_w, h, 3, 3)


# ── 动作目的与提示映射 ────────────────────────────────────

ACTION_METADATA = {
    "扩胸打开": {
        "purpose": "打开胸腔，缓解久坐含胸",
        "tips": "双手向两侧打开，肩膀放松，保持 3 秒",
        "slow_hint": "动作可以慢一点，系统会等待你完成",
    },
    "双手上举": {
        "purpose": "拉伸肩臂，改善上肢血液循环",
        "tips": "双手同时向上伸直，尽量举过头顶",
        "slow_hint": "动作可以慢一点，系统会等待你完成",
    },
    "左手上举": {
        "purpose": "拉伸左侧肩臂，缓解单侧疲劳",
        "tips": "左手向上伸展，右手自然放松",
        "slow_hint": "动作可以慢一点，系统会等待你完成",
    },
    "右手上举": {
        "purpose": "拉伸右侧肩臂，缓解单侧疲劳",
        "tips": "右手向上伸展，左手自然放松",
        "slow_hint": "动作可以慢一点，系统会等待你完成",
    },
    "左侧拉伸(举右手)": {
        "purpose": "拉伸右侧身体，缓解侧腰紧张",
        "tips": "右手向上举过头顶，身体轻轻向左弯",
        "slow_hint": "动作可以慢一点，系统会等待你完成",
    },
    "右侧拉伸(举左手)": {
        "purpose": "拉伸左侧身体，缓解侧腰紧张",
        "tips": "左手向上举过头顶，身体轻轻向右弯",
        "slow_hint": "动作可以慢一点，系统会等待你完成",
    },
    "颈部左转(提示)": {
        "purpose": "活动颈椎，缓解颈部僵硬",
        "tips": "缓慢转头看向左侧，保持肩膀不动",
        "slow_hint": "轻轻转动即可，不需要用力",
    },
    "颈部右转(提示)": {
        "purpose": "活动颈椎，缓解颈部僵硬",
        "tips": "缓慢转头看向右侧，保持肩膀不动",
        "slow_hint": "轻轻转动即可，不需要用力",
    },
    "蹲下": {
        "purpose": "活动下肢关节，促进腿部循环",
        "tips": "膝盖弯曲降低身体重心，保持背部挺直",
        "slow_hint": "动作可以慢一点，注意安全",
    },
    "左抬腿": {
        "purpose": "活动左腿，改善下肢循环",
        "tips": "左腿向前或向上抬起，保持身体稳定",
        "slow_hint": "动作可以慢一点，注意安全",
    },
    "右抬腿": {
        "purpose": "活动右腿，改善下肢循环",
        "tips": "右腿向前或向上抬起，保持身体稳定",
        "slow_hint": "动作可以慢一点，注意安全",
    },
}


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

        self._score_label = QLabel("完成度  0")
        self._score_label.setObjectName("scoreLabel")
        self._score_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._combo_label = QLabel("")
        self._combo_label.setObjectName("comboLabel")
        self._combo_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self._time_label = QLabel("本轮剩余  30s")
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
        self._score_label.setText(f"完成度  {score}")
        if combo >= 2:
            self._combo_label.setText(f"连续完成  x{combo}")
            self._combo_label.setObjectName("comboLabelHigh" if combo >= 5 else "comboLabel")
            self._combo_label.style().unpolish(self._combo_label)
            self._combo_label.style().polish(self._combo_label)
        else:
            self._combo_label.setText("")

        self._time_label.setText(f"本轮剩余  {int(time_left)}s")
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
        mode_names = {'practice': '轻松放松', 'normal': '标准放松', 'hard': '活力放松'}
        display = mode_names.get(mode, mode)
        color = get_difficulty_color(mode)
        self._mode_label.setText(f"{display}")
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



# ---------- 训练主页面 ----------

class GamePage(QWidget):
    """训练主页面：视频背景 + Loading/倒计时覆盖层 + 训练 HUD"""
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

        # AI 助手面板
        self.assistant_panel = AssistantPanel(self)

        self._dialog_manager = None
        self._ready_to_start = False

        self._last_combo = 0
        self._last_score = 0

        # 初始状态：显示 loading
        self._loading_overlay.show()
        self._loading_overlay.raise_()

    def set_dialog_manager(self, dm):
        """注入 DialogManager 实例"""
        self._dialog_manager = dm

    def set_workout_ready(self):
        """用户已通过语音/按钮确认开始，允许进入举手倒计时"""
        self._ready_to_start = True

    def reset_loading(self):
        """重置到 loading 状态（训练重新开始时调用）"""
        self._phase = 'idle'
        self._ready_to_start = False
        self._last_score = 0
        self._last_combo = 0
        self._loading_overlay.reset()
        self._loading_overlay.show()
        self._loading_overlay.raise_()
        self._loading_overlay.start_fake_loading()
        self._countdown_overlay.hide()

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
        self.assistant_panel.setGeometry(
            12, 88, 280, h - 160)
        self.assistant_panel.raise_()

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

        if phase == 'camera_error':
            # 运行中相机断联
            err_msg = status.get('camera_error_message',
                               '摄像头连接中断。')
            hint = ('按 ESC 返回主菜单后重新进入；\n'
                    '或关闭程序重新启动。\n'
                    '如需临时使用普通摄像头:\n'
                    '$env:OFFICEFIT_CAMERA="webcam"\n'
                    'python run_game_frontend.py')
            self._loading_overlay.show()
            self._loading_overlay.raise_()
            self._loading_overlay.show_error(
                '摄像头连接中断',
                err_msg,
                hint,
            )
            self._countdown_overlay.hide()
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
            self.assistant_panel.raise_()

        if phase == 'office':
            # 办公模式 — 隐藏训练 HUD
            self._prompt_bar.update_prompt('', None, 'forward')
            self._score_panel.hide()
            self._time_bar.hide()
            if self._loading_overlay.isVisible():
                self._loading_overlay.finish_loading()
            # 中央文案由 MainWindow 通过 show_dynamic_hint 管理
            # （办公守护中 / 正在听 / 请坐到画面中央 等）。
            # 这里不使用训练用 show_waiting_ready()，仅在无任何提示时回退到办公守护文案。
            if not self._countdown_overlay.isVisible():
                self._countdown_overlay.show_office_idle()
                self._countdown_overlay.raise_()
            return

        if phase == 'countdown':
            # 显示训练 HUD
            self._score_panel.show()
            self._time_bar.show()
            cd = status.get('countdown', 3)
            if self._countdown_overlay._mode != 'countdown':
                self._countdown_overlay.start_countdown(cd)
            else:
                self._countdown_overlay.update_countdown(cd)
            if self._ready_to_start:
                self._prompt_bar.update_prompt(
                    '已准备开始，请跟随引导进入放松训练', None, 'forward')
            else:
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

        prompt_text = status.get('prompt_text', '')
        self._prompt_bar.update_prompt(
            prompt_text,
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

        # 同步 AI 助手面板
        self.assistant_panel.update_camera_info(
            status.get('camera_mode_text', ''),
            status.get('depth_available', False),
        )
        # 训练中显示当前动作的目的和提示
        if prompt_text and prompt_text != getattr(self, '_last_prompt_text', ''):
            self._last_prompt_text = prompt_text
            self.assistant_panel.set_current_action(
                prompt_text, status.get('mode', ''))

        new_score = status.get('score', 0)
        new_combo = status.get('combo', 0)
        if new_score > prev_score:
            lines = [f"+{new_score - prev_score}"]
            if new_combo >= 2:
                lines.append(f"连续完成 x{new_combo}")
            text = "  ".join(lines)
            if new_combo >= 4:
                color = (255, 165, 0)
            elif new_combo >= 2:
                color = (0, 255, 0)
            else:
                color = (255, 255, 0)
            self.float_text_requested.emit(text, color)

    def is_video_ready(self):
        return self._video._frame is not None


# ---------- AI 助手面板 ----------

class AssistantPanel(QWidget):
    """AI 助手面板：麦克风状态 / 语音指令 / 回复 / 相机信息 / 模式选择"""

    listen_clicked = Signal()
    simulate_command = Signal(str)  # 携带指令字符串
    mode_selected = Signal(str)     # upper_body / full_body

    _SIM_COMMANDS = [
        "我肩膀酸", "我脖子酸", "腰背僵硬",
        "开始放松", "换一个动作", "结束",
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("glassPanel")
        self.setFixedWidth(280)
        self._dialog_entries = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(6)

        # 标题
        title = QLabel("AI 久坐健康助手")
        title.setStyleSheet(
            "color: #00d4ff; font-size: 18px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        layout.addWidget(title)

        # 麦克风状态
        self._mic_label = QLabel("语音: 准备就绪")
        self._mic_label.setStyleSheet(
            "color: #8899aa; font-size: 13px; font-family: 'Microsoft YaHei';")
        layout.addWidget(self._mic_label)

        # 最近输入（我听到 / 理解为）
        recent_title = QLabel("最近指令")
        recent_title.setStyleSheet(
            "color: #6677aa; font-size: 13px; font-family: 'Microsoft YaHei';")
        layout.addWidget(recent_title)
        self._voice_label = QLabel("我听到：(无)")
        self._voice_label.setStyleSheet(
            "color: #ffaa00; font-size: 14px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        self._voice_label.setWordWrap(True)
        layout.addWidget(self._voice_label)
        self._understood_label = QLabel("理解为：(无)")
        self._understood_label.setStyleSheet(
            "color: #00d4ff; font-size: 13px; font-family: 'Microsoft YaHei';")
        self._understood_label.setWordWrap(True)
        layout.addWidget(self._understood_label)

        # 对话日志：完整语音和 AI 回复进入滚动区域，避免长文本被截断。
        dialog_title = QLabel("对话日志")
        dialog_title.setStyleSheet(
            "color: #6677aa; font-size: 13px; font-family: 'Microsoft YaHei';")
        layout.addWidget(dialog_title)
        self._dialog_log = QTextEdit()
        self._dialog_log.setReadOnly(True)
        self._dialog_log.setAcceptRichText(False)
        self._dialog_log.setMinimumHeight(130)
        self._dialog_log.setMaximumHeight(180)
        self._dialog_log.setStyleSheet(
            "QTextEdit { color: #c0d0e0; font-size: 13px;"
            " font-family: 'Microsoft YaHei';"
            " background-color: rgba(15,52,96,0.4); border-radius: 8px;"
            " border: 1px solid rgba(255,255,255,0.08); padding: 8px; }")
        layout.addWidget(self._dialog_log)
        self.add_dialog_entry(
            "AI",
            "我已进入 AI 办公守护模式。感到不适时可以说「我肩膀酸」，"
            "我会推荐适合的放松动作。")

        # 推荐动作链
        self._recommend_label = QLabel("推荐动作: --")
        self._recommend_label.setWordWrap(True)
        self._recommend_label.setStyleSheet(
            "color: #00ff88; font-size: 13px; font-family: 'Microsoft YaHei';")
        layout.addWidget(self._recommend_label)

        # 相机模式 + 深度
        self._camera_info_label = QLabel("相机: --")
        self._camera_info_label.setStyleSheet(
            "color: #6677aa; font-size: 12px; font-family: 'Microsoft YaHei';")
        layout.addWidget(self._camera_info_label)
        self._depth_label = QLabel("深度: --")
        self._depth_label.setStyleSheet(
            "color: #6677aa; font-size: 12px; font-family: 'Microsoft YaHei';")
        layout.addWidget(self._depth_label)

        # 会话状态
        self._session_state_label = QLabel("等待指令")
        self._session_state_label.setStyleSheet(
            "color: #00d4ff; font-size: 14px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        layout.addWidget(self._session_state_label)

        layout.addSpacing(6)

        # ── 模式选择 ──
        mode_title = QLabel("放松模式")
        mode_title.setStyleSheet(
            "color: #00d4ff; font-size: 14px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        layout.addWidget(mode_title)

        mode_row = QHBoxLayout()
        self._upper_btn = QPushButton("上半身")
        self._upper_btn.setObjectName("smallBtn")
        self._upper_btn.setCursor(Qt.PointingHandCursor)
        self._upper_btn.setCheckable(True)
        self._upper_btn.setChecked(True)
        self._upper_btn.clicked.connect(lambda: self._select_mode("upper_body"))
        mode_row.addWidget(self._upper_btn)

        self._full_btn = QPushButton("全身")
        self._full_btn.setObjectName("smallBtn")
        self._full_btn.setCursor(Qt.PointingHandCursor)
        self._full_btn.setCheckable(True)
        self._full_btn.clicked.connect(lambda: self._select_mode("full_body"))
        mode_row.addWidget(self._full_btn)
        layout.addLayout(mode_row)

        layout.addSpacing(6)

        # ── 语音输入按钮 ──
        self._listen_btn = QPushButton("语音输入")
        self._listen_btn.setObjectName("smallBtn")
        self._listen_btn.setCursor(Qt.PointingHandCursor)
        self._listen_btn.clicked.connect(self.listen_clicked.emit)
        layout.addWidget(self._listen_btn)

        # 音量条（录音时显示）
        self._volume_bar = VolumeBar()
        self._volume_bar.setVisible(False)
        layout.addWidget(self._volume_bar)

        self._listen_status = QLabel(
            "点击后说一句话，再次点击可结束录音。")
        self._listen_status.setStyleSheet(
            "color: #6677aa; font-size: 12px; font-family: 'Microsoft YaHei';")
        self._listen_status.setWordWrap(True)
        layout.addWidget(self._listen_status)

        # ── 快捷输入按钮 ──
        sim_title = QLabel("快捷输入")
        sim_title.setStyleSheet(
            "color: #6677aa; font-size: 13px; font-family: 'Microsoft YaHei';")
        layout.addWidget(sim_title)

        sim_grid = QHBoxLayout()
        sim_grid.setSpacing(4)
        col1 = QVBoxLayout()
        col1.setSpacing(3)
        col2 = QVBoxLayout()
        col2.setSpacing(3)
        for i, cmd in enumerate(self._SIM_COMMANDS):
            btn = QPushButton(cmd)
            btn.setObjectName("smallBtn")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                "QPushButton { font-size: 11px; padding: 3px 6px;"
                " font-family: 'Microsoft YaHei'; }")
            btn.clicked.connect(lambda checked, c=cmd: self.simulate_command.emit(c))
            if i < 3:
                col1.addWidget(btn)
            else:
                col2.addWidget(btn)
        sim_grid.addLayout(col1)
        sim_grid.addLayout(col2)
        layout.addLayout(sim_grid)

        layout.addStretch()

    def _select_mode(self, mode):
        self.mode_selected.emit(mode)
        if mode == "upper_body":
            self._upper_btn.setChecked(True)
            self._full_btn.setChecked(False)
        else:
            self._upper_btn.setChecked(False)
            self._full_btn.setChecked(True)

    def set_ai_source(self, source: str):
        """设置 AI 来源（本地 / 外部 AI）"""
        self._ai_source = source
        self._refresh_mic_label()

    def set_mic_available(self, available: bool):
        if available:
            self._mic_available = True
            self._listen_btn.setEnabled(True)
            self._listen_btn.setText("语音输入")
            self._listen_status.setText(
                "点击后说一句话，再次点击可结束录音。")
        else:
            self._mic_available = False
            self._listen_btn.setEnabled(False)
            self._listen_btn.setText("麦克风不可用")
            self._listen_status.setText("请使用下方快捷输入按钮")
        self._refresh_mic_label()

    def set_mic_device(self, name: str, index: int):
        self._mic_device_name = name
        self._mic_device_index = index
        self._refresh_mic_label()

    def _refresh_mic_label(self):
        """根据 AI 来源和麦克风状态更新标签"""
        ai_source = getattr(self, '_ai_source', '本地')
        mic_ok = getattr(self, '_mic_available', False)
        name = getattr(self, '_mic_device_name', '')
        idx = getattr(self, '_mic_device_index', None)

        if mic_ok:
            dev = f"[{idx}] {name}" if idx is not None and name else ""
            if ai_source == '本地':
                text = f"语音: 本地 | 麦克风: {dev}" if dev else "语音: 本地 | 麦克风: 可用"
            else:
                text = f"语音: {ai_source} | 麦克风: {dev}" if dev else f"语音: {ai_source}"
            color = "#00ff88"
        else:
            hint = ("可在 PowerShell 设置: $env:OFFICEFIT_MIC_INDEX='编号'"
                    if ai_source == '本地' else "")
            text = ("麦克风: 不可用\n" + hint).rstrip()
            color = "#ffaa00"

        self._mic_label.setText(text)
        self._mic_label.setStyleSheet(
            f"color: {color}; font-size: 13px; font-family: 'Microsoft YaHei';")

    def set_listening_active(self, active: bool):
        if active:
            self._listen_btn.setText("正在录音，点击结束")
            self._listen_btn.setStyleSheet(
                "background: #0a2a1a; color: #00ff88;"
                " border: 1px solid #00ff88; border-radius: 10px;"
                " padding: 10px 28px; font-size: 16px;"
                " font-family: 'Microsoft YaHei';")
            self._listen_status.setText(
                "正在录音，请说话...")
            self._listen_status.setStyleSheet(
                "color: #00ff88; font-size: 12px; font-family: 'Microsoft YaHei';")
            self._volume_bar.setVisible(True)
        else:
            self._listen_btn.setText("语音输入")
            self._listen_btn.setStyleSheet("")
            self._listen_status.setText(
                "点击后说一句话，再次点击可结束录音。")
            self._volume_bar.setVisible(False)

    def set_voice_understanding(self):
        """录音结束，正在识别中"""
        self._listen_btn.setText("正在理解...")
        self._listen_btn.setStyleSheet(
            "background: #1a1a3e; color: #ffaa00;"
            " border: 1px solid #ffaa00; border-radius: 10px;"
            " padding: 10px 28px; font-size: 16px;"
            " font-family: 'Microsoft YaHei';")
        self._listen_status.setText("正在理解你说的话...")
        self._listen_status.setStyleSheet(
            "color: #ffaa00; font-size: 12px; font-family: 'Microsoft YaHei';")

    def update_voice_level(self, level: float):
        """更新音量条（0.0~1.0）"""
        self._volume_bar.set_level(level)

    def set_listen_error(self, message: str):
        """显示监听错误/状态信息"""
        self._listen_status.setText(message)
        self._listen_status.setStyleSheet(
            "color: #ffaa00; font-size: 12px; font-family: 'Microsoft YaHei';")
        self.add_dialog_entry("系统", message)

    def set_recent_input(self, raw: str, understood: str):
        """最近输入两行：我听到 / 理解为"""
        self._voice_label.setText(f"我听到：{raw or '(无)'}")
        self._voice_label.setStyleSheet(
            "color: #ffaa00; font-size: 14px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        self._understood_label.setText(f"理解为：{understood or '(无)'}")
        if raw:
            self.add_dialog_entry("用户", raw)

    def set_recommended_actions(self, actions):
        """推荐动作链：动作1 → 动作2 → ..."""
        if actions:
            self._recommend_label.setText(
                "推荐动作: " + " → ".join(list(actions)[:6]))
        else:
            self._recommend_label.setText("推荐动作: --")

    def set_voice_result(self, text: str, success: bool):
        """设置语音识别结果（失败提示用）"""
        if success:
            self._voice_label.setText(f"我听到：{text}")
            self._voice_label.setStyleSheet(
                "color: #00ff88; font-size: 14px; font-weight: bold;"
                " font-family: 'Microsoft YaHei';")
        else:
            self._listen_status.setText(
                "没有识别到有效指令，请再说一次，或使用快捷输入按钮")
            self._listen_status.setStyleSheet(
                "color: #ffaa00; font-size: 12px; font-family: 'Microsoft YaHei';")
            self.add_dialog_entry("系统", "没有识别到有效指令，请再说一次，或使用快捷输入按钮")

    def set_voice_command(self, text: str):
        self._voice_label.setText(f"我听到：{text}")

    def set_assistant_message(self, text: str):
        source = getattr(self, '_ai_source', '本地')
        if source != '本地':
            self.add_dialog_entry("AI", f"[{source}] {text}")
        else:
            self.add_dialog_entry("AI", text)

    def add_dialog_entry(self, role: str, text: str):
        """追加对话日志，最多保留最近 20 条并自动滚到底部。"""
        text = (text or "").strip()
        if not text:
            return
        self._dialog_entries.append(f"[{role}] {text}")
        self._dialog_entries = self._dialog_entries[-20:]
        self._dialog_log.setPlainText("\n\n".join(self._dialog_entries))
        scrollbar = self._dialog_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def set_current_action(self, display_text: str, mode: str = ""):
        """训练中更新当前动作信息：目的 + 提示 + 慢速提醒"""
        meta = ACTION_METADATA.get(display_text)
        if not meta:
            # 反向动作或未知动作：只显示基本提示
            if display_text:
                self.add_dialog_entry(
                    "AI",
                    f"当前动作：{display_text}\n"
                    f"请跟随屏幕提示完成动作。")
            return
        slow_hint = meta["slow_hint"]
        # 轻松/标准模式才显示慢速提示
        if mode in ("practice", "normal", "轻松", "标准"):
            slow_line = f"\n识别提示：{slow_hint}"
        else:
            slow_line = ""
        self.add_dialog_entry(
            "AI",
            f"当前动作：{display_text}\n"
            f"动作目的：{meta['purpose']}\n"
            f"动作提示：{meta['tips']}"
            f"{slow_line}"
        )

    def set_session_state(self, state: str):
        """更新会话状态标签"""
        labels = {
            "office": "办公守护中",
            "ready": "等待指令",
            "listening": "正在录音",
            "training": "训练中",
            "relax_guidance": "放松引导",
            "paused": "已暂停",
            "summary": "本次完成",
        }
        self._session_state_label.setText(labels.get(state, state))

    def update_camera_info(self, camera_mode_text: str, depth_available: bool):
        self._camera_info_label.setText(f"相机: {camera_mode_text or '--'}")
        self._depth_label.setText(
            f"深度: {'可用' if depth_available else '不可用'}"
        )
        self._depth_label.setStyleSheet(
            f"color: {'#00ff88' if depth_available else '#ffaa00'};"
            " font-size: 12px; font-family: 'Microsoft YaHei';"
        )
