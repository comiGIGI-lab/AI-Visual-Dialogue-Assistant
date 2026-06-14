# -*- coding: utf-8 -*-
"""
动作说明数据与对话框
=====================

提供所有放松动作的文字说明，支持图片占位。
图片放在 assets/exercise_guides/ 目录下。
"""

import os
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QPushButton,
    QScrollArea, QWidget, QHBoxLayout,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QFont

# 图片占位目录
_ASSETS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets", "exercise_guides",
)


def _img_path(name: str) -> str:
    """获取图片完整路径"""
    return os.path.join(_ASSETS_DIR, name)


def _try_load_pixmap(name: str, max_w: int = 200, max_h: int = 140):
    """尝试加载图片，失败返回 None"""
    path = _img_path(name)
    if os.path.isfile(path):
        pix = QPixmap(path)
        if not pix.isNull():
            return pix.scaled(max_w, max_h, Qt.KeepAspectRatio,
                            Qt.SmoothTransformation)
    return None


# ── 动作说明数据 ─────────────────────────────────────────

UPPER_BODY_ACTIONS = [
    {
        "name": "颈部左转",
        "image": "neck_turn_left.png",
        "desc": "缓慢转头看向左侧，保持肩膀放松。",
    },
    {
        "name": "颈部右转",
        "image": "neck_turn_right.png",
        "desc": "缓慢转头看向右侧，保持肩膀放松。",
    },
    {
        "name": "双手上举",
        "image": "both_hands_up.png",
        "desc": "双手同时举过肩膀，尽量伸直手臂，保持 2 秒。",
    },
    {
        "name": "左手上举",
        "image": "left_hand_up.png",
        "desc": "左手向上伸展，右手自然放松，身体正对摄像头。",
    },
    {
        "name": "右手上举",
        "image": "right_hand_up.png",
        "desc": "右手向上伸展，左手自然放松，身体正对摄像头。",
    },
    {
        "name": "扩胸打开",
        "image": "chest_open.png",
        "desc": "双手向身体两侧打开，肩膀放松，打开胸腔。",
    },
    {
        "name": "左侧拉伸",
        "image": "left_stretch.png",
        "desc": "右手上举，身体轻轻向左侧弯，感受右侧身体拉伸。",
    },
    {
        "name": "右侧拉伸",
        "image": "right_stretch.png",
        "desc": "左手上举，身体轻轻向右侧弯，感受左侧身体拉伸。",
    },
]

FULL_BODY_ACTIONS = [
    {
        "name": "左抬腿",
        "image": "left_leg_raise.png",
        "desc": "左腿向前或向上抬起，保持身体稳定。",
    },
    {
        "name": "右抬腿",
        "image": "right_leg_raise.png",
        "desc": "右腿向前或向上抬起，保持身体稳定。",
    },
    {
        "name": "蹲下",
        "image": "squat.png",
        "desc": "膝盖弯曲，身体重心下降，背部尽量保持自然挺直。",
    },
]


# ── 对话框 ───────────────────────────────────────────────

class ActionGuideDialog(QDialog):
    """动作说明弹窗"""

    def __init__(self, parent=None, mode: str = "upper_body"):
        super().__init__(parent)
        self.setWindowTitle("动作说明")
        self.setMinimumSize(600, 500)
        self.setStyleSheet("""
            QDialog { background-color: #0a0a1a; }
            QLabel { color: #d0d0e0; font-family: 'Microsoft YaHei'; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical { background: #0a0a1a; width: 8px; }
            QScrollBar::handle:vertical { background: #1a3050; border-radius: 4px; }
        """)

        layout = QVBoxLayout(self)

        # 标题
        title = QLabel(
            "上半身放松 — 动作说明"
            if mode == "upper_body"
            else "全身互动 — 动作说明"
        )
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            "color: #00d4ff; font-size: 22px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        layout.addWidget(title)

        # 快捷键说明
        shortcuts_text = (
            "快捷键:  F11 切换全屏  |  ESC 返回首页  |  "
            "R 重新开始本轮放松  |  Q 结束本轮放松"
        )
        shortcuts = QLabel(shortcuts_text)
        shortcuts.setAlignment(Qt.AlignCenter)
        shortcuts.setStyleSheet(
            "color: #6677aa; font-size: 12px; font-family: 'Microsoft YaHei';"
            " padding: 4px;")
        layout.addWidget(shortcuts)

        # 反向模式说明
        reverse_text = (
            '绿色提示：做屏幕显示的动作。\n'
            '红色提示：做相反方向的动作。\n'
            '例如：提示「反向 左手上举」，请做右手上举。\n'
            '例如：提示「反向 右侧拉伸」，请做左侧拉伸。'
        )
        reverse_info = QLabel(reverse_text)
        reverse_info.setWordWrap(True)
        reverse_info.setAlignment(Qt.AlignCenter)
        reverse_info.setStyleSheet(
            "color: #ff6688; font-size: 13px; font-family: 'Microsoft YaHei';"
            " background: rgba(255,68,102,0.1); border-radius: 8px; padding: 8px;")
        layout.addWidget(reverse_info)

        # 滚动区域
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(16)

        actions = (UPPER_BODY_ACTIONS if mode == "upper_body"
                   else FULL_BODY_ACTIONS)

        for item in actions:
            card = self._make_card(item)
            scroll_layout.addWidget(card)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # 关闭按钮
        close_btn = QPushButton("关闭")
        close_btn.setObjectName("smallBtn")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)

    def _make_card(self, item: dict) -> QWidget:
        """构建单个动作说明卡片"""
        card = QWidget()
        card.setStyleSheet(
            "background: rgba(15,52,96,0.3); border-radius: 12px;"
            " padding: 12px;")
        card_layout = QHBoxLayout(card)
        card_layout.setSpacing(14)

        # 图片占位
        img_name = item.get("image", "")
        pix = _try_load_pixmap(img_name) if img_name else None
        if pix is not None:
            img_label = QLabel()
            img_label.setPixmap(pix)
            img_label.setFixedSize(200, 140)
            img_label.setAlignment(Qt.AlignCenter)
        else:
            img_label = QLabel("[ 图片占位 ]")
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setFixedSize(200, 140)
            img_label.setStyleSheet(
                "color: #446688; font-size: 14px;"
                " border: 2px dashed #1a3050; border-radius: 10px;"
                " font-family: 'Microsoft YaHei';")

        card_layout.addWidget(img_label)

        # 文字说明
        text_widget = QWidget()
        text_layout = QVBoxLayout(text_widget)
        text_layout.setContentsMargins(0, 0, 0, 0)

        name_label = QLabel(item["name"])
        name_label.setStyleSheet(
            "color: #00ff88; font-size: 18px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        text_layout.addWidget(name_label)

        desc_label = QLabel(item["desc"])
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet(
            "color: #c0d0e0; font-size: 14px;"
            " font-family: 'Microsoft YaHei';")
        text_layout.addWidget(desc_label)

        text_layout.addStretch()
        card_layout.addWidget(text_widget, stretch=1)

        return card
