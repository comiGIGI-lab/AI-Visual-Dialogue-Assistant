"""菜单页面控件"""
from PySide6.QtWidgets import (QWidget, QLabel, QPushButton, QVBoxLayout,
                                QHBoxLayout, QCheckBox, QFrame)
from PySide6.QtCore import Qt, Signal, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QFont


class MenuButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("menuBtn")
        self.setCursor(Qt.PointingHandCursor)


class SmallButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("smallBtn")
        self.setCursor(Qt.PointingHandCursor)


class MenuPage(QWidget):
    start_clicked = Signal()
    leaderboard_clicked = Signal()
    exit_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("menuPage")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(16)

        # 顶部装饰线
        accent = QFrame()
        accent.setFixedSize(80, 3)
        accent.setStyleSheet("background: #00d4ff; border-radius: 2px; border: none;")
        layout.addWidget(accent, alignment=Qt.AlignCenter)
        layout.addSpacing(10)

        # 标题
        title = QLabel("OfficeFit AI 视觉对话放松助手")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("面向久坐办公人群的 3D 视觉互动放松助手")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        desc = QLabel("通过 Orbbec 3D 相机、YOLO 人体检测和 MediaPipe 姿态估计，引导用户完成轻量化身体活动。")
        desc.setStyleSheet("color: #556688; font-size: 14px; font-family: 'Microsoft YaHei';")
        desc.setAlignment(Qt.AlignCenter)
        layout.addWidget(desc)

        layout.addSpacing(28)

        self.btn_start = MenuButton("  开始 2 分钟放松  ")
        self.btn_start.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.btn_start, alignment=Qt.AlignCenter)

        self.btn_leaderboard = MenuButton("  放松记录  ")
        self.btn_leaderboard.clicked.connect(self.leaderboard_clicked.emit)
        layout.addWidget(self.btn_leaderboard, alignment=Qt.AlignCenter)

        self.btn_exit = QPushButton("  退出  ")
        self.btn_exit.setObjectName("exitBtn")
        self.btn_exit.setCursor(Qt.PointingHandCursor)
        self.btn_exit.clicked.connect(self.exit_clicked.emit)
        layout.addWidget(self.btn_exit, alignment=Qt.AlignCenter)

        layout.addSpacing(30)

        hint = QLabel("Orbbec 深度相机  |  MediaPipe + YOLO 姿态识别  |  F11 全屏")
        hint.setStyleSheet("color: #3a3a5a; font-size: 13px; font-family: 'Microsoft YaHei';")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)


class DifficultyPage(QWidget):
    difficulty_selected = Signal(str)
    back_clicked = Signal()

    DIFFICULTIES = [
        ('practice', '轻松模式', '适合办公间隙，动作节奏较慢', '#00ff88'),
        ('normal',   '标准模式', '推荐日常使用，动作节奏适中', '#ffaa00'),
        ('hard',     '活力模式', '节奏更快，适合状态较好时挑战', '#ff4466'),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("menuPage")
        self._selected = 'normal'

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)

        accent = QFrame()
        accent.setFixedSize(60, 3)
        accent.setStyleSheet("background: #ffaa00; border-radius: 2px; border: none;")
        layout.addWidget(accent, alignment=Qt.AlignCenter)
        layout.addSpacing(8)

        title = QLabel("选择放松模式")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(18)

        self._buttons = {}
        for diff_id, diff_name, diff_desc, _ in self.DIFFICULTIES:
            btn = QPushButton(f"  {diff_name}\n  {diff_desc}")
            btn.setObjectName("diffCard")
            btn.setFixedWidth(420)
            btn.setMinimumHeight(80)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, d=diff_id: self._select(d))
            layout.addWidget(btn, alignment=Qt.AlignCenter)
            self._buttons[diff_id] = btn

        layout.addSpacing(22)

        self._confirm_btn = MenuButton("  开始游戏  ")
        self._confirm_btn.clicked.connect(lambda: self.difficulty_selected.emit(self._selected))
        layout.addWidget(self._confirm_btn, alignment=Qt.AlignCenter)

        back_btn = SmallButton("  返回  ")
        back_btn.clicked.connect(self.back_clicked.emit)
        layout.addWidget(back_btn, alignment=Qt.AlignCenter)

        self._update_highlights()

    def _select(self, diff_id):
        self._selected = diff_id
        self._update_highlights()

    def _update_highlights(self):
        colors = {d[0]: d[3] for d in self.DIFFICULTIES}
        for diff_id, btn in self._buttons.items():
            if diff_id == self._selected:
                c = colors[diff_id]
                btn.setProperty("selected", True)
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 rgba(0,40,20,200), stop:1 #0f3460);
                        color: {c};
                        border: 2px solid {c};
                        border-radius: 16px;
                        padding: 18px 28px;
                        font-size: 17px;
                        font-weight: bold;
                        font-family: "Microsoft YaHei";
                        text-align: left;
                    }}
                """)
            else:
                btn.setProperty("selected", False)
                btn.setStyleSheet("")


class SettingsPage(QWidget):
    sound_toggled = Signal(bool)
    back_clicked = Signal()

    def __init__(self, sound_enabled=True, parent=None):
        super().__init__(parent)
        self.setObjectName("menuPage")

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(20)

        accent = QFrame()
        accent.setFixedSize(60, 3)
        accent.setStyleSheet("background: #00d4ff; border-radius: 2px; border: none;")
        layout.addWidget(accent, alignment=Qt.AlignCenter)
        layout.addSpacing(8)

        title = QLabel("设置")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addSpacing(20)

        self._sound_check = QCheckBox("  启用音效")
        self._sound_check.setChecked(sound_enabled)
        self._sound_check.setCursor(Qt.PointingHandCursor)
        self._sound_check.toggled.connect(self.sound_toggled.emit)
        layout.addWidget(self._sound_check, alignment=Qt.AlignCenter)

        layout.addSpacing(40)

        back_btn = SmallButton("  返回  ")
        back_btn.clicked.connect(self.back_clicked.emit)
        layout.addWidget(back_btn, alignment=Qt.AlignCenter)


class GameOverPage(QWidget):
    restart_clicked = Signal()
    menu_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("gameOverOverlay")
        self.setAttribute(Qt.WA_TranslucentBackground, False)

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(14)

        self._title = QLabel("本次放松完成")
        self._title.setObjectName("gameOverTitle")
        self._title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._title)

        self._score_label = QLabel()
        self._score_label.setObjectName("gameOverScore")
        self._score_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._score_label)

        self._best_label = QLabel()
        self._best_label.setObjectName("gameOverBest")
        self._best_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._best_label)

        # 装饰线
        sep = QFrame()
        sep.setFixedSize(200, 2)
        sep.setStyleSheet("background: #00d4ff; border-radius: 1px; border: none;")
        layout.addWidget(sep, alignment=Qt.AlignCenter)
        layout.addSpacing(8)

        self._restart_btn = MenuButton("  再来一局  ")
        self._restart_btn.clicked.connect(self.restart_clicked.emit)
        layout.addWidget(self._restart_btn, alignment=Qt.AlignCenter)

        self._menu_btn = SmallButton("  返回菜单  ")
        self._menu_btn.clicked.connect(self.menu_clicked.emit)
        layout.addWidget(self._menu_btn, alignment=Qt.AlignCenter)

    def set_results(self, score, high_score, mode):
        self._score_label.setText(f"得分  {score}")
        self._best_label.setText(f"最高  {high_score}    ·    {mode.upper()}")

