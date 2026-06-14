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
    office_clicked = Signal()
    leaderboard_clicked = Signal()
    exit_clicked = Signal()
    guide_clicked = Signal()

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

        subtitle = QLabel("面向久坐办公人群的 3D 视觉健康陪伴助手")
        subtitle.setObjectName("subtitleLabel")
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        desc = QLabel(
            'AI 会观察你的坐姿与活动状态。你可以说出身体不适，'
            '例如“我肩膀酸”“我脖子紧”“腰背僵硬”，系统会推荐适合的放松动作。')
        desc.setStyleSheet("color: #556688; font-size: 14px; font-family: 'Microsoft YaHei';")
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        layout.addWidget(desc)

        layout.addSpacing(28)

        _sub_style = ("color: #6677aa; font-size: 12px;"
                      " font-family: 'Microsoft YaHei';")

        self.btn_start = MenuButton("  开始办公  ")
        self.btn_start.clicked.connect(self.office_clicked.emit)
        layout.addWidget(self.btn_start, alignment=Qt.AlignCenter)
        office_sub = QLabel("进入 AI 办公守护模式，持续观察久坐状态")
        office_sub.setStyleSheet(_sub_style)
        office_sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(office_sub)

        self.btn_relax = MenuButton("  开始 2 分钟放松  ")
        self.btn_relax.clicked.connect(self.start_clicked.emit)
        layout.addWidget(self.btn_relax, alignment=Qt.AlignCenter)
        relax_sub = QLabel("进入结构化放松训练")
        relax_sub.setStyleSheet(_sub_style)
        relax_sub.setAlignment(Qt.AlignCenter)
        layout.addWidget(relax_sub)

        self.btn_guide = MenuButton("  使用说明  ")
        self.btn_guide.clicked.connect(self.guide_clicked.emit)
        layout.addWidget(self.btn_guide, alignment=Qt.AlignCenter)

        self.btn_leaderboard = MenuButton("  放松记录  ")
        self.btn_leaderboard.clicked.connect(self.leaderboard_clicked.emit)
        layout.addWidget(self.btn_leaderboard, alignment=Qt.AlignCenter)

        self.btn_exit = QPushButton("  退出  ")
        self.btn_exit.setObjectName("exitBtn")
        self.btn_exit.setCursor(Qt.PointingHandCursor)
        self.btn_exit.clicked.connect(self.exit_clicked.emit)
        layout.addWidget(self.btn_exit, alignment=Qt.AlignCenter)

        layout.addSpacing(20)

        hint = QLabel("Orbbec 3D 相机｜MediaPipe 姿态识别｜本地语音与可选 AI 回复")
        hint.setStyleSheet("color: #3a3a5a; font-size: 13px; font-family: 'Microsoft YaHei';")
        hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(hint)


class DifficultyPage(QWidget):
    difficulty_selected = Signal(str)
    workout_mode_selected = Signal(str)   # upper_body / full_body
    action_guide_clicked = Signal(str)     # 传递当前 workout_mode
    back_clicked = Signal()

    DIFFICULTIES = [
        ('practice', '轻松模式', '节奏最慢，适合初次使用或识别较慢时', '#00ff88'),
        ('normal',   '标准模式', '推荐日常使用，节奏较慢，识别更稳', '#ffaa00'),
        ('hard',     '活力模式', '节奏适中偏快，适合状态较好时挑战', '#ff4466'),
    ]

    WORKOUT_MODES = [
        ('upper_body', '上半身放松',
         '适合坐在办公桌前完成，只需露出头部、肩部和双手。', '#00ff88'),
        ('full_body', '全身互动',
         '需要站立并完整入镜，包含更多身体动作。', '#ffaa00'),
        ('ai_recommend', 'AI 推荐',
         '根据当前观察状态和语音描述生成动作组合。', '#00d4ff'),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("menuPage")
        self._selected = 'normal'
        self._workout_mode = 'upper_body'

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setSpacing(12)

        accent = QFrame()
        accent.setFixedSize(60, 3)
        accent.setStyleSheet("background: #ffaa00; border-radius: 2px; border: none;")
        layout.addWidget(accent, alignment=Qt.AlignCenter)
        layout.addSpacing(6)

        # 标题
        title = QLabel("选择本次放松方案")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        # ── 放松类型 ──
        type_label = QLabel("— 放松类型 —")
        type_label.setStyleSheet(
            "color: #6677aa; font-size: 14px; font-family: 'Microsoft YaHei';")
        type_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(type_label)

        self._mode_buttons = {}
        mode_row = QHBoxLayout()
        mode_row.setSpacing(12)
        for mode_id, mode_name, mode_desc, color in self.WORKOUT_MODES:
            btn = QPushButton(f"  {mode_name}\n  {mode_desc}")
            btn.setObjectName("diffCard")
            btn.setFixedWidth(280)
            btn.setMinimumHeight(65)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, m=mode_id: self._select_mode(m))
            mode_row.addWidget(btn, alignment=Qt.AlignCenter)
            self._mode_buttons[mode_id] = (btn, color)
        layout.addLayout(mode_row)

        # 动作说明按钮
        guide_btn = QPushButton("  查看动作说明  ")
        guide_btn.setObjectName("smallBtn")
        guide_btn.setCursor(Qt.PointingHandCursor)
        guide_btn.clicked.connect(
            lambda: self.action_guide_clicked.emit(self._workout_mode))
        layout.addWidget(guide_btn, alignment=Qt.AlignCenter)

        layout.addSpacing(8)

        # ── 难度选择 ──
        diff_label = QLabel("— 难度 —")
        diff_label.setStyleSheet(
            "color: #6677aa; font-size: 14px; font-family: 'Microsoft YaHei';")
        diff_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(diff_label)

        self._buttons = {}
        for diff_id, diff_name, diff_desc, _ in self.DIFFICULTIES:
            btn = QPushButton(f"  {diff_name}\n  {diff_desc}")
            btn.setObjectName("diffCard")
            btn.setFixedWidth(420)
            btn.setMinimumHeight(70)
            btn.setCursor(Qt.PointingHandCursor)
            btn.clicked.connect(lambda checked, d=diff_id: self._select(d))
            layout.addWidget(btn, alignment=Qt.AlignCenter)
            self._buttons[diff_id] = btn

        layout.addSpacing(16)

        self._confirm_btn = MenuButton("  开始放松  ")
        self._confirm_btn.clicked.connect(
            lambda: self.difficulty_selected.emit(self._selected))
        layout.addWidget(self._confirm_btn, alignment=Qt.AlignCenter)

        back_btn = SmallButton("  返回  ")
        back_btn.clicked.connect(self.back_clicked.emit)
        layout.addWidget(back_btn, alignment=Qt.AlignCenter)

        self._update_highlights()
        self._update_mode_highlights()

    def _select(self, diff_id):
        self._selected = diff_id
        self._update_highlights()

    def _select_mode(self, mode_id):
        self._workout_mode = mode_id
        self._update_mode_highlights()
        self.workout_mode_selected.emit(mode_id)

    def _update_mode_highlights(self):
        for mode_id, (btn, color) in self._mode_buttons.items():
            if mode_id == self._workout_mode:
                btn.setProperty("selected", True)
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                            stop:0 rgba(0,40,20,200), stop:1 #0f3460);
                        color: {color};
                        border: 2px solid {color};
                        border-radius: 16px;
                        padding: 14px 20px;
                        font-size: 15px;
                        font-weight: bold;
                        font-family: "Microsoft YaHei";
                        text-align: left;
                    }}
                """)
            else:
                btn.setProperty("selected", False)
                btn.setStyleSheet("")

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

        self._restart_btn = MenuButton("  再来一轮  ")
        self._restart_btn.clicked.connect(self.restart_clicked.emit)
        layout.addWidget(self._restart_btn, alignment=Qt.AlignCenter)

        self._menu_btn = SmallButton("  返回首页  ")
        self._menu_btn.clicked.connect(self.menu_clicked.emit)
        layout.addWidget(self._menu_btn, alignment=Qt.AlignCenter)

    def set_results(self, score, high_score, mode):
        mode_names = {'practice': '轻松', 'normal': '标准', 'hard': '活力'}
        display = mode_names.get(mode, mode)
        self._score_label.setText(f"完成度  {score}")
        self._best_label.setText(f"最佳  {high_score}    ·    {display}")

    def set_session_ended(self):
        """用户手动结束本轮放松"""
        self._title.setText("本轮放松已结束")
        self._title.setStyleSheet(
            "color: #00d4ff; font-size: 42px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        self._score_label.setText("可返回首页或重新开始一轮")
        self._score_label.setStyleSheet(
            "color: #c0d0e0; font-size: 22px; font-family: 'Microsoft YaHei';")
        self._best_label.setText("")
        self._restart_btn.setText("  再来一轮  ")
        self._menu_btn.setText("  返回首页  ")

    def reset_title_style(self):
        """恢复默认标题样式（用于重新开始/自然结束时）"""
        self._title.setText("本次放松完成")
        self._title.setStyleSheet(
            "color: #ff4466; font-size: 52px; font-weight: bold;"
            " font-family: 'Microsoft YaHei';")
        self._score_label.setStyleSheet(
            "color: #ffffff; font-size: 30px; font-family: 'Microsoft YaHei';")

