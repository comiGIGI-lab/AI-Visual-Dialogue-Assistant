"""全局 QSS 暗色主题 — 赛博朋克风格"""

GLOBAL_QSS = """
/* ====== 全局 ====== */
QMainWindow {
    background-color: #0a0a1a;
}

/* ====== 菜单页 ====== */
QWidget#menuPage {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #0a0a1a, stop:0.5 #12122a, stop:1 #0a0a1a);
}

QLabel#titleLabel {
    color: #ffffff;
    font-size: 42px;
    font-weight: bold;
    font-family: "Microsoft YaHei";
}

QLabel#subtitleLabel {
    color: #7a7aaa;
    font-size: 16px;
    font-family: "Microsoft YaHei";
}

/* ====== 菜单按钮 — 渐变 + 发光悬停 ====== */
QPushButton#menuBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0f3460, stop:1 #16213e);
    color: #c0c0d0;
    border: 1px solid #1a5276;
    border-radius: 14px;
    padding: 16px 48px;
    font-size: 22px;
    font-weight: bold;
    font-family: "Microsoft YaHei";
    min-width: 280px;
}
QPushButton#menuBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #1a5276, stop:1 #0f3460);
    border-color: #00d4ff;
    color: #ffffff;
}
QPushButton#menuBtn:pressed {
    background: #0a2a4a;
    border-color: #00ff88;
}

/* ====== 难度选择按钮（卡片风格） ====== */
QPushButton#diffCard {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0d1b3e, stop:1 #16213e);
    color: #a0a0c0;
    border: 2px solid #1a3050;
    border-radius: 16px;
    padding: 18px 28px;
    font-size: 17px;
    font-family: "Microsoft YaHei";
    text-align: left;
}
QPushButton#diffCard:hover {
    border-color: #00d4ff;
    color: #e0e0f0;
}
QPushButton#diffCard[selected="true"] {
    border-color: #00ff88;
    color: #ffffff;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #0a2a1a, stop:1 #0f3460);
}

/* ====== 返回/小按钮 ====== */
QPushButton#smallBtn {
    background: transparent;
    color: #6666aa;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 10px 28px;
    font-size: 16px;
    font-family: "Microsoft YaHei";
}
QPushButton#smallBtn:hover {
    background: #1a1a3e;
    border-color: #5555aa;
    color: #aaaadd;
}

/* ====== 退出按钮 ====== */
QPushButton#exitBtn {
    background: transparent;
    color: #aa5566;
    border: 2px solid #aa5566;
    border-radius: 14px;
    padding: 14px 44px;
    font-size: 20px;
    font-family: "Microsoft YaHei";
    min-width: 280px;
}
QPushButton#exitBtn:hover {
    background: #aa5566;
    color: #ffffff;
}

/* ====== 玻璃面板 — HUD 计分板 ====== */
QWidget#glassPanel {
    background: rgba(10, 10, 30, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
}

QLabel#scoreLabel {
    color: #00ff88;
    font-size: 34px;
    font-weight: bold;
    font-family: "Consolas", "Microsoft YaHei";
}
QLabel#fpsLabel {
    color: #6677aa;
    font-size: 34px;
    font-weight: bold;
    font-family: "Consolas", "Microsoft YaHei";
}
QLabel#comboLabel {
    color: #ffaa00;
    font-size: 26px;
    font-weight: bold;
    font-family: "Consolas", "Microsoft YaHei";
}
QLabel#comboLabelHigh {
    color: #ff6600;
    font-size: 30px;
    font-weight: bold;
    font-family: "Consolas", "Microsoft YaHei";
}
QLabel#timeLabel {
    color: #00d4ff;
    font-size: 30px;
    font-weight: bold;
    font-family: "Consolas", "Microsoft YaHei";
}
QLabel#timeLabelWarn {
    color: #ff4444;
    font-size: 32px;
    font-weight: bold;
    font-family: "Consolas", "Microsoft YaHei";
}
QLabel#modeLabel {
    color: #6677aa;
    font-size: 16px;
    font-family: "Microsoft YaHei";
}

/* ====== 底部时间条 ====== */
QWidget#timeBarWidget {
    background: rgba(10, 10, 30, 0.7);
    border-top: 1px solid rgba(255, 255, 255, 0.06);
}

/* ====== Action 提示条 ====== */
QWidget#promptBarWidget {
    background: rgba(10, 10, 30, 0.7);
    border-bottom: 1px solid rgba(255, 255, 255, 0.06);
}

/* ====== 游戏结束覆盖层 ====== */
QWidget#gameOverOverlay {
    background: rgba(5, 5, 20, 0.92);
}
QLabel#gameOverTitle {
    color: #ff4466;
    font-size: 52px;
    font-weight: bold;
    font-family: "Microsoft YaHei";
}
QLabel#gameOverScore {
    color: #ffffff;
    font-size: 30px;
    font-family: "Microsoft YaHei";
}
QLabel#gameOverBest {
    color: #ffaa00;
    font-size: 24px;
    font-family: "Microsoft YaHei";
}

/* ====== 排行榜 ====== */
QTableWidget {
    background-color: #0d1b3e;
    color: #d0d0e0;
    border: 1px solid #1a3050;
    gridline-color: #1a3050;
    font-family: "Consolas", "Microsoft YaHei";
    font-size: 14px;
    border-radius: 10px;
}
QTableWidget::item {
    padding: 8px;
}
QTableWidget::item:selected {
    background-color: #0f3460;
}
QHeaderView::section {
    background-color: #0f3460;
    color: #ffaa00;
    padding: 8px;
    border: none;
    font-weight: bold;
    font-size: 15px;
}
QDialog {
    background-color: #0a0a1a;
}

/* ====== Radio / Check ====== */
QCheckBox {
    color: #c0c0d0;
    font-size: 20px;
    font-family: "Microsoft YaHei";
}
QCheckBox::indicator {
    width: 26px; height: 26px;
    background-color: #0d1b3e;
    border: 2px solid #1a3050;
    border-radius: 6px;
}
QCheckBox::indicator:checked {
    background-color: #0f3460;
    border-color: #00ff88;
}

/* ====== ScrollBar ====== */
QScrollBar:vertical {
    background: #0a0a1a;
    width: 8px;
    border-radius: 4px;
}
QScrollBar::handle:vertical {
    background: #1a3050;
    border-radius: 4px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background: #0f3460;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

"""


def get_difficulty_color(mode):
    return {'practice': '#00ff88', 'normal': '#ffaa00', 'hard': '#ff4466'}.get(mode, '#ffffff')
