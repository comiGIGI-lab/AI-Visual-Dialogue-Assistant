"""排行榜对话框"""
import json
import os
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QTableWidget,
                                QTableWidgetItem, QLabel, QPushButton, QHeaderView)
from PySide6.QtCore import Qt

LEADERBOARD_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'game_leaderboard.json')


def load_leaderboard():
    try:
        with open(LEADERBOARD_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def save_leaderboard(entries):
    with open(LEADERBOARD_FILE, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)


class LeaderboardDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('放松记录')
        self.setMinimumSize(500, 400)
        self.setStyleSheet("""
            QDialog { background-color: #1a1a2e; }
            QLabel { color: #ffffff; font-family: "Microsoft YaHei"; }
            QTableWidget {
                background-color: #16213e; color: #e0e0e0;
                border: 1px solid #0f3460; gridline-color: #1a1a3e;
                font-family: "Consolas", "Microsoft YaHei"; font-size: 14px;
            }
            QTableWidget::item { padding: 6px; }
            QHeaderView::section {
                background-color: #0f3460; color: #ffaa00;
                padding: 6px; border: none; font-weight: bold;
            }
            QPushButton {
                background-color: #16213e; color: #e0e0e0;
                border: 1px solid #0f3460; border-radius: 8px;
                padding: 8px 24px; font-size: 16px;
                font-family: "Microsoft YaHei";
            }
            QPushButton:hover { background-color: #1a3a6e; color: #ffffff; }
        """)
        self._setup_ui()
        self._refresh()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        title = QLabel('OfficeFit 放松记录')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet('font-size: 24px; font-weight: bold; color: #ffaa00;')
        layout.addWidget(title)

        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(['排名', '分数', '难度', '日期'])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        layout.addWidget(self.table)

        close_btn = QPushButton('关闭')
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignCenter)

    def _refresh(self):
        entries = load_leaderboard()
        self.table.setRowCount(len(entries))
        for i, e in enumerate(entries):
            rank = QTableWidgetItem(str(i + 1))
            rank.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 0, rank)

            score = QTableWidgetItem(str(e.get('score', 0)))
            score.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 1, score)

            diff = QTableWidgetItem(e.get('difficulty', 'NORMAL'))
            diff.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 2, diff)

            date = QTableWidgetItem(e.get('date', ''))
            date.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(i, 3, date)
