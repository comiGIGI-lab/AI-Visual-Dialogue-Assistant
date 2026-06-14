"""主窗口：QStackedWidget 管理 menu → difficulty → game 流程

Phase 流程:
  loading   — 后端加载中，前端显示进度条
  countdown — 3 秒倒计时，"请举起双手开始游戏"
  playing   — 正常游戏
  game_over — 游戏结束覆盖层
"""
import datetime
import winsound

from PySide6.QtWidgets import (QMainWindow, QStackedWidget, QMessageBox,
                                QApplication)
from PySide6.QtCore import Qt, QTimer

from game_frontend.styles import GLOBAL_QSS
from game_frontend.menu_widgets import MenuPage, DifficultyPage, SettingsPage, GameOverPage
from game_frontend.game_widgets import GamePage
from game_frontend.leaderboard_dialog import LeaderboardDialog, load_leaderboard, save_leaderboard
from game_frontend.backend_thread import BackendThread

PAGE_MENU = 0
PAGE_DIFFICULTY = 1
PAGE_SETTINGS = 2
PAGE_GAME = 3


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OfficeFit AI 视觉对话放松助手")
        self.setMinimumSize(1024, 768)
        self.resize(1280, 860)
        self.setStyleSheet(GLOBAL_QSS)
        self.showFullScreen()

        self._difficulty = 'normal'
        self._sound_enabled = True
        self._backend = None
        self._prev_score = 0
        self._prev_combo = 0
        self._leaderboard_saved = False
        self._game_phase = 'idle'  # idle | loading | countdown | playing | game_over

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self._menu_page = MenuPage()
        self._difficulty_page = DifficultyPage()
        self._settings_page = SettingsPage()
        self._game_page = GamePage()
        self._game_over_page = GameOverPage()

        self._stack.addWidget(self._menu_page)
        self._stack.addWidget(self._difficulty_page)
        self._stack.addWidget(self._settings_page)
        self._stack.addWidget(self._game_page)

        # 菜单
        self._menu_page.start_clicked.connect(self._on_start_clicked)
        self._menu_page.leaderboard_clicked.connect(self._show_leaderboard)
        self._menu_page.exit_clicked.connect(self.close)

        # 难度
        self._difficulty_page.difficulty_selected.connect(self._on_difficulty_selected)
        self._difficulty_page.back_clicked.connect(lambda: self._stack.setCurrentIndex(PAGE_MENU))

        # 设置
        self._settings_page.back_clicked.connect(lambda: self._stack.setCurrentIndex(PAGE_MENU))
        self._settings_page.sound_toggled.connect(self._on_sound_toggled)

        # 游戏结束覆盖层
        self._game_over_page.restart_clicked.connect(self._restart_game)
        self._game_over_page.menu_clicked.connect(self._back_to_menu)
        self._game_over_page.hide()

        # 倒计时完成
        self._game_page.countdown_finished.connect(self._on_countdown_finished)

        self.setFocusPolicy(Qt.StrongFocus)

    # ---------- 页面切换 ----------

    def _on_start_clicked(self):
        self._stack.setCurrentIndex(PAGE_DIFFICULTY)

    def _on_difficulty_selected(self, difficulty):
        self._difficulty = difficulty
        self._leaderboard_saved = False
        self._prev_score = 0
        self._prev_combo = 0
        self._game_phase = 'idle'

        # 重置 GamePage 到 loading 状态，然后立即显示
        self._game_page.reset_loading()
        self._stack.setCurrentIndex(PAGE_GAME)
        self._start_backend()

    def _on_countdown_finished(self):
        """倒计时结束，游戏正式开始"""
        self._game_phase = 'playing'

    def _show_leaderboard(self):
        dlg = LeaderboardDialog(self)
        dlg.exec()

    def _on_sound_toggled(self, enabled):
        self._sound_enabled = enabled

    def _back_to_menu(self):
        self._stop_backend()
        self._game_over_page.hide()
        self._game_phase = 'idle'
        self._stack.setCurrentIndex(PAGE_MENU)

    # ---------- 后端管理 ----------

    def _start_backend(self):
        self._stop_backend()
        self._backend = BackendThread(
            mode=self._difficulty,
            sound_enabled=False,
            parent=self,
        )
        self._backend.status_updated.connect(self._on_status_updated)
        self._backend.start()

    def _stop_backend(self):
        if self._backend is None:
            return
        try:
            self._backend.status_updated.disconnect(self._on_status_updated)
        except (TypeError, RuntimeError):
            pass
        if self._backend.isRunning():
            self._backend.stop()
        self._backend = None

    def _restart_game(self):
        self._game_over_page.hide()
        self._leaderboard_saved = False
        self._prev_score = 0
        self._prev_combo = 0
        self._game_phase = 'idle'
        self._game_page.reset_loading()
        self._start_backend()

    def closeEvent(self, event):
        self._stop_backend()
        super().closeEvent(event)

    # ---------- 状态更新 ----------

    def _on_status_updated(self, status: dict):
        phase = status.get('phase', 'idle')
        self._game_page.update_status(status)

        # 仅在 playing 阶段处理后端分数变化
        if phase == 'playing':
            score = status.get('score', 0)
            combo = status.get('combo', 0)
            if score > self._prev_score and self._sound_enabled:
                if combo >= 3:
                    winsound.Beep(1200, 80)
                winsound.Beep(880, 100)
                winsound.Beep(1100, 120)
            self._prev_score = score
            self._prev_combo = combo

        # 单人游戏结束
        if phase == 'game_over' and not self._leaderboard_saved:
            self._on_game_over(status)

    def _on_game_over(self, status: dict):
        self._leaderboard_saved = True
        score = status.get('score', 0)
        high_score = status.get('high_score', 0)
        mode = status.get('mode', 'practice')

        if score > 0:
            entries = load_leaderboard()
            entries.append({
                'score': score,
                'difficulty': mode.upper(),
                'date': datetime.date.today().isoformat(),
            })
            entries.sort(key=lambda x: x['score'], reverse=True)
            entries = entries[:20]
            save_leaderboard(entries)

        if self._sound_enabled:
            winsound.Beep(200, 500)

        self._game_over_page.set_results(score, high_score, mode)
        self._game_over_page.setParent(self._game_page)
        self._game_over_page.setGeometry(0, 0,
                                         self._game_page.width(),
                                         self._game_page.height())
        self._game_over_page.show()
        self._game_over_page.raise_()

    # ---------- 键盘快捷键 ----------

    def keyPressEvent(self, event):
        key = event.key()

        if key == Qt.Key_F11:
            if self.isFullScreen():
                self.showNormal()
            else:
                self.showFullScreen()
            return

        if self._stack.currentIndex() == PAGE_GAME:
            if key == Qt.Key_R:
                self._restart_game()
            elif key == Qt.Key_Escape or key == Qt.Key_Q:
                self._back_to_menu()
        elif self._stack.currentIndex() == PAGE_MENU:
            if key == Qt.Key_Escape:
                self.close()

        super().keyPressEvent(event)
