"""PySide6 游戏前端入口"""
import sys
import os

# 切换到脚本所在目录，确保 models/ 等相对路径正确
_here = os.path.dirname(os.path.abspath(__file__))
os.chdir(_here)
if _here not in sys.path:
    sys.path.insert(0, _here)

from PySide6.QtWidgets import QApplication
from game_frontend.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("动作模仿挑战游戏")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == '__main__':
    main()
