@echo off
chcp 65001 >nul
cd /d "%~dp0"
call C:\Users\34356\anaconda3\envs\ob\python.exe run_game_frontend.py
pause
