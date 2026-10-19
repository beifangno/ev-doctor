@echo off
chcp 65001 >nul
cd /d %~dp0
start "" http://127.0.0.1:8000
where python >nul 2>nul && (python server.py) || (py server.py)
pause
