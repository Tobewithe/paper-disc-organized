@echo off
setlocal
"C:\Users\28358\anaconda3\envs\pytorch\python.exe" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8871/api/catalog',timeout=2)" >nul 2>&1
if not errorlevel 1 (
  start "" "http://127.0.0.1:8871"
  exit /b 0
)
cd /d "%~dp0code"
echo TraceLab: http://127.0.0.1:8871
echo Keep this window open while using the demo.
start "" "http://127.0.0.1:8871"
"C:\Users\28358\anaconda3\envs\pytorch\python.exe" -m shared.dense_video_identity serve --root "%~dp0." --port 8871
pause
