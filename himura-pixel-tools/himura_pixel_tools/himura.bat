@echo off
REM Himura Pixel Tools — convenience launcher.
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo Running first-time setup...
    powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_windows.ps1"
)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start_windows.ps1" %*
endlocal
