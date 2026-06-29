@echo off
REM ============================================================================
REM  Himura Pixel Tools - START.BAT
REM  Launches the local backend (FastAPI on 127.0.0.1:8765) using the project
REM  venv and opens the desktop web UI in your default browser.
REM
REM  Usage:  double-click  OR  START.BAT [--no-browser] [--port N] [--host H]
REM  If the venv is missing, it runs SETUP.BAT first.
REM ============================================================================

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set "PF_DIR=%~dp0himura_pixel_tools"
set "VENV=%PF_DIR%\.venv"
set "PYEXE=%VENV%\Scripts\python.exe"

REM --- defaults ---------------------------------------------------------------
set "HOST=127.0.0.1"
set "PORT=8765"
set "OPEN_BROWSER=1"

REM --- parse args -------------------------------------------------------------
:args
if "%~1"=="" goto doneargs
if /i "%~1"=="--no-browser" ( set "OPEN_BROWSER=0" & shift & goto args )
if /i "%~1"=="--port"       ( set "PORT=%~2" & shift & shift & goto args )
if /i "%~1"=="--host"       ( set "HOST=%~2" & shift & shift & goto args )
shift
goto args
:doneargs

echo ================================================================
echo   Himura Pixel Tools - Start
echo ================================================================
echo   Host: %HOST%
echo   Port: %PORT%
echo ----------------------------------------------------------------

REM --- ensure venv exists AND works, otherwise run setup ---------------------
REM Validate Python compatibility and that PyTorch imports. CUDA, DirectML,
REM and CPU fallback are all valid runtime backends.
set "NEED_SETUP=0"
if not exist "%PYEXE%" (
    set "NEED_SETUP=1"
) else (
    "%PYEXE%" -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
    if errorlevel 1 set "NEED_SETUP=1"
    if "!NEED_SETUP!"=="0" (
        "%PYEXE%" -c "import torch" >nul 2>nul
        if errorlevel 1 (
            echo Existing venv cannot import PyTorch. Running SETUP.BAT first ...
            set "NEED_SETUP=1"
        )
    )
)
if "%NEED_SETUP%"=="1" (
    echo Virtual environment missing or incompatible. Running SETUP.BAT first ...
    call "%~dp0SETUP.BAT"
    "%PYEXE%" -c "import torch" >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Setup did not produce a working PyTorch venv. Please run SETUP.BAT manually.
        pause
        exit /b 1
    )
)

REM --- publish port/host to the app ------------------------------------------
set "HIMURA_HOST=%HOST%"
set "HIMURA_PORT=%PORT%"

REM --- print GPU diagnostics --------------------------------------------------
echo Checking PyTorch runtime ...
"%PYEXE%" -c "import importlib.util, torch; dml=importlib.util.find_spec('torch_directml') is not None; print('  torch=' + torch.__version__ + ' cuda_build=' + str(torch.version.cuda) + ' cuda_available=' + str(torch.cuda.is_available()) + ((' device=' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else '') + ' directml_available=' + str(dml))"
if errorlevel 1 echo [WARN] Could not query PyTorch runtime.

REM --- if the API is already running, just open it instead of flashing closed --
netstat -ano -p tcp | findstr /R /C:":%PORT% .*LISTENING" >nul 2>nul
if not errorlevel 1 (
    echo Himura Pixel Tools already appears to be running on http://%HOST%:%PORT%/
    if "%OPEN_BROWSER%"=="1" start "" "http://%HOST%:%PORT%/"
    echo.
    echo Close this window when you are done.
    pause
    exit /b 0
)

REM --- open browser after a short delay (background) -------------------------
if "%OPEN_BROWSER%"=="1" (
    start "" /b cmd /c "timeout /t 2 >nul && start """" http://%HOST%:%PORT%/"
)

REM --- launch the API server (foreground) ------------------------------------
echo Starting Himura Pixel Tools API on http://%HOST%:%PORT%/
echo Press Ctrl+C to stop.
echo ----------------------------------------------------------------
"%PYEXE%" -m himura_pixel_tools.api.server --host %HOST% --port %PORT%
set "EXITCODE=%ERRORLEVEL%"

if not "%EXITCODE%"=="0" (
    echo.
    echo [ERROR] Himura Pixel Tools stopped with exit code %EXITCODE%.
    echo The message above is the real startup error.
    pause
) else (
    echo.
    echo Himura Pixel Tools stopped.
    pause
)

endlocal & exit /b %EXITCODE%


