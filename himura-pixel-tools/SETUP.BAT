@echo off
REM ============================================================================
REM  Himura Pixel Tools - SETUP.BAT
REM  Creates a local virtual environment (.venv), installs all dependencies,
REM  installs the package, and offers to download the recommended models.
REM
REM  Usage:  double-click  OR  SETUP.BAT [extra args]
REM  Re-runnable: skips steps that are already done.
REM ============================================================================

setlocal enableextensions enabledelayedexpansion
cd /d "%~dp0"

set "PF_DIR=%~dp0himura_pixel_tools"
set "VENV=%PF_DIR%\.venv"
set "PYEXE=%VENV%\Scripts\python.exe"
set "PIPEXE=%VENV%\Scripts\pip.exe"

echo ================================================================
echo   Himura Pixel Tools - Setup
echo ================================================================
echo   Project:  %PF_DIR%
echo   Venv:     %VENV%
echo ----------------------------------------------------------------

REM --- 1. locate a PyTorch-compatible Python ---------------------------------
REM Prefer Python 3.12/3.11. PyTorch wheels may not be available for newer
REM Python versions yet, and using them can silently install an unusable runtime.
set "PYCMD="
if exist "%~dp0python312-runtime\python.exe" (
    "%~dp0python312-runtime\python.exe" -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYCMD=%~dp0python312-runtime\python.exe"
)
if not defined PYCMD where py >nul 2>nul
if not errorlevel 1 (
    py -3.12 -c "import sys" >nul 2>nul
    if not errorlevel 1 set "PYCMD=py -3.12"
    if not defined PYCMD (
        py -3.11 -c "import sys" >nul 2>nul
        if not errorlevel 1 set "PYCMD=py -3.11"
    )
)
if not defined PYCMD (
    python -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYCMD=python"
)
if not defined PYCMD (
    echo [ERROR] Python 3.11 or 3.12 is required for the local PyTorch runtime.
    echo         Python 3.13/3.14 were intentionally skipped because matching torch wheels may not be available.
    echo         Install Python 3.12 from https://python.org and re-run SETUP.BAT.
    echo.
    pause
    exit /b 1
)
echo [1/5] Using Python launcher: %PYCMD%
%PYCMD% -c "import sys; print('      Python ' + sys.version.split()[0] + ' at ' + sys.executable)"

REM --- 2. create / validate venv ---------------------------------------------
REM A venv whose parent folder was renamed has broken .exe launchers that still
REM point at the old path. Validate by actually running the venv python; if it
REM fails, delete and recreate the venv from scratch. Using goto-flow avoids the
REM nested-parenthesis parsing pitfalls of batch IF blocks.
set "NEED_CREATE=0"
if not exist "%PYEXE%" goto :venv_create
"%PYEXE%" -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
if errorlevel 1 goto :venv_broken
"%PYEXE%" -c "import sys" >nul 2>nul
if errorlevel 1 goto :venv_broken
echo [2/5] Virtual environment already exists and is valid, reusing it.
goto :venv_done

:venv_broken
echo [2/5] Existing venv is broken ^(stale launchers^). Recreating it ...
rmdir /s /q "%VENV%"

:venv_create
echo [2/5] Creating virtual environment at .venv ...
%PYCMD% -m venv "%VENV%"
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

:venv_done

REM --- 3. upgrade pip ---------------------------------------------------------
echo [3/5] Upgrading pip / wheel / setuptools ...
REM Pin setuptools below 82 because torch 2.x requires setuptools<82.
"%PYEXE%" -m pip install --upgrade "pip" "wheel" "setuptools<82" >nul

REM --- 4. install torch backend ----------------------------------------------
set "GPU_BACKEND=%HIMURA_GPU_BACKEND%"
if not defined GPU_BACKEND set "GPU_BACKEND=auto"
if /i "%GPU_BACKEND%"=="auto" (
    where nvidia-smi >nul 2>nul
    if not errorlevel 1 (
        set "GPU_BACKEND=cuda"
    ) else (
        powershell -NoProfile -ExecutionPolicy Bypass -Command "$g=(Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ' '; if ($g -match 'AMD|Radeon|Intel|Arc|Iris|UHD|Xe') { exit 0 } else { exit 1 }" >nul 2>nul
        if not errorlevel 1 (set "GPU_BACKEND=directml") else set "GPU_BACKEND=cpu"
    )
)

echo [4/5] Installing PyTorch backend: %GPU_BACKEND%
if /i "%GPU_BACKEND%"=="cuda" (
    echo       Installing PyTorch CUDA 12.1 wheels ...
    "%PIPEXE%" install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if errorlevel 1 (
        echo [ERROR] CUDA PyTorch wheel install failed.
        echo         Check internet access and Python 3.12/3.11, then re-run SETUP.BAT.
        pause
        exit /b 1
    )
    "%PYEXE%" -c "import torch, sys; print('      torch=' + torch.__version__ + ' cuda_build=' + str(torch.version.cuda) + ' cuda_available=' + str(torch.cuda.is_available()) + ((' device=' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else '')); raise SystemExit(0 if torch.cuda.is_available() else 1)"
    if errorlevel 1 (
        echo [ERROR] CUDA was selected, but PyTorch cannot see an NVIDIA CUDA GPU.
        echo         Update the NVIDIA driver or rerun with HIMURA_GPU_BACKEND=directml or cpu.
        pause
        exit /b 1
    )
) else if /i "%GPU_BACKEND%"=="directml" (
    echo       Installing PyTorch CPU wheels plus torch-directml ...
    "%PIPEXE%" install --upgrade --force-reinstall torch torchvision
    if errorlevel 1 exit /b 1
    "%PIPEXE%" install --upgrade torch-directml
    if errorlevel 1 (
        echo [ERROR] torch-directml install failed. Rerun with HIMURA_GPU_BACKEND=cpu to use CPU fallback.
        pause
        exit /b 1
    )
    "%PYEXE%" -c "import torch, torch_directml; print('      torch=' + torch.__version__ + ' directml_device=' + str(torch_directml.device()))"
) else (
    set "GPU_BACKEND=cpu"
    echo       Installing PyTorch CPU wheels ...
    "%PIPEXE%" install --upgrade --force-reinstall torch torchvision
    if errorlevel 1 exit /b 1
    "%PYEXE%" -c "import torch; print('      torch=' + torch.__version__ + ' cpu fallback')"
)

echo       Installing Himura Pixel Tools dependencies ...
"%PIPEXE%" install -r "%PF_DIR%\requirements.txt"

echo       Installing himura_pixel_tools (editable) ...
"%PIPEXE%" install -e "%PF_DIR%"
"%PYEXE%" -c "from himura_pixel_tools.config import RuntimeConfig; cfg=RuntimeConfig.load(); cfg.extras['gpu_backend']='%GPU_BACKEND%'; cfg.save(); print('      saved gpu_backend=%GPU_BACKEND%')"

REM --- 5. offer model download ------------------------------------------------
echo ================================================================
echo   Setup complete.
echo ================================================================
echo.
echo Download the recommended models now? (SDXL base + fp16-fix VAE)
echo   [Y] yes - download now (recommended, ~7 GB)
echo   [N] no  - skip, download later from the Models tab or START menu
set /p "ANS=Choice (Y/N): "
if /i "%ANS%"=="Y" (
    "%PYEXE%" -m himura_pixel_tools.runtime.download_models --all
) else if /i "%ANS%"=="YES" (
    "%PYEXE%" -m himura_pixel_tools.runtime.download_models --all
) else (
    echo Skipped. You can download later with:
    echo   "%PYEXE%" -m himura_pixel_tools.runtime.download_models --all
)

echo.
echo ----------------------------------------------------------------
echo Done. Start the app with:  START.BAT
echo ----------------------------------------------------------------
pause
endlocal



