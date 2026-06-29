@echo off
REM ============================================================================
REM  Build the Himura Pixel Tools Tauri launcher and place the exe next to the
REM  portable app folder. Requires the Rust toolchain (rustup, MSVC host).
REM ============================================================================
setlocal enableextensions
cd /d "%~dp0"

where cargo >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Rust/cargo not found. Install from https://rustup.rs and re-run.
    pause
    exit /b 1
)

echo Building the launcher (release) ...
pushd "%~dp0tauri-launcher\src-tauri"
cargo build --release
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
    echo [ERROR] Build failed with code %RC%.
    pause
    exit /b %RC%
)

set "EXE=%~dp0tauri-launcher\src-tauri\target\release\himura-pixel-tools.exe"
if exist "%EXE%" (
    copy /y "%EXE%" "%~dp0himura-pixel-tools.exe" >nul
    echo.
    echo Done. Launcher built and copied to:
    echo   %~dp0himura-pixel-tools.exe
    echo Keep it next to the himura-pixel-tools folder.
) else (
    echo [ERROR] Build reported success but the exe was not found at:
    echo   %EXE%
    exit /b 1
)
endlocal
