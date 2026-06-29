<#
.SYNOPSIS
  Starts the Himura Pixel Tools backend (FastAPI on 127.0.0.1:8765) and opens
  the desktop web UI in the default browser. Reuses the .venv from setup.
#>
param(
    [int]$Port = 8765,
    [string]$Host_ = "127.0.0.1",
    [switch]$NoBrowser,
    [switch]$SkipModels
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root
$Venv  = Join-Path $Root ".venv"
$PyExe = Join-Path $Venv "Scripts\python.exe"
$Setup = Join-Path $Root "setup_windows.ps1"

function Test-WorkingRuntime {
    param([string]$PythonPath)
    if (-not (Test-Path $PythonPath)) { return $false }
    & $PythonPath -c "import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" 2>$null
    if ($LASTEXITCODE -ne 0) { return $false }
    & $PythonPath -c "import torch" 2>$null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-WorkingRuntime -PythonPath $PyExe)) {
    Write-Host "Virtual environment missing or incompatible. Running setup_windows.ps1 first ..." -ForegroundColor Yellow
    if ($SkipModels) {
        & $Setup -SkipModels
    } else {
        & $Setup
    }
    if (-not (Test-WorkingRuntime -PythonPath $PyExe)) {
        Write-Host "Setup did not produce a working PyTorch runtime. Please run setup_windows.ps1 manually." -ForegroundColor Red
        exit 1
    }
}

$env:HIMURA_HOST = $Host_
$env:HIMURA_PORT = "$Port"

Write-Host "Checking PyTorch runtime ..." -ForegroundColor Yellow
& $PyExe -c "import importlib.util, torch; dml=importlib.util.find_spec('torch_directml') is not None; print('  torch=' + torch.__version__ + ' cuda_build=' + str(torch.version.cuda) + ' cuda_available=' + str(torch.cuda.is_available()) + ((' device=' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else '') + ' directml_available=' + str(dml))"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Could not query PyTorch runtime." -ForegroundColor Yellow
}

if (-not $NoBrowser) {
    $url = "http://$Host_`:$Port/"
    Write-Host "Opening browser at $url in 2 seconds ..." -ForegroundColor Yellow
    Start-Job -ScriptBlock { param($u) Start-Sleep -Seconds 2; Start-Process $u } -ArgumentList $url | Out-Null
}

Write-Host "Starting Himura Pixel Tools API on http://$Host_`:$Port" -ForegroundColor Cyan
& $PyExe -m himura_pixel_tools.api.server --host $Host_ --port $Port

