<#
.SYNOPSIS
  Himura Pixel Tools - one-time setup on Windows.
.DESCRIPTION
  Creates a local Python virtual environment (.venv) inside the project,
  installs dependencies, configures CUDA, DirectML, or CPU PyTorch, and optionally downloads
  recommended models into the local model store on first run.
#>
param(
    [switch]$SkipModels,
    [string]$Python = '',
    [ValidateSet('auto', 'cuda', 'directml', 'cpu')]
    [string]$GpuBackend = 'auto'
)

$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

Write-Host '================================================' -ForegroundColor Cyan
Write-Host '  Himura Pixel Tools - setup' -ForegroundColor Cyan
Write-Host '================================================' -ForegroundColor Cyan

function Get-CompatiblePython {
    param([string]$Requested)

    $candidates = @()
    if ($Requested) { $candidates += @{ Exe = $Requested; Args = @() } }
    $localPython = Join-Path (Split-Path -Parent $Root) 'python312-runtime\python.exe'
    if (Test-Path $localPython) { $candidates += @{ Exe = $localPython; Args = @() } }
    $candidates += @{ Exe = 'py'; Args = @('-3.12') }
    $candidates += @{ Exe = 'py'; Args = @('-3.11') }
    $candidates += @{ Exe = 'python'; Args = @() }

    $versionCheck = 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'
    foreach ($candidate in $candidates) {
        $cmd = Get-Command $candidate.Exe -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $probeArgs = @($candidate.Args) + @('-c', $versionCheck)
        & $candidate.Exe @probeArgs 2>$null
        if ($LASTEXITCODE -eq 0) {
            $pathArgs = @($candidate.Args) + @('-c', 'import sys; print(sys.executable)')
            $exe = & $candidate.Exe @pathArgs
            return $exe.Trim()
        }
    }
    return $null
}

function Resolve-GpuBackend {
    param([string]$Requested)

    if ($Requested -ne 'auto') { return $Requested }
    $nvidia = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($nvidia) { return 'cuda' }
    try {
        $names = (Get-CimInstance Win32_VideoController | Select-Object -ExpandProperty Name) -join ' '
        if ($names -match 'AMD|Radeon|Intel|Arc|Iris|UHD|Xe') { return 'directml' }
    } catch {
        Write-Host 'Could not query GPU adapters, using CPU backend.' -ForegroundColor Yellow
    }
    return 'cpu'
}

$PythonExe = Get-CompatiblePython -Requested $Python
if (-not $PythonExe) {
    Write-Host 'Python 3.11 or 3.12 is required for the local PyTorch runtime.' -ForegroundColor Red
    Write-Host 'Python 3.13/3.14 are skipped because matching torch wheels may not be available.'
    exit 1
}
Write-Host "Using Python: $PythonExe"

$Venv = Join-Path $Root '.venv'
if (Test-Path $Venv) {
    $ExistingPy = Join-Path $Venv 'Scripts\python.exe'
    if (-not (Test-Path $ExistingPy)) {
        Remove-Item -Recurse -Force $Venv
    } else {
        $versionCheck = 'import sys; raise SystemExit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)'
        & $ExistingPy -c $versionCheck 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'Existing venv uses an unsupported Python. Recreating it ...' -ForegroundColor Yellow
            Remove-Item -Recurse -Force $Venv
        }
    }
}

if (-not (Test-Path $Venv)) {
    Write-Host 'Creating virtual environment at .venv ...' -ForegroundColor Yellow
    & $PythonExe -m venv $Venv
} else {
    Write-Host 'Virtual environment already exists at .venv' -ForegroundColor Green
}

$PyExe = Join-Path $Venv 'Scripts\python.exe'
$PipExe = Join-Path $Venv 'Scripts\pip.exe'

Write-Host 'Upgrading pip ...' -ForegroundColor Yellow
& $PyExe -m pip install --upgrade pip wheel 'setuptools<82' | Out-Null

$ResolvedBackend = Resolve-GpuBackend -Requested $GpuBackend
Write-Host "Installing PyTorch backend: $ResolvedBackend" -ForegroundColor Yellow
if ($ResolvedBackend -eq 'cuda') {
    & $PipExe install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'CUDA PyTorch wheel install failed. Check internet access and Python 3.12/3.11, then re-run setup.' -ForegroundColor Red
        exit 1
    }
    $torchCheck = "import torch, sys; print('torch=' + torch.__version__ + ' cuda_build=' + str(torch.version.cuda) + ' cuda_available=' + str(torch.cuda.is_available()) + ((' device=' + torch.cuda.get_device_name(0)) if torch.cuda.is_available() else '')); raise SystemExit(0 if torch.cuda.is_available() else 1)"
    & $PyExe -c $torchCheck
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'CUDA was selected, but PyTorch cannot see an NVIDIA CUDA GPU.' -ForegroundColor Red
        Write-Host 'Update the NVIDIA driver or run setup with -GpuBackend directml or -GpuBackend cpu.'
        exit 1
    }
} elseif ($ResolvedBackend -eq 'directml') {
    & $PipExe install --upgrade --force-reinstall torch torchvision
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $PipExe install --upgrade torch-directml
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'torch-directml install failed. Run setup with -GpuBackend cpu to use CPU fallback.' -ForegroundColor Red
        exit 1
    }
    & $PyExe -c "import torch, torch_directml; print('torch=' + torch.__version__ + ' directml_device=' + str(torch_directml.device()))"
} else {
    $ResolvedBackend = 'cpu'
    & $PipExe install --upgrade --force-reinstall torch torchvision
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $PyExe -c "import torch; print('torch=' + torch.__version__ + ' cpu fallback')"
}

Write-Host 'Installing Himura Pixel Tools dependencies ...' -ForegroundColor Yellow
& $PipExe install -r (Join-Path $Root 'requirements.txt')

Write-Host 'Installing himura_pixel_tools (editable) ...' -ForegroundColor Yellow
& $PipExe install -e $Root
& $PyExe -c "from himura_pixel_tools.config import RuntimeConfig; cfg=RuntimeConfig.load(); cfg.extras['gpu_backend']='$ResolvedBackend'; cfg.save(); print('saved gpu_backend=$ResolvedBackend')"

Write-Host ''
Write-Host 'Setup complete.' -ForegroundColor Green

if (-not $SkipModels) {
    Write-Host ''
    Write-Host 'Download the recommended models now? (y/N) ' -NoNewline -ForegroundColor Yellow
    $reply = Read-Host
    if ($reply -match '^[yY]') {
        & $PyExe -m himura_pixel_tools.runtime.download_models --all
    } else {
        Write-Host 'Skipped. You can download models later with:' -ForegroundColor Gray
        Write-Host '  .venv\Scripts\python.exe -m himura_pixel_tools.runtime.download_models' -ForegroundColor Gray
    }
}

Write-Host ''
Write-Host 'Run the app with:  .\start_windows.ps1' -ForegroundColor Cyan
Write-Host 'Run the app with:  .\himura.bat' -ForegroundColor Cyan
