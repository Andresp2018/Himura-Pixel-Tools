#!/usr/bin/env bash
# Himura Pixel Tools — one-time setup on Linux/macOS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
SKIP_MODELS="${SKIP_MODELS:-0}"

echo "================================================"
echo "  Himura Pixel Tools — setup"
echo "================================================"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "Python ('$PYTHON') was not found on PATH."
    echo "Install Python 3.11/3.12 and re-run, or set PYTHON=/path/to/python"
    exit 1
fi
echo "Using Python: $(command -v "$PYTHON")"

VENV="$ROOT/.venv"
if [[ ! -d "$VENV" ]]; then
    echo "Creating virtual environment at .venv ..."
    "$PYTHON" -m venv "$VENV"
else
    echo "Virtual environment already exists at .venv"
fi

PYEXE="$VENV/bin/python"
PIPEXE="$VENV/bin/pip"

echo "Upgrading pip ..."
"$PYEXE" -m pip install --upgrade pip wheel setuptools >/dev/null

echo "Installing PyTorch (CUDA 12.1 wheels — falls back to CPU if unavailable) ..."
"$PIPEXE" install torch torchvision --index-url https://download.pytorch.org/whl/cu121 || \
    "$PIPEXE" install torch torchvision

echo "Installing Himura Pixel Tools dependencies ..."
"$PIPEXE" install -r "$ROOT/requirements.txt"

echo "Installing himura_pixel_tools (editable) ..."
"$PIPEXE" install -e "$ROOT"

echo ""
echo "Setup complete."

if [[ "$SKIP_MODELS" != "1" ]]; then
    echo ""
    read -r -p "Download the recommended models now? (y/N) " reply
    if [[ "$reply" =~ ^[yY] ]]; then
        "$PYEXE" -m himura_pixel_tools.runtime.download_models --all
    else
        echo "Skipped. You can download models later with:"
        echo "  ./.venv/bin/python -m himura_pixel_tools.runtime.download_models"
    fi
fi

echo ""
echo "Run the app with:  ./start.sh"
