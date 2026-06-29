#!/usr/bin/env bash
# Himura Pixel Tools — launcher (Linux/macOS).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYEXE="$ROOT/.venv/bin/python"
if [[ ! -x "$PYEXE" ]]; then
    echo "Virtual environment not found. Run ./setup.sh first."
    exit 1
fi

HOST="${HIMURA_HOST:-127.0.0.1}"
PORT="${HIMURA_PORT:-8765}"

if [[ "${1:-}" != "--no-browser" ]]; then
    URL="http://${HOST}:${PORT}/"
    echo "Opening browser at $URL in 2 seconds ..."
    ( sleep 2; if command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL"; fi ) &
fi

export HIMURA_HOST="$HOST"
export HIMURA_PORT="$PORT"

echo "Starting Himura Pixel Tools API on http://${HOST}:${PORT}"
exec "$PYEXE" -m himura_pixel_tools.api.server --host "$HOST" --port "$PORT"
