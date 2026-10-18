#!/usr/bin/env bash
set -euo pipefail

root=${1:?persistent project root required}
export PATH="$HOME/.local/bin:$PATH"

code=$(cd "$(dirname "$0")/.." && pwd)

mkdir -p "$root/models" "$root/data" "$root/runs"

if [ ! -x "$code/.venv/bin/python" ]; then
    uv venv --python 3.12 "$code/.venv"
fi

uv pip install --python "$code/.venv/bin/python" -r "$code/requirements-gpu.txt"

uv pip install --python "$code/.venv/bin/python" --no-deps -e "$code"
