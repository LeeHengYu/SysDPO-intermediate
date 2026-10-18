#!/usr/bin/env bash
set -euo pipefail

remote=${1:-cuhk-compute}
root=${2:-/hpchome/hengyulee/sysdpo}
source_dir=$(cd "$(dirname "$0")/.." && pwd)
[[ "$root" =~ ^/[a-zA-Z0-9_./-]+$ && "$remote" =~ ^[a-zA-Z0-9_.@-]+$ ]] || exit 2

release="$root/code/$(date -u +%Y%m%dT%H%M%S)-$$"

ssh "$remote" mkdir -p "$release"
rsync -az --exclude='.git' --exclude='.venv' --exclude='.uv-cache' \
    --exclude='.hf-cache' --exclude='*.egg-info' --exclude='build' --exclude='dist' --exclude='__pycache__' --exclude='runs' --exclude='.DS_Store' \
    --exclude='.env*' "$source_dir/" "$remote:$release/"

ssh "$remote" bash "$release/multi-llm/setup.sh" "$root"

ssh "$remote" ln -sfn "$release" "$root/current"

printf 'Deployed %s\n' "$release"
