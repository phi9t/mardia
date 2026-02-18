#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LAUNCH="${ROOT}/infra/zephyr/container/launch_container.sh"
MODEL="${MODEL:-Qwen/Qwen3-VL-2B-Instruct}"
PORT="${PORT:-30000}"

cd "${ROOT}"
"${LAUNCH}" bash -lc "cd /workspace && python -m sglang.launch_server --model ${MODEL} --host 0.0.0.0 --port ${PORT}"
