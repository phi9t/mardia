#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LAUNCH="${ROOT}/infra/zephyr/container/launch_container.sh"

cd "${ROOT}"
"${LAUNCH}" bash -lc "cd /workspace && python -m qwen3_tts.validation.run_educational_smoke && python -m qwen3_tts.validation.compare_structure"
"${LAUNCH}" bash -lc "cd /workspace && python -m qwen3_vl.validation.run_educational_smoke && python -m qwen3_vl.validation.compare_structure"
