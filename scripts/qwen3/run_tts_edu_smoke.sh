#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
python -m qwen3_tts.validation.run_educational_smoke
python -m qwen3_tts.validation.compare_structure
