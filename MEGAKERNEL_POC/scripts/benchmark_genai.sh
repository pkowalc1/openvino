#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
megakernel_root="$(cd "${script_dir}/.." && pwd)"
cd "${megakernel_root}/python"

ZE_AFFINITY_MASK=0 python3 run_openvino_genai.py "$@"