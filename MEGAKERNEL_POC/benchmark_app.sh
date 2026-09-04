#!/usr/bin/env bash
set -euo pipefail

attention_mask_size=1
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
model_path="${script_dir}/python/qwen3-0.6b-openvino-ir/openvino_model.xml"
benchmark_app_path="${script_dir}/../bin/intel64/Release/benchmark_app"

if [[ ! -x "${benchmark_app_path}" ]]; then
	echo "Error: benchmark_app executable not found or not executable: ${benchmark_app_path} -> fix hardcoded path in the script!" >&2
	exit 1
fi

input_dir="$(mktemp -d)"
trap 'rm -rf "${input_dir}"' EXIT

INPUT_DIR="${input_dir}" ATTENTION_MASK_SIZE="${attention_mask_size}" python3 - <<'PY'
import os
from pathlib import Path

import numpy as np

input_dir = Path(os.environ["INPUT_DIR"])
attention_mask_size = int(os.environ["ATTENTION_MASK_SIZE"])
np.save(input_dir / "input_ids.npy", np.array([[13]], dtype=np.int64))
np.save(input_dir / "attention_mask.npy", np.ones((1, attention_mask_size), dtype=np.int64))
np.save(input_dir / "position_ids.npy", np.array([[attention_mask_size - 1]], dtype=np.int64))
np.save(input_dir / "beam_idx.npy", np.array([0], dtype=np.int32))
PY

bench_args=(
	-m "${model_path}"
	-d GPU
	-hint latency
	-api sync
	-nireq 1
	-niter 100
	-data_shape "input_ids[1,1],attention_mask[1,${attention_mask_size}],position_ids[1,1],beam_idx[1]"
	-i
	"input_ids:${input_dir}/input_ids.npy"
	"attention_mask:${input_dir}/attention_mask.npy"
	"position_ids:${input_dir}/position_ids.npy"
	"beam_idx:${input_dir}/beam_idx.npy"
)

set +e
ZE_AFFINITY_MASK=0 "${benchmark_app_path}" "${bench_args[@]}"
status=$?
set -e

if [[ ${status} -ne 0 ]]; then
	echo "benchmark_app exited with status ${status}; re-running under gdb for a backtrace..." >&2
	ZE_AFFINITY_MASK=0 gdb -q -batch -ex run -ex "thread apply all bt full" -ex quit \
		--args "${benchmark_app_path}" "${bench_args[@]}" || true
	exit "${status}"
fi