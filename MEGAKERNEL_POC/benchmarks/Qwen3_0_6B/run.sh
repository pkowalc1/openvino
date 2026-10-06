#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${BUILD_DIR:-${script_dir}/build}"
jobs="${JOBS:-$(nproc)}"

cmake -S "${script_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
cmake --build "${build_dir}" -j "${jobs}"

exec "${build_dir}/qwen06b_random_decode_benchmark" "$@"