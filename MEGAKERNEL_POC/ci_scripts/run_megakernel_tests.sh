#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/mk_config.sh"
source "${script_dir}/mk_utility.sh"

main() {
    local repo_root="$1"
    shift
    local build_dir="${repo_root}/build"
    local venv_dir="${build_dir}/venv"
    local -A options mk_config
    local -a extra_args export_args
    parse_megakernel_args test "${repo_root}/MEGAKERNEL/EXPORTED_MODELS" options extra_args "$@" || {
        megakernel_usage "${0##*/}" test >&2
        return 2
    }
    if [[ "${options[mode]}" == help ]]; then
        megakernel_usage "${0##*/}" test
        return
    fi
    load_mk_config "${options[model]}" "${options[models_dir]}" mk_config export_args

    local benchmark_args=(--frameworks decode_only optimum genai --torch-threads 20)
    local runner=(python)
    source "${venv_dir}/bin/activate"

    export PYTHONPATH="${build_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${build_dir}/openvino_genai${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    if [[ "${options[mode]}" == quick ]]; then
        local runtime_dir="${repo_root}/bin/intel64/Release"
        export PYTHONPATH="${runtime_dir}/python:${PYTHONPATH}"
        export LD_LIBRARY_PATH="${runtime_dir}:${runtime_dir}/lib:${LD_LIBRARY_PATH}"
        python - "${runtime_dir}/python" <<'PY'
import sys
from pathlib import Path
import openvino

assert Path(openvino.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()), \
    "Quick mode requires local-build Python bindings; refusing to benchmark the installed wheel"
PY
        benchmark_args=(--frameworks genai --tokens 8 --gen-warmup 1 --gen-iters 2 --torch-threads 20)
        runner=(timeout --signal=TERM --kill-after=10s "${options[quick_timeout]}s" python)
    else
        python -m pip install "${build_dir}"/wheels/*.whl --force-reinstall
        python -c "import openvino_genai; print(openvino_genai.__version__)"
    fi

    "${runner[@]}" "${repo_root}/MEGAKERNEL_POC/python/e2e_performance_measurement.py" \
        --model-dir "${mk_config[model_dir]}" "${benchmark_args[@]}" "${extra_args[@]}"
}

    main "$(cd "${script_dir}/../.." && pwd)" "$@"