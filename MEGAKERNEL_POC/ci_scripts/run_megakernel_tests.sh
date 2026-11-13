#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
build_dir="${repo_root}/build"
venv_dir="${build_dir}/venv"
model_id="${MODEL_ID:-Qwen/Qwen3-0.6B}"
model_name="${model_id##*/}"
model_dir="${MODELS_DIR:-${repo_root}/MEGAKERNEL_POC/python}/${model_name,,}-openvino-ir"

main() {
    python3 -m venv --system-site-packages "${venv_dir}"
    source "${venv_dir}/bin/activate"

    export PYTHONPATH="${build_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${build_dir}/openvino_genai${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    python -m pip install "${build_dir}"/wheels/*.whl --force-reinstall

    python -c "import openvino_genai; print(openvino_genai.__version__)"
    # benchmark_app.sh feeds input_ids; multimodal IRs take inputs_embeds instead.
    if [[ -f "${model_dir}/openvino_model.xml" ]]; then
        MODEL_PATH="${model_dir}/openvino_model.xml" bash "${repo_root}/MEGAKERNEL_POC/scripts/benchmark_app.sh"
    fi
    # shellcheck disable=SC2086  # E2E_ARGS is a list of CLI flags
    python "${repo_root}/MEGAKERNEL_POC/python/e2e_performance_measurement.py" \
        --model-dir "${model_dir}" \
        --frameworks decode_only optimum genai \
        --torch-threads 20 ${E2E_ARGS:-}
}

main "$@"