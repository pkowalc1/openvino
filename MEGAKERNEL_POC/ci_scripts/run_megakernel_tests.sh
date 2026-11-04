#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
build_dir="${repo_root}/build"
venv_dir="${build_dir}/venv"

source "${script_dir}/megakernels.sh"

build_and_install() {
    cmake -S "${repo_root}" -B "${build_dir}" -DMEGAKERNEL_IMPLEMENTATION="$1"
    # The wheel has no file dependency on the plugin, so remove it to have it repacked
    rm -f "${build_dir}"/wheels/openvino-[0-9]*.whl
    cmake --build "${build_dir}" --parallel 16
    python -m pip install "${build_dir}"/wheels/*.whl --force-reinstall
}

main() {
    bash "${repo_root}/install_build_dependencies.sh"

    python3 -m venv --system-site-packages "${venv_dir}"
    source "${venv_dir}/bin/activate"

    export PYTHONPATH="${build_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${build_dir}/openvino_genai${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

    "${repo_root}/bin/intel64/Release/ov_gpu_unit_tests" --gtest_filter='megakernel_*'

    local entry megakernel model_id
    for entry in "${megakernels[@]}"; do
        read -r megakernel model_id _ <<< "${entry}"
        echo "=== Megakernel ${megakernel} on ${model_id}"
        build_and_install "${megakernel}"

        python -c "import openvino_genai; print(openvino_genai.__version__)"
        bash "${repo_root}/MEGAKERNEL_POC/scripts/benchmark_app.sh" "$(model_dir "${model_id}")/openvino_model.xml"
        python "${repo_root}/MEGAKERNEL_POC/python/e2e_performance_measurement.py" \
            --model-dir "$(model_dir "${model_id}")" \
            --frameworks decode_only optimum genai \
            --torch-threads 20
    done
}

main "$@"