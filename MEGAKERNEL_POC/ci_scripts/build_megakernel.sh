#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/../.." && pwd)"
build_dir="${repo_root}/build"
venv_dir="${build_dir}/venv"

main() {
    python3 -m venv --system-site-packages "${venv_dir}"
    source "${venv_dir}/bin/activate"

    chmod +x "${repo_root}/install_build_dependencies.sh"
    "${repo_root}/install_build_dependencies.sh"

    python -m pip install --upgrade optimum-intel
    python "${repo_root}/MEGAKERNEL_POC/python/convert_to_openvino_ir.py" \
        --output-dir "${repo_root}/MEGAKERNEL_POC/python/qwen3-0.6b-openvino-ir"

    git clone https://github.com/openvinotoolkit/openvino.genai.git

    cmake -S "${repo_root}" -B "${build_dir}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_DEBUG_CAPS=ON \
        -DENABLE_CPU_DEBUG_CAPS=OFF \
        -DENABLE_GPU_DEBUG_CAPS=ON \
        -DENABLE_TESTS=ON \
        -DENABLE_INTEL_CPU=ON \
        -DENABLE_INTEL_GPU=ON \
        -DENABLE_OV_ONNX_FRONTEND=OFF \
        -DENABLE_PYTHON=ON \
        -DENABLE_OV_PADDLE_FRONTEND=OFF \
        -DENABLE_OV_PYTORCH_FRONTEND=ON \
        -DENABLE_OV_JAX_FRONTEND=OFF \
        -DENABLE_OV_TF_FRONTEND=OFF \
        -DENABLE_OV_TF_LITE_FRONTEND=OFF \
        -DENABLE_JS=OFF \
        -DENABLE_WHEEL=ON \
        -DENABLE_TEMPLATE_REGISTRATION=OFF \
        -DOPENVINO_EXTRA_MODULES=./openvino.genai \
        -DCPACK_ARCHIVE_COMPONENT_INSTALL=OFF \
        -DMEGAKERNEL_IMPLEMENTATION=Qwen06BPOC 
    cmake --build "${build_dir}" --parallel 16
}

main "$@"