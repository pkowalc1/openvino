#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/mk_config.sh"
source "${script_dir}/mk_utility.sh"

setup_dependencies() {
    local install_script="$1"
    local requirements_file="$2"
    local stamp_file="$3"
    local openvino_version="2026.3.0"
    local tokenizers_version="2026.3.0.0"
    local fingerprint
    fingerprint="$({
        sha256sum "${install_script}" "${requirements_file}" || exit 1
        printf '%s\n' "${openvino_version}" "${tokenizers_version}"
    } | sha256sum)"
    if [[ -f "${stamp_file}" && "$(<"${stamp_file}")" == "${fingerprint}" ]]; then
        return
    fi

    if (( EUID == 0 )); then
        bash "${install_script}"
    else
        sudo -E bash "${install_script}"
    fi

    python -m pip install --upgrade pip setuptools wheel
    python -m pip install --upgrade -r "${requirements_file}" "optimum-intel[openvino]" accelerate
    python -m pip install \
        "openvino==${openvino_version}" \
        "openvino-tokenizers==${tokenizers_version}" \
        --force-reinstall --no-deps
    printf '%s\n' "${fingerprint}" > "${stamp_file}"
}

setup_genai() {
    local genai_dir="$1"
    local patch_commit=fb6461a27c3c90050281b4d7816fbf6e5ba2534a
    local patch_status
    if [[ ! -d "${genai_dir}/.git" ]]; then
        git clone --recursive --branch 2026.3.0.0 \
            https://github.com/openvinotoolkit/openvino.genai.git "${genai_dir}"
    fi
    patch_status="$(git -C "${genai_dir}" cherry HEAD "${patch_commit}" "${patch_commit}^")"
    if [[ "${patch_status}" == "+ ${patch_commit}" ]]; then
        git -C "${genai_dir}" cherry-pick "${patch_commit}"
    fi
}

main() {
    local repo_root="$1"
    shift
    local build_dir="${repo_root}/build"
    local venv_dir="${build_dir}/venv"
    local genai_dir="${build_dir}/openvino.genai"
    local stamp_file="${build_dir}/megakernel-dependencies.sha256"
    local -A options mk_config
    local -a benchmark_args export_args
    parse_megakernel_args build "${repo_root}/MEGAKERNEL/EXPORTED_MODELS" options benchmark_args "$@" || {
        megakernel_usage "${0##*/}" build >&2
        return 2
    }
    if [[ "${options[mode]}" == help ]]; then
        megakernel_usage "${0##*/}" build
        return
    fi
    load_mk_config "${options[model]}" "${options[models_dir]}" mk_config export_args
    local megakernel="${mk_config[kernel]}"
    local model_dir="${mk_config[model_dir]}"

    if [[ "${options[mode]}" == quick ]]; then
        if ! grep -sEq "^MEGAKERNEL_IMPLEMENTATION:[^=]+=${megakernel}$" "${build_dir}/CMakeCache.txt"; then
            echo "Quick mode needs a build configured for ${megakernel}; run without --quick-build first." >&2
            return 1
        fi
        timeout --signal=TERM --kill-after=10s "${options[quick_timeout]}s" \
            cmake --build "${build_dir}" --parallel "${options[jobs]}" --target openvino_intel_gpu_plugin benchmark_app
        return
    fi

    if [[ ! -x "${venv_dir}/bin/python" ]]; then
        python3 -m venv "${venv_dir}"
        rm -f -- "${stamp_file}"
    fi
    source "${venv_dir}/bin/activate"
    setup_genai "${genai_dir}"
    setup_dependencies "${repo_root}/install_build_dependencies.sh" "${genai_dir}/requirements-build.txt" "${stamp_file}"

    if [[ ! -d "${model_dir}" ]]; then
        python "${repo_root}/MEGAKERNEL_POC/python/convert_to_openvino_ir.py" --overwrite \
            --model-id "${mk_config[model_id]}" --output-dir "${model_dir}.partial" "${export_args[@]}"
        mv "${model_dir}.partial" "${model_dir}"
    fi

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
        -DPython3_EXECUTABLE="${venv_dir}/bin/python" \
        -DENABLE_OV_PADDLE_FRONTEND=OFF \
        -DENABLE_OV_PYTORCH_FRONTEND=ON \
        -DENABLE_OV_JAX_FRONTEND=OFF \
        -DENABLE_OV_TF_FRONTEND=OFF \
        -DENABLE_OV_TF_LITE_FRONTEND=OFF \
        -DENABLE_JS=OFF \
        -DENABLE_WHEEL=ON \
        -DENABLE_TEMPLATE_REGISTRATION=OFF \
        -DOPENVINO_EXTRA_MODULES="${genai_dir}" \
        -DCPACK_ARCHIVE_COMPONENT_INSTALL=OFF \
        -DMEGAKERNEL_IMPLEMENTATION="${megakernel}"
    cmake --build "${build_dir}" --parallel "${options[jobs]}"
}

main "$(cd "${script_dir}/../.." && pwd)" "$@"