#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
megakernel_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${megakernel_root}/.." && pwd)"
build_dir="${repo_root}/build"
venv_dir="${build_dir}/venv"
setup_state_dir="${build_dir}/megakernel_setup"
genai_dir="${build_dir}/openvino.genai"
model_dir="${megakernel_root}/python/qwen3-0.6b-openvino-ir"
python_bin="${venv_dir}/bin/python"
build_jobs="${MEGAKERNEL_BUILD_JOBS:-16}"
openvino_version="2026.3.0"
tokenizers_version="2026.3.0.0"

fingerprint_inputs() {
    sha256sum "$@" | sha256sum | cut -d ' ' -f 1
}

stamp_matches() {
    local stamp_file="$1"
    local expected="$2"
    [[ -f "${stamp_file}" ]] && [[ "$(<"${stamp_file}")" == "${expected}" ]]
}

clean_generated_artifacts() {
    local artifact
    local artifacts=(
        "${build_dir}"
        "${repo_root}/bin"
        "${repo_root}/temp"
        "${genai_dir}"
        "${model_dir}"
    )

    for artifact in "${artifacts[@]}"; do
        case "${artifact}" in
            "${build_dir}"|"${repo_root}/bin"|"${repo_root}/temp"|"${genai_dir}"|"${model_dir}") ;;
            *)
                echo "Refusing to remove unexpected path: ${artifact}" >&2
                exit 1
                ;;
        esac

        if [[ -e "${artifact}" || -L "${artifact}" ]]; then
            echo "Removing ${artifact}"
            rm -rf -- "${artifact}"
        fi
    done
}

activate_venv() {
    if [[ ! -x "${python_bin}" ]]; then
        python3 -m venv "${venv_dir}"
    fi
    export VIRTUAL_ENV="${venv_dir}"
    export PATH="${venv_dir}/bin:${PATH}"
    unset PYTHONHOME
}

install_system_dependencies() {
    local stamp_file="${setup_state_dir}/system-dependencies.sha256"
    local fingerprint
    fingerprint="$(fingerprint_inputs "${repo_root}/install_build_dependencies.sh")"
    if stamp_matches "${stamp_file}" "${fingerprint}"; then
        echo "System build dependencies are already installed"
        return
    fi

    if (( EUID == 0 )); then
        bash "${repo_root}/install_build_dependencies.sh"
    elif command -v sudo >/dev/null; then
        sudo -E bash "${repo_root}/install_build_dependencies.sh"
    else
        echo "System build dependencies require root. Re-run as root or install them manually." >&2
        exit 1
    fi

    mkdir -p "${setup_state_dir}"
    printf '%s\n' "${fingerprint}" > "${stamp_file}"
}

install_python_dependencies() {
    local stamp_file="${setup_state_dir}/python-dependencies.sha256"
    local fingerprint
    fingerprint="$({
        fingerprint_inputs "${genai_dir}/requirements-build.txt"
        printf '%s\n' "${openvino_version}" "${tokenizers_version}" "optimum-intel[openvino]" "accelerate"
    } | sha256sum | cut -d ' ' -f 1)"
    if stamp_matches "${stamp_file}" "${fingerprint}" &&
       "${python_bin}" -c "import accelerate, openvino, openvino_tokenizers, optimum, transformers"; then
        echo "Python dependencies are already installed"
        return
    fi

    "${python_bin}" -m pip install --upgrade pip setuptools wheel
    "${python_bin}" -m pip install --upgrade \
        -r "${genai_dir}/requirements-build.txt" \
        "optimum-intel[openvino]" \
        accelerate
    "${python_bin}" -m pip install \
        "openvino==${openvino_version}" \
        "openvino-tokenizers==${tokenizers_version}" \
        --force-reinstall --no-deps

    mkdir -p "${setup_state_dir}"
    printf '%s\n' "${fingerprint}" > "${stamp_file}"
}

download_genai() {
    local patch_commit="fb6461a27c3c90050281b4d7816fbf6e5ba2534a"

    if [[ ! -d "${genai_dir}/.git" ]]; then
        rm -rf "${genai_dir}"
        git clone --recursive --branch 2026.3.0.0 \
            https://github.com/openvinotoolkit/openvino.genai.git "${genai_dir}"
    else
        git -C "${genai_dir}" submodule update --init --recursive
    fi

    if git -C "${genai_dir}" rev-parse -q --verify CHERRY_PICK_HEAD >/dev/null &&
       git -C "${genai_dir}" diff --quiet &&
       git -C "${genai_dir}" diff --cached --quiet; then
        git -C "${genai_dir}" cherry-pick --skip
    fi

    if ! git -C "${genai_dir}" merge-base --is-ancestor "${patch_commit}" HEAD; then
        if ! git -C "${genai_dir}" cherry-pick "${patch_commit}"; then
            if git -C "${genai_dir}" rev-parse -q --verify CHERRY_PICK_HEAD >/dev/null &&
               git -C "${genai_dir}" diff --quiet &&
               git -C "${genai_dir}" diff --cached --quiet; then
                git -C "${genai_dir}" cherry-pick --skip
            else
                return 1
            fi
        fi
    fi
}

generate_model() {
    if [[ ! -f "${model_dir}/openvino_model.xml" ]]; then
        local overwrite=()
        [[ -d "${model_dir}" ]] && overwrite=(--overwrite)
        "${python_bin}" "${megakernel_root}/python/convert_to_openvino_ir.py" \
            --output-dir "${model_dir}" "${overwrite[@]}"
    fi

    if ! grep -q "openvino_tokenizers_version value=\"${tokenizers_version}" \
        "${model_dir}/openvino_tokenizer.xml" 2>/dev/null; then
        "${python_bin}" - "${model_dir}" <<'PY'
import sys
from pathlib import Path

import openvino as ov
from openvino_tokenizers import convert_tokenizer
from transformers import AutoTokenizer

model_dir = Path(sys.argv[1])
tokenizer = AutoTokenizer.from_pretrained(model_dir)
ov_tokenizer, ov_detokenizer = convert_tokenizer(tokenizer, with_detokenizer=True)
ov.save_model(ov_tokenizer, model_dir / "openvino_tokenizer.xml")
ov.save_model(ov_detokenizer, model_dir / "openvino_detokenizer.xml")
PY
    fi
}

set_runtime_environment() {
    export PYTHONPATH="${repo_root}/bin/intel64/Release/python:${build_dir}${PYTHONPATH:+:${PYTHONPATH}}"
    export LD_LIBRARY_PATH="${repo_root}/bin/intel64/Release:${repo_root}/bin/intel64/Release/lib:${build_dir}/openvino_genai${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
}

if [[ "${1:-}" == "--clean" ]]; then
    if (( $# != 1 )); then
        echo "--clean does not accept additional arguments" >&2
        exit 2
    fi
    clean_generated_artifacts
    exit
fi

if [[ "${1:-}" == "--quick" ]]; then
    shift
    activate_venv
    quick_timeout="${MEGAKERNEL_QUICK_TIMEOUT:-180}s"
    timeout --signal=TERM --kill-after=10s "${quick_timeout}" \
        cmake --build "${build_dir}" --parallel "${build_jobs}" --target openvino_intel_gpu_plugin benchmark_app
    set_runtime_environment
    "${python_bin}" - "${repo_root}/bin/intel64/Release/python" <<'PY'
import sys
from pathlib import Path
import openvino

assert Path(openvino.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve()), \
    "Quick mode requires local-build Python bindings; refusing to benchmark the installed wheel"
PY
    timeout --signal=TERM --kill-after=10s "${quick_timeout}" \
        "${python_bin}" "${megakernel_root}/python/e2e_performance_measurement.py" \
        --frameworks genai --tokens 8 --gen-warmup 1 --gen-iters 2 \
        --torch-threads 20 "$@"
    exit
fi

install_system_dependencies
activate_venv
download_genai
install_python_dependencies
generate_model
rm -rf "${build_dir}/wheels"

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
    -DOPENVINO_EXTRA_MODULES="${genai_dir}" \
    -DCPACK_ARCHIVE_COMPONENT_INSTALL=OFF \
    -DMEGAKERNEL_IMPLEMENTATION=Qwen06BPOC_prefill_separate_kernels
cmake --build "${build_dir}" --parallel "${build_jobs}"

"${python_bin}" -m pip install "${build_dir}"/wheels/*.whl --force-reinstall

set_runtime_environment
"${python_bin}" -c "import openvino, openvino_tokenizers; print(openvino.__version__); print(openvino_tokenizers.__version__)"
"${python_bin}" -c "import openvino_genai; print(openvino_genai.__version__)"

bash "${script_dir}/benchmark_app.sh"

"${python_bin}" "${megakernel_root}/python/e2e_performance_measurement.py" \
    --frameworks decode_only optimum genai \
    --torch-threads 20 
