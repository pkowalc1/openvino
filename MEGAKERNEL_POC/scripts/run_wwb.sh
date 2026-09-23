#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
megakernel_root="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${megakernel_root}/.." && pwd)"
build_dir="${BUILD_DIR:-${repo_root}/build}"
venv_dir="${VENV_DIR:-}"
python_bin="${PYTHON_BIN:-python3}"
wwb_dir="${megakernel_root}/openvino.genai/tools/who_what_benchmark"

model_id="${MODEL_ID:-Qwen/Qwen3-0.6B}"
target_model="${TARGET_MODEL:-${megakernel_root}/python/qwen3-0.6b-openvino-ir}"
results_dir="${RESULTS_DIR:-${megakernel_root}/wwb_results}"
gt_data="${GT_DATA:-${results_dir}/qwen3-0.6b-gt.csv}"
num_samples="${NUM_SAMPLES:-10}"
max_new_tokens="${MAX_NEW_TOKENS:-128}"
target_device="${TARGET_DEVICE:-GPU}"
install_deps=0
regenerate_gt=0

usage() {
    cat <<EOF
Usage: $(basename "$0") [--install-deps] [--regenerate-gt] [-- WWB_TARGET_ARGS...]

Run Who What Benchmark against the OpenVINO and OpenVINO GenAI built from this
checkout. Ground truth is generated with the Hugging Face Qwen3-0.6B model and
reused on later runs. WWB is loaded directly from the local GenAI checkout.

Options:
  --install-deps   Install WWB dependencies, then reinstall local build wheels.
  --regenerate-gt  Overwrite the cached Hugging Face ground truth.
  -h, --help       Show this help.

Environment overrides:
    BUILD_DIR, VENV_DIR, PYTHON_BIN, MODEL_ID, TARGET_MODEL, RESULTS_DIR,
    GT_DATA, NUM_SAMPLES, MAX_NEW_TOKENS, TARGET_DEVICE, ZE_AFFINITY_MASK
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-deps)
            install_deps=1
            shift
            ;;
        --regenerate-gt)
            regenerate_gt=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done
target_args=("$@")

if [[ -n "${venv_dir}" ]]; then
    if [[ ! -f "${venv_dir}/bin/activate" ]]; then
        echo "Missing virtual environment: ${venv_dir}" >&2
        exit 1
    fi
    # shellcheck source=/dev/null
    source "${venv_dir}/bin/activate"
    python_bin="python"
fi
if [[ ! -d "${target_model}" ]]; then
    echo "Missing converted model: ${target_model}" >&2
    echo "Build it with ${megakernel_root}/python/convert_to_openvino_ir.py" >&2
    exit 1
fi

export PYTHONPATH="${build_dir}:${wwb_dir}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${build_dir}/openvino_genai${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export ZE_AFFINITY_MASK="${ZE_AFFINITY_MASK:-0}"
unset OV_MEGAKERNEL_DISABLE

if [[ ! -d "${build_dir}/wheels" ]]; then
    echo "Missing local wheel directory: ${build_dir}/wheels" >&2
    echo "Build the branch first: bash ${script_dir}/compile_run_megakernel.sh" >&2
    exit 1
fi
mapfile -t local_wheels < <(find "${build_dir}/wheels" -maxdepth 1 -name '*.whl' -type f -print | sort)
if [[ ${#local_wheels[@]} -eq 0 ]]; then
    echo "No locally built wheels found in ${build_dir}/wheels" >&2
    echo "Build the branch first: bash ${script_dir}/compile_run_megakernel.sh" >&2
    exit 1
fi

pip_args=(--disable-pip-version-check)
if [[ -z "${venv_dir}" ]]; then
    pip_args+=(--break-system-packages)
fi

if (( install_deps )); then
    "${python_bin}" -m pip install "${pip_args[@]}" -r <(grep -v -E '^[[:space:]]*openvino-genai([[:space:]<>=]|$)' "${wwb_dir}/requirements.txt")
    "${python_bin}" -m pip install "${pip_args[@]}" --editable "${wwb_dir}" --no-deps
fi

if ! "${python_bin}" -c "import importlib.util, sys; sys.exit(importlib.util.find_spec('whowhatbench') is None)"; then
    echo "WWB is not installed for ${python_bin}. Re-run with --install-deps." >&2
    exit 1
fi

"${python_bin}" -m pip install "${pip_args[@]}" --force-reinstall --no-deps "${local_wheels[@]}"
"${python_bin}" -m pip install "${pip_args[@]}" --no-deps openvino-tokenizers==2026.3.0.0

"${python_bin}" - <<'PY'
import openvino
import openvino_genai
import openvino_tokenizers
import whowhatbench

print(f"OpenVINO:       {openvino.__file__}")
print(f"OpenVINO GenAI: {openvino_genai.__file__}")
print(f"Tokenizers:    {openvino_tokenizers.__version__} ({openvino_tokenizers.__file__})")
PY

mkdir -p "${results_dir}" "$(dirname "${gt_data}")"

if [[ ! -f "${gt_data}" || ${regenerate_gt} -eq 1 ]]; then
    echo "Generating Hugging Face ground truth: ${gt_data}"
    "${python_bin}" -m whowhatbench.wwb \
        --base-model "${model_id}" \
        --gt-data "${gt_data}" \
        --model-type text \
        --device CPU \
        --hf
else
    echo "Reusing ground truth: ${gt_data}"
fi

echo "Evaluating megakernel target on ${target_device}: ${target_model}"
"${python_bin}" -m whowhatbench.wwb \
    --target-model "${target_model}" \
    --gt-data "${gt_data}" \
    --model-type text \
    --device "${target_device}" \
    --genai \
    --output "${results_dir}" \
    "${target_args[@]}"