#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/../ci_scripts/mk_config.sh"
source "${script_dir}/../ci_scripts/mk_utility.sh"

main() {
    local repo_root="$1"
    shift
    if (( $# == 0 )); then
        megakernel_usage "${0##*/}" local
        return
    fi
    local ci_dir="${repo_root}/MEGAKERNEL_POC/ci_scripts"
    local local_models_dir="${repo_root}/MEGAKERNEL/EXPORTED_MODELS"
    local -A options mk_config
    local -a benchmark_args export_args
    parse_megakernel_args local "${local_models_dir}" options benchmark_args "$@" || {
        megakernel_usage "${0##*/}" local >&2
        return 2
    }

    if [[ "${options[mode]}" == help ]]; then
        megakernel_usage "${0##*/}" local
        return
    fi
    if [[ -z "${options[model]}" ]]; then
        echo "Select a megakernel (qwen or gemma)." >&2
        megakernel_usage "${0##*/}" local >&2
        return 2
    fi
    if [[ "${options[mode]}" == clean ]]; then
        load_mk_config "${options[model]}" "${local_models_dir}" mk_config export_args
        local model_dir="${mk_config[model_dir]}"
        local artifact
        for artifact in "${repo_root}/build" "${repo_root}/bin" "${repo_root}/temp" "${model_dir}" "${model_dir}.partial"; do
            if [[ -e "${artifact}" || -L "${artifact}" ]]; then
                echo "Removing ${artifact}"
                rm -rf -- "${artifact}"
            fi
        done
        return
    fi
    local common_args=("${options[model]}" --models-dir "${options[models_dir]}")
    if [[ "${options[mode]}" == quick ]]; then
        common_args+=(--quick-timeout "${options[quick_timeout]}")
        bash "${ci_dir}/build_megakernel.sh" "${common_args[@]}" --jobs "${options[jobs]}" --quick-build
        bash "${ci_dir}/run_megakernel_tests.sh" "${common_args[@]}" --quick-build "${benchmark_args[@]}"
    else
        bash "${ci_dir}/build_megakernel.sh" "${common_args[@]}" --jobs "${options[jobs]}"
        bash "${ci_dir}/run_megakernel_tests.sh" "${common_args[@]}"
    fi
}

main "$(cd "${script_dir}/../.." && pwd)" "$@"