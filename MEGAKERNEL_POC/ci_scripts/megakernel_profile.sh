parse_megakernel_args() {
    local entrypoint="$1"
    local default_models_dir="$2"
    local -n options_output="$3"
    local -n benchmark_output="$4"
    shift 4
    options_output=([model]=qwen [models_dir]="${default_models_dir}" [jobs]=16 [quick_timeout]=180 [mode]=full)
    benchmark_output=()
    if [[ "${entrypoint}" == local ]]; then
        options_output[model]=''
    fi

    case "${1:-}" in
        qwen|gemma)
            options_output[model]="$1"
            shift
            ;;
    esac

    while (( $# )); do
        case "$1" in
            --models-dir)
                [[ -n "${2:-}" && "$2" != --* ]] || { echo "$1 requires a path" >&2; return 2; }
                options_output[models_dir]="$2"
                shift 2
                ;;
            --jobs|--quick-timeout)
                [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || { echo "$1 requires a positive integer" >&2; return 2; }
                case "$1" in
                    --jobs) options_output[jobs]="$2" ;;
                    --quick-timeout) options_output[quick_timeout]="$2" ;;
                esac
                shift 2
                ;;
            --quick-build)
                options_output[mode]=quick
                shift
                if [[ "${entrypoint}" != build ]]; then
                    benchmark_output=("$@")
                    return
                fi
                ;;
            --clean)
                if [[ "${entrypoint}" != local || $# != 1 || "${options_output[mode]}" != full ]]; then
                    echo "--clean is only supported as the final option of the local wrapper" >&2
                    return 2
                fi
                options_output[mode]=clean
                shift
                ;;
            --help|-h)
                options_output[mode]=help
                return
                ;;
            *)
                echo "Unexpected argument: $1" >&2
                return 2
                ;;
        esac
    done
}

megakernel_usage() {
    local mode_options='[--quick-build [benchmark options]]'
    local model_options='[qwen|gemma]'
    case "$2" in
        local)
            mode_options='[--clean | --quick-build [benchmark options]]'
            model_options='<qwen|gemma>'
            ;;
        build) mode_options='[--quick-build]' ;;
    esac
    echo "Usage: $1 ${model_options} [--models-dir PATH] [--jobs N] [--quick-timeout SECONDS] ${mode_options}"
    if [[ "$2" == local ]]; then
        echo "Available megakernels:"
        local model
        local -A usage_profile
        local -a usage_export_args
        for model in qwen gemma; do
            load_megakernel_profile "${model}" '' usage_profile usage_export_args
            printf '  %s  %s\n' "${model}" "${usage_profile[kernel]}"
        done
    fi
}

load_megakernel_profile() {
    local -n profile_output="$3"
    local -n export_output="$4"
    case "$1" in
        qwen)
            profile_output[kernel]=Qwen06BPOC_prefill_separate_kernels
            profile_output[model_id]=Qwen/Qwen3-0.6B
            export_output=()
            ;;
        gemma)
            profile_output[kernel]=Gemma4POC
            profile_output[model_id]=google/gemma-4-26B-A4B-it
            export_output=(--task image-text-to-text --weight-format int4)
            ;;
        *)
            echo "Unknown megakernel model: $1 (expected qwen or gemma)" >&2
            return 2
            ;;
    esac

    local model_name="${profile_output[model_id]##*/}"
    profile_output[model_dir]="$2/${model_name,,}-openvino-ir"
}