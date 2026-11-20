parse_megakernel_args() {
    local entrypoint="$1"
    local default_models_dir="$2"
    local -n options_output="$3"
    local -n benchmark_output="$4"
    shift 4
    options_output=([model]="${MK_CONFIGS[0]}" [models_dir]="${default_models_dir}" [jobs]=16 [quick_timeout]=180 [mode]=full)
    benchmark_output=()
    if [[ "${entrypoint}" == local ]]; then
        options_output[model]=''
    fi

    if [[ -n "${1:-}" ]] && is_mk_config "$1"; then
        options_output[model]="$1"
        shift
    fi

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
    local model_list
    local model_options
    printf -v model_list '%s|' "${MK_CONFIGS[@]}"
    model_list="${model_list%|}"
    model_options="[${model_list}]"
    case "$2" in
        local)
            mode_options='[--clean | --quick-build [benchmark options]]'
            model_options="<${model_list}>"
            ;;
        build) mode_options='[--quick-build]' ;;
    esac
    echo "Usage: $1 ${model_options} [--models-dir PATH] [--jobs N] [--quick-timeout SECONDS] ${mode_options}"
    if [[ "$2" == local ]]; then
        echo "Available megakernels:"
        local model
        local -A usage_config
        local -a usage_export_args
        for model in "${MK_CONFIGS[@]}"; do
            load_mk_config "${model}" '' usage_config usage_export_args
            printf '  %s  %s\n' "${model}" "${usage_config[kernel]}"
        done
    fi
}