MK_CONFIGS=(qwen3 gemma4)

is_mk_config() {
    local model
    for model in "${MK_CONFIGS[@]}"; do
        [[ "$1" == "${model}" ]] && return 0
    done
    return 1
}

load_mk_config() {
    local -n config_output="$3"
    local -n export_output="$4"
    is_mk_config "$1" || return 2
    case "$1" in
        qwen3)
            config_output[kernel]=Qwen06BPOC
            config_output[model_id]=Qwen/Qwen3-0.6B
            export_output=()
            ;;
        gemma4)
            config_output[kernel]=Gemma4POC
            config_output[model_id]=google/gemma-4-26B-A4B-it
            export_output=(--task image-text-to-text --weight-format int4)
            ;;
        *)
            local model_list
            printf -v model_list '%s or ' "${MK_CONFIGS[@]}"
            model_list="${model_list% or }"
            echo "Unknown megakernel model: $1 (expected ${model_list})" >&2
            return 2
            ;;
    esac

    local model_name="${config_output[model_id]##*/}"
    config_output[model_dir]="$2/${model_name,,}-openvino-ir"
}