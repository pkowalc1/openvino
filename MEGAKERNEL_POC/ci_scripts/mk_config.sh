load_mk_config() {
    local -n config_output="$3"
    local -n export_output="$4"
    case "$1" in
        qwen)
            config_output[kernel]=Qwen06BPOC_prefill_separate_kernels
            config_output[model_id]=Qwen/Qwen3-0.6B
            export_output=()
            ;;
        gemma)
            config_output[kernel]=Gemma4POC
            config_output[model_id]=google/gemma-4-26B-A4B-it
            export_output=(--task image-text-to-text --weight-format int4)
            ;;
        *)
            echo "Unknown megakernel model: $1 (expected qwen or gemma)" >&2
            return 2
            ;;
    esac

    local model_name="${config_output[model_id]##*/}"
    config_output[model_dir]="$2/${model_name,,}-openvino-ir"
}