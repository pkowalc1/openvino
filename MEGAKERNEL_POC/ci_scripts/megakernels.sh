# Sourced by the CI scripts; expects repo_root to be set.

# Megakernels tested in CI: "<MEGAKERNEL_POC/megakernels/<name>> <Hugging Face model id> <optimum-cli export task>".
# The first entry is the one built by build_megakernel.sh.
megakernels=(
    "Qwen06BPOC_prefill_separate_kernels Qwen/Qwen3-0.6B            text-generation-with-past"
    "Qwen06BPOC_prefill_megakernel       Qwen/Qwen3-0.6B            text-generation-with-past"
    "Qwen06BPOC                          Qwen/Qwen3-0.6B            text-generation-with-past"
    "Gemma4POC                           google/gemma-4-26B-A4B-it  image-text-to-text"
)

# OpenVINO IR directory of a Hugging Face model id, e.g. Qwen/Qwen3-0.6B -> .../qwen3-0.6b-openvino-ir.
model_dir() {
    local name="${1##*/}"
    echo "${repo_root}/MEGAKERNEL_POC/python/${name,,}-openvino-ir"
}
