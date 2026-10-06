# Qwen06BPOC random decode benchmark

Standalone benchmark for the Qwen06BPOC runtime. It needs no captured fixtures:
all FP16 weights and hidden states are
generated from a fixed seed; positions are initialized to zero for context
building and to the context length for decode. The output buffers are allocated
on the GPU and checked for finite values after setup, warmup and timing. This
does not compare against a reference model or test numerical correctness.

Run from the repository root:

```sh
MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/run.sh --device=0 --iterations=200
```

The runner builds in `Qwen3_0_6B/build` (override with `BUILD_DIR`, `JOBS`, or
`CMAKE_BUILD_TYPE`). Options: `--device=<index|name substring>` (first OpenCL
GPU by default), `--list-devices`, `--context-tokens=N` (default 4000),
`--iterations=N` (default 100), `--warmup=N` (default 5), `--seed=N`
(default 42). Requires an Intel OpenCL GPU with USM support.

Select the megakernel at CMake configure time (default: `Qwen06BPOC`):

```sh
cmake -S MEGAKERNEL_POC/benchmarks/Qwen3_0_6B -B /tmp/qwen06b-bench2-build \
  -DMEGAKERNEL_IMPLEMENTATION=Qwen06BPOC_prefill_megakernel
cmake --build /tmp/qwen06b-bench2-build -j8
/tmp/qwen06b-bench2-build/qwen06b_random_decode_benchmark --context-tokens=128
```

The `Qwen06BPOC_prefill_separate_kernels` implementation additionally requires
the OpenVINO build's `onednn_gpu_tgt` target and cannot be linked by this
standalone CMake project.

Latency uses GPU timestamps around an in-order queue's repeated
single-token decode calls, excluding setup and transfers. Minimum transfer
counts weights once, previous K/V reads once, new K/V writes once, and decode
input, position and output bytes. Effective bandwidth divides that lower bound
by average GPU latency (decimal GB/s); it does not measure actual memory traffic.
For the Arc Pro B60, SOL Memory reports effective bandwidth as a percentage of
its 456 GB/s peak bandwidth, using the same minimum-transfer estimate.