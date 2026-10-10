# Qwen06BPOC random prefill and decode benchmark

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
`--prefill-iterations=N` (default 5), `--prefill-warmup=N` (default 1),
`--iterations=N` (decode, default 100), `--warmup=N` (decode, default 5), `--seed=N`
(default 42). Requires an Intel OpenCL GPU with USM support.
`--context-tokens` must be between 2 and 4095 so the runtime uses its prefill path
and leaves room for a decode token.

Select the megakernel at CMake configure time (default: `Qwen06BPOC`):

```sh
cmake -S MEGAKERNEL_POC/benchmarks/Qwen3_0_6B -B /tmp/qwen06b-bench2-build \
  -DMEGAKERNEL_IMPLEMENTATION=Qwen06BPOC_prefill_megakernel
cmake --build /tmp/qwen06b-bench2-build -j8
/tmp/qwen06b-bench2-build/qwen06b_random_decode_benchmark --context-tokens=128
```

The `Qwen06BPOC_prefill_separate_kernels` implementation builds the repository's
oneDNN GPU dependency as part of this standalone CMake project, so its initial
build takes longer than the other implementations.

Prefill and decode are timed separately using GPU timestamps on an in-order
queue. The prefill measurement repeats the same context at position zero after
one setup call and optional warmup, then decode repeats the same single-token
call at the next position. Each section reports average GPU latency, estimated
memory bandwidth, and B60 SOL Memory. Times exclude host setup and transfers.
The prefill transfer estimate counts
weights once, K/V writes once, and input, position and output bytes, but omits
K/V reads that may be served from on-chip storage. Decode counts weights once,
previous K/V reads once, new K/V writes once, and input, position and output
bytes. Effective bandwidth divides each estimate by average GPU latency
(decimal GB/s); it does not measure actual memory traffic. On an Arc Pro B60,
each phase's SOL Memory compares its effective bandwidth with the 456 GB/s peak.