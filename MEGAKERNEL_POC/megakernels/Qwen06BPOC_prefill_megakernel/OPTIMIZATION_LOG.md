# Qwen3 0.6B B60 Prefill Optimization Log

## Scope

- Target: Intel Arc Pro B60, Qwen3 0.6B, batch size 1.
- Changed only the multi-token prefill path in `impl/qwen06BPOCRuntime.cpp`.
- Kept the existing single-token decode task queue unchanged.
- Preserved all runtime interfaces.
- Validation command: `bash MEGAKERNEL_POC/scripts/compile_run_megakernel.sh qwen` from the repository root.

The B60 reports 160 compute units, a maximum work-group size of 1024, 128 KiB of local memory, and support for `cl_intel_subgroup_matrix_multiply_accumulate`.

## Baseline

Prefill reused the decode implementation once per prompt token. This made all dense projections GEMVs and serialized prompt processing on the host.

| Prompt tokens | Standard OpenVINO TTFT | Megakernel TTFT | Relative speed |
|---:|---:|---:|---:|
| 19 | about 12 ms | about 59 ms | 0.20x |
| 58 | about 17 ms | about 170 ms | 0.10x |
| 281 | about 40 ms | about 821 ms | 0.05x |

Decode was already fast: approximately 1.52x standard OpenVINO, with matching argmax and cosine similarity 1.0000.

## Iteration 1: Batched XMX Prefill

Implemented a separate prefill pipeline selected only when `newTokens > 1`:

- Batched copy and RMSNorm kernels.
- SIMD16 XMX GEMMs using `intel_sub_group_f16_f16_matrix_mad_k16`.
- One subgroup computes an 8-token by 16-output tile.
- Batched Q, K, V, output, gate, up, and down projections.
- Batched RoPE and KV-cache population.
- Batched causal attention and residual/SiLU elementwise stages.
- File-local runtime state and scratch allocations avoided interface changes.

The first execution returned `CL_OUT_OF_RESOURCES`. All tensors and weights are reached indirectly through a USM context, but the new kernels had not enabled indirect USM access. Applying `CL_KERNEL_EXEC_INFO_INDIRECT_DEVICE_ACCESS_INTEL`, host access, and shared access to every prefill kernel fixed the failure.

Results:

| Prompt tokens | Before | Iteration 1 | Improvement over old prefill |
|---:|---:|---:|---:|
| 19 | 59 ms | 21 ms | 2.8x |
| 58 | 170 ms | 34 ms | 5.0x |
| 281 | 821 ms | 110 ms | 7.5x |

Outcome: retained. Batching converted the compute-bound prefill projections from serial GEMV work into XMX GEMM work.

## Iteration 2: SLM Activation Reuse

Grouped eight SIMD16 subgroups into a 128-thread work-group. Each group staged an 8-token activation tile in local memory and reused it across 128 output columns. The largest projection required 48 KiB of SLM per work-group.

Results:

| Prompt tokens | Iteration 1 | Iteration 2 |
|---:|---:|---:|
| 19 | 21 ms | 20 ms |
| 58 | 34 ms | 42 ms |
| 281 | 110 ms | 149 ms |

Outcome: reverted. Activation reuse slightly helped the shortest prompt, but the 48 KiB SLM allocation reduced occupancy and regressed medium and long prompts.

## Iteration 3: Projection Launch Fusion

Kept the Iteration 1 XMX tile and fused projection families at dispatch level:

- Q, K, and V use one combined XMX launch.
- Gate and up projections use one combined XMX launch.
- Output boundaries remain aligned to 16 columns, so no subgroup crosses between tensors.
- Removed 56 kernel launches across the 28 transformer layers.

Results from the final full run:

| Prompt tokens | Standard OpenVINO TTFT | Final megakernel TTFT | Versus standard | Versus old prefill |
|---:|---:|---:|---:|---:|
| 19 | 12.227 ms | 17.517 ms | 0.70x | 3.4x faster |
| 58 | 16.989 ms | 31.578 ms | 0.54x | 5.4x faster |
| 281 | 43.788 ms | 110.672 ms | 0.40x | 7.4x faster |

Outcome: retained. It improved short and medium TTFT without materially changing the long-prompt result.

## Iteration 4: Small-Panel Blocked GEMM

Replaced independent output subgroups with oneDNN-style work-group blocking. Four SIMD16 subgroups share an 8-token by 128-feature activation panel in 2 KiB of SLM and compute an 8-token by 64-output tile. This preserves occupancy while reusing each activation panel across four output subgroups.

| Prompt tokens | Iteration 3 | Iteration 4 |
|---:|---:|---:|
| 19 | 17.5 ms | 15.0 ms |
| 58 | 31.6 ms | 28.4 ms |
| 281 | 110.7 ms | 102.8 ms |

Outcome: retained. Small K panels provided activation reuse without the occupancy loss of Iteration 2's full activation tile.

## Iteration 5: 128-Column Output Block

Widened the work-group output block to 128 columns using eight SIMD16 subgroups. The K panel remained 128 features and 2 KiB, doubling activation reuse without increasing SLM.

| Prompt tokens | Iteration 4 | Iteration 5 |
|---:|---:|---:|
| 19 | 15.0 ms | 13.0 ms |
| 58 | 28.4 ms | 26.5 ms |
| 281 | 102.8 ms | 97.4 ms |

Outcome: retained. The larger N block improved every prompt size.

## Iteration 6: 256-Column Output Block

Widened the output block again to 256 columns using sixteen SIMD16 subgroups, still sharing the same 2 KiB activation panel.

| Prompt tokens | Iteration 5 | Iteration 6 |
|---:|---:|---:|
| 19 | 13.0 ms | 12.4 ms |
| 58 | 26.5 ms | 25.8 ms |
| 281 | 97.4 ms | 95.5 ms |

Outcome: retained. This was the best measured shape: M=8 tokens, N=256 outputs, K=128 per panel.

## Iteration 7: 512-Column Output Block

Tested 32 subgroups and a 512-column output block while keeping the 2 KiB K panel.

| Prompt tokens | Iteration 6 | Iteration 7 |
|---:|---:|---:|
| 19 | 12.4 ms | 16.0 ms |
| 58 | 25.8 ms | 26.7 ms |
| 281 | 95.5 ms | 99.3 ms |

Outcome: reverted. The coarser block reduced the number of output work-groups too far, underexposing parallelism on the B60, especially for short prompts.

## Iteration 8: 256-Wide K Panel

Returned to the 256-column output block and doubled the activation panel from K=128 to K=256. This used 4 KiB SLM and halved K-loop barrier frequency.

| Prompt tokens | Iteration 6 | Iteration 8 |
|---:|---:|---:|
| 19 | 12.4 ms | 12.5 ms |
| 58 | 25.8 ms | 25.9 ms |
| 281 | 95.5 ms | 96.3 ms |

Outcome: reverted. Fewer barriers did not offset the larger panel and slightly reduced performance. The final K panel remains 128 features.

## Iteration 9: 16-Token GEMM Tile

Doubled the token tile from M8 to M16 while retaining N256 and K128. Each loaded weight operand is reused by two XMX accumulators, and the 4 KiB activation panel still permits high occupancy.

| Prompt tokens | M8/N256/K128 | M16/N256/K128 |
|---:|---:|---:|
| 19 | 12.644 ms | 13.376 ms |
| 58 | 25.599 ms | 21.190 ms |
| 281 | 97.556 ms | 70.444 ms |

Outcome: retained. M16 substantially improves medium and long prompts while keeping short TTFT close to the M8 result.

## Iteration 10: 32-Token GEMM Tile

Doubled the token tile again to M32. Four XMX accumulators reuse each weight operand across 32 prompt rows, with an 8 KiB activation panel.

| Prompt tokens | M16 | M32 |
|---:|---:|---:|
| 19 | 13.376 ms | 16.356 ms |
| 58 | 21.190 ms | 23.082 ms |
| 281 | 70.444 ms | 61.851 ms |

Outcome: reverted for the general path. M32 is the best long-prompt shape, but its register pressure and coarse token grid regress short and medium prompts.

## Iteration 11: Adaptive M16/M32 Kernel

Tested M16 for prompts up to 64 tokens and M32 above that threshold in one kernel. The first version exposed a store race: M16 work-groups wrote rows 16-31 with zero accumulators. Conditional stores fixed correctness.

| Prompt tokens | Specialized shape | Adaptive kernel |
|---:|---:|---:|
| 19 | 13.376 ms | 15.391 ms |
| 58 | 21.190 ms | 24.485 ms |
| 281 | 61.851 ms | 63.157 ms |

Outcome: reverted. The unified kernel retains the static register footprint of four accumulators even on its M16 path, losing the benefit of specialization.

## Iteration 12: Subgroup Weight Prefetch

Restored specialized M32 and issued `intel_sub_group_block_prefetch_us8` for each output row's next 128-half weight panel while the current panel executed.

| Prompt tokens | M32 | M32 with prefetch |
|---:|---:|---:|
| 19 | 16.356 ms | 16.214 ms |
| 58 | 23.082 ms | 22.127 ms |
| 281 | 61.851 ms | 62.204 ms |

Outcome: reverted. Explicit prefetch is effectively neutral and slightly worsens the most important long-prompt result, indicating that sequential weight access is already handled adequately by B60 caching and hardware prefetch.

## Iteration 13: Four-Subgroup Attention

Partitioned each causal attention sequence across four SIMD16 subgroups. Each subgroup computed an online-softmax partial, then the work-group merged the four stable max/denominator/accumulator states through 2 KiB of SLM.

| Prompt tokens | M32 baseline | Parallel attention |
|---:|---:|---:|
| 19 | 16.356 ms | 16.401 ms |
| 58 | 23.082 ms | 22.434 ms |
| 281 | 61.851 ms | 62.265 ms |

Outcome: reverted. The extra subgroup work and SLM synchronization do not improve TTFT, confirming that causal attention is not the dominant bottleneck at these sequence lengths.

## Iteration 14: Separate M16 and M32 Kernels

Added independent M16 and M32 OpenCL entry points and selected M32 only above 64 tokens. This lets the compiler allocate two accumulators for M16 and four for M32, avoiding the unified kernel's static M32 register footprint from Iteration 11.

| Prompt tokens | M16-only baseline | Split M16/M32 |
|---:|---:|---:|
| 19 | 13.643 ms | 13.540 ms |
| 58 | 21.693 ms | 20.946 ms |
| 281 | 71.071 ms | 64.290 ms |

Outcome: retained. The split preserves M16's short/medium behavior and recovers M32's long-prompt weight reuse with correct argmax and cosine similarity 1.0000.

## Iteration 15: M24 Medium-Prompt Specialization

Added a third independently compiled kernel with three accumulators and a 24-token tile. The 58-token prompt requires three M24 work-groups instead of four M16 work-groups.

| Prompt tokens | Split M16/M32 | M16/M24/M32 |
|---:|---:|---:|
| 19 | 13.540 ms | 13.664 ms |
| 58 | 20.946 ms | 21.650 ms |
| 281 | 64.290 ms | 62.153 ms |

Outcome: reverted. M24's additional accumulator pressure outweighs its reduction in token work-groups for the medium prompt. The long difference is run-to-run variation because both variants use the same M32 path.

## Iteration 16: M32/N512 Output Block

Widened only the long-prompt M32 output block from 256 to 512 columns using 32 SIMD16 subgroups. The 8 KiB K128 activation panel is then shared across twice as many output rows, while the work-group remains within B60's 1024-thread and 64-subgroup limits.

| Prompt tokens | M32/N256 | M32/N512 |
|---:|---:|---:|
| 19 | 13.540 ms | 13.407 ms |
| 58 | 20.946 ms | 20.962 ms |
| 281 | 64.290 ms | 61.611 ms |

Outcome: retained. N512 improves the long prompt while short and medium continue to use the unchanged M16/N256 kernel.

## Iteration 17: M32/N512/K256

Doubled the long kernel's activation panel to K256, using 16 KiB SLM and halving panel barriers at the highest tested token/output reuse point.

| Prompt tokens | M32/N512/K128 | M32/N512/K256 |
|---:|---:|---:|
| 19 | 13.407 ms | 13.760 ms |
| 58 | 20.962 ms | 21.259 ms |
| 281 | 61.611 ms | 66.454 ms |

Outcome: reverted. Fewer barriers do not offset the larger panel's occupancy and loading cost. K128 remains preferable even at M32/N512.

## Iteration 18: M32/N384 Output Block

Tested a 384-column midpoint using 24 SIMD16 subgroups, seeking more grid parallelism than N512 and more activation reuse than N256.

| Prompt tokens | M32/N512 | M32/N384 |
|---:|---:|---:|
| 19 | 13.407 ms | 13.540 ms |
| 58 | 20.962 ms | 20.799 ms |
| 281 | 61.611 ms | 68.505 ms |

Outcome: reverted. N384 loses substantially on the long prompt; N512's additional activation reuse is more valuable than the finer output grid.

## Iteration 19: Subgroup RMSNorm Reduction

Replaced RMSNorm's 256-thread SLM reduction tree with SIMD16 subgroup reductions followed by one subgroup-level merge. This reduced the work-group reduction from eight barriers to two.

| Prompt tokens | Original RMSNorm | Subgroup RMSNorm |
|---:|---:|---:|
| 19 | 13.562 ms | 13.676 ms |
| 58 | 20.976 ms | 21.067 ms |
| 281 | 63.728 ms | 63.572 ms |

Outcome: reverted. The differences are within run-to-run noise, showing that RMSNorm synchronization is not a material TTFT bottleneck.

## Iteration 20: M32/N768 Output Block

Widened the long-prompt output block to 768 columns using 48 SIMD16 subgroups, increasing activation-panel reuse by 50% over N512.

| Prompt tokens | M32/N512 | M32/N768 |
|---:|---:|---:|
| 19 | 13.562 ms | 13.909 ms |
| 58 | 20.976 ms | 21.221 ms |
| 281 | 63.728 ms | 67.517 ms |

Outcome: reverted. The coarser output grid outweighs the additional panel reuse.

## Iteration 21: M32/N1024 Output Block

Tested B60's maximum legal 1024-thread work-group and all 64 available SIMD16 subgroups, sharing each activation panel across 1024 output columns.

| Prompt tokens | M32/N512 | M32/N1024 |
|---:|---:|---:|
| 19 | 13.562 ms | 13.526 ms |
| 58 | 20.976 ms | 21.585 ms |
| 281 | 63.728 ms | 71.627 ms |

Outcome: reverted. N1024 underexposes output-grid parallelism further and confirms N512 as the best tested long-prompt N block.

## Iteration 22: M32/N512/K64

Halved the long kernel's activation panel to K64, reducing SLM from 8 KiB to 4 KiB while doubling K-loop barriers.

| Prompt tokens | M32/N512/K128 | M32/N512/K64 |
|---:|---:|---:|
| 19 | 13.562 ms | 13.733 ms |
| 58 | 20.976 ms | 21.161 ms |
| 281 | 63.728 ms | 62.935 ms |

A clean repeat measured 63.194 ms for the long prompt, overlapping K128's recent 61.611-63.728 ms range.

Outcome: reverted. The apparent sub-millisecond gain was not reproducible enough to justify doubling barriers; K128 remains the simpler retained panel.

## Iteration 23: M32 Threshold at 32 Tokens

Lowered the specialized M32/N512/K64 dispatch threshold from 64 to 32 tokens so the 58-token prompt used the higher-reuse long kernel.

| Prompt tokens | Threshold 64 | Threshold 32 |
|---:|---:|---:|
| 19 | 13.733 ms | 14.487 ms |
| 58 | 21.161 ms | 23.946 ms |
| 281 | 62.935 ms | 61.788 ms |

Outcome: reverted. M32's four-accumulator register pressure and coarse token grid regress the medium prompt, validating the 64-token crossover.

## Iteration 24: Packed Down-Projection Weights

Allocated 168 MiB for a duplicate of all 28 down-projection matrices and repacked them synchronously in `Init()`. The layout was `[N/16][K/16][N lane][K lane]`, placing a subgroup's sixteen 16-half row vectors in one 512-byte tile. Decode retained the original weights; only prefill `op == 6` used the packed copy.

| Prompt tokens | Row-major baseline | Packed down |
|---:|---:|---:|
| 19 | 13.733 ms | 16.357 ms |
| 58 | 21.004 ms | 24.462 ms |
| 281 | 63.615 ms | 63.377 ms |

Outcome: reverted. Short and medium TTFT regressed substantially, and the long generated output diverged despite the dedicated decode argmax match and cosine 1.0000.

## Iteration 25: Packed Gate/Up Weights

Restored down projection and isolated the same Init-time layout on gate and up weights. This required 336 MiB of duplicate storage and redirected only fused prefill `op == 8`.

| Prompt tokens | Row-major baseline | Packed gate/up |
|---:|---:|---:|
| 19 | 13.733 ms | 14.986 ms |
| 58 | 21.004 ms | 23.478 ms |
| 281 | 63.615 ms | 63.873 ms |

Outcome: reverted. All prompt sizes were neutral or slower, and the long generated output diverged.

## Iteration 26: Packed QKV Weights

Restored gate/up and packed Q, K, and V separately during `Init()`, requiring about 224 MiB. The fused QKV dispatch remapped its output index before using the packed address.

| Prompt tokens | Row-major baseline | Packed QKV |
|---:|---:|---:|
| 19 | 13.733 ms | 14.896 ms |
| 58 | 21.004 ms | 21.961 ms |
| 281 | 63.615 ms | 64.063 ms |

Outcome: reverted. Packing regressed every prompt and the long generated output diverged.

## Iteration 27: Packed Output-Projection Weights

Restored QKV and isolated the row-interleaved packed layout on output-projection weights, adding about 112 MiB during `Init()`.

| Prompt tokens | Row-major baseline | Packed output |
|---:|---:|---:|
| 19 | 13.733 ms | 15.905 ms |
| 58 | 21.004 ms | 23.315 ms |
| 281 | 63.615 ms | 63.127 ms |

Outcome: reverted. The small long-prompt difference did not offset severe short/medium regressions or long-output divergence. Results across four projection families show that merely interleaving row vectors does not improve B60's weight loads.

## Iteration 28: K-Major Tiles and Subgroup Block Reads

Changed the output-projection packed tile to `[N/16][K/16][K lane][N lane]`. Two `intel_sub_group_block_read_us8` operations loaded each subgroup's 16x16 weight operand from a contiguous 512-byte tile, replacing sixteen adjacent per-lane `vload16` operations. Packing and allocation still occurred only in `Init()`.

| Prompt tokens | Row-major baseline | Transposed tile/block read |
|---:|---:|---:|
| 19 | 13.733 ms | 13.988 ms |
| 58 | 21.004 ms | 21.713 ms |
| 281 | 63.615 ms | 63.547 ms |

Outcome: reverted. Subgroup block reads recovered most of the row-interleaved regression but did not beat the original layout, and the long generated output still diverged. All duplicate allocations and packing kernels were removed.

## Final Correctness and Decode Check

Iterations 24-28 completed the full build and benchmark workflow. Rejected variants were restored before the final run.

Final retained-code run:

| Prompt tokens | Standard OpenVINO TTFT | Final megakernel TTFT | Prefill speed |
|---:|---:|---:|---:|
| 19 | 13.542 ms | 13.615 ms | 0.99x |
| 58 | 16.979 ms | 21.517 ms | 0.79x |
| 281 | 39.297 ms | 62.702 ms | 0.63x |

- Decode speedup remained approximately 1.5x across the runs.
- Final dedicated decode-only speedup: 1.50x.
- Decode argmax match: true.
- Decode cosine similarity: 1.0000.
- Existing decode tasks and their dispatch path were not modified.
- Target-only rebuild also passed:
  `cmake --build ../build --target Qwen06BPOCv2 -j$(nproc)`.

## Remaining Bottleneck

The retained prompt-size dispatch uses M16/N256/K128 through 64 tokens and M32/N512/K128 above 64. N768 and N1024 proved that further activation reuse loses too much grid parallelism, while K64/K256 showed that K128 is the panel-size balance. Init-time packing of every major projection family was slower, and even a K-major 16x16 layout with subgroup block reads did not beat row-major `vload16`. The original row-major tensors therefore already provide the best tested B60 weight path without hundreds of MiB of duplicate storage. Further progress requires stage-level device profiling or a different GEMM decomposition rather than another local layout permutation.

## Iteration 29: Dedicated Prefill Task System

Replaced the per-stage prefill kernel dispatches with independent SIMD16 task
workers. Each former work-group is represented by one static `TaskDesc`; exact
stage task counts form the semaphore dependencies. Task queues are cached by
prompt length, and M16/LWS256 remains selected through 64 tokens while
M32/LWS512 handles longer prompts. Decode task types, workers, and queue
construction were not changed.

Validation exposed two task-specific correctness requirements. Tail tasks must
guard inactive lanes instead of returning before the task completion barrier,
and the M32 RMS reduction must combine all 32 subgroup partial sums rather than
only the first SIMD16 group. Both were corrected before measurement.

| Prompt tokens | Standard OpenVINO TTFT | Task prefill TTFT | Prefill speed |
|---:|---:|---:|---:|
| 19 | 11.852 ms | 21.235 ms | 0.56x |
| 58 | 16.095 ms | 33.596 ms | 0.48x |
| 281 | 38.502 ms | 79.369 ms | 0.49x |

Outcome: retained to provide the requested prefill task architecture. The task
worker removes the repeated host-side stage dispatches but is slower than the
specialized standalone-kernel path because all stages use the GEMM worker's
large work-group and each fine-grained task pays queue and semaphore overhead.
Dedicated decode remained 1.58x faster with argmax match true and cosine
similarity 1.0000. All generated outputs matched.

## Iterations 30-34: Late and Fine-Grained Prefill Task Waits

These iterations changed only the dedicated prefill task system. Decode task
types, task bodies, queue construction, and dispatch were not modified.

### Iteration 30: Operation-Local Late Waits

Removed the unconditional wait from `ExecutePrefillTaskM16/M32`. Each prefill
operation now waits immediately before its first read of producer-owned mutable
data, allowing task decoding, address calculation, constant-weight selection,
and output setup to execute while the dependency is pending.

Correctness passed. The benchmark host was heavily loaded during this run:
standard OpenVINO TTFT rose to `107.910 / 76.751 / 115.951 ms`, so the measured
megakernel TTFT of `96.228 / 147.640 / 87.589 ms` was not used for comparison.
The structural change was retained for the following iterations.

### Iteration 31: First Weight Vector Register Preload

Loaded the first constant K16 weight vector into registers before the late GEMM
wait and reused it for the first XMX operation. Correctness passed, but system
load remained abnormal and the long megakernel TTFT reached `178.765 ms`.
Outcome: rejected because the extra live `int8` increased register pressure and
did not demonstrate a stable benefit.

### Iteration 32: First Weight Tile SLM Preload

Cooperatively staged the complete first K16 weight tile before waiting: 8 KiB
for M16 and 16 KiB for M32. Worker SLM was increased from 16 KiB to 32 KiB.

| Prompt tokens | Standard OpenVINO TTFT | SLM preload TTFT |
|---:|---:|---:|
| 19 | 11.926 ms | 25.107 ms |
| 58 | 17.336 ms | 39.173 ms |
| 281 | 43.073 ms | 84.446 ms |

Outcome: rejected. Cooperative preload traffic and lower occupancy cost more
than overlapping one K16 weight tile with the dependency wait.

### Iteration 33: Per-Token Residual Synchronization

Added a second optional completion counter to each prefill task. Copy and final
residual tasks signal both their stage counter and a token-specific counter in
one release barrier. Each next-layer RMS task waits only for the groups that
produce its token, rather than the complete previous-layer residual matrix.

Initial TTFT was `21.050 / 32.443 / 82.954 ms`. This design was retained.

### Iteration 34: Per-Tile RMS-to-GEMM Synchronization

RMS tasks additionally signaled one counter per 16-token M16 or 32-token M32
tile. QKV and gate/up GEMM tasks waited only for their own RMS input tile.

| Prompt tokens | Standard OpenVINO TTFT | Tile-sync TTFT |
|---:|---:|---:|
| 19 | 12.538 ms | 20.937 ms |
| 58 | 16.579 ms | 32.660 ms |
| 281 | 40.304 ms | 83.342 ms |

Outcome: rejected. The additional RMS atomics did not offset medium/long task
overhead. Iteration 33 was restored.

Final retained-code run:

| Prompt tokens | Standard OpenVINO TTFT | Final task prefill TTFT | Prefill speed |
|---:|---:|---:|---:|
| 19 | 12.267 ms | 20.920 ms | 0.59x |
| 58 | 17.597 ms | 32.823 ms | 0.54x |
| 281 | 38.406 ms | 79.905 ms | 0.48x |

Compared with the final Iteration 29 task-only run (`21.366 / 32.898 / 80.508
ms`), late waits plus per-token residual synchronization improved short and
long TTFT slightly without changing numerical results. Dedicated decode speedup
was 1.59x, argmax matched, cosine similarity was 1.0000, and generated outputs
matched for all prompts.
