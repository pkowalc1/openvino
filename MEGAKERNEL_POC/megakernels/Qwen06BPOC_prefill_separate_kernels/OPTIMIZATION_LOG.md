# Qwen3 0.6B B60 Prefill Optimization Log

## Scope

- Target: Intel Arc Pro B60, Qwen3 0.6B, batch size 1.
- Changed only the multi-token prefill path in `impl/qwen06BPOCRuntime.cpp`.
- Kept the existing single-token decode task queue unchanged.
- Preserved all runtime interfaces.
- Validation command: `bash compile_run_megakernel.sh`.

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

## Re-baseline: Longer "long" Prompt

The `long` benchmark prompt in `python/e2e_performance_measurement.py` was extended (repeated background text), raising it from 281 to 1889 tokens. This exposed the prefill path to much larger sequence lengths and changed which stage dominates TTFT, so prefill was re-measured before further work.

| Prompt tokens | Standard OpenVINO TTFT | Megakernel TTFT | Prefill speed |
|---:|---:|---:|---:|
| 19 | 13.135 ms | 6.923 ms | 1.90x |
| 58 | 17.433 ms | 11.206 ms | 1.56x |
| 1889 | 251.105 ms | 398.685 ms | 0.63x |

Short and medium prompts remained faster than standard OpenVINO from the retained iterations above; the long prompt, now much longer, regressed to 0.63x. This is the new target.

## Iterations 29-33: KV-Cache Update Path

These iterations targeted `prefill_rope_cache`, the kernel that applies RoPE to Q/K and writes K/V into the persistent KV cache once per (layer, token). At 1889 tokens it dispatched `(NH+KVH)=24` work-groups of only 16 threads each, per layer, per token — 28 * 24 * 1889 ≈ 1.27M tiny work-group launches for the whole prompt.

### Iteration 29: Fused Per-Token RoPE/Cache Work-Group

Combined all `NH+KVH=24` heads for one token into a single work-group (one SIMD16 subgroup per head, local size 384) instead of one 16-thread work-group per head. This cuts the number of launched work-groups for this stage by 24x, from `24 * tokens` to `tokens` per layer.

| Prompt tokens | Before | Iteration 29 |
|---:|---:|---:|
| 19 | 6.923 ms | 6.906 ms |
| 58 | 11.206 ms | 11.447 ms |
| 1889 | 398.685 ms | 384.204 ms |

Outcome: retained. The long prompt improved by about 3.6%; short/medium were unchanged within noise.

### Iteration 30: Vectorized Block Read/Write

On top of Iteration 29, replaced the four-iteration scalar per-lane loop (loading/storing `rf`, `qn`/`kn`, and Q/K/V one `half` at a time) with `float4`/`half4` math and `intel_sub_group_block_read_us4` / `intel_sub_group_block_write_us4`, since each lane's address is already `base + component*16 + lane` — exactly the layout these intrinsics expect.

| Prompt tokens | Iteration 29 | Iteration 30 |
|---:|---:|---:|
| 19 | 6.906 ms | 6.834-6.938 ms |
| 58 | 11.447 ms | 11.097-11.264 ms |
| 1889 | 384.204 ms | 389.1-392.1 ms (two runs) |

Outcome: reverted. The scalar addressing pattern was already auto-coalesced by the compiler/hardware; the explicit vector intrinsics were neutral to slightly worse on the long prompt across repeated runs.

### Iteration 31: Separate Bulk V-Cache Copy Kernel

V requires no RoPE or normalization, only a position-indexed scatter into the cache. Split it out of `prefill_rope_cache` into a new flat `prefill_cache_copy_v` kernel dispatched once per layer over all tokens at once with normal 256-thread work-groups, leaving only Q/K RoPE in the per-token/head kernel.

| Prompt tokens | Iteration 29 | Iteration 31 |
|---:|---:|---:|
| 19 | 6.906 ms | 6.834 ms |
| 58 | 11.447 ms | 11.264 ms |
| 1889 | 384.204 ms | 395.647 ms |

Outcome: reverted. The extra kernel launch (28 more dispatches, one per layer) outweighed the saved work in the fused kernel; the long prompt regressed.

### Iteration 32: Two Tokens per Work-Group

On top of Iteration 29, packed `ROPE_TPB=2` tokens into each `prefill_rope_cache` work-group (768 threads, 48 subgroups) to further amortize launch and barrier overhead, with the second token's subgroups mapped via `sub_group_id() / (NH+KVH)`.

| Prompt tokens | Iteration 29 | Iteration 32 |
|---:|---:|---:|
| 19 | 6.906 ms | 6.938 ms |
| 58 | 11.447 ms | 11.097 ms |
| 1889 | 384.204 ms | 393.090 ms |

Outcome: reverted. Doubling the work-group size did not improve the long prompt; the coarser dispatch grid (half as many work-groups, twice the size) was not better than the plain per-token fusion from Iteration 29.

### Iteration 33: Precomputed RoPE Cos/Sin Table

RoPE angles depend only on `(token position, dim)`, not on layer, but `prefill_rope_cache` recomputed `native_cos`/`native_sin` independently in all 28 layers. Added a small `prefill_rope_table` kernel that computes `cos(pos * rope_inv_freq)` / `sin(...)` once per prefill call into a `[tokens][64]` table, and had the per-layer kernel read from it instead of recomputing.

| Prompt tokens | Iteration 29 | Iteration 33 |
|---:|---:|---:|
| 19 | 6.906 ms | 6.783 ms |
| 58 | 11.447 ms | 10.937 ms |
| 1889 | 384.204 ms | 391.672 ms |

Outcome: reverted. Despite eliminating 27x redundant transcendental-function calls per token, the long prompt did not improve; `native_cos`/`native_sin` were evidently not a measurable cost next to the memory traffic already dominating this kernel.

### Conclusion

Only Iteration 29 (fusing the 24 per-head work-groups into one per-token work-group) produced a measurable, reproducible gain (~3.6% on the long prompt). Iterations 30-33 each tried a different way to reduce KV-cache-update cost further — wider vector memory ops, separating the RoPE-free V write, coarser token batching, and removing redundant per-layer trig — and none moved the long-prompt TTFT outside run-to-run noise (roughly ±2%). This indicates `prefill_rope_cache` is no longer the dominant cost at 1889 tokens once Iteration 29 is applied; the remaining prefill/standard-OpenVINO gap for long prompts is most likely the per-token, per-query-head serial causal-attention kernel (`prefill_attention`, O(n²) in sequence length with only 16 threads per query) or the oneDNN GEMM cost scaling with token count, not the cache-write path itself.

Final retained state for this set of iterations (Iteration 29 only):

| Prompt tokens | Standard OpenVINO TTFT | Megakernel TTFT | Prefill speed |
|---:|---:|---:|---:|
| 19 | 12.689 ms | 6.907 ms | 1.84x |
| 58 | 15.005 ms | 11.300 ms | 1.33x |
| 1889 | 231.979 ms | 385.253 ms | 0.60x |

- Decode-only speedup: 1.51x, argmax match: true, cosine similarity: 1.0000.
- Existing decode tasks and their dispatch path were not modified.
- Target-only rebuild also passed:
  `cmake --build ../build --target Qwen06BPOC_prefill_separate_kernels -j$(nproc)`.

## Remaining Bottleneck

The retained prompt-size dispatch uses M16/N256/K128 through 64 tokens and M32/N512/K128 above 64. N768 and N1024 proved that further activation reuse loses too much grid parallelism, while K64/K256 showed that K128 is the panel-size balance. Init-time packing of every major projection family was slower, and even a K-major 16x16 layout with subgroup block reads did not beat row-major `vload16`. The original row-major tensors therefore already provide the best tested B60 weight path without hundreds of MiB of duplicate storage. Further progress requires stage-level device profiling or a different GEMM decomposition rather than another local layout permutation.

## Iterations 29-33: oneDNN GPU GEMM

Iteration 29 replaced the custom prefill GEMMs with cached oneDNN GPU matmul
primitives using OpenCL interoperability and the existing USM allocations. The
row-major model weights use transposed-stride descriptors, so the path requires
no copies or repacking. `DNNL_VERBOSE=1` confirmed `jit:gemm:any` on the B60.
The custom kernels remain available with `OV_MEGAKERNEL_PREFILL_ONEDNN=0`.

| Prompt tokens | Previous custom GEMM | oneDNN GEMM |
|---:|---:|---:|
| 19 | 13.615 ms | 6.810 ms |
| 58 | 21.517 ms | 11.620 ms |
| 281 | 62.702 ms | 41.446 ms |

Outcome: retained. Generated output matched the standard OpenVINO path.

Iteration 30 fused the O-projection residual through a oneDNN sum post-op. It
measured 7.149 / 11.203 / 41.088 ms, but corrupted the hidden state and generated
only repeated `!` tokens. Outcome: reverted.

Iteration 31 removed the forced fp32 accumulation attribute and allowed oneDNN
to select its default accumulation mode. It measured 6.810 / 11.218 / 41.136 ms
with matching generated output. Outcome: retained.

Iteration 32 computed the up projection first and fused SiLU plus binary multiply
into the gate matmul using oneDNN post-ops. Primitive creation succeeded, but
execution failed on the B60. Outcome: reverted.

Iteration 33 cached each primitive's immutable argument map instead of rebuilding
196 hash maps per inference. It measured 6.964 / 11.344 / 41.288 ms with matching
generated output. The difference from Iteration 31 is within run-to-run noise;
the allocation-free dispatch path was retained.

The retained oneDNN path improves megakernel prefill TTFT by approximately 2.0x,
1.9x, and 1.5x for the 19-, 58-, and 281-token prompts respectively. It beats
standard OpenVINO prefill on short and medium prompts and is near parity on the
long prompt in the measured runs.

## Iterations 34-38: KV-Cache Work-Group Occupancy

The current Iteration 29 cache-update kernel assigns one SIMD16 subgroup to each
of the 16 query heads and 8 KV heads, producing a 384-thread work-group per
token. Before changing it, the 1889-token prompt was measured over five calls:

| Prompt tokens | Standard OpenVINO TTFT | Megakernel TTFT | Prefill speed |
|---:|---:|---:|---:|
| 19 | 14.323 ms | 6.753 ms | 2.12x |
| 58 | 14.799 ms | 10.490 ms | 1.41x |
| 1889 | 233.620 ms | 388.965 ms | 0.60x |

Iterations 34-36 changed the kernel so each subgroup could process multiple
strided Q or KV heads, then swept smaller work-groups. The memory access within
each head remained subgroup-coalesced.

| Iteration | Subgroups | Head assignment | 1889-token TTFT |
|---:|---:|---|---:|
| 34 | 12 | Strided Q and KV heads | 385.508 ms |
| 35 | 8 | Two Q heads and one KV head per subgroup | 389.461 ms |
| 36 | 16 | One Q head; first eight also update one KV head | 383.230 ms |

Iteration 34's 0.9% difference was within normal variation. Iteration 35
regressed long, short, and medium prompts. Iteration 36 initially appeared best
at 383.230 ms.

Iteration 37 retained 16 subgroups but performed the KV work before query RoPE,
attempting to expose cache-store latency earlier. It regressed the long prompt
to 392.047 ms and was reverted.

Iteration 38 restored query-first execution and tested a 14-subgroup midpoint.
It measured 389.517 ms and was reverted.

A clean seven-call repeat of Iteration 36 measured 388.403 ms, effectively the
same as the 388.965 ms baseline. It therefore failed the reproducibility check
and was also reverted. The final source retains Iteration 29's simpler
one-subgroup-per-head mapping.

The final validation reported decode argmax match `true`, cosine similarity
`1.0000`, and a 1.61x decode-only speedup. No generated-output mismatch was
reported. These five experiments reinforce the earlier result: after fusing
per-head work-groups, KV-cache update scheduling is not the remaining long-
prefill bottleneck. Profiling and parallelizing the quadratic attention stage is
the next higher-value direction.

## Iterations 39-43: Sequence-Parallel Attention

The long benchmark prompt had grown again, from 1889 to 3229 tokens. Its serial
SIMD16 attention baseline measured 943.074 ms megakernel TTFT. Each experiment
split one `(query_head, token)` row into contiguous sequence ranges processed by
multiple SIMD16 subgroups, then merged their online-softmax maximum,
denominator, and eight per-lane accumulators through SLM.

| Iteration | Subgroups per row | Short TTFT | Medium TTFT | 3229-token TTFT |
|---:|---:|---:|---:|---:|
| 39 | 8 | 6.652 ms | 10.814 ms | 938.947 ms |
| 40 | 2 | 6.747 ms | 10.455 ms | 935.499 ms |
| 41 | 4 | 6.648 ms | 10.613 ms | 936.812 ms |
| 42 | 16 | 6.924 ms | 13.993 ms | 943.639 ms |
| 43 | 32 | 6.837 ms | 12.071 ms | 946.118 ms |

All variants built successfully and generated output matching the standard
OpenVINO path. Two subgroups produced the lowest single measurement, but its
0.8% improvement over serial was within normal run-to-run variation. Larger
work-groups added SLM reduction and occupancy cost without reducing TTFT.

Outcome: all five variants reverted. Sequence traversal within a single
attention row is not limited by a lack of subgroup parallelism. The retained
cache layout is already coalesced at the hardware transaction level: for every
K/V component, the SIMD16 lanes access 16 consecutive fp16 head-dimension
elements. A future layout experiment should therefore target reuse across the
two GQA query heads sharing one KV head, rather than rearranging each subgroup's
already contiguous reads.

## Iterations 44-48: GQA Reuse and Block Messages

A fresh serial-attention baseline measured 6.854 / 11.379 / 941.185 ms for the
19-, 58-, and 3229-token prompts. These iterations tested whether the two GQA
query heads sharing each KV head could reuse cache loads, and whether explicit
Intel subgroup block messages improved the existing strided source expression.

| Iteration | Variant | Short TTFT | Medium TTFT | 3229-token TTFT |
|---:|---|---:|---:|---:|
| 44 | One SIMD16 subgroup computes both GQA query heads | 7.094 ms | 11.092 ms | 946.252 ms |
| 45 | Paired heads plus `us8` Q/K/V block messages | 6.848 ms | 11.172 ms | 940.782 ms |
| 46 | One head per subgroup plus `us8` block messages | 6.695 ms | 11.565 ms | 935.357 ms |
| 47 | Iteration 46 plus 2x sequence-loop unrolling | 6.843 ms | 10.650 ms | 946.607 ms |
| 48 | Iteration 46 plus 4x sequence-loop unrolling | 6.659 ms | 10.503 ms | 935.781 ms |

All five variants built successfully and generated output matching standard
OpenVINO. Paired-head reuse halved logical K/V loads but doubled live query and
accumulator state, producing no gain. Explicit `us8` messages and 4x unrolling
each produced isolated 0.6% long-prompt improvements, while 2x unrolling
regressed by 0.6%; these deltas are below normal measurement variation and are
not reproducible evidence of a faster kernel.

Outcome: all five variants reverted. The compiler and memory subsystem already
coalesce the lane-strided scalar expression effectively, and the shared cache
likely serves the second GQA head without requiring explicit in-kernel reuse.
The remaining 3229-token gap is therefore unlikely to come from head-dimension
cache layout or redundant GQA reads in this attention kernel.

## Iterations 49-53: Batched Prefill Attention

The serial SIMD16 attention baseline was 941.185 ms at 3229 prompt tokens.
Iteration 49 replaced its per-row GEMV loop with two cached 4D oneDNN matmuls:
`Q * K^T`, followed by a causal row-softmax OpenCL kernel, followed by `P * V`.
The GQA K/V batch dimension is broadcast across its two query heads. Scores and
probabilities share one prompt-sized FP16 buffer; at 3229 tokens this is about
334 MB. Short prompts keep the original serial kernel.

Iterations 50-53 tested a fused alternative that stages K/V in SLM and reuses
each tile across both GQA heads and multiple query tokens. This avoids the full
square score buffer and upper-triangle computation, but adds work-group barriers
and SLM traffic.

| Iteration | Variant | 3229-token TTFT |
|---:|---|---:|
| 49 | oneDNN batched QK/PV plus causal softmax | 605.456 ms |
| 50 | Fused GQA, Q8/K32 tiles | 1009.501 ms |
| 51 | Fused GQA, Q8/K16 tiles | 1017.427 ms |
| 52 | Fused GQA, Q8/K64 tiles | 997.799 ms |
| 53 | Fused GQA, Q4/K64 tiles | 1385.313 ms |

Outcome: retained Iteration 49. It reduces long-prompt TTFT by 35.7% versus the
serial attention baseline and generated matching output in the benchmark. The
fused variants were all reverted: reduced global K/V traffic did not repay SLM
and synchronization costs on B60. Set
`OV_MEGAKERNEL_PREFILL_BATCHED_ATTENTION=0` to select the serial fallback.

During measurement, the Python benchmark was initially found to load the
installed wheel's shared object rather than the target-only build output. The
reported numbers above were collected only after installing the rebuilt
`libQwen06BPOC_prefill_separate_kernels.so` into the active OpenVINO package.

## Decode: Five Optimization Steps (2026-10-10)

Scope: single-token decode only. Prefill kernels and dispatch were not changed.
Measurements used `MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/run.sh` on Arc Pro B60,
with 32 persistent workers, 512 threads per worker, and SIMD32 subgroups.

### Final Results

| Context tokens | Baseline latency | Final latency | Baseline SOL Memory | Final SOL Memory, three runs |
|---:|---:|---:|---:|---:|
| 40 | 2.74 ms | 2.73-2.74 ms | 70.94% | 70.97-71.02% |
| 400 | 2.99 ms | 2.86 ms | 67.99% | 71.12-71.13% |
| 4000 | 5.60 ms | 4.17-4.18 ms | 52.48% | 70.35-70.38% |

All nine final runs exceeded 70%. Long-context decode latency fell by about
25%. The 4000-token result has only a small margin above the target; these
measurements establish the result on this B60, not on other devices or workloads.

Baseline runs used 100 measured decode iterations, five decode warmups, one
measured prefill, and no prefill warmup. Final runs used 500 measured decode
iterations with the runner's default prefill setup and five decode warmups:

```sh
for context in 40 400 4000; do
    ./MEGAKERNEL_POC/benchmarks/Qwen3_0_6B/run.sh \
        --context-tokens="$context" --iterations=500
done
```

SOL Memory retains the benchmark's original definition: estimated minimum
transfer bytes divided by GPU latency and B60's 456 GB/s peak. It is effective
bandwidth, not a hardware-counter measurement of actual memory traffic. Neither
the byte estimate nor the peak-bandwidth denominator was changed.

### What Worked and Why

1. **Right-size decode SLM.** Reduced each worker's reservation from 64 KiB to
   the 8,320 bytes needed for attention partials. Retained as a resource reduction,
   but it did not independently improve performance: 4000-token latency stayed
   at 5.60 ms. Tests with 24 and 48 workers did not beat the default 32.

2. **Partition flash decode and fuse the per-head merge.** The original attention
   used only 16 query-head tasks, each scanning a long context. Two context
   partitions per head expose 32 attention tasks, matching the worker pool and
   shortening each task's serial scan. Partitions publish FP32 maximum, sum,
   and accumulator values. The last partition of each head performs the merge
   directly, avoiding a separately queued merge task and its wait. Four
   partitions initially improved 4000-token latency to 4.49 ms, but penalized
   short contexts. Two partitions with an in-task merge reached about 4.28 ms
   with first-strip preloading, before the final shared-normalization change.

3. **Parallelize the subgroup softmax merge.** Replaced the serial chain of
   running-maximum updates and repeated accumulator rescaling with subgroup
   reductions for the common maximum and denominator. Each partial is then
   rescaled once. This preserves stable softmax mathematics while shortening
   the merge dependency chain. The measured improvement was small, roughly
   0.02 ms at 4000 tokens in the intermediate configuration; it was not the
   main source of the overall gain.

4. **Share FP32 normalization across projection tasks.** Compute input and
   post-attention normalization once per layer stage into a shared 4 KiB FP32
   buffer. Previously, every QKV and gate/up task repeated the RMS reduction
   and normalization work. Projection tasks now wait for the normalized vector
   and consume it with subgroup block reads. Keeping FP32 avoids introducing
   an extra FP16 rounding step. This removed about 0.1 ms across contexts and
   brought all three SOL results above 70%.

5. **Preload immutable weights before activation waits.** QKV, gate, output,
   and down projection tasks load their first weight strip into FP16 registers
   before acquiring the producer's activation-ready semaphore. This permits
   weight traffic to overlap the preceding task's remaining work; dependent
   activation reads remain after the acquire. The initial version saved about
   0.04 ms in the split-attention configuration. First-strip loading was retained;
   preloading the full RMS-projection tile was slower.

These are cumulative experiments with follow-up refinements, not five isolated
speedups that can be added together.

### What Did Not Work

- Four-token attention batching was slightly slower; an eight-lane-per-token
  layout was substantially slower. Both passed numerical checks, but the
  original SIMD32 K/V block-read loop remained faster.
- Sharing gate/up activations through SLM added stores and a barrier without
  recovering their cost. Fusing gate/up with register reuse was also slightly
  slower. Both variants were removed in favor of stage-shared normalization.
- Interleaving QKV and attention using per-projection-head readiness regressed
  4000-token latency to 5.23 ms. The overlap did not compensate for losing the
  throughput of stage-ordered work. Stage ordering was restored; fine-grained
  synchronization remains only for the attention partition merge.
- Full-tile weight preloading regressed all contexts, consistent with increased
  live register state outweighing the extra overlap. No register-pressure
  profiling was collected to establish that mechanism conclusively.

### Synchronization and Validation

Partial results are published before an acquire-release completion increment.
The last partition acquires earlier producers' writes, merges its head, and
signals the output projection. A local barrier is required after every subgroup
reads the shared last-partition flag: without it, a fast subgroup can reuse the
same SLM for task dispatch before others have consumed the flag. This race was
exposed by the interleaved-scheduling experiment and fixed in the retained code.
Empty subgroup partials contribute zero during the softmax merge.

Original decode outputs were captured before changing the math, using seed 42
at contexts 40, 400, and 4000. The benchmark now supports untimed
`--write-decode-reference=PATH` and `--decode-reference=PATH` checks. Final runs
passed both numerical comparison and exact warmup-versus-final output equality
at the repeated decode position. Relative L2 errors against the original were
`7.64e-4`, `7.94e-4`, and `6.49e-4`; maximum absolute errors were approximately
`0.00244`, `0.00395`, and `0.00292`, respectively. Results are not bit-identical
to the original because attention reduction ordering changed.

Additional seed-17 runs at contexts 2, 3, 33, and 4095 passed finite-output and
repeat-determinism checks, covering empty subgroup ranges, uneven partitions,
and the final valid cache position. These boundary runs did not compare against
the original implementation. No reference-model accuracy or sequential
multi-token generation validation was performed in this decode experiment.
