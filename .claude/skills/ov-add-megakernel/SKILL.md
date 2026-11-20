---
name: ov-add-megakernel
description: Add, port, build, test, profile and optimize a new MegaKernel (fused kernel for LLM decode/prefill) for a Hugging Face model in MEGAKERNEL_POC on the OpenVINO Intel GPU plugin. Use when given a Hugging Face model id (and optionally a prompt) to megakernelize, when writing a megakernel transformation or runtime, when speeding up an existing megakernel, when validating megakernel accuracy, profiling megakernel performance, or adding a megakernel to CI.
---

# Add a new MegaKernel

Input of an ideal session: a Hugging Face model id plus a prompt (and optionally a GPU device).
Output: `MEGAKERNEL_POC/megakernels/<Name>/` that builds with `-DMEGAKERNEL_IMPLEMENTATION=<Name>`,
produces correct output, is measurably faster than stock OpenVINO on decode and prefill, is listed in CI, and
has an `OPTIMIZATION_LOG.md` with every measured iteration.

## Priorities

1. **Performance AT accuracy.** A speedup with wrong inaccurate output is worth nothing - revert it.
   - Some accuracy degradation may occur due to different precision levels, execution order and other factors, but it should be 
     minimal and disclosed to the user.
   - Weights must stay byte-identical to the IR. Repacking may only reshuffle bytes (assert dtypes, shapes, group size).
   - Activation precision must be >= baseline: f32 accumulation, and if you quantize activations use the same
     or finer group than the stock GPU plugin (`compiled.get_property("DYNAMIC_QUANTIZATION_GROUP_SIZE")`, ex. 64 on Intel B60).
   - Reductions must be deterministic (see split-K in [optimization playbook](references/optimization-playbook.md)).
2. **Slow and correct first, fast second.** Do not write a clever kernel before a simple one passes the accuracy checks.
3. **Measure, never assume.** Every optimization is one change, kept only if the gain is 
   above noise (~2%), logged in `OPTIMIZATION_LOG.md` either way (rejected attempts are valuable).
4. "Done" = reproducible, measurable decode speedup (`decode_x` > 1 beyond noise in `decode_only` and `genai`) with correct
   output. Then keep optimizing; prior megakernels reached 1.4-1.6x decode on B60.
5. If an optimization does not yield a measurable speedup, revert it even if it seems theoretically beneficial.
6. If a successful optimization has been found which is not included in [optimization playbook](references/optimization-playbook.md), suggest the user to add it there.

## Reference implementations (read the mentioned kernels only if needed)

| Folder | What to learn from it |
|---|---|
| `megakernels/Qwen06BPOC` | Smallest complete example (~1k lines). fp16 Qwen3-0.6B, one MegaKernel op, name-based transformation, task-system persistent decode kernel, MK-owned KV cache. **Start by copying this layout.** |
| `megakernels/Qwen06BPOC_prefill_separate_kernels` | Same decode + batched prefill (XMX GEMM, oneDNN matmul interop, batched attention). Read its `OPTIMIZATION_LOG.md` - 50+ measured experiments. |
| `megakernels/Qwen06BPOC_prefill_megakernel` | Prefill also as a task system; shows why fine-grained tasks lost to separate kernels for prefill. |
| `megakernels/Gemma4POC` | int4 MoE 26B model. Structural (op-type) matching, exact graph rewrites, per-block MegaKernels (`kind` in params) collapsed into one whole-model persistent kernel, int4 repack at transformation time, cross-barrier weight prefetch, Q32.32 deterministic split-K, DPAS prefill + flash attention, in-kernel barrier profiling. Long header comments explain every decision with measurements. |
| `megakernels/README.txt`, `iOVIntegration.h`, `iMegakernelRuntime.h`, `megakernelTransformationUtils.h` | The contract every megakernel must satisfy and shared transformation helpers. |
| `research/preloading_gemv` | Task system (`taskSystem/`), semaphores, load-warp/compute-warp double-buffered SLM GEMV (`gemvOpt/gemvBlock.hcl`), in-kernel cycle profiling, ISA dump script. |
| `benchmarks/Qwen3_0_6B` | Standalone C++ harness: runtime without OpenVINO, GPU timestamps, bandwidth vs roofline. |

## Workflow

### 0. Setup
- Pick the device: use the one the user named; otherwise the discrete GPU:
  `python -c "import openvino as ov; c=ov.Core(); [print(d, c.get_property(d,'FULL_DEVICE_NAME'), c.get_property(d,'DEVICE_TYPE')) for d in c.available_devices]"`
  Tune for that device, but query limits (`CL_DEVICE_MAX_COMPUTE_UNITS`, SLM, max WG size) instead of hard-coding.
- Read [gotchas](references/gotchas.md) environment section for clues.

### 1. Understand the model
- Read the HF `config.json` and the modeling code in `transformers` (layers, hidden/intermediate sizes, heads, kv heads, head_dim, norms incl. q/k norms, RoPE variant/theta/partial rotary, sliding window, soft-caps, MoE, tied embeddings, eps).
- Export: `python MEGAKERNEL_POC/python/convert_to_openvino_ir.py --model-id <hf_id> --output-dir MEGAKERNEL_POC/python/<basename lowercased>-openvino-ir --task <text-generation-with-past|image-text-to-text> --weight-format <fp16 for <=1B models, int4 for larger>`
  (dir naming must match `model_dir()` in `ci_scripts/megakernels.sh`).
- Baseline: run the e2e benchmark (step 5) with no transformation for your model - both paths are stock, which gives
  baseline numbers and proves the harness works with this model (VLM layout and extra inputs may need `Feeder` tweaks).
- Profile the baseline decode step ([profiling](references/profiling.md)): primitive count, device time vs wall time, per-op breakdown, memory-roofline per GEMV. This decides what the megakernel must improve.
- Dump the graph as the pass sees it (`OV_MEGAKERNEL_DUMP=1 OV_MEGAKERNEL_DUMP_DIR=...` via `mk::util::verify_and_dump`, or `ov::pass::VisualizeTree`/`ov::serialize` inside your transformation). The pass runs in `src/plugins/intel_gpu/src/plugin/transformations_pipeline.cpp` (`InsertMegaKernel`): after ConvertPrecision, RMSFusion, weight-compression fusion (`FullyConnectedCompressed`, `MOECompressed`, `GatherCompressed`), SDPAScaleFusion; before KVCacheFusion and TransposeFusion. So `ReadValue`/`Assign` KV state is still in the graph, FCs are plugin ops.

### 2. Scaffold `MEGAKERNEL_POC/megakernels/<Name>/` (you may copy Qwen06BPOC)
- The implementation details may change over time as the plugin and transformation evolve, so use existing megakernels for reference.
- Only one implementation is built into the plugin at a time; switching = reconfigure + rebuild.
- Use task system for enqueueing kernel executions and managing dependencies efficiently.

### 3. Transformation rules
- **Leave every other model untouched.** Fingerprint first (e.g. `count_kv_read_values == 2*NUM_LAYERS` plus a
  model-unique structure, like `looks_like_gemma4`). Match and validate EVERYTHING before mutating anything; on any mismatch return `false` with the graph unchanged.
- Prefer structural matching on op types/shapes (Gemma4) over friendly names (Qwen); names from the exporter
  (`self.model.layers.<l>.self_attn.q_proj.weight`, `__module.model.layers.27/aten::add/Add_1`) change between
  optimum/transformers versions. Assert exact dtypes/shapes of weights (e.g. int4 = `u4 [N,K]` + `f16 [N,K/G]` scale + `u8` zp).
- MegaKernel op: `ov::intel_gpu::op::MegaKernel(inputs, MegaKernelAttrs)`. Inputs are any outputs (activations, Parameters, Constants = weights). `attrs.outputs = {{type, {MegaKernelDim::of_input(port, axis) | MegaKernelDim::constant(n), ...}}}`. `attrs.params = mk::util::make_params(<trivially copyable struct>)`; if you have several op kinds, make the first field `kind`.
- Weights become device USM pointers of the Constant inputs in `FillMegaKernelConstantParams`. Repack/stack at
  transformation time into NEW Constants (host side, once) so the device keeps a single copy - the original constants only feed the replaced subgraph, which becomes dead and is never compiled.
- Rewire the consumers of the replaced output (Convert to the downstream precision if needed),
  `m->remove_sink(assign)` for KV state the MK owns, `m->validate_nodes_and_infer_types()`, then `verify_and_dump`.
- Handle what frameworks add: genai inserts a last-token `StridedSlice` before `lm_head`; genai VLM passes
  `inputs_embeds` as non-USM `cl_mem` (plugin stages it, costs a copy); genai is run with `ATTENTION_BACKEND=SDPA`. PagedAttention is not yet supported.

### 4. Version 0: slow but accurate
Build the simplest correct implementation of the whole decoder, then validate it before any tuning:
- Plain kernels, enqueued in order from `Execute` on the given `stream`. f32 math/accumulation, weights read in their original layout, simple attention loop.
- MK owns its KV cache (device buffers `[L, KVH, cap, D]`). Derive `past` on device or from shapes
  (`position_ids`, or `attention_mask` length `L - T`); a sequence arriving with `past == 0` overwrites the cache.
  The runtime instance is shared by all infer requests of a compiled model.
- Prefill (`tokens > 1`): looping the decode path per token is acceptable for V0 - `clFinish` every ~64 tokens or long prompts die with `CL_OUT_OF_RESOURCES`.
- Validate with step 5 until outputs match.
- Alternative incremental route: replace one block at a time (FFN, attention input, mask) with per-block MegaKernels while stock SDPA/KV cache stays in between, validate each, then collapse into one whole-model kernel.

### 5. Build and test (every iteration)
Build (only one implementation per build tree):
```bash
cmake -S . -B build -DMEGAKERNEL_IMPLEMENTATION=<Name>
cmake --build build --target openvino_intel_gpu_plugin <Name> -j"$(nproc)"
```

Test:
```bash
python MEGAKERNEL_POC/python/e2e_performance_measurement.py \
    --model-dir MEGAKERNEL_POC/python/<model>-openvino-ir --device <GPU.N> \
    --frameworks decode_only optimum genai --tokens 100 --torch-threads 20 [--prompt "<session prompt>"]
```
- Each framework runs baseline (`OV_MEGAKERNEL_DISABLE=1`) and megakernel in subprocesses with `CACHE_DIR=""`.
- `decode_only`: native API, fixed-position decode microbenchmark; prints `decode_x`, `argmatch`/`cos` of the
  last-token prompt logits and the greedy text. Use `--decode-mode greedy` for MoE (routing changes per token) and `--decode-ctx 1000` (or more) to check long-context decode.
- `OUTPUT MISMATCH` blocks: identical text on short prompts is required. Divergence late in long generations can be a near-tie argmax flip, but make sure it's not a serious accuracy issue before dismissing - often the baseline is the less precise one.

### 6. Optimization passes
Only after V0 is correct: Loop: profile -> pick the biggest measured cost -> one change -> rebuild ->
accuracy + speed -> keep or revert -> append to `OPTIMIZATION_LOG.md` (hypothesis, change, table of numbers, outcome, why).
Read [optimization playbook](references/optimization-playbook.md) for what worked and what did not (with numbers); do not repeat a recorded dead end without a new reason. Weight preloading/prefetching across sync points is usually one of the most profitable decode techniques. If you find a new successful optimization, add it to the optimization playbook. The final model is supposed to be a single, persistent monokernel for decode - one kernel call per decoded token.

### 7. Add to CI
- Append `"<Name> <hf_model_id> <optimum export task>"` to the `megakernels` array in `MEGAKERNEL_POC/ci_scripts/megakernels.sh` (the first entry is the one `build_mk.sh` configures; `run_mk_tests.sh` rebuilds/reinstalls per entry, exports the IR to `model_dir()` if missing, runs `benchmark_app.sh` and `e2e_performance_measurement.py --frameworks decode_only optimum genai --torch-threads 20`).
- CI exports with `convert_to_openvino_ir.py`'s default `--weight-format fp16`. If your transformation needs int4, extend the entry format with a weight-format field and pass it through in `build_mk.sh`/`run_mk_tests.sh`, otherwise the transformation silently does not fire and CI just compares baseline with baseline.
- New gtests for the op/transformation must be named `megakernel_*` to run in CI.
- CI (`.github/workflows/megakernel-ci.yml`) only fails on crashes; it does not assert speedup or matching text - remind the user to read the tables.
- Run CI locally if in doubt: clone to /tmp, init submodules, run both steps inside the `ci_megakernel_image` docker
  exactly as the workflow does (image built from `ci_scripts/dockerfile/ci_megakernel.dockerfile`).

### 8. Report
Final e2e tables (all frameworks/prompts), accuracy evidence, the optimization log summary, the remaining bottleneck
with numbers, reproducible benchmark in a single command, and anything not checked.
