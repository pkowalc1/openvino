#include "megakernelImpl.h"

#include "impl/qwen06BPOCRuntime.h"

/////////////////////////////////////////////////////////////////////////
extern "C" mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime() {
  return new mk::Qwen06BPOCRuntime();
}

/////////////////////////////////////////////////////////////////////////
extern "C" void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime) {
  if (runtime) delete runtime;
}

/////////////////////////////////////////////////////////////////////////
// Port order must match qwen06BPOCTransformation.cpp.
extern "C" void FillMegaKernelConstantParams(mk::IConstantParams* params,
                                             const mk::MegakernelIo* io) {
  auto& w = *static_cast<mk::Qwen06BConstantParams*>(params);
  w.q_proj_w = io->input_ptr(io->ctx, 5);
  w.k_proj_w = io->input_ptr(io->ctx, 6);
  w.v_proj_w = io->input_ptr(io->ctx, 7);
  w.o_proj_w = io->input_ptr(io->ctx, 8);
  w.gate_proj_w = io->input_ptr(io->ctx, 9);
  w.up_proj_w = io->input_ptr(io->ctx, 10);
  w.down_proj_w = io->input_ptr(io->ctx, 11);
  w.input_ln_w = io->input_ptr(io->ctx, 12);
  w.post_attn_ln_w = io->input_ptr(io->ctx, 13);
  w.q_norm_w = io->input_ptr(io->ctx, 14);
  w.k_norm_w = io->input_ptr(io->ctx, 15);
  w.rope_inv_freq = io->input_ptr(io->ctx, 16);
}

/////////////////////////////////////////////////////////////////////////
extern "C" void FillMegaKernelRuntimeParams(mk::IRuntimeParams* params,
                                            const mk::MegakernelIo* io) {
  auto& p = *static_cast<mk::Qwen06BRuntimeParams*>(params);
  p.hidden_states = io->input_ptr(io->ctx, 0);
  p.position_ids = io->input_ptr(io->ctx, 1);
  p.hidden_states_out = io->output_ptr(io->ctx, 0);
  p.newTokens = static_cast<int>(io->input_dim(io->ctx, 0, 1));
}
