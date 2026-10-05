#include "megakernelImpl.h"

#include "impl/gemma4POCRuntime.h"

namespace {
// Gemma-4-26B-A4B lm_head: 262144 x 2816, u8 with a per-row scale/zero point.
constexpr int VOCAB = 262144;
constexpr int HIDDEN = 2816;
}  // namespace

extern "C" mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime() {
  return new mk::Gemma4POCRuntime();
}

extern "C" void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime) {
  if (runtime) delete runtime;
}

extern "C" void FillMegaKernelConstantParams(mk::IConstantParams* params,
                                             const mk::MegakernelIo* io) {
  auto& w = *static_cast<mk::Gemma4ConstantParams*>(params);
  w.kind = static_cast<int>(io->kind);
  if (w.kind == mk::kFfn) {
    for (int p = mk::kFfnWPostAttn; p < mk::kFfnPorts; ++p)
      w.ffn[p] = io->input_ptr(io->ctx, p);
    return;
  }
  if (w.kind == mk::kAttnIn) {
    for (int p = mk::kAttnWIn; p < mk::kAttnPorts; ++p)
      w.attn[p] = io->input_ptr(io->ctx, p);
    // Repacked records: 36 bytes per row and 64-element group.
    w.attn_rows = static_cast<int>(io->input_dim(io->ctx, mk::kAttnQkv, 0) / (HIDDEN / 64 * 36));
    return;
  }
  if (w.kind == mk::kMaskFull || w.kind == mk::kMaskSliding) {
    w.mask_meta = io->input_ptr(io->ctx, mk::kMaskMeta);
    return;
  }
  w.lm_head_q = io->input_ptr(io->ctx, 1);
  w.lm_head_zp = io->input_ptr(io->ctx, 2);
  w.lm_head_scale = io->input_ptr(io->ctx, 3);
  w.N = VOCAB;
  w.K = HIDDEN;
}

extern "C" void FillMegaKernelRuntimeParams(mk::IRuntimeParams* params,
                                            const mk::MegakernelIo* io) {
  auto& p = *static_cast<mk::Gemma4RuntimeParams*>(params);
  p.in = io->input_ptr(io->ctx, 0);
  p.in2 = io->kind == mk::kFfn ? io->input_ptr(io->ctx, mk::kFfnH) : nullptr;
  p.out = io->output_ptr(io->ctx, 0);
  // Activation is [B, S, K]; the kernels treat B*S as the token count.
  p.tokens = static_cast<int>(io->input_dim(io->ctx, 0, 0) * io->input_dim(io->ctx, 0, 1));
  if (io->kind == mk::kFfn)
    p.ka = static_cast<int>(io->input_dim(io->ctx, 0, 2));
  if (io->kind == mk::kAttnIn) {
    p.in2 = io->input_ptr(io->ctx, mk::kAttnPos);
    p.s = static_cast<int>(io->input_dim(io->ctx, 0, 1));
  }
  if (io->kind == mk::kMaskFull || io->kind == mk::kMaskSliding) {
    p.in2 = io->input_ptr(io->ctx, mk::kMaskTti);
    p.tokens = static_cast<int>(io->input_dim(io->ctx, mk::kMaskAm, 0));
    p.ka = static_cast<int>(io->input_dim(io->ctx, mk::kMaskAm, 1));
    p.s = static_cast<int>(io->input_dim(io->ctx, mk::kMaskTti, 1));
  }
}
