#include "megakernelImpl.h"

#include "impl/gemma4POCRuntime.h"

namespace {
// Gemma-4-26B-A4B lm_head: 262144 x 2816, u8 with a per-row scale/zero point.
constexpr int HIDDEN = 2816;

// Every Gemma4 MegaKernel op's params blob is a Gemma4KindMeta (see
// gemma4POCTransformation.cpp); num_layers is only set for kModel.
bool read_kind(const mk::MegakernelIo* io, mk::Gemma4KindMeta& km) {
  return mk::ReadParams(*io, km);
}
}  // namespace

extern "C" mk::IMegakernelRuntime* CreateMegaKernelPOCRuntime() {
  return new mk::Gemma4POCRuntime();
}

extern "C" void DestroyMegaKernelPOCRuntime(mk::IMegakernelRuntime* runtime) {
  if (runtime) delete runtime;
}

extern "C" mk::TErrorcode FillMegaKernelConstantParams(mk::IConstantParams* params,
                                                       const mk::MegakernelIo* io) {
  auto& w = *static_cast<mk::Gemma4ConstantParams*>(params);
  mk::Gemma4KindMeta km{};
  if (!read_kind(io, km))
    return 1;
  w.kind = km.kind;
  if (w.kind == mk::kModel) {
    for (int p = mk::kModelMaskMeta; p < mk::kModelLayer0; ++p)
      w.model[p] = io->input_ptr(io->ctx, p);
    w.layers = km.num_layers;
    if (w.layers > mk::kModelMaxLayers)
      w.layers = 0;  // rejected by Init
    for (int l = 0; l < w.layers; ++l) {
      const int base = mk::kModelLayer0 + l * mk::kModelLayerPorts;
      for (int p = 0; p < mk::kModelLayerPorts; ++p)
        w.layer[l][p] = io->input_ptr(io->ctx, base + p);
      // Repacked records: 36 bytes per row and 64-element group.
      w.layer_qkv_rows[l] = static_cast<int>(io->input_dim(io->ctx, base + (mk::kAttnQkv - mk::kAttnWIn), 0) /
                                             (HIDDEN / 64 * 36));
    }
    return 0;
  }
  if (w.kind == mk::kFfn) {
    for (int p = mk::kFfnWPostAttn; p < mk::kFfnPorts; ++p)
      w.ffn[p] = io->input_ptr(io->ctx, p);
    return 0;
  }
  if (w.kind == mk::kAttnIn) {
    for (int p = mk::kAttnWIn; p < mk::kAttnPorts; ++p)
      w.attn[p] = io->input_ptr(io->ctx, p);
    // Repacked records: 36 bytes per row and 64-element group.
    w.attn_rows = static_cast<int>(io->input_dim(io->ctx, mk::kAttnQkv, 0) / (HIDDEN / 64 * 36));
    return 0;
  }
  if (w.kind == mk::kMaskFull || w.kind == mk::kMaskSliding) {
    w.mask_meta = io->input_ptr(io->ctx, mk::kMaskMeta);
    return 0;
  }
  return 1;
}

extern "C" mk::TErrorcode FillMegaKernelRuntimeParams(mk::IRuntimeParams* params,
                                                      const mk::MegakernelIo* io) {
  auto& p = *static_cast<mk::Gemma4RuntimeParams*>(params);
  mk::Gemma4KindMeta km{};
  if (!read_kind(io, km))
    return 1;
  if (km.kind == mk::kModel) {
    p.in = io->input_ptr(io->ctx, mk::kModelEmbeds);
    p.in2 = io->input_ptr(io->ctx, mk::kModelPos);
    p.in3 = io->input_ptr(io->ctx, mk::kModelAm);
    p.in4 = io->input_ptr(io->ctx, mk::kModelTti);
    p.out = io->output_ptr(io->ctx, 0);
    p.batch = static_cast<int>(io->input_dim(io->ctx, mk::kModelEmbeds, 0));
    p.tokens = static_cast<int>(io->input_dim(io->ctx, mk::kModelEmbeds, 1));
    p.s = p.tokens;
    p.ka = static_cast<int>(io->input_dim(io->ctx, mk::kModelAm, 1));
    return 0;
  }
  p.in = io->input_ptr(io->ctx, 0);
  p.in2 = km.kind == mk::kFfn ? io->input_ptr(io->ctx, mk::kFfnH) : nullptr;
  p.out = io->output_ptr(io->ctx, 0);
  // Activation is [B, S, K]; the kernels treat B*S as the token count.
  p.tokens = static_cast<int>(io->input_dim(io->ctx, 0, 0) * io->input_dim(io->ctx, 0, 1));
  if (km.kind == mk::kFfn)
    p.ka = static_cast<int>(io->input_dim(io->ctx, 0, 2));
  if (km.kind == mk::kAttnIn) {
    p.in2 = io->input_ptr(io->ctx, mk::kAttnPos);
    p.s = static_cast<int>(io->input_dim(io->ctx, 0, 1));
  }
  if (km.kind == mk::kMaskFull || km.kind == mk::kMaskSliding) {
    p.in2 = io->input_ptr(io->ctx, mk::kMaskTti);
    p.tokens = static_cast<int>(io->input_dim(io->ctx, mk::kMaskAm, 0));
    p.ka = static_cast<int>(io->input_dim(io->ctx, mk::kMaskAm, 1));
    p.s = static_cast<int>(io->input_dim(io->ctx, mk::kMaskTti, 1));
  }
  return 0;
}
