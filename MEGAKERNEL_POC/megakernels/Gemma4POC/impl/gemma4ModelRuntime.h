#pragma once

#include <CL/cl.h>
#include <CL/cl_ext.h>

#include <memory>
#include <vector>

#include "../gemma4POCParams.h"

namespace mk {

class Gemma4POCRuntime;

// kModel: the whole language model behind one MegaKernel op (see
// gemma4ModelKernels.h). It owns the KV cache; the cache length of a call is
// attention_mask's length minus the new tokens, so a fresh sequence (past 0)
// simply overwrites it.
//   * 1 token  -> one persistent mk_decode launch.
//   * T tokens -> the per-layer batched kernels of kAttnIn / kFfn runtimes
//                 (one per layer, built from the same weights), plus prompt
//                 attention and lm_head helpers.
class Gemma4ModelRuntime {
public:
    struct Usm {
        clDeviceMemAllocINTEL_fn alloc;
        clMemFreeINTEL_fn free;
        clMemBlockingFreeINTEL_fn blocking_free;
        clSetKernelArgMemPointerINTEL_fn set_arg;
        clEnqueueMemcpyINTEL_fn memcpy;
    };

    Gemma4ModelRuntime();
    ~Gemma4ModelRuntime();

    int Init(const Gemma4ConstantParams& w, cl_context ctx, cl_device_id dev, cl_command_queue q, const Usm& usm);
    int Execute(const Gemma4RuntimeParams& p);
    void Destroy();

private:
    int Decode(const Gemma4RuntimeParams& p, int past, int L);
    int Prompt(const Gemma4RuntimeParams& p, int past, int L);
    int EnsureCapacity(int L);
    int UploadTables();
    void Profile();
    void* Alloc(size_t bytes);
    void* Grow(void*& buf, size_t& cap, size_t bytes);

    cl_context ctx_ = nullptr;
    cl_device_id dev_ = nullptr;
    cl_command_queue q_ = nullptr;
    Usm usm_{};
    cl_program prog_ = nullptr;
    cl_kernel kDecode_ = nullptr, kCvt_ = nullptr, kKvAppend_ = nullptr, kMaskRange_ = nullptr, kPfAttn_ = nullptr, kPfFa_ = nullptr,
              kLmRows_ = nullptr;

    struct Layer {
        int R = 0, HQ = 0, HK = 0, D = 0, RN = 0, q_off = 0, k_off = 0, v_off = 0, half_c = 0, pos_f16 = 0;
        int sliding = 0;
        void* kc = nullptr;
        void* vc = nullptr;
    };
    std::vector<Layer> layers_;
    Gemma4ConstantParams w_{};
    int nl_ = 0, vocab_ = 0, last_only_ = 0, nwg_ = 0;
    float cap_div_ = 0, cap_mul_ = 0, scale_ = 1, window_ = 0, causal_off_ = 1, p_mul_ = 0, p_add_ = 0;
    int pos_stride_ = 1, am_stride_ = 1;
    int kv_cap_ = 0, kv_valid_ = 0;
    int epoch_ = 0;
    int att_base_[2] = {0, 0};  // sliding, full
    int max_kv_heads_[2] = {0, 0};  // sliding, full

    // Decode scratch (one block) and device tables.
    void* scratch_ = nullptr;
    void* tab_ = nullptr;      // ModelTab
    void* ltab_ = nullptr;     // LayerTab[nl_]
    std::vector<uint8_t> htab_, hltab_;
    size_t off_[16] = {};
    bool prof_on_ = false;
    bool debug_cmp_ = false;
    std::vector<uint16_t>* dbg_x_ = nullptr;
    void* prof_ = nullptr;
    std::vector<double> prof_acc_, prof_dbg_, prof_d_;
    int prof_n_ = 0;

    // Prompt path.
    std::vector<std::unique_ptr<Gemma4POCRuntime>> attn_rt_, ffn_rt_, mask_rt_;
    enum PfBuf { kXa, kXb, kHm, kO, kMaskF, kMaskS, kRange, kPfBufs };
    void* pf_[kPfBufs] = {};
    size_t pf_cap_[kPfBufs] = {};
};

}  // namespace mk
