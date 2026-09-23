***===== MEGA KERNEL POC 2026 ====***

**1. IDEA**

We want to explore the possibility of optimizing OV model inference by introducing MegaKernel.

In short, MegaKernel combines a pipeline of many independent kernel calls into one GPU kernel, which is called once.

This may sound like simple kernel fusion, and in fact MegaKernel has all the pros and cons of kernel fusion, but it also leverages fine-grained control over scheduling and work partitioning to execute faster. A similar idea underlies, for example, flash attention.

This POC will try to improve latency with MegaKernel by:
1) preloading weights for the next operation while the previous operation is being computed
2) introducing fine-grained synchronization
3) eliminating the tail effect
4) leveraging all benefits of standard kernel fusion, such as eliminating kernel launch time and full synchronization

Of course, this may fail; in that case, we want to find out why.
Possible risks:
- The B60 GPU is a small GPU. Papers in the References section usually use powerful server-class GPUs like MI300X or B200, which are certainly underutilized for small models like Llama 1B or Qwen3 1.7B
- The B60 GPU lacks advanced features like async operations and direct transfers to shared memory, which such kernels heavily use
- Tooling risk, for example, lack of ASM support in Intel's OpenCL, which means lack of access to advanced hardware features

**2. What will be explored during the POC**

For this POC, we want to create a single mega kernel for the decode step of Qwen3 0.6B on B60 GPU. The kernel will only support that particular model; no generalization will be possible at this step.
The kernel will be integrated with the GPU plugin and, as such, can be used with frameworks such as GenAI and Optimum to run the Qwen3 0.6B model.

Assumptions:

- Model: Qwen3 0.6B
- HW: B60 GPU
- Batch size = 1 only
- Decode step only

**3. Expected speedup**

Similar work (see references) usually claims speedups of 1.2x to 1.7x.
We aim for a similar speedup.

**4. How to build**

For a local build, use `./MEGAKERNEL_POC/compile_run_megakernel.sh`. The first run installs the required dependencies; later runs reuse them and only rebuild, reinstall the generated wheel, and run the benchmarks. Use `./MEGAKERNEL_POC/compile_run_megakernel.sh --clean` to remove generated artifacts and cached setup state before a clean build.

**5. Code structure**

* MEGAKERNEL_POC/megakernels - contains various megakernels implemented for Qwen3 0.6B - each megakernel is implemented as a separate shared library. You can select megakernel implementation with cmake build flag, e.g. -DMEGAKERNEL_IMPLEMENTATION=Qwen06BPOC_prefill_separate_kernels

* MEGAKERNEL_POC/python - contains python tools to e.g. export the model

* MEGAKERNEL_POC/ci_scripts - contains scripts for CI

* MEGAKERNEL_POC/research - contains mini R&D projects to test some of the ideas that megakernel can use, e.g. warp-specialized gemv, preloading, etc

* On the OV side we have added a custom transformation and a separate Megakernel operator that calls shared lib with actual megakernel impl.

**6. References:**

1) Mirage Persistent Kernel: A Compiler and Runtime for Mega-Kernelizing Tensor Programs (https://arxiv.org/pdf/2512.22219)
2) Ada-MK: Adaptive MegaKernel Optimization via Automated
DAG-based Search for LLM Inference (https://arxiv.org/pdf/2605.11581)
3) Building a single-kernel, latency-optimized LLM inference engine on AMD MI300X GPUs (https://blog.kog.ai/building-a-single-kernel-latency-optimized-llm-inference-engine-on-amd-mi300x-gpus/)
4) Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B (https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles)
