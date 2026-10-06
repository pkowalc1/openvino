#include "megakernelImpl.h"
#include "qwen06BPOCParams.h"

#include <CL/cl_ext.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace fixture {
constexpr size_t hidden = 1024;
constexpr size_t layers = 28;
constexpr size_t fp16 = 2;

struct Weight {
    const char* name;
    size_t bytes;
    void* mk::Qwen06BConstantParams::*member;
};

constexpr std::array<Weight, 12> weights{{
    {"q_proj_w", layers * 2048 * hidden * fp16, &mk::Qwen06BConstantParams::q_proj_w},
    {"k_proj_w", layers * hidden * hidden * fp16, &mk::Qwen06BConstantParams::k_proj_w},
    {"v_proj_w", layers * hidden * hidden * fp16, &mk::Qwen06BConstantParams::v_proj_w},
    {"o_proj_w", layers * hidden * 2048 * fp16, &mk::Qwen06BConstantParams::o_proj_w},
    {"gate_proj_w", layers * 3072 * hidden * fp16, &mk::Qwen06BConstantParams::gate_proj_w},
    {"up_proj_w", layers * 3072 * hidden * fp16, &mk::Qwen06BConstantParams::up_proj_w},
    {"down_proj_w", layers * hidden * 3072 * fp16, &mk::Qwen06BConstantParams::down_proj_w},
    {"input_ln_w", layers * hidden * fp16, &mk::Qwen06BConstantParams::input_ln_w},
    {"post_attn_ln_w", layers * hidden * fp16, &mk::Qwen06BConstantParams::post_attn_ln_w},
    {"q_norm_w", layers * 128 * fp16, &mk::Qwen06BConstantParams::q_norm_w},
    {"k_norm_w", layers * 128 * fp16, &mk::Qwen06BConstantParams::k_norm_w},
    {"rope_inv_freq", 64 * fp16, &mk::Qwen06BConstantParams::rope_inv_freq},
}};
}

struct Options {
    std::string device;
    int context_tokens = 4000;
    int prefill_iterations = 5;
    int prefill_warmup = 1;
    int warmup = 5;
    int iterations = 100;
    uint32_t seed = 42;
    bool list_devices = false;
} options;

void check(cl_int status) {
    if (status != CL_SUCCESS) throw std::runtime_error("OpenCL error: " + std::to_string(status));
}

struct GpuEntry {
    cl_platform_id platform;
    cl_device_id device;
    std::string name;
};

std::vector<GpuEntry> enumerate_gpus() {
    std::vector<GpuEntry> gpus;
    cl_uint count = 0;
    if (clGetPlatformIDs(0, nullptr, &count) != CL_SUCCESS) return gpus;
    std::vector<cl_platform_id> platforms(count);
    check(clGetPlatformIDs(count, platforms.data(), nullptr));
    for (auto platform : platforms) {
        cl_uint devices_count = 0;
        if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &devices_count) != CL_SUCCESS) continue;
        std::vector<cl_device_id> devices(devices_count);
        check(clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, devices_count, devices.data(), nullptr));
        for (auto device : devices) {
            size_t size = 0;
            check(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &size));
            std::string name(size, '\0');
            check(clGetDeviceInfo(device, CL_DEVICE_NAME, size, name.data(), nullptr));
            name.resize(std::strlen(name.c_str()));
            gpus.push_back({platform, device, name});
        }
    }
    return gpus;
}

std::string lower(std::string text) {
    for (auto& c : text) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return text;
}

const GpuEntry* select_gpu(const std::vector<GpuEntry>& gpus, const std::string& selector) {
    if (!selector.empty() && std::all_of(selector.begin(), selector.end(), [](unsigned char c) { return std::isdigit(c); })) {
        const size_t index = std::stoul(selector);
        return index < gpus.size() ? &gpus[index] : nullptr;
    }
    for (const auto& gpu : gpus)
        if (gpu.name == selector) return &gpu;
    for (const auto& gpu : gpus)
        if (lower(gpu.name).find(lower(selector)) != std::string::npos) return &gpu;
    return nullptr;
}

void print_gpus(const std::vector<GpuEntry>& gpus) {
    std::cout << "Available OpenCL GPUs:\n";
    for (size_t index = 0; index < gpus.size(); ++index) std::cout << "  [" << index << "] " << gpus[index].name << '\n';
}

struct Device {
    cl_device_id id = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    clDeviceMemAllocINTEL_fn alloc = nullptr;
    clMemFreeINTEL_fn free = nullptr;
    clEnqueueMemcpyINTEL_fn copy = nullptr;
    std::vector<void*> buffers;

    explicit Device(const GpuEntry& gpu) : id(gpu.device) {
        cl_int status = CL_SUCCESS;
        context = clCreateContext(nullptr, 1, &id, nullptr, nullptr, &status);
        check(status);
        const cl_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        queue = clCreateCommandQueueWithProperties(context, id, properties, &status);
        check(status);
        alloc = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
            clGetExtensionFunctionAddressForPlatform(gpu.platform, "clDeviceMemAllocINTEL"));
        free = reinterpret_cast<clMemFreeINTEL_fn>(
            clGetExtensionFunctionAddressForPlatform(gpu.platform, "clMemFreeINTEL"));
        copy = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(
            clGetExtensionFunctionAddressForPlatform(gpu.platform, "clEnqueueMemcpyINTEL"));
        if (!alloc || !free || !copy) throw std::runtime_error("Intel GPU USM extension unavailable");
    }

    ~Device() {
        if (queue) clFinish(queue);
        for (auto* buffer : buffers) free(context, buffer);
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    void* upload(const void* data, size_t bytes) {
        cl_int status = CL_SUCCESS;
        void* buffer = alloc(context, id, nullptr, bytes, 0, &status);
        check(status);
        if (!buffer) throw std::runtime_error("USM allocation returned null");
        buffers.push_back(buffer);
        check(copy(queue, CL_TRUE, buffer, data, bytes, 0, nullptr, nullptr));
        return buffer;
    }

    void* allocate(size_t bytes) {
        cl_int status = CL_SUCCESS;
        void* buffer = alloc(context, id, nullptr, bytes, 0, &status);
        check(status);
        if (!buffer) throw std::runtime_error("USM allocation returned null");
        buffers.push_back(buffer);
        return buffer;
    }
};

struct RuntimeDeleter {
    void operator()(mk::IMegakernelRuntime* runtime) const {
        if (runtime) {
            runtime->Destroy();
            DestroyMegaKernelPOCRuntime(runtime);
        }
    }
};

std::vector<uint16_t> random_half(size_t count, std::mt19937& rng, bool norm = false) {
    std::uniform_int_distribution<unsigned> mantissa(0, 1023);
    std::uniform_int_distribution<unsigned> sign(0, 1);
    std::vector<uint16_t> data(count);
    for (auto& value : data)
        value = static_cast<uint16_t>((norm ? 14u : 8u) << 10 | mantissa(rng) | (norm ? 0u : sign(rng) << 15));
    return data;
}

bool is_norm(const char* name) {
    return std::strstr(name, "_ln_w") || std::strstr(name, "_norm_w");
}

cl_ulong profiling_ns(cl_event event, cl_profiling_info info) {
    cl_ulong value = 0;
    check(clGetEventProfilingInfo(event, info, sizeof(value), &value, nullptr));
    return value;
}

void print_phase(const char* name, cl_ulong gpu_ns, size_t minimum_bytes, int iterations, int warmup, bool is_b60) {
    const double bandwidth = static_cast<double>(minimum_bytes) * iterations / gpu_ns;
    std::cout << std::fixed << std::setprecision(2)
              << "\n" << name << " (" << iterations << " measured, " << warmup << " warmup)\n"
              << "  Latency:        " << gpu_ns / 1e6 / iterations << " ms\n"
              << "  Min transfer:   " << minimum_bytes / 1e6 << " MB\n"
              << "  Mem bandwidth:  " << bandwidth << " GB/s\n"
              << "  SOL Memory:     ";
    if (is_b60) {
        constexpr double b60_peak_bandwidth = 456.0;
        std::cout << 100.0 * bandwidth / b60_peak_bandwidth << "% (B60 peak: " << b60_peak_bandwidth << " GB/s)\n";
    } else {
        std::cout << "N/A (B60 only)\n";
    }
}

void expect_finite_output(Device& device, const void* buffer, size_t bytes) {
    std::vector<float> output(bytes / sizeof(float));
    check(device.copy(device.queue, CL_TRUE, output.data(), buffer, bytes, 0, nullptr, nullptr));
    for (size_t index = 0; index < output.size(); ++index)
        ASSERT_TRUE(std::isfinite(output[index])) << "Non-finite output at element " << index;
}

TEST(Qwen06B, RandomPrefillAndDecodeLatencyAndBandwidth) {
    const auto gpus = enumerate_gpus();
    if (gpus.empty()) GTEST_SKIP() << "No OpenCL GPU available";
    const GpuEntry* gpu = options.device.empty() ? &gpus.front() : select_gpu(gpus, options.device);
    if (!gpu) {
        print_gpus(gpus);
        FAIL() << "No GPU matches '" << options.device << "'. Pass --device=<index|name>.";
    }
    std::cout << "Device: " << gpu->name << "; seed: " << options.seed
              << "; context tokens: " << options.context_tokens << '\n';
    Device device(*gpu);
    std::mt19937 rng(options.seed);
    mk::Qwen06BConstantParams weights{};
    size_t weight_bytes = 0;
    for (const auto& weight : fixture::weights) {
        const auto data = random_half(weight.bytes / fixture::fp16, rng, is_norm(weight.name));
        weights.*(weight.member) = device.upload(data.data(), weight.bytes);
        weight_bytes += weight.bytes;
    }
    mk::Qwen06BPlatformParams platform{};
    platform.deviceId = device.id;
    platform.context = device.context;
    platform.stream = device.queue;
    std::unique_ptr<mk::IMegakernelRuntime, RuntimeDeleter> runtime(CreateMegaKernelPOCRuntime());
    ASSERT_EQ(runtime->Init(&weights, &platform), 0);

    const auto context_hidden = random_half(static_cast<size_t>(options.context_tokens) * fixture::hidden, rng);
    const int64_t context_position = 0;
    mk::Qwen06BRuntimeParams context{};
    context.hidden_states = device.upload(context_hidden.data(), context_hidden.size() * fixture::fp16);
    context.position_ids = device.upload(&context_position, sizeof(context_position));
    context.hidden_states_out = device.allocate(context_hidden.size() * sizeof(float));
    context.newTokens = options.context_tokens;
    ASSERT_EQ(runtime->Execute(&context), 0);
    check(clFinish(device.queue));
    expect_finite_output(device, context.hidden_states_out, context_hidden.size() * sizeof(float));

    for (int index = 0; index < options.prefill_warmup; ++index) ASSERT_EQ(runtime->Execute(&context), 0);
    check(clFinish(device.queue));
    cl_event prefill_begin = nullptr;
    cl_event prefill_end = nullptr;
    check(clEnqueueMarkerWithWaitList(device.queue, 0, nullptr, &prefill_begin));
    for (int index = 0; index < options.prefill_iterations; ++index) ASSERT_EQ(runtime->Execute(&context), 0);
    check(clEnqueueMarkerWithWaitList(device.queue, 0, nullptr, &prefill_end));
    check(clFinish(device.queue));
    const cl_ulong prefill_ns = profiling_ns(prefill_end, CL_PROFILING_COMMAND_START) -
                                profiling_ns(prefill_begin, CL_PROFILING_COMMAND_END);
    check(clReleaseEvent(prefill_begin));
    check(clReleaseEvent(prefill_end));
    expect_finite_output(device, context.hidden_states_out, context_hidden.size() * sizeof(float));

    constexpr size_t kv_bytes_per_token = fixture::layers * 2 * 8 * 128 * fixture::fp16;
    const size_t prefill_bytes = weight_bytes + static_cast<size_t>(options.context_tokens) * kv_bytes_per_token +
                                 context_hidden.size() * fixture::fp16 + sizeof(context_position) +
                                 context_hidden.size() * sizeof(float);

    const auto decode_hidden = random_half(fixture::hidden, rng);
    const int64_t decode_position = options.context_tokens;
    mk::Qwen06BRuntimeParams decode{};
    decode.hidden_states = device.upload(decode_hidden.data(), decode_hidden.size() * fixture::fp16);
    decode.position_ids = device.upload(&decode_position, sizeof(decode_position));
    decode.hidden_states_out = device.allocate(decode_hidden.size() * sizeof(float));
    decode.newTokens = 1;
    for (int index = 0; index < options.warmup; ++index) ASSERT_EQ(runtime->Execute(&decode), 0);
    check(clFinish(device.queue));
    expect_finite_output(device, decode.hidden_states_out, decode_hidden.size() * sizeof(float));

    cl_event begin = nullptr;
    cl_event end = nullptr;
    check(clEnqueueMarkerWithWaitList(device.queue, 0, nullptr, &begin));
    for (int index = 0; index < options.iterations; ++index) ASSERT_EQ(runtime->Execute(&decode), 0);
    check(clEnqueueMarkerWithWaitList(device.queue, 0, nullptr, &end));
    check(clFinish(device.queue));
    const cl_ulong gpu_ns = profiling_ns(end, CL_PROFILING_COMMAND_START) - profiling_ns(begin, CL_PROFILING_COMMAND_END);
    check(clReleaseEvent(begin));
    check(clReleaseEvent(end));
    expect_finite_output(device, decode.hidden_states_out, decode_hidden.size() * sizeof(float));

    const size_t minimum_bytes = weight_bytes + (static_cast<size_t>(decode_position) + 1) * kv_bytes_per_token +
                                 decode_hidden.size() * fixture::fp16 + sizeof(decode_position) +
                                 decode_hidden.size() * sizeof(float);
    const bool is_b60 = gpu->name.find("Arc(TM) Pro B60") != std::string::npos;
    print_phase("Prefill", prefill_ns, prefill_bytes, options.prefill_iterations, options.prefill_warmup, is_b60);
    print_phase("Decode", gpu_ns, minimum_bytes, options.iterations, options.warmup, is_b60);
}
}  // namespace

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    for (int index = 1; index < argc; ++index) {
        const std::string arg = argv[index];
        auto value = [&](const char* flag) -> const char* {
            const std::string prefix = std::string(flag) + "=";
            return arg.rfind(prefix, 0) == 0 ? arg.c_str() + prefix.size() : nullptr;
        };
        if (const char* v = value("--device")) options.device = v;
        else if (const char* v = value("--context-tokens")) options.context_tokens = std::stoi(v);
        else if (const char* v = value("--prefill-iterations")) options.prefill_iterations = std::stoi(v);
        else if (const char* v = value("--prefill-warmup")) options.prefill_warmup = std::stoi(v);
        else if (const char* v = value("--iterations")) options.iterations = std::stoi(v);
        else if (const char* v = value("--warmup")) options.warmup = std::stoi(v);
        else if (const char* v = value("--seed")) options.seed = static_cast<uint32_t>(std::stoul(v));
        else if (arg == "--list-devices") options.list_devices = true;
        else {
            std::cerr << "Unknown argument: " << arg << "\nUsage: qwen06b_random_decode_benchmark "
                         "[--device=<index|name>] [--context-tokens=N] [--prefill-iterations=N] "
                         "[--prefill-warmup=N] [--iterations=N] "
                         "[--warmup=N] [--seed=N] [--list-devices] [gtest flags]\n";
            return 2;
        }
    }
    if (options.list_devices) {
        print_gpus(enumerate_gpus());
        return 0;
    }
    if (options.iterations <= 0 || options.warmup < 0 || options.prefill_iterations <= 0 ||
        options.prefill_warmup < 0 || options.context_tokens < 2 || options.context_tokens >= 4096) {
        std::cerr << "--iterations and --prefill-iterations must be > 0; --warmup and --prefill-warmup >= 0; "
                     "--context-tokens must be between 2 and 4095\n";
        return 2;
    }
    return RUN_ALL_TESTS();
}