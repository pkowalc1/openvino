#include <CL/cl_ext.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <vector>

#include "../../../../../common/oclTestFixture.h"
#include "../../../ocl/taskSystem/host/taskManagerHost.h"
#include "ocl/tasks/pow2Task.h"
#include "ocl/tasks/siluTask.h"

namespace {

constexpr size_t WORKERS = 80;
constexpr size_t THREADS = 512;
constexpr size_t TILE_COUNT = 20;

class TaskSystemTests : public ocltest::OclTestFixture {};

const std::string TASK_SYSTEM_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) +
    "../tests/taskSystem/unaryChainTest/ocl/";

inline float Pow2(float x) { return x * x; }
inline float Silu(float x) { return x / (1.0f + std::exp(-x)); }

TEST_F(TaskSystemTests, UnaryChainTest) {
  const OCLBinary binary = createProgramAndKernel(
      TASK_SYSTEM_KERNEL_PATH + "taskManagerKernel.cl", "taskManagerKernel",
      "-I " + std::string(OPENCL_KERNEL_SOURCE_PATH) + " -I " +
          std::string(TASK_SYSTEM_KERNEL_PATH));

  // Create buffers:
  cl_int status = CL_SUCCESS;
  cl_platform_id platform = nullptr;
  ASSERT_OCL_SUCCESS(clGetDeviceInfo(deviceId(), CL_DEVICE_PLATFORM,
                                     sizeof(platform), &platform, nullptr));
  const auto deviceMemAlloc = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform,
                                               "clDeviceMemAllocINTEL"));
  const auto enqueueMemcpy = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform,
                                               "clEnqueueMemcpyINTEL"));
  const auto setKernelArgMemPointer =
      reinterpret_cast<clSetKernelArgMemPointerINTEL_fn>(
          clGetExtensionFunctionAddressForPlatform(
              platform, "clSetKernelArgMemPointerINTEL"));
  const auto memFree = reinterpret_cast<clMemFreeINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL"));

  std::vector<float> inputHost(TILE_COUNT * THREADS);
  for (size_t i = 0; i < inputHost.size(); ++i) {
    inputHost[i] = (static_cast<int>(i % 401) - 200) / 100.0f;
  }

  float* inputGPU = static_cast<float*>(deviceMemAlloc(
      context(), deviceId(), nullptr, inputHost.size() * sizeof(float),
      alignof(float), &status));
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(enqueueMemcpy(queue(), CL_TRUE, inputGPU, inputHost.data(),
                                   inputHost.size() * sizeof(float), 0, nullptr,
                                   nullptr));

  float* intermediateGPU = static_cast<float*>(deviceMemAlloc(
      context(), deviceId(), nullptr, inputHost.size() * sizeof(float),
      alignof(float), &status));
  ASSERT_OCL_SUCCESS(status);

  int* syncGPU = static_cast<int*>(
      deviceMemAlloc(context(), deviceId(), nullptr, TILE_COUNT * sizeof(int),
                     alignof(int), &status));
  ASSERT_OCL_SUCCESS(status);
  std::vector<int> syncClearHOST(TILE_COUNT, 0);
  ASSERT_OCL_SUCCESS(
      enqueueMemcpy(queue(), CL_TRUE, syncGPU, syncClearHOST.data(),
                    syncClearHOST.size() * sizeof(int), 0, nullptr, nullptr));

  float* outputGPU = static_cast<float*>(deviceMemAlloc(
      context(), deviceId(), nullptr, inputHost.size() * sizeof(float),
      alignof(float), &status));
  ASSERT_OCL_SUCCESS(status);

  std::vector<float> intermediateClearHOST(inputHost.size(), 0.0f);
  std::vector<float> outputClearHOST(inputHost.size(), 0.0f);
  std::vector<float> outputHOST(inputHost.size(), 0.0f);

  // -------------------------------------------------

  // Create task queue on the host and submit it:
  std::vector<TaskDesc> topologicallySortedTaskQueue(TILE_COUNT * 2);
  for (size_t index = 0; index < TILE_COUNT; ++index) {
    topologicallySortedTaskQueue[index].type = 0;
    Pow2Task task;
    task.size = THREADS;
    task.outputReady = syncGPU + index;
    task.input = inputGPU + index * THREADS;
    task.output = intermediateGPU + index * THREADS;
    static_assert(sizeof(task) <= PAYLOAD_SIZE,
                  "Pow2Task size exceeds payload size");
    std::memcpy(topologicallySortedTaskQueue[index].payload, &task,
                sizeof(task));

    topologicallySortedTaskQueue[TILE_COUNT + index].type = 1;
    SiluTask siluTask;
    siluTask.size = THREADS;
    siluTask.syncValue = 1;
    siluTask.inputReady = syncGPU + index;
    siluTask.input = intermediateGPU + index * THREADS;
    siluTask.output = outputGPU + index * THREADS;
    static_assert(sizeof(siluTask) <= PAYLOAD_SIZE,
                  "SiluTask size exceeds payload size");
    std::memcpy(topologicallySortedTaskQueue[TILE_COUNT + index].payload,
                &siluTask, sizeof(siluTask));
  }

  TaskManager taskManager;
  ASSERT_OCL_SUCCESS(HostInitalizeTaskSystem(
      taskManager, topologicallySortedTaskQueue, syncGPU, TILE_COUNT,
      deviceId(), context(), queue()));
  cl_mem taskManagerBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(taskManager), &taskManager, &status);
  ASSERT_OCL_SUCCESS(status);
  void* indirectPointers[] = {inputGPU, intermediateGPU, syncGPU, outputGPU};
  ASSERT_OCL_SUCCESS(
      clSetKernelExecInfo(binary.kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
                          sizeof(indirectPointers), indirectPointers));
  // -------------------------------------------------

  size_t workers = WORKERS;
  for (int i = 0; i < 100; ++i) {
    // How to get rid of this sync buff clear?
    ASSERT_OCL_SUCCESS(
        enqueueMemcpy(queue(), CL_TRUE, syncGPU, syncClearHOST.data(),
                      syncClearHOST.size() * sizeof(int), 0, nullptr, nullptr));
    ASSERT_OCL_SUCCESS(enqueueMemcpy(
        queue(), CL_TRUE, intermediateGPU, intermediateClearHOST.data(),
        intermediateClearHOST.size() * sizeof(float), 0, nullptr, nullptr));
    ASSERT_OCL_SUCCESS(enqueueMemcpy(
        queue(), CL_TRUE, outputGPU, outputClearHOST.data(),
        outputClearHOST.size() * sizeof(float), 0, nullptr, nullptr));

    ASSERT_OCL_SUCCESS(
        clSetKernelArg(binary.kernel, 0, sizeof(cl_mem), &taskManagerBuffer));

    const size_t globalWorkSize = workers * THREADS;
    ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue(), binary.kernel, 1,
                                              nullptr, &globalWorkSize,
                                              &THREADS, 0, nullptr, nullptr));

    ASSERT_OCL_SUCCESS(
        enqueueMemcpy(queue(), CL_TRUE, outputHOST.data(), outputGPU,
                      outputHOST.size() * sizeof(float), 0, nullptr, nullptr));

    for (size_t i = 0; i < outputHOST.size(); ++i) {
      ASSERT_NEAR(outputHOST[i], Silu(Pow2(inputHost[i])), 1e-5f)
          << "Task " << i << " was not executed correctly";
    }

    workers = (workers + 53) % WORKERS +
              1;  // Change number of workers for next iteration
  }

  ASSERT_OCL_SUCCESS(clReleaseMemObject(taskManagerBuffer));
  ASSERT_OCL_SUCCESS(memFree(context(), inputGPU));
  ASSERT_OCL_SUCCESS(memFree(context(), intermediateGPU));
  ASSERT_OCL_SUCCESS(memFree(context(), syncGPU));
  ASSERT_OCL_SUCCESS(memFree(context(), outputGPU));
  ASSERT_OCL_SUCCESS(HostReleaseTaskSystem(taskManager, deviceId(), context()));
  releaseOCLBinary(binary);
}

}  // namespace