#include <CL/cl_ext.h>
#include <CL/cl_half.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "../../../../../../common/utils.h"
#include "../../../../ocl/taskSystem/host/taskManagerHost.h"
#include "../../testCommon/gemvBenchmark.h"
#include "ocl/tasks/gemv1024x2048Task.h"
#include "ocl/tasks/gemv1024x3072Task.h"
#include "ocl/tasks/gemv3072x1024Task.h"

namespace {

static constexpr float ABS_ERROR = 1e-5f;

constexpr size_t WORKERS = 35;
constexpr size_t WORK_GROUP_SIZE = 1024;

const std::string GEMV_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) + "../tests/gemv/static/ocl/";
const std::string TASK_SYSTEM_GEMV_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) +
    "../tests/gemv/taskSystem/chain/ocl/";

const std::vector<ocltest::GemvParams> TASK_SYSTEM_GEMV_PARAMS = {
    {1024, 2048, 32, 8, 8},
    {3072, 1024, 64, 16, 8},
    {1024, 3072, 32, 4, 4},
};

const std::vector<ocltest::GemvParams> GEMV_OPT_PARAMS = {
    {1024, 2048, 32, 4, 4},
    {3072, 1024, 32, 4, 4},
    {1024, 3072, 32, 2, 2},
};

std::vector<cl_half> ConvertToHalf(const std::vector<float>& input) {
  std::vector<cl_half> output(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = cl_half_from_float(input[index], CL_HALF_RTE);
  }
  return output;
}

std::vector<float> ConvertToFloat(const std::vector<cl_half>& input) {
  std::vector<float> output(input.size());
  for (size_t index = 0; index < input.size(); ++index) {
    output[index] = cl_half_to_float(input[index]);
  }
  return output;
}

void PrintTaskQueue(const std::vector<TaskDesc>& tasks) {
  // Print tasks for debugging:
  for (size_t taskIndex = 0; taskIndex < tasks.size(); ++taskIndex) {
    const TaskDesc& task = tasks[taskIndex];
    std::cout << "Task " << taskIndex << ": type=" << task.type
              << ", payloadSize=" << sizeof(task.payload) << "\n";
    // Print payload casted to the appropriate GEMV task struct based on the
    // task type
    switch (task.type) {
      case 3: {
        const Gemv1024x2048Task* gemvTask =
            reinterpret_cast<const Gemv1024x2048Task*>(task.payload);
        std::cout << "  Gemv1024x2048Task: tileId=" << gemvTask->tileId
                  << ", wantedInputSyncValue=" << gemvTask->wantedInputSyncValue
                  << "\n";
        break;
      }
      case 4: {
        const Gemv3072x1024Task* gemvTask =
            reinterpret_cast<const Gemv3072x1024Task*>(task.payload);
        std::cout << "  Gemv3072x1024Task: tileId=" << gemvTask->tileId
                  << ", wantedInputSyncValue=" << gemvTask->wantedInputSyncValue
                  << "\n";
        break;
      }
      case 5: {
        const Gemv1024x3072Task* gemvTask =
            reinterpret_cast<const Gemv1024x3072Task*>(task.payload);
        std::cout << "  Gemv1024x3072Task: tileId=" << gemvTask->tileId
                  << ", wantedInputSyncValue=" << gemvTask->wantedInputSyncValue
                  << "\n";
        break;
      }
      default:
        std::cerr << "Unknown task type: " << task.type << "\n";
        break;
    }
  }
}

template <typename GemvTask>
TaskDesc CreateGemvTaskDesc(int type, const cl_half* matrix,
                            const cl_half* vector, cl_half* output,
                            int* inputSemaphore, int* outputSemaphore,
                            int wantedInputSyncValue, int tileId) {
  TaskDesc taskDesc{};
  taskDesc.type = type;
  const GemvTask task = {matrix,         vector,          output,
                         inputSemaphore, outputSemaphore, wantedInputSyncValue,
                         tileId};
  static_assert(sizeof(task) <= PAYLOAD_SIZE,
                "GEMV task size exceeds task payload size");
  std::memcpy(taskDesc.payload, &task, sizeof(task));
  return taskDesc;
}

class ChainTaskSystemGemvTest : public ocltest::GemvTestFixture {
 protected:
  ocltest::GemvBenchmarkResult benchmarkTaskSystemChain(
      const std::vector<std::vector<float>>& matrices,
      const std::vector<float>& input,
      const std::vector<ocltest::GemvParams>& params);
};

ocltest::GemvBenchmarkResult ChainTaskSystemGemvTest::benchmarkTaskSystemChain(
    const std::vector<std::vector<float>>& matrices,
    const std::vector<float>& input,
    const std::vector<ocltest::GemvParams>& params) {
  std::vector<std::vector<cl_half>> matricesHalf(params.size());
  for (size_t layer = 0; layer < params.size(); ++layer) {
    matricesHalf[layer] = ConvertToHalf(matrices[layer]);
  }
  const std::vector<cl_half> inputHalf = ConvertToHalf(input);

  int BLOCK_TILE_ROWS_1024x2048 = params[0].rowsPerBlock;
  int PHASE_TILE_ROWS_1024x2048 = params[0].gemvPhaseTileRows;
  int COMPUTE_WARPS_1024x2048 = params[0].gemvComputeWarps;

  int BLOCK_TILE_ROWS_3072x1024 =
      params.size() > 1 ? params[1].rowsPerBlock : 32;
  int PHASE_TILE_ROWS_3072x1024 =
      params.size() > 1 ? params[1].gemvPhaseTileRows : 4;
  int COMPUTE_WARPS_3072x1024 =
      params.size() > 1 ? params[1].gemvComputeWarps : 4;

  int BLOCK_TILE_ROWS_1024x3072 =
      params.size() > 2 ? params[2].rowsPerBlock : 32;
  int PHASE_TILE_ROWS_1024x3072 =
      params.size() > 2 ? params[2].gemvPhaseTileRows : 2;
  int COMPUTE_WARPS_1024x3072 =
      params.size() > 2 ? params[2].gemvComputeWarps : 2;

  const OCLBinary binary = createProgramAndKernel(
      TASK_SYSTEM_GEMV_KERNEL_PATH + "chainTaskSystemGemvKernel.cl",
      "chainTaskSystemGemvKernel",
      "-I " + std::string(OPENCL_KERNEL_SOURCE_PATH) + " -I " +
          TASK_SYSTEM_GEMV_KERNEL_PATH +
          " -igc_opts 'VISAOptions=-hybridRAWithSpill -fastCompileRA'" +
          " -DTOTAL_WARPS=" + std::to_string(WORK_GROUP_SIZE / 32) +
          " -DBLOCK_TILE_ROWS_3072x1024=" +
          std::to_string(BLOCK_TILE_ROWS_3072x1024) +
          " -DPHASE_TILE_ROWS_3072x1024=" +
          std::to_string(PHASE_TILE_ROWS_3072x1024) +
          " -DCOMPUTE_WARPS_3072x1024=" +
          std::to_string(COMPUTE_WARPS_3072x1024) +
          " -DBLOCK_TILE_ROWS_1024x3072=" +
          std::to_string(BLOCK_TILE_ROWS_1024x3072) +
          " -DPHASE_TILE_ROWS_1024x3072=" +
          std::to_string(PHASE_TILE_ROWS_1024x3072) +
          " -DCOMPUTE_WARPS_1024x3072=" +
          std::to_string(COMPUTE_WARPS_1024x3072) +
          " -DBLOCK_TILE_ROWS_1024x2048=" +
          std::to_string(BLOCK_TILE_ROWS_1024x2048) +
          " -DPHASE_TILE_ROWS_1024x2048=" +
          std::to_string(PHASE_TILE_ROWS_1024x2048) +
          " -DCOMPUTE_WARPS_1024x2048=" +
          std::to_string(COMPUTE_WARPS_1024x2048));

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
  const auto memFree = reinterpret_cast<clMemFreeINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL"));
  if (deviceMemAlloc == nullptr || enqueueMemcpy == nullptr ||
      memFree == nullptr) {
    ADD_FAILURE() << "Required Intel USM extension functions are unavailable";
    return {};
  }

  std::vector<cl_half*> matrixGpu(params.size());
  for (size_t layer = 0; layer < params.size(); ++layer) {
    matrixGpu[layer] = static_cast<cl_half*>(
        deviceMemAlloc(context(), deviceId(), nullptr,
                       matricesHalf[layer].size() * sizeof(cl_half),
                       alignof(cl_half), &status));
    ASSERT_OCL_SUCCESS(status);
    ASSERT_OCL_SUCCESS(enqueueMemcpy(
        queue(), CL_TRUE, matrixGpu[layer], matricesHalf[layer].data(),
        matricesHalf[layer].size() * sizeof(cl_half), 0, nullptr, nullptr));
  }

  std::vector<cl_half*> vectorsGpu(params.size() + 1);
  vectorsGpu.front() = static_cast<cl_half*>(deviceMemAlloc(
      context(), deviceId(), nullptr, inputHalf.size() * sizeof(cl_half),
      alignof(cl_half), &status));
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(
      enqueueMemcpy(queue(), CL_TRUE, vectorsGpu.front(), inputHalf.data(),
                    inputHalf.size() * sizeof(cl_half), 0, nullptr, nullptr));
  for (size_t layer = 0; layer < params.size(); ++layer) {
    vectorsGpu[layer + 1] = static_cast<cl_half*>(deviceMemAlloc(
        context(), deviceId(), nullptr,
        params[layer].rowCount * sizeof(cl_half), alignof(cl_half), &status));
    ASSERT_OCL_SUCCESS(status);
  }

  const std::vector<int> clearedCompletionCounts(params.size(), 0);
  int* completionCountsGpu = static_cast<int*>(deviceMemAlloc(
      context(), deviceId(), nullptr,
      clearedCompletionCounts.size() * sizeof(int), alignof(int), &status));
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(enqueueMemcpy(
      queue(), CL_TRUE, completionCountsGpu, clearedCompletionCounts.data(),
      clearedCompletionCounts.size() * sizeof(int), 0, nullptr, nullptr));

  size_t totalTaskCount = 0;
  for (const ocltest::GemvParams& layerParams : params) {
    totalTaskCount += layerParams.rowCount / layerParams.rowsPerBlock;
  }
  std::vector<TaskDesc> tasks;
  tasks.reserve(totalTaskCount);
  size_t prevLayerOutputTiles = 0;
  for (size_t layer = 0; layer < params.size(); ++layer) {
    const ocltest::GemvParams& layerParams = params[layer];
    const size_t taskCount = layerParams.rowCount / layerParams.rowsPerBlock;
    std::cout << "Layer " << layer << ": " << taskCount << " tasks\n";
    for (size_t tileId = 0; tileId < taskCount; ++tileId) {
      switch (layer) {
        case 0:
          tasks.push_back(CreateGemvTaskDesc<Gemv1024x2048Task>(
              3, matrixGpu[layer], vectorsGpu[layer], vectorsGpu[layer + 1],
              nullptr, completionCountsGpu, 0, static_cast<int>(tileId)));
          break;
        case 1:
          tasks.push_back(CreateGemvTaskDesc<Gemv3072x1024Task>(
              4, matrixGpu[layer], vectorsGpu[layer], vectorsGpu[layer + 1],
              &completionCountsGpu[0], &completionCountsGpu[1],
              prevLayerOutputTiles, static_cast<int>(tileId)));
          break;
        case 2:
          tasks.push_back(CreateGemvTaskDesc<Gemv1024x3072Task>(
              5, matrixGpu[layer], vectorsGpu[layer], vectorsGpu[layer + 1],
              &completionCountsGpu[1], nullptr, prevLayerOutputTiles,
              static_cast<int>(tileId)));
          break;
      }
    }

    prevLayerOutputTiles = taskCount;
  }

  TaskManager taskManager;
  ASSERT_OCL_SUCCESS(
      HostInitalizeTaskSystem(taskManager, tasks, completionCountsGpu,
                              static_cast<int>(clearedCompletionCounts.size()),
                              deviceId(), context(), queue()));
  cl_mem taskManagerBuffer =
      clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     sizeof(taskManager), &taskManager, &status);
  ASSERT_OCL_SUCCESS(status);
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(binary.kernel, 0, sizeof(cl_mem), &taskManagerBuffer));
  std::vector<void*> indirectPointers;
  indirectPointers.reserve(matrixGpu.size() + vectorsGpu.size() + 1);
  indirectPointers.insert(indirectPointers.end(), matrixGpu.begin(),
                          matrixGpu.end());
  indirectPointers.insert(indirectPointers.end(), vectorsGpu.begin(),
                          vectorsGpu.end());
  indirectPointers.push_back(completionCountsGpu);
  ASSERT_OCL_SUCCESS(clSetKernelExecInfo(
      binary.kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
      indirectPointers.size() * sizeof(void*), indirectPointers.data()));
  const auto selectedWorkers = std::min(WORKERS, totalTaskCount);
  const size_t globalWorkSize = selectedWorkers * WORK_GROUP_SIZE;

  std::cout << "Total task count: " << totalTaskCount
            << ", workers: " << selectedWorkers << "\n";

  // Get initial output to check correctness after benchmarking:
  ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(queue(), binary.kernel, 1, nullptr,
                                            &globalWorkSize, &WORK_GROUP_SIZE,
                                            0, nullptr, nullptr));

  std::vector<cl_half> outputBeforeHalf(params.back().rowCount);
  ASSERT_OCL_SUCCESS(enqueueMemcpy(
      queue(), CL_TRUE, outputBeforeHalf.data(), vectorsGpu.back(),
      outputBeforeHalf.size() * sizeof(cl_half), 0, nullptr, nullptr));

  const ocltest::ProfileResult taskSystemProfile =
      ocltest::ProfileOpenCL<ocltest::CLEAR_CACHE_BEFORE_BENCHMARK>(

          [&]() {
            ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(
                queue(), binary.kernel, 1, nullptr, &globalWorkSize,
                &WORK_GROUP_SIZE, 0, nullptr, nullptr));
          },
          queue(), ocltest::WARMUP_ITERATIONS, ocltest::BENCHMARK_ITERATIONS);

  std::vector<cl_half> outputAfterHalf(params.back().rowCount);
  ASSERT_OCL_SUCCESS(enqueueMemcpy(
      queue(), CL_TRUE, outputAfterHalf.data(), vectorsGpu.back(),
      outputAfterHalf.size() * sizeof(cl_half), 0, nullptr, nullptr));

  ASSERT_OCL_SUCCESS(clReleaseMemObject(taskManagerBuffer));
  ASSERT_OCL_SUCCESS(HostReleaseTaskSystem(taskManager, deviceId(), context()));
  ASSERT_OCL_SUCCESS(memFree(context(), completionCountsGpu));
  for (cl_half* vectorGpu : vectorsGpu) {
    ASSERT_OCL_SUCCESS(memFree(context(), vectorGpu));
  }
  for (cl_half* matrix : matrixGpu) {
    ASSERT_OCL_SUCCESS(memFree(context(), matrix));
  }
  releaseOCLBinary(binary);

  // Compare output before and after benchmarking to ensure correctness:
  const auto beforeFloat = ConvertToFloat(outputBeforeHalf);
  const auto afterFloat = ConvertToFloat(outputAfterHalf);
  for (size_t index = 0; index < beforeFloat.size(); ++index) {
    EXPECT_NEAR(beforeFloat[index], afterFloat[index], ABS_ERROR)
        << "Output mismatch at index " << index;
  }

  return {taskSystemProfile, ConvertToFloat(outputAfterHalf)};
}

TEST_F(ChainTaskSystemGemvTest, ThreeGemvChain) {
  std::vector<std::vector<float>> matrices(GEMV_OPT_PARAMS.size());
  for (size_t layer = 0; layer < GEMV_OPT_PARAMS.size(); ++layer) {
    matrices[layer] = utils::createRandomBuffer(
        GEMV_OPT_PARAMS[layer].rowCount * GEMV_OPT_PARAMS[layer].columnCount,
        layer);
    const float scale =
        1.0f /
        std::sqrt(static_cast<float>(GEMV_OPT_PARAMS[layer].columnCount));
    for (float& value : matrices[layer]) {
      value *= scale;
    }
  }
  const std::vector<float> input =
      utils::createRandomBuffer(GEMV_OPT_PARAMS.front().columnCount, 3);

  std::cout << "Benchmarking three-GEMV task-system chain...\n";
  const ocltest::GemvBenchmarkResult taskSystemResult =
      benchmarkTaskSystemChain(matrices, input, TASK_SYSTEM_GEMV_PARAMS);
  std::cout << "Benchmarking three-GEMV OpenCL chain...\n";
  const ocltest::GemvBenchmarkResult openClResult = benchmarkOpenClGemvChain(
      matrices, input, GEMV_OPT_PARAMS, GEMV_KERNEL_PATH);
  std::cout << "Benchmarking three-GEMV oneDNN chain...\n";
  const ocltest::GemvBenchmarkResult dnnlResult =
      benchmarkDnnlGemvChain(matrices, input, GEMV_OPT_PARAMS);

  taskSystemResult.profileResult.print("3-GEMV task-system chain");
  openClResult.profileResult.print("3-GEMV OpenCL chain");
  dnnlResult.profileResult.print("3-GEMV oneDNN chain");
  std::cout << "OpenCL / task-system speedup: "
            << openClResult.profileResult.averageUs /
                   taskSystemResult.profileResult.averageUs
            << "x\n";
  std::cout << "oneDNN / task-system speedup: "
            << dnnlResult.profileResult.averageUs /
                   taskSystemResult.profileResult.averageUs
            << "x\n";

  ASSERT_EQ(taskSystemResult.output.size(), dnnlResult.output.size());
  ASSERT_EQ(taskSystemResult.output.size(), openClResult.output.size());
  for (size_t index = 0; index < taskSystemResult.output.size(); ++index) {
    ASSERT_NEAR(taskSystemResult.output[index], openClResult.output[index],
                ABS_ERROR)
        << "Task-system GEMV chain differs from OpenCL at index " << index;
    ASSERT_NEAR(taskSystemResult.output[index], dnnlResult.output[index],
                ABS_ERROR)
        << "Task-system GEMV chain differs from oneDNN at index " << index;
  }
}

}  // namespace