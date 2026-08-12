#include <CL/cl_ext.h>
#include <CL/cl_half.h>

#include <cstring>
#include <string>
#include <vector>

#include "../../../../../../common/utils.h"
#include "../../../../ocl/taskSystem/host/taskManagerHost.h"
#include "../../testCommon/gemvBenchmark.h"
#include "ocl/tasks/gemvTask.h"

namespace {

constexpr size_t WORKERS = 80;
constexpr size_t WORK_GROUP_SIZE = 512;

const std::string GEMV_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) + "../tests/gemv/static/ocl/";
const std::string TASK_SYSTEM_GEMV_KERNEL_PATH =
    std::string(OPENCL_KERNEL_SOURCE_PATH) +
    "../tests/gemv/taskSystem/single/ocl/";

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

class TaskSystemGemvTest : public ocltest::GemvTestFixture {
 public:
  void RunGemvBenchmark(size_t rows, size_t columns, size_t rowsPerBlock,
                        size_t gemvPhaseTileRows, size_t gemvComputeWarps);

 protected:
  ocltest::GemvBenchmarkResult BenchmarkTaskSystemGemv(
      const std::vector<float>& matrix, const std::vector<float>& vector,
      size_t rows, size_t columns, size_t rowsPerBlock,
      size_t gemvPhaseTileRows, size_t gemvComputeWarps) {
    const std::string buildOptions =
        "-I " + std::string(OPENCL_KERNEL_SOURCE_PATH) + " -I " +
        TASK_SYSTEM_GEMV_KERNEL_PATH +
        " -DMATRIX_ROWS=" + std::to_string(rows) +
        " -DMATRIX_COLUMNS=" + std::to_string(columns) +
        " -DBLOCK_TILE_ROWS=" + std::to_string(rowsPerBlock) +
        " -DGEMV_PHASE_TILE_ROWS=" + std::to_string(gemvPhaseTileRows) +
        " -DGEMV_COMPUTE_WARPS=" + std::to_string(gemvComputeWarps);
    const OCLBinary binary = createProgramAndKernel(
        TASK_SYSTEM_GEMV_KERNEL_PATH + "taskSystemGemvKernel.cl",
        "taskSystemGemvKernel", buildOptions);

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
    EXPECT_NE(deviceMemAlloc, nullptr);
    EXPECT_NE(enqueueMemcpy, nullptr);
    EXPECT_NE(memFree, nullptr);
    if (deviceMemAlloc == nullptr || enqueueMemcpy == nullptr ||
        memFree == nullptr) {
      releaseOCLBinary(binary);
      return {};
    }

    const std::vector<cl_half> matrixHalf = ConvertToHalf(matrix);
    const std::vector<cl_half> vectorHalf = ConvertToHalf(vector);
    cl_half* matrixGpu = static_cast<cl_half*>(deviceMemAlloc(
        context(), deviceId(), nullptr, matrixHalf.size() * sizeof(cl_half),
        alignof(cl_half), &status));
    ASSERT_OCL_SUCCESS(status);
    cl_half* vectorGpu = static_cast<cl_half*>(deviceMemAlloc(
        context(), deviceId(), nullptr, vectorHalf.size() * sizeof(cl_half),
        alignof(cl_half), &status));
    ASSERT_OCL_SUCCESS(status);
    cl_half* outputGpu = static_cast<cl_half*>(
        deviceMemAlloc(context(), deviceId(), nullptr, rows * sizeof(cl_half),
                       alignof(cl_half), &status));
    ASSERT_OCL_SUCCESS(status);
    ASSERT_OCL_SUCCESS(enqueueMemcpy(
        queue(), CL_TRUE, matrixGpu, matrixHalf.data(),
        matrixHalf.size() * sizeof(cl_half), 0, nullptr, nullptr));
    ASSERT_OCL_SUCCESS(enqueueMemcpy(
        queue(), CL_TRUE, vectorGpu, vectorHalf.data(),
        vectorHalf.size() * sizeof(cl_half), 0, nullptr, nullptr));

    const size_t taskCount = rows / rowsPerBlock;
    std::vector<TaskDesc> tasks(taskCount);
    for (size_t tileId = 0; tileId < taskCount; ++tileId) {
      tasks[tileId].type = 0;
      const GemvTask task = {matrixGpu, vectorGpu, outputGpu,
                             static_cast<int>(tileId)};
      static_assert(sizeof(task) <= PAYLOAD_SIZE,
                    "GemvTask size exceeds task payload size");
      std::memcpy(tasks[tileId].payload, &task, sizeof(task));
    }

    const size_t workerCount = std::min(WORKERS, taskCount);
    std::cout << "Benchmarking task-system GEMV kernel with " << taskCount
              << " tasks and " << workerCount << " workers...\n";

    TaskManager taskManager;
    ASSERT_OCL_SUCCESS(HostInitalizeTaskSystem(taskManager, tasks, nullptr, 0,
                                               deviceId(), context(), queue()));
    cl_mem taskManagerBuffer =
        clCreateBuffer(context(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                       sizeof(taskManager), &taskManager, &status);
    ASSERT_OCL_SUCCESS(status);
    ASSERT_OCL_SUCCESS(
        clSetKernelArg(binary.kernel, 0, sizeof(cl_mem), &taskManagerBuffer));
    void* indirectPointers[] = {matrixGpu, vectorGpu, outputGpu};
    ASSERT_OCL_SUCCESS(
        clSetKernelExecInfo(binary.kernel, CL_KERNEL_EXEC_INFO_USM_PTRS_INTEL,
                            sizeof(indirectPointers), indirectPointers));

    const size_t globalWorkSize = workerCount * WORK_GROUP_SIZE;
    const ocltest::ProfileResult profileResult =
        ocltest::ProfileOpenCL<ocltest::CLEAR_CACHE_BEFORE_BENCHMARK>(
            [&]() {
              ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(
                  queue(), binary.kernel, 1, nullptr, &globalWorkSize,
                  &WORK_GROUP_SIZE, 0, nullptr, nullptr));
            },
            queue(), ocltest::WARMUP_ITERATIONS, ocltest::BENCHMARK_ITERATIONS);

    std::vector<cl_half> outputHalf(rows);
    ASSERT_OCL_SUCCESS(enqueueMemcpy(
        queue(), CL_TRUE, outputHalf.data(), outputGpu,
        outputHalf.size() * sizeof(cl_half), 0, nullptr, nullptr));

    ASSERT_OCL_SUCCESS(clReleaseMemObject(taskManagerBuffer));
    ASSERT_OCL_SUCCESS(
        HostReleaseTaskSystem(taskManager, deviceId(), context()));
    ASSERT_OCL_SUCCESS(memFree(context(), outputGpu));
    ASSERT_OCL_SUCCESS(memFree(context(), vectorGpu));
    ASSERT_OCL_SUCCESS(memFree(context(), matrixGpu));
    releaseOCLBinary(binary);

    return {profileResult, ConvertToFloat(outputHalf)};
  }
};

void TaskSystemGemvTest::RunGemvBenchmark(size_t rows, size_t columns,
                                          size_t rowsPerBlock,
                                          size_t gemvPhaseTileRows,
                                          size_t gemvComputeWarps) {
  const std::vector<float> matrix =
      utils::createRandomBuffer(rows * columns, 0);
  const std::vector<float> vector = utils::createRandomBuffer(columns, 1);
  const std::vector<ocltest::GemvParams> shapes = {
      {rows, columns, rowsPerBlock, gemvPhaseTileRows, gemvComputeWarps}};

  const ocltest::GemvBenchmarkResult taskSystemResult =
      BenchmarkTaskSystemGemv(matrix, vector, rows, columns, rowsPerBlock,
                              gemvPhaseTileRows, gemvComputeWarps);
  ASSERT_FALSE(HasFailure());
  const ocltest::GemvBenchmarkResult gemvOptResult =
      benchmarkOpenClGemvChain({matrix}, vector, shapes, GEMV_KERNEL_PATH);
  const ocltest::GemvBenchmarkResult dnnlResult =
      benchmarkDnnlGemvChain({matrix}, vector, shapes);

  taskSystemResult.profileResult.print("GEMV task-system kernel");
  gemvOptResult.profileResult.print("GEMV optimized OpenCL kernel");
  dnnlResult.profileResult.print("GEMV oneDNN kernel");
  std::cout << "GemvOpt / task-system speedup: "
            << gemvOptResult.profileResult.averageUs /
                   taskSystemResult.profileResult.averageUs
            << "x\n";
  std::cout << "oneDNN / task-system speedup: "
            << dnnlResult.profileResult.averageUs /
                   taskSystemResult.profileResult.averageUs
            << "x\n";

  ASSERT_EQ(taskSystemResult.output.size(), dnnlResult.output.size());
  ASSERT_EQ(taskSystemResult.output.size(), gemvOptResult.output.size());
  for (size_t index = 0; index < rows; ++index) {
    ASSERT_NEAR(taskSystemResult.output[index], dnnlResult.output[index],
                ocltest::ABS_ERROR)
        << "Task-system GEMV differs from oneDNN at index " << index;
    ASSERT_NEAR(taskSystemResult.output[index], gemvOptResult.output[index],
                ocltest::ABS_ERROR)
        << "Task-system GEMV differs from GemvOpt at index " << index;
  }
}

#define RUN_GEMV_BENCHMARK(rows, columns, rowsPerBlock, gemvPhaseTileRows, \
                           gemvComputeWarps)                               \
  TEST_F(TaskSystemGemvTest, Gemv##rows##x##columns) {                     \
    RunGemvBenchmark(rows, columns, rowsPerBlock, gemvPhaseTileRows,       \
                     gemvComputeWarps);                                    \
  }

RUN_GEMV_BENCHMARK(2048, 1024, 32, 4, 4)
RUN_GEMV_BENCHMARK(1024, 2048, 32, 4, 4)
RUN_GEMV_BENCHMARK(1024, 3072, 32, 2, 2)
RUN_GEMV_BENCHMARK(3072, 1024, 32, 4, 4)

}  // namespace