#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "../../../../common/benchmarkOpenCLpipeline.h"
#include "../../../../common/oclTestFixture.h"

namespace {

constexpr size_t MiB = 1024 * 1024;
constexpr size_t vectorSizeBytes = 4 * sizeof(cl_uint);
constexpr size_t bufferSizeBytes =
    256 * MiB;  // 256 MiB buffer size for the memory bandwidth test
constexpr size_t vectorCount = bufferSizeBytes / vectorSizeBytes;
constexpr size_t localWorkSize = 512;
constexpr size_t warmupIterations = 10;
constexpr size_t benchmarkIterations = 100;
constexpr size_t blocks = 40;

class MemoryBandwidthTest : public ocltest::OclTestFixture {};

TEST_F(MemoryBandwidthTest, DISABLED_MaxGpuBandwidthB60GPU_occupancy50Percent) {
  const std::string kernelPath = std::string(OPENCL_KERNEL_SOURCE_PATH) +
                                 "../tests/membandwith/memoryBandwidth.cl";
  const OCLBinary binary =
      createProgramAndKernel(kernelPath, "memory_bandwidth");

  cl_int status = CL_SUCCESS;
  cl_mem buffer = clCreateBuffer(context(), CL_MEM_READ_WRITE, bufferSizeBytes,
                                 nullptr, &status);
  ASSERT_OCL_SUCCESS(status);

  ASSERT_OCL_SUCCESS(clSetKernelArg(binary.kernel, 0, sizeof(buffer), &buffer));
  ASSERT_OCL_SUCCESS(
      clSetKernelArg(binary.kernel, 1, sizeof(vectorCount), &vectorCount));

  const size_t globalWorkSize = blocks * localWorkSize;
  const ocltest::ProfileResult profile = ocltest::ProfileOpenCL<false>(
      [&]() {
        ASSERT_OCL_SUCCESS(clEnqueueNDRangeKernel(
            queue(), binary.kernel, 1, nullptr, &globalWorkSize, &localWorkSize,
            0, nullptr, nullptr));
      },
      queue(), warmupIterations, benchmarkIterations);

  const double transferredBytes =
      static_cast<double>(bufferSizeBytes + globalWorkSize * vectorSizeBytes);
  const double bandwidthGBs = transferredBytes / (profile.averageUs * 1000.0);
  std::cout << std::fixed << std::setprecision(2)
            << "GPU memory bandwidth: " << bandwidthGBs << " GB/s"
            << " (buffer: " << bufferSizeBytes / MiB << " MiB, read + write)\n";

  EXPECT_GT(bandwidthGBs, 0.0);
  ASSERT_OCL_SUCCESS(clReleaseMemObject(buffer));
  releaseOCLBinary(binary);
}

}  // namespace