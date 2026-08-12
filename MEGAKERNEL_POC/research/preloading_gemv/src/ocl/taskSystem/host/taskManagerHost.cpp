#include "taskManagerHost.h"

#define CHECK_OCL_SUCCESS(stmt)    \
  {                                \
    const cl_int _status = (stmt); \
    if (_status != CL_SUCCESS) {   \
      return _status;              \
    }                              \
  }

/////////////////////////////////////////////////////////
cl_int HostInitalizeTaskSystem(TaskManager& taskManager,
                               std::vector<TaskDesc>& tasksQueue,
                               int* gpuSyncBufferToClear, int bufferSize,
                               cl_device_id deviceId, cl_context context,
                               cl_command_queue queue) {
  const int ZERO = 0;

  cl_platform_id platform = nullptr;
  CHECK_OCL_SUCCESS(clGetDeviceInfo(deviceId, CL_DEVICE_PLATFORM,
                                    sizeof(platform), &platform, nullptr));
  const auto deviceMemAlloc = reinterpret_cast<clDeviceMemAllocINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform,
                                               "clDeviceMemAllocINTEL"));
  const auto enqueueMemcpy = reinterpret_cast<clEnqueueMemcpyINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform,
                                               "clEnqueueMemcpyINTEL"));
  cl_int status = CL_SUCCESS;
  TaskDesc* taskQueueGPU = static_cast<TaskDesc*>(deviceMemAlloc(
      context, deviceId, nullptr, tasksQueue.size() * sizeof(TaskDesc),
      alignof(TaskDesc), &status));
  CHECK_OCL_SUCCESS(status);
  int* nextTaskIDGPU = static_cast<int*>(deviceMemAlloc(
      context, deviceId, nullptr, sizeof(int), alignof(int), &status));
  CHECK_OCL_SUCCESS(status);

  CHECK_OCL_SUCCESS(
      enqueueMemcpy(queue, CL_TRUE, taskQueueGPU, tasksQueue.data(),
                    tasksQueue.size() * sizeof(TaskDesc), 0, nullptr, nullptr));
  CHECK_OCL_SUCCESS(enqueueMemcpy(queue, CL_TRUE, nextTaskIDGPU, &ZERO,
                                  sizeof(ZERO), 0, nullptr, nullptr));

  taskManager.workQueue = taskQueueGPU;
  taskManager.workQueueSize = static_cast<int>(tasksQueue.size());
  taskManager.processedTaskCount = nextTaskIDGPU;
  taskManager.syncBuffer = gpuSyncBufferToClear;
  taskManager.syncBufferSize = bufferSize;

  return CL_SUCCESS;
}

/////////////////////////////////////////////////////////
cl_int HostReleaseTaskSystem(TaskManager& taskManager, cl_device_id deviceId,
                             cl_context context) {
  cl_platform_id platform = nullptr;
  CHECK_OCL_SUCCESS(clGetDeviceInfo(deviceId, CL_DEVICE_PLATFORM,
                                    sizeof(platform), &platform, nullptr));
  const auto memFree = reinterpret_cast<clMemFreeINTEL_fn>(
      clGetExtensionFunctionAddressForPlatform(platform, "clMemFreeINTEL"));
  CHECK_OCL_SUCCESS(
      memFree(context, const_cast<TaskDesc*>(taskManager.workQueue)));
  CHECK_OCL_SUCCESS(memFree(context, taskManager.processedTaskCount));

  return CL_SUCCESS;
}