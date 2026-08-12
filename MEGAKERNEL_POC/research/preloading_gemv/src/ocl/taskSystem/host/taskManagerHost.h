#pragma once

#include <CL/cl_ext.h>

#include <vector>

#include "../shared/taskManager.h"

// Initialize task system on the host, allocate memory for task queue and
// processed task count on the device, and copy task queue to the device.
cl_int HostInitalizeTaskSystem(TaskManager& taskManager,
                               std::vector<TaskDesc>& tasksQueue,
                               int* gpuSyncBufferToClear, int bufferSize,
                               cl_device_id deviceId, cl_context context,
                               cl_command_queue queue);

// Release task system on the host, free memory for task queue and processed
// task count on the device.
cl_int HostReleaseTaskSystem(TaskManager& taskManager, cl_device_id deviceId,
                             cl_context context);