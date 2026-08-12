#pragma once

#include "taskDesc.h"

// Assumptions of task system for MEGAKERNEL:
// 1. Tasks queue is preallocated on the host and it is already sorted
// topologically
// 2. Task queue is a const buffer, no dynamic task creation is allowed.
// 3. Any runtime task dependencies are solved by the task itself.
// 5. Task is executed from the beginning to the end by given task worker, no
// preemption is allowed.
// 6. When all tasks are executed, kernel is finished.
// 7. Tasks are scheduled to workers dynamically, FIFO order.
// 8. Task workers are blocks, each block executes one task at a time,
// cooperatively.
// 9. Task workers are guaranteed to be executed in parallel, otherwise deadLock
// may occur.
typedef struct TaskManager {
  GLOBAL_DEVICE_PTR const TaskDesc* workQueue;
  GLOBAL_DEVICE_PTR int* processedTaskCount;
  GLOBAL_DEVICE_PTR int* syncBuffer;
  int syncBufferSize;
  int workQueueSize;
} TaskManager;
