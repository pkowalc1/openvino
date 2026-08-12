#pragma once

#include "../shared/taskManager.h"

// GetNext task to execute.
// Returns invalid task(NULL) if no more tasks are available.
__global const TaskDesc* GetNextTask_block(TaskManager taskManager,
                                           __local char* slmBuffer);

// Clear the state of the task manager.
void LastWorkerClearTaskManagerState_block(TaskManager taskManager,
                                           __local char* slmBuffer);

///////////////////////////////////////////////////////////////
//
// INLINES:
//
/////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////
inline __global const TaskDesc* GetNextTask_thread(TaskManager taskManager) {
  const int slotId = atomic_inc(taskManager.processedTaskCount);
  if (slotId >= taskManager.workQueueSize) {
    return NULL;
  }
  return taskManager.workQueue + slotId;
}

/////////////////////////////////////////////////////////////
inline __global const TaskDesc* GetNextTask_block(TaskManager taskManager,
                                                  __local char* slmBuffer) {
  __global const TaskDesc* task = NULL;
  __local ulong* taskAddress = (__local ulong*)slmBuffer;

  // Broadcast the task pointer to all threads in the block, without using
  // work_group_broadcast, which uses SLM indirectly and decreseas occupancy in
  // case where block uses all SLM for its own purposes.

  if (get_local_id(0) == 0) {
    task = GetNextTask_thread(taskManager);
    *taskAddress = (ulong)task;
  }

  barrier(CLK_LOCAL_MEM_FENCE);
  task = (__global const TaskDesc*)(*taskAddress);
  return task;
}

/////////////////////////////////////////////////////////////
inline void ClearTaskManagerState_thread(TaskManager taskManager) {
  atomic_xchg(taskManager.processedTaskCount, 0);
}

/////////////////////////////////////////////////////////////
inline void LastWorkerClearTaskManagerState_block(TaskManager taskManager,
                                                  __local char* slmBuffer) {
  barrier(CLK_LOCAL_MEM_FENCE);

  __local bool* isLastWorker_local = (__local bool*)slmBuffer;

  if (get_local_id(0) == 0) {
    volatile __global atomic_int* syncBuffer =
        (volatile __global atomic_int*)(taskManager.processedTaskCount);
    const int processed = atomic_load_explicit(syncBuffer, memory_order_acquire,
                                               memory_scope_device);
    const int workers =
        get_num_groups(0) * get_num_groups(1) * get_num_groups(2);
    isLastWorker_local[0] = (processed == workers + taskManager.workQueueSize);
  }

  barrier(CLK_LOCAL_MEM_FENCE);

  bool isLastWorker = isLastWorker_local[0];

  if (isLastWorker) {
    if (get_local_id(0) == 0) {
      ClearTaskManagerState_thread(taskManager);
    }
    // CLear sync buffer for next launch:
    for (int i = get_local_id(0); i < taskManager.syncBufferSize;
         i += get_local_size(0)) {
      taskManager.syncBuffer[i] = 0;
    }
  }
}