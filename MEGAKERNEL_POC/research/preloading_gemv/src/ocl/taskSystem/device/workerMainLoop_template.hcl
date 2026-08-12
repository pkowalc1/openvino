#include "common/template.hcl"
#include "taskManager.hcl"

#ifndef WorkerMainLoop_block_SUFFIX
#define WorkerMainLoop_block_SUFFIX
#endif

// Template function.
// Main worker loo for each block.

// Requires template parameters:
// #define WorkerMainLoop_block_EXEC_FUN -> policy function: void FUNC(TaskDesc
// task, __local char* slmBuffer)

// Optional parameter to give unique name of template instantiation.
// #define WorkerMainLoop_block_SUFFIX
inline void TEMPLATE(WorkerMainLoop_block, WorkerMainLoop_block_SUFFIX)(
    __constant const TaskManager* taskManager, __local char* slmBuffer);

////////////////////////////////////////////////////////////////
//
// IMPLEMENTATION
//
////////////////////////////////////////////////////////////////

#ifndef WorkerMainLoop_block_EXEC_FUN
#error "WorkerMainLoop_block_EXEC_FUN is not defined"
#endif

inline void TEMPLATE(WorkerMainLoop_block, WorkerMainLoop_block_SUFFIX)(
    __constant const TaskManager* taskManagerPtr, __local char* slmBuffer) {
  __global const TaskDesc* taskPtr = NULL;
  TaskManager taskManager = *taskManagerPtr;
  taskPtr = GetNextTask_block(taskManager, slmBuffer);

  while (taskPtr != NULL) {
    TaskDesc task = *taskPtr;
    WorkerMainLoop_block_EXEC_FUN(task, slmBuffer);
    taskPtr = GetNextTask_block(taskManager, slmBuffer);
  }

  LastWorkerClearTaskManagerState_block(taskManager, slmBuffer);
}

#undef WorkerMainLoop_block_EXEC_FUN
#undef WorkerMainLoop_block_SUFFIX