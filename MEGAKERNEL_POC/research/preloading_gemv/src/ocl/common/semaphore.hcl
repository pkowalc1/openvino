#pragma once

// Whole block waits for the semaphore to reach the wanted value.
// Sync is done by thread 0 of selected warp.
inline void WaitForSemaphore_block(int warpID,
                                   volatile __global atomic_int* syncMemory,
                                   int wantedSyncVal);

// Whole block signals the semaphore by incrementing it by 1.
// Sync is done by thread 0 of selected warp.
inline void SignalSemaphore_block(int warpID,
                                  volatile __global atomic_int* syncMemory);

//////////////////////////////////////////////////////////////////////////////////////
//
// INLINES:
//
//////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////
inline void WaitForSemaphore_block(int warpID,
                                   volatile __global atomic_int* syncMemory,
                                   int wantedSyncVal) {
  const bool semaphoreActive = (syncMemory != NULL);
  if (semaphoreActive) {
    if (get_sub_group_id() == warpID && get_sub_group_local_id() == 0) {
      while (atomic_load_explicit(syncMemory, memory_order_relaxed,
                                  memory_scope_device) < wantedSyncVal) {
      }
      // This is needed to ensure global visibility of the operations of
      // producer, since barrier(CLK_GLOBAL_MEM_FENCE) only works at the
      // work-group level.
      atomic_load_explicit(syncMemory, memory_order_acquire,
                           memory_scope_device);
    }
    barrier(CLK_GLOBAL_MEM_FENCE);
  }
}

//////////////////////////////////////////////////////////////////////////////////////
inline void SignalSemaphore_block(int warpID,
                                  volatile __global atomic_int* syncMemory) {
  const bool semaphoreActive = (syncMemory != NULL);
  if (semaphoreActive) {
    barrier(CLK_GLOBAL_MEM_FENCE);
    if (syncMemory != NULL && get_sub_group_id() == warpID &&
        get_sub_group_local_id() == 0) {
      atomic_fetch_add_explicit(syncMemory, 1, memory_order_release,
                                memory_scope_device);
    }
  }
}