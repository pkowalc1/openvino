
__attribute__((intel_reqd_sub_group_size(32))) __kernel void memory_bandwidth(
    __global uint4* data, const uint vector_count) {
  const size_t index = get_global_id(0);
  const size_t total_work_items = get_global_size(0);
  uint4 acc = (uint4)(0);
  for (size_t i = index; i < vector_count; i += total_work_items) {
    acc += data[i] * (uint4)(1664525u) + (uint4)(1013904223u);
  }
  // I want to write much less data than I read, to measure only read bandwidth.
  data[index] = acc;
}