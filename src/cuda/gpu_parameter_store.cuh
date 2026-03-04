#pragma once

#include <cstdint>

namespace paramserver {
namespace cuda {

// GPU-resident parameter store.
// Manages weight matrices in device memory and applies gradient updates
// using CUDA atomic operations for lock-free concurrency.
class GPUParameterStore {
public:
  // Initialize the store with a flat parameter array of the given size.
  // Allocates device memory and zeroes it.
  GPUParameterStore(int32_t total_params);
  ~GPUParameterStore();

  // Non-copyable, movable.
  GPUParameterStore(const GPUParameterStore &) = delete;
  GPUParameterStore &operator=(const GPUParameterStore &) = delete;
  GPUParameterStore(GPUParameterStore &&other) noexcept;
  GPUParameterStore &operator=(GPUParameterStore &&other) noexcept;

  // Apply a dense gradient update on the GPU using atomic operations.
  // Launches on the given CUDA stream for async execution.
  // parameters[i] -= learning_rate * gradient_d[i]
  void ApplyGradientAsync(const float *gradient_d, float learning_rate,
                          int32_t count, void *stream = nullptr);

  // Apply a sparse (CSR) gradient update on the GPU using atomics.
  // Only non-zero gradient entries are applied.
  void ApplyCSRGradientAsync(const float *values_d,
                             const int32_t *col_indices_d,
                             const int32_t *row_offsets_d, int32_t num_rows,
                             int32_t num_cols, float learning_rate,
                             void *stream = nullptr);

  // Copy parameters from device to host (synchronous, pageable memory).
  void CopyToHost(float *host_dst, int32_t count) const;

  // Copy parameters from host to device (synchronous, pageable memory).
  void CopyFromHost(const float *host_src, int32_t count);

  // Async copy to pinned host memory on a CUDA stream.
  void CopyToHostAsync(float *pinned_dst, int32_t count, void *stream) const;

  // Async copy from pinned host memory on a CUDA stream.
  void CopyFromHostAsync(const float *pinned_src, int32_t count, void *stream);

  // Get raw device pointer (for kernel use).
  float *device_data() { return d_params_; }
  const float *device_data() const { return d_params_; }

  int32_t size() const { return total_params_; }

private:
  float *d_params_ = nullptr; // Device memory for parameters
  int32_t total_params_ = 0;
};

} // namespace cuda
} // namespace paramserver
