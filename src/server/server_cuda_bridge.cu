#include "src/server/server_cuda_bridge.h"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "src/cuda/gpu_parameter_store.cuh"

namespace paramserver {
namespace server_cuda {

namespace {
void CheckCuda(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(err));
  }
}
} // namespace

// The opaque handle wraps the actual GPUParameterStore.
struct GPUStoreHandle {
  cuda::GPUParameterStore store;
  GPUStoreHandle(int32_t total) : store(total) {}
};

// ============================================================================
// Stream pool
// ============================================================================

StreamHandle *CreateStreamPool(int count) {
  auto *pool = new StreamHandle[count];
  for (int i = 0; i < count; ++i) {
    cudaStream_t s;
    CheckCuda(cudaStreamCreate(&s), "CreateStreamPool");
    pool[i] = static_cast<StreamHandle>(s);
  }
  return pool;
}

void DestroyStreamPool(StreamHandle *pool, int count) {
  for (int i = 0; i < count; ++i) {
    cudaStreamDestroy(static_cast<cudaStream_t>(pool[i]));
  }
  delete[] pool;
}

// ============================================================================
// GPU Store
// ============================================================================

GPUStoreHandle *CreateGPUStore(int32_t total_params) {
  return new GPUStoreHandle(total_params);
}

void DestroyGPUStore(GPUStoreHandle *store) { delete store; }

int32_t GetStoreSize(const GPUStoreHandle *store) {
  return store->store.size();
}

// ============================================================================
// CSR Gradient Application
// ============================================================================

void ApplyCSRGradient(GPUStoreHandle *store, const float *h_values,
                      const int32_t *h_col_indices,
                      const int32_t *h_row_offsets, int32_t nnz,
                      int32_t num_rows, int32_t num_cols, float learning_rate,
                      StreamHandle stream) {
  auto s = static_cast<cudaStream_t>(stream);

  // Async H→D transfer from pinned host memory.
  float *d_values = nullptr;
  int32_t *d_col_indices = nullptr;
  int32_t *d_row_offsets = nullptr;

  CheckCuda(cudaMalloc(&d_values, nnz * sizeof(float)),
            "ApplyCSRGradient: alloc d_values");
  CheckCuda(cudaMalloc(&d_col_indices, nnz * sizeof(int32_t)),
            "ApplyCSRGradient: alloc d_col_indices");
  CheckCuda(cudaMalloc(&d_row_offsets, (num_rows + 1) * sizeof(int32_t)),
            "ApplyCSRGradient: alloc d_row_offsets");

  CheckCuda(cudaMemcpyAsync(d_values, h_values, nnz * sizeof(float),
                            cudaMemcpyHostToDevice, s),
            "ApplyCSRGradient: copy values");
  CheckCuda(cudaMemcpyAsync(d_col_indices, h_col_indices, nnz * sizeof(int32_t),
                            cudaMemcpyHostToDevice, s),
            "ApplyCSRGradient: copy col_indices");
  CheckCuda(cudaMemcpyAsync(d_row_offsets, h_row_offsets,
                            (num_rows + 1) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, s),
            "ApplyCSRGradient: copy row_offsets");

  // Apply with lock-free atomics.
  store->store.ApplyCSRGradientAsync(d_values, d_col_indices, d_row_offsets,
                                     num_rows, num_cols, learning_rate, s);

  // Synchronize before freeing temp device buffers.
  CheckCuda(cudaStreamSynchronize(s), "ApplyCSRGradient: sync");

  cudaFree(d_values);
  cudaFree(d_col_indices);
  cudaFree(d_row_offsets);
}

// ============================================================================
// Parameter Retrieval
// ============================================================================

void CopyParamsToHost(const GPUStoreHandle *store, float *pinned_dst,
                      int32_t count, StreamHandle stream) {
  auto s = static_cast<cudaStream_t>(stream);
  store->store.CopyToHostAsync(pinned_dst, count, s);
  CheckCuda(cudaStreamSynchronize(s), "CopyParamsToHost: sync");
}

// ============================================================================
// Pinned Memory
// ============================================================================

void *AllocPinned(size_t bytes) {
  void *ptr = nullptr;
  CheckCuda(cudaMallocHost(&ptr, bytes), "AllocPinned");
  return ptr;
}

void FreePinned(void *ptr) {
  if (ptr) {
    cudaFreeHost(ptr);
  }
}

} // namespace server_cuda
} // namespace paramserver
