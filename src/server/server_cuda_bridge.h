#pragma once

#include <cstdint>

// Bridge header for CUDA operations used by the server.
// This header uses only plain C++ types (no CUDA headers) so it can be
// included from files compiled by MSVC (cc_library) without going through NVCC.

namespace paramserver {
namespace server_cuda {

// Opaque handle for CUDA stream.
using StreamHandle = void *;

// Create and destroy a CUDA stream pool.
StreamHandle *CreateStreamPool(int count);
void DestroyStreamPool(StreamHandle *pool, int count);

// GPU parameter store operations.
struct GPUStoreHandle;

GPUStoreHandle *CreateGPUStore(int32_t total_params);
void DestroyGPUStore(GPUStoreHandle *store);
int32_t GetStoreSize(const GPUStoreHandle *store);

// Transfer CSR gradient from pinned host memory to GPU and apply with atomics.
// h_values, h_col_indices, h_row_offsets must be pinned memory
// (allocated via AllocPinned / FreePinned).
void ApplyCSRGradient(GPUStoreHandle *store, const float *h_values,
                      const int32_t *h_col_indices,
                      const int32_t *h_row_offsets, int32_t nnz,
                      int32_t num_rows, int32_t num_cols, float learning_rate,
                      StreamHandle stream);

// Copy parameters from GPU to pinned host buffer (async, synchronizes stream).
void CopyParamsToHost(const GPUStoreHandle *store, float *pinned_dst,
                      int32_t count, StreamHandle stream);

// Pinned memory allocation / deallocation.
void *AllocPinned(size_t bytes);
void FreePinned(void *ptr);

} // namespace server_cuda
} // namespace paramserver
