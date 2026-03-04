#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

namespace paramserver {
namespace cuda {

// ============================================================================
// PinnedBuffer — RAII wrapper for cudaMallocHost / cudaFreeHost
// ============================================================================

template <typename T> class PinnedBuffer {
public:
  PinnedBuffer() = default;

  explicit PinnedBuffer(size_t count) : count_(count) {
    if (count > 0) {
      cudaError_t err = cudaMallocHost(&ptr_, count * sizeof(T));
      if (err != cudaSuccess) {
        throw std::runtime_error(std::string("cudaMallocHost failed: ") +
                                 cudaGetErrorString(err));
      }
    }
  }

  ~PinnedBuffer() {
    if (ptr_) {
      cudaFreeHost(ptr_);
    }
  }

  // Non-copyable, movable.
  PinnedBuffer(const PinnedBuffer &) = delete;
  PinnedBuffer &operator=(const PinnedBuffer &) = delete;

  PinnedBuffer(PinnedBuffer &&other) noexcept
      : ptr_(other.ptr_), count_(other.count_) {
    other.ptr_ = nullptr;
    other.count_ = 0;
  }

  PinnedBuffer &operator=(PinnedBuffer &&other) noexcept {
    if (this != &other) {
      if (ptr_)
        cudaFreeHost(ptr_);
      ptr_ = other.ptr_;
      count_ = other.count_;
      other.ptr_ = nullptr;
      other.count_ = 0;
    }
    return *this;
  }

  T *data() { return ptr_; }
  const T *data() const { return ptr_; }
  size_t size() const { return count_; }

private:
  T *ptr_ = nullptr;
  size_t count_ = 0;
};

// ============================================================================
// DeviceCSR — GPU-resident Compressed Sparse Row
// ============================================================================

struct DeviceCSR {
  float *d_values = nullptr;        // Device: non-zero values
  int32_t *d_col_indices = nullptr; // Device: column indices
  int32_t *d_row_offsets = nullptr; // Device: row offsets
  int32_t num_rows = 0;
  int32_t num_cols = 0;
  int32_t nnz = 0;

  // Free device memory.
  void Free();
};

// ============================================================================
// Host-side pinned CSR — for efficient PCIe transfers
// ============================================================================

struct HostCSRPinned {
  PinnedBuffer<float> values;
  PinnedBuffer<int32_t> col_indices;
  PinnedBuffer<int32_t> row_offsets;
  int32_t num_rows = 0;
  int32_t num_cols = 0;
  int32_t nnz = 0;
};

// ============================================================================
// GPU CSR Compression / Decompression
// ============================================================================

// Compress a dense device matrix to CSR entirely on the GPU.
// Elements with |value| <= threshold are treated as zero.
// Uses Thrust prefix-scan and compaction — no D→H copies.
DeviceCSR CompressToCSRDevice(const float *d_dense, int32_t num_rows,
                              int32_t num_cols, float threshold = 1e-8f,
                              void *stream = nullptr);

// Decompress a CSR matrix to dense format on the GPU.
// d_dense must be pre-allocated with num_rows * num_cols floats.
void DecompressFromCSRDevice(const DeviceCSR &csr, float *d_dense,
                             void *stream = nullptr);

// ============================================================================
// Host ↔ Device Transfers
// ============================================================================

// Transfer CSR data from device to host vectors (pageable memory).
void DeviceCSRToHost(const DeviceCSR &csr, std::vector<float> &values,
                     std::vector<int32_t> &col_indices,
                     std::vector<int32_t> &row_offsets);

// Transfer CSR data from host vectors to a DeviceCSR (pageable memory).
DeviceCSR HostToDeviceCSR(const std::vector<float> &values,
                          const std::vector<int32_t> &col_indices,
                          const std::vector<int32_t> &row_offsets,
                          int32_t num_rows, int32_t num_cols);

// ============================================================================
// Pinned-Memory Async Transfers (PCIe-optimized)
// ============================================================================

// Async D→H into pinned host buffers. Caller must synchronize the stream
// before reading the returned HostCSRPinned.
HostCSRPinned DeviceCSRToHostPinned(const DeviceCSR &csr,
                                    void *stream = nullptr);

// Async H→D from pinned host buffers into a new DeviceCSR.
// Caller must synchronize the stream before using the returned DeviceCSR.
DeviceCSR HostToDeviceCSRAsync(const HostCSRPinned &host_csr,
                               void *stream = nullptr);

} // namespace cuda
} // namespace paramserver
