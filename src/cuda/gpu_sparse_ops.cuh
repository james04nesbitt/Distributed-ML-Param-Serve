#pragma once

#include <cstdint>
#include <vector>

namespace paramserver {
namespace cuda {

// GPU-accelerated CSR sparse operations.
// Provides device-side compression and decompression of gradient matrices.

// Compress a dense matrix (already in device memory) into CSR format.
// Allocates and returns device pointers for CSR components.
// The caller is responsible for freeing the returned device memory.
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

// Compress a dense device matrix to CSR on the GPU.
// Elements with |value| <= threshold are treated as zero.
DeviceCSR CompressToCSRDevice(const float *d_dense, int32_t num_rows,
                              int32_t num_cols, float threshold = 1e-8f,
                              void *stream = nullptr);

// Decompress a CSR matrix to dense format on the GPU.
// d_dense must be pre-allocated with num_rows * num_cols floats.
void DecompressFromCSRDevice(const DeviceCSR &csr, float *d_dense,
                             void *stream = nullptr);

// Transfer CSR data from device to host vectors.
void DeviceCSRToHost(const DeviceCSR &csr, std::vector<float> &values,
                     std::vector<int32_t> &col_indices,
                     std::vector<int32_t> &row_offsets);

// Transfer CSR data from host vectors to a DeviceCSR.
// Allocates device memory; caller must Free() the result.
DeviceCSR HostToDeviceCSR(const std::vector<float> &values,
                          const std::vector<int32_t> &col_indices,
                          const std::vector<int32_t> &row_offsets,
                          int32_t num_rows, int32_t num_cols);

} // namespace cuda
} // namespace paramserver
