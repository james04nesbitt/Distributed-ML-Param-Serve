#include "src/cuda/gpu_sparse_ops.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>
#include <string>

namespace paramserver {
namespace cuda {

namespace {
void CheckCuda(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(err));
  }
}
} // namespace

// ============================================================================
// DeviceCSR
// ============================================================================

void DeviceCSR::Free() {
  if (d_values) {
    cudaFree(d_values);
    d_values = nullptr;
  }
  if (d_col_indices) {
    cudaFree(d_col_indices);
    d_col_indices = nullptr;
  }
  if (d_row_offsets) {
    cudaFree(d_row_offsets);
    d_row_offsets = nullptr;
  }
  nnz = 0;
}

// ============================================================================
// GPU CSR Compression — Stub (full implementation in Milestone 5)
//
// For now, compression is done on the CPU and transferred to the device.
// A future milestone will implement prefix-sum-based GPU compression.
// ============================================================================

DeviceCSR CompressToCSRDevice(const float *d_dense, int32_t num_rows,
                              int32_t num_cols, float threshold, void *stream) {
  // TODO(milestone-5): Implement GPU-side CSR compression using:
  //   1. Per-element threshold kernel → binary mask
  //   2. Prefix sum (thrust::exclusive_scan) → row_offsets + scatter indices
  //   3. Compact kernel → values + col_indices
  //
  // For now, fall back to CPU:
  int32_t total = num_rows * num_cols;
  std::vector<float> host_dense(total);
  CheckCuda(cudaMemcpy(host_dense.data(), d_dense, total * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "CompressToCSRDevice: copy to host");

  // CPU compression
  std::vector<float> values;
  std::vector<int32_t> col_indices;
  std::vector<int32_t> row_offsets;
  row_offsets.reserve(num_rows + 1);

  for (int32_t row = 0; row < num_rows; ++row) {
    row_offsets.push_back(static_cast<int32_t>(values.size()));
    for (int32_t col = 0; col < num_cols; ++col) {
      float val = host_dense[row * num_cols + col];
      if (std::fabs(val) > threshold) {
        values.push_back(val);
        col_indices.push_back(col);
      }
    }
  }
  row_offsets.push_back(static_cast<int32_t>(values.size()));

  return HostToDeviceCSR(values, col_indices, row_offsets, num_rows, num_cols);
}

// ============================================================================
// GPU Decompression Kernel
// ============================================================================

__global__ void DecompressCSRKernel(float *dense, const float *values,
                                    const int32_t *col_indices,
                                    const int32_t *row_offsets,
                                    int32_t num_rows, int32_t num_cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows)
    return;

  // Zero this row first
  for (int col = 0; col < num_cols; ++col) {
    dense[row * num_cols + col] = 0.0f;
  }

  // Fill non-zero entries
  for (int idx = row_offsets[row]; idx < row_offsets[row + 1]; ++idx) {
    dense[row * num_cols + col_indices[idx]] = values[idx];
  }
}

void DecompressFromCSRDevice(const DeviceCSR &csr, float *d_dense,
                             void *stream) {
  if (csr.num_rows == 0)
    return;

  constexpr int kBlockSize = 256;
  int num_blocks = (csr.num_rows + kBlockSize - 1) / kBlockSize;

  DecompressCSRKernel<<<num_blocks, kBlockSize, 0,
                        static_cast<cudaStream_t>(stream)>>>(
      d_dense, csr.d_values, csr.d_col_indices, csr.d_row_offsets, csr.num_rows,
      csr.num_cols);

  CheckCuda(cudaGetLastError(), "DecompressCSRKernel launch");
}

// ============================================================================
// Host ↔ Device Transfers
// ============================================================================

void DeviceCSRToHost(const DeviceCSR &csr, std::vector<float> &values,
                     std::vector<int32_t> &col_indices,
                     std::vector<int32_t> &row_offsets) {
  values.resize(csr.nnz);
  col_indices.resize(csr.nnz);
  row_offsets.resize(csr.num_rows + 1);

  if (csr.nnz > 0) {
    CheckCuda(cudaMemcpy(values.data(), csr.d_values, csr.nnz * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "DeviceCSRToHost: values");
    CheckCuda(cudaMemcpy(col_indices.data(), csr.d_col_indices,
                         csr.nnz * sizeof(int32_t), cudaMemcpyDeviceToHost),
              "DeviceCSRToHost: col_indices");
  }
  CheckCuda(cudaMemcpy(row_offsets.data(), csr.d_row_offsets,
                       (csr.num_rows + 1) * sizeof(int32_t),
                       cudaMemcpyDeviceToHost),
            "DeviceCSRToHost: row_offsets");
}

DeviceCSR HostToDeviceCSR(const std::vector<float> &values,
                          const std::vector<int32_t> &col_indices,
                          const std::vector<int32_t> &row_offsets,
                          int32_t num_rows, int32_t num_cols) {
  DeviceCSR csr;
  csr.num_rows = num_rows;
  csr.num_cols = num_cols;
  csr.nnz = static_cast<int32_t>(values.size());

  if (csr.nnz > 0) {
    CheckCuda(cudaMalloc(&csr.d_values, csr.nnz * sizeof(float)),
              "HostToDeviceCSR: alloc values");
    CheckCuda(cudaMemcpy(csr.d_values, values.data(), csr.nnz * sizeof(float),
                         cudaMemcpyHostToDevice),
              "HostToDeviceCSR: copy values");

    CheckCuda(cudaMalloc(&csr.d_col_indices, csr.nnz * sizeof(int32_t)),
              "HostToDeviceCSR: alloc col_indices");
    CheckCuda(cudaMemcpy(csr.d_col_indices, col_indices.data(),
                         csr.nnz * sizeof(int32_t), cudaMemcpyHostToDevice),
              "HostToDeviceCSR: copy col_indices");
  }

  CheckCuda(cudaMalloc(&csr.d_row_offsets, (num_rows + 1) * sizeof(int32_t)),
            "HostToDeviceCSR: alloc row_offsets");
  CheckCuda(cudaMemcpy(csr.d_row_offsets, row_offsets.data(),
                       (num_rows + 1) * sizeof(int32_t),
                       cudaMemcpyHostToDevice),
            "HostToDeviceCSR: copy row_offsets");

  return csr;
}

} // namespace cuda
} // namespace paramserver
