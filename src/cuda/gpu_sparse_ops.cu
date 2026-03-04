#include "src/cuda/gpu_sparse_ops.cuh"

#include <cuda_runtime.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>

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
// GPU CSR Compression — Fully on-device using Thrust
//
// Algorithm:
//   1. Build a binary mask: mask[i] = (|dense[i]| > threshold) ? 1 : 0
//   2. Per-row NNZ: reduce_by_key over row indices with mask values
//   3. Row offsets: exclusive_scan over per-row NNZ counts
//   4. Per-element scatter index: exclusive_scan over the flat mask
//   5. Compact kernel: scatter non-zero values + column indices
// ============================================================================

// Kernel: threshold mask — 1 if |val| > threshold, else 0
__global__ void ThresholdMaskKernel(const float *dense, int32_t *mask,
                                    int32_t total, float threshold) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total) {
    float val = dense[idx];
    mask[idx] = (val > threshold || val < -threshold) ? 1 : 0;
  }
}

// Kernel: scatter non-zero values and column indices into compact arrays
__global__ void CompactKernel(const float *dense, const int32_t *mask,
                              const int32_t *scatter_indices, float *values,
                              int32_t *col_indices, int32_t total,
                              int32_t num_cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < total && mask[idx]) {
    int out_idx = scatter_indices[idx];
    values[out_idx] = dense[idx];
    col_indices[out_idx] = idx % num_cols;
  }
}

// Kernel: count non-zeros per row
__global__ void CountNNZPerRowKernel(const int32_t *mask, int32_t *row_nnz,
                                     int32_t num_rows, int32_t num_cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < num_rows) {
    int count = 0;
    const int32_t *row_start = mask + row * num_cols;
    for (int col = 0; col < num_cols; ++col) {
      count += row_start[col];
    }
    row_nnz[row] = count;
  }
}

DeviceCSR CompressToCSRDevice(const float *d_dense, int32_t num_rows,
                              int32_t num_cols, float threshold, void *stream) {
  int32_t total = num_rows * num_cols;
  constexpr int kBlockSize = 256;

  // --- Step 1: Build binary mask on device ---
  int32_t *d_mask = nullptr;
  CheckCuda(cudaMalloc(&d_mask, total * sizeof(int32_t)),
            "CompressToCSR: alloc mask");

  int mask_blocks = (total + kBlockSize - 1) / kBlockSize;
  ThresholdMaskKernel<<<mask_blocks, kBlockSize>>>(d_dense, d_mask, total,
                                                   threshold);
  CheckCuda(cudaGetLastError(), "ThresholdMaskKernel");

  // --- Step 2: Prefix sum over flat mask → scatter indices ---
  int32_t *d_scatter = nullptr;
  CheckCuda(cudaMalloc(&d_scatter, total * sizeof(int32_t)),
            "CompressToCSR: alloc scatter");

  thrust::device_ptr<int32_t> mask_ptr(d_mask);
  thrust::device_ptr<int32_t> scatter_ptr(d_scatter);
  thrust::exclusive_scan(mask_ptr, mask_ptr + total, scatter_ptr);

  // Total NNZ = scatter[last] + mask[last]
  int32_t last_scatter = 0, last_mask = 0;
  CheckCuda(cudaMemcpy(&last_scatter, d_scatter + total - 1, sizeof(int32_t),
                       cudaMemcpyDeviceToHost),
            "CompressToCSR: read last scatter");
  CheckCuda(cudaMemcpy(&last_mask, d_mask + total - 1, sizeof(int32_t),
                       cudaMemcpyDeviceToHost),
            "CompressToCSR: read last mask");
  int32_t nnz = last_scatter + last_mask;

  // --- Step 3: Per-row NNZ → row_offsets via exclusive_scan ---
  int32_t *d_row_nnz = nullptr;
  CheckCuda(cudaMalloc(&d_row_nnz, num_rows * sizeof(int32_t)),
            "CompressToCSR: alloc row_nnz");

  int row_blocks = (num_rows + kBlockSize - 1) / kBlockSize;
  CountNNZPerRowKernel<<<row_blocks, kBlockSize>>>(d_mask, d_row_nnz, num_rows,
                                                   num_cols);
  CheckCuda(cudaGetLastError(), "CountNNZPerRowKernel");

  // row_offsets has num_rows+1 entries
  int32_t *d_row_offsets = nullptr;
  CheckCuda(cudaMalloc(&d_row_offsets, (num_rows + 1) * sizeof(int32_t)),
            "CompressToCSR: alloc row_offsets");

  thrust::device_ptr<int32_t> row_nnz_ptr(d_row_nnz);
  thrust::device_ptr<int32_t> row_off_ptr(d_row_offsets);
  thrust::exclusive_scan(row_nnz_ptr, row_nnz_ptr + num_rows, row_off_ptr);

  // Set row_offsets[num_rows] = nnz
  CheckCuda(cudaMemcpy(d_row_offsets + num_rows, &nnz, sizeof(int32_t),
                       cudaMemcpyHostToDevice),
            "CompressToCSR: set final row offset");

  cudaFree(d_row_nnz);

  // --- Step 4: Compact values and col_indices ---
  DeviceCSR csr;
  csr.num_rows = num_rows;
  csr.num_cols = num_cols;
  csr.nnz = nnz;
  csr.d_row_offsets = d_row_offsets;

  if (nnz > 0) {
    CheckCuda(cudaMalloc(&csr.d_values, nnz * sizeof(float)),
              "CompressToCSR: alloc values");
    CheckCuda(cudaMalloc(&csr.d_col_indices, nnz * sizeof(int32_t)),
              "CompressToCSR: alloc col_indices");

    CompactKernel<<<mask_blocks, kBlockSize>>>(d_dense, d_mask, d_scatter,
                                               csr.d_values, csr.d_col_indices,
                                               total, num_cols);
    CheckCuda(cudaGetLastError(), "CompactKernel");
    CheckCuda(cudaDeviceSynchronize(), "CompressToCSR: sync after compact");
  }

  cudaFree(d_mask);
  cudaFree(d_scatter);

  return csr;
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
// Host ↔ Device Transfers (pageable memory — original API)
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

// ============================================================================
// Pinned-Memory Async Transfers (PCIe-optimized)
// ============================================================================

HostCSRPinned DeviceCSRToHostPinned(const DeviceCSR &csr, void *stream) {
  auto s = static_cast<cudaStream_t>(stream);

  HostCSRPinned host;
  host.num_rows = csr.num_rows;
  host.num_cols = csr.num_cols;
  host.nnz = csr.nnz;

  host.row_offsets = PinnedBuffer<int32_t>(csr.num_rows + 1);
  CheckCuda(cudaMemcpyAsync(host.row_offsets.data(), csr.d_row_offsets,
                            (csr.num_rows + 1) * sizeof(int32_t),
                            cudaMemcpyDeviceToHost, s),
            "DeviceCSRToHostPinned: row_offsets");

  if (csr.nnz > 0) {
    host.values = PinnedBuffer<float>(csr.nnz);
    host.col_indices = PinnedBuffer<int32_t>(csr.nnz);

    CheckCuda(cudaMemcpyAsync(host.values.data(), csr.d_values,
                              csr.nnz * sizeof(float), cudaMemcpyDeviceToHost,
                              s),
              "DeviceCSRToHostPinned: values");
    CheckCuda(cudaMemcpyAsync(host.col_indices.data(), csr.d_col_indices,
                              csr.nnz * sizeof(int32_t), cudaMemcpyDeviceToHost,
                              s),
              "DeviceCSRToHostPinned: col_indices");
  }

  return host;
}

DeviceCSR HostToDeviceCSRAsync(const HostCSRPinned &host_csr, void *stream) {
  auto s = static_cast<cudaStream_t>(stream);

  DeviceCSR csr;
  csr.num_rows = host_csr.num_rows;
  csr.num_cols = host_csr.num_cols;
  csr.nnz = host_csr.nnz;

  CheckCuda(
      cudaMalloc(&csr.d_row_offsets, (csr.num_rows + 1) * sizeof(int32_t)),
      "HostToDeviceCSRAsync: alloc row_offsets");
  CheckCuda(cudaMemcpyAsync(csr.d_row_offsets, host_csr.row_offsets.data(),
                            (csr.num_rows + 1) * sizeof(int32_t),
                            cudaMemcpyHostToDevice, s),
            "HostToDeviceCSRAsync: copy row_offsets");

  if (csr.nnz > 0) {
    CheckCuda(cudaMalloc(&csr.d_values, csr.nnz * sizeof(float)),
              "HostToDeviceCSRAsync: alloc values");
    CheckCuda(cudaMemcpyAsync(csr.d_values, host_csr.values.data(),
                              csr.nnz * sizeof(float), cudaMemcpyHostToDevice,
                              s),
              "HostToDeviceCSRAsync: copy values");

    CheckCuda(cudaMalloc(&csr.d_col_indices, csr.nnz * sizeof(int32_t)),
              "HostToDeviceCSRAsync: alloc col_indices");
    CheckCuda(cudaMemcpyAsync(csr.d_col_indices, host_csr.col_indices.data(),
                              csr.nnz * sizeof(int32_t), cudaMemcpyHostToDevice,
                              s),
              "HostToDeviceCSRAsync: copy col_indices");
  }

  return csr;
}

} // namespace cuda
} // namespace paramserver
