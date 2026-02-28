#include "src/cuda/gpu_parameter_store.cuh"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace paramserver {
namespace cuda {

// ============================================================================
// CUDA Kernels
// ============================================================================

// Dense gradient application: params[i] -= lr * grad[i]
// Uses atomicAdd for lock-free concurrent updates from multiple streams.
__global__ void ApplyDenseGradientKernel(float *params, const float *gradient,
                                         float learning_rate, int32_t count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < count) {
    atomicAdd(&params[idx], -learning_rate * gradient[idx]);
  }
}

// CSR sparse gradient application.
// Each thread handles one non-zero element from the CSR matrix.
__global__ void ApplyCSRGradientKernel(float *params, const float *values,
                                       const int32_t *col_indices,
                                       const int32_t *row_offsets,
                                       int32_t num_rows, int32_t num_cols,
                                       float learning_rate) {
  int nnz = row_offsets[num_rows]; // Total non-zero elements
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nnz)
    return;

  // Find which row this element belongs to (binary search)
  int row = 0;
  int lo = 0, hi = num_rows;
  while (lo < hi) {
    int mid = (lo + hi) / 2;
    if (row_offsets[mid + 1] <= idx) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  row = lo;

  int col = col_indices[idx];
  int param_idx = row * num_cols + col;
  atomicAdd(&params[param_idx], -learning_rate * values[idx]);
}

// ============================================================================
// GPUParameterStore Implementation
// ============================================================================

namespace {
void CheckCuda(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(err));
  }
}
} // namespace

GPUParameterStore::GPUParameterStore(int32_t total_params)
    : total_params_(total_params) {
  CheckCuda(cudaMalloc(&d_params_, total_params * sizeof(float)),
            "cudaMalloc params");
  CheckCuda(cudaMemset(d_params_, 0, total_params * sizeof(float)),
            "cudaMemset params");
}

GPUParameterStore::~GPUParameterStore() {
  if (d_params_) {
    cudaFree(d_params_);
    d_params_ = nullptr;
  }
}

GPUParameterStore::GPUParameterStore(GPUParameterStore &&other) noexcept
    : d_params_(other.d_params_), total_params_(other.total_params_) {
  other.d_params_ = nullptr;
  other.total_params_ = 0;
}

GPUParameterStore &
GPUParameterStore::operator=(GPUParameterStore &&other) noexcept {
  if (this != &other) {
    if (d_params_)
      cudaFree(d_params_);
    d_params_ = other.d_params_;
    total_params_ = other.total_params_;
    other.d_params_ = nullptr;
    other.total_params_ = 0;
  }
  return *this;
}

void GPUParameterStore::ApplyGradientAsync(const float *gradient_d,
                                           float learning_rate, int32_t count,
                                           void *stream) {
  if (count > total_params_) {
    throw std::runtime_error("Gradient count exceeds parameter store size");
  }

  constexpr int kBlockSize = 256;
  int num_blocks = (count + kBlockSize - 1) / kBlockSize;

  ApplyDenseGradientKernel<<<num_blocks, kBlockSize, 0,
                             static_cast<cudaStream_t>(stream)>>>(
      d_params_, gradient_d, learning_rate, count);

  CheckCuda(cudaGetLastError(), "ApplyDenseGradientKernel launch");
}

void GPUParameterStore::ApplyCSRGradientAsync(
    const float *values_d, const int32_t *col_indices_d,
    const int32_t *row_offsets_d, int32_t num_rows, int32_t num_cols,
    float learning_rate, void *stream) {
  // We need to know nnz to size the grid. Copy row_offsets[num_rows] from
  // device — this is a single int32.
  int32_t nnz = 0;
  CheckCuda(cudaMemcpyAsync(&nnz, row_offsets_d + num_rows, sizeof(int32_t),
                            cudaMemcpyDeviceToHost,
                            static_cast<cudaStream_t>(stream)),
            "copy nnz");
  CheckCuda(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)),
            "sync for nnz");

  if (nnz == 0)
    return;

  constexpr int kBlockSize = 256;
  int num_blocks = (nnz + kBlockSize - 1) / kBlockSize;

  ApplyCSRGradientKernel<<<num_blocks, kBlockSize, 0,
                           static_cast<cudaStream_t>(stream)>>>(
      d_params_, values_d, col_indices_d, row_offsets_d, num_rows, num_cols,
      learning_rate);

  CheckCuda(cudaGetLastError(), "ApplyCSRGradientKernel launch");
}

void GPUParameterStore::CopyToHost(float *host_dst, int32_t count) const {
  CheckCuda(cudaMemcpy(host_dst, d_params_, count * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "CopyToHost");
}

void GPUParameterStore::CopyFromHost(const float *host_src, int32_t count) {
  CheckCuda(cudaMemcpy(d_params_, host_src, count * sizeof(float),
                       cudaMemcpyHostToDevice),
            "CopyFromHost");
}

} // namespace cuda
} // namespace paramserver
