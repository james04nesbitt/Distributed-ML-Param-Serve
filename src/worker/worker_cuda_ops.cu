#include "src/worker/worker_cuda_ops.cuh"

#include <stdexcept>
#include <string>

namespace paramserver {
namespace worker {

namespace {
void CheckCuda(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(err));
  }
}
} // namespace

// Simple kernel to populate a dense array with sparse random-ish data.
// Uses a deterministic hash so results are reproducible per iteration.
__global__ void GenerateSparseGradientKernel(float *dense, int32_t total,
                                             float sparsity,
                                             unsigned int seed) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total)
    return;

  // Simple hash-based pseudo-random: keep ~(1-sparsity) fraction non-zero
  unsigned int h = (seed * 2654435761u) ^ (idx * 2246822519u);
  h = ((h >> 16) ^ h) * 0x45d9f3b;
  h = ((h >> 16) ^ h);
  float r = static_cast<float>(h & 0xFFFF) / 65535.0f;

  if (r < sparsity) {
    dense[idx] = 0.0f;
  } else {
    float val =
        (static_cast<float>((h >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 0.02f;
    dense[idx] = val;
  }
}

float *AllocateGradientDevice(int32_t total) {
  float *d_gradient = nullptr;
  CheckCuda(cudaMalloc(&d_gradient, total * sizeof(float)),
            "AllocateGradientDevice");
  return d_gradient;
}

void GenerateSparseGradient(float *d_gradient, int32_t total, float sparsity,
                            unsigned int seed) {
  constexpr int kBlockSize = 256;
  int num_blocks = (total + kBlockSize - 1) / kBlockSize;
  GenerateSparseGradientKernel<<<num_blocks, kBlockSize>>>(d_gradient, total,
                                                           sparsity, seed);
  CheckCuda(cudaGetLastError(), "GenerateSparseGradientKernel");
}

} // namespace worker
} // namespace paramserver
