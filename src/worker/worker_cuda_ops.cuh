#pragma once

#include <cstdint>
#include <cuda_runtime.h>


namespace paramserver {
namespace worker {

// Allocate a dense gradient array on device memory.
float *AllocateGradientDevice(int32_t total);

// Generate sparse random gradient data on the GPU.
// Fills ~(1-sparsity) fraction with non-zero values.
void GenerateSparseGradient(float *d_gradient, int32_t total, float sparsity,
                            unsigned int seed);

} // namespace worker
} // namespace paramserver
