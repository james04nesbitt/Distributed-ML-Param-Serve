// =============================================================================
// Benchmark 2: CSR Compression Ratio & GPU Compression Throughput
//
// Measures:
//   - Payload size reduction (%) at various sparsity levels
//   - GPU vs CPU compression speed
//   - Round-trip accuracy (compress → decompress)
//
// This directly populates the resume line:
//   "reducing network payload sizes by [X]%"
// =============================================================================

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/core/sparse_format.h"
#include "src/cuda/gpu_sparse_ops.cuh"

namespace {

void CheckCuda(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(err));
  }
}

// Generate a dense matrix with a given sparsity (fraction of zeros).
std::vector<float> GenerateSparse(int32_t rows, int32_t cols, float sparsity,
                                  unsigned int seed = 42) {
  int32_t total = rows * cols;
  std::vector<float> dense(total);

  for (int32_t i = 0; i < total; ++i) {
    unsigned int h = (seed * 2654435761u) ^ (i * 2246822519u);
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h);
    float r = static_cast<float>(h & 0xFFFF) / 65535.0f;

    if (r < sparsity) {
      dense[i] = 0.0f;
    } else {
      float val =
          (static_cast<float>((h >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 0.02f;
      dense[i] = val;
    }
  }
  return dense;
}

} // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::cerr << "No CUDA device found." << std::endl;
    return 1;
  }

  int32_t num_rows = 1000;
  int32_t num_cols = 1000;
  int32_t total = num_rows * num_cols;
  float sparsities[] = {0.50f, 0.70f, 0.80f, 0.90f, 0.95f, 0.99f};

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "=== CSR Compression Benchmark (" << num_rows << "x" << num_cols
            << " = " << total << " elements) ===" << std::endl;
  std::cout << std::setw(12) << "Sparsity" << std::setw(10) << "NNZ"
            << std::setw(15) << "Dense (KB)" << std::setw(15) << "CSR (KB)"
            << std::setw(15) << "Reduction %" << std::setw(12) << "Ratio"
            << std::setw(15) << "CPU (ms)" << std::setw(15) << "GPU (ms)"
            << std::setw(12) << "Speedup" << std::endl;
  std::cout << std::string(121, '-') << std::endl;

  for (float sp : sparsities) {
    auto dense = GenerateSparse(num_rows, num_cols, sp);
    double dense_bytes = total * sizeof(float);

    // --- CPU compression benchmark ---
    auto cpu_start = std::chrono::high_resolution_clock::now();
    auto csr_cpu = paramserver::CompressToCSR(dense, num_rows, num_cols, 1e-8f);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    double cpu_ms =
        std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    int32_t nnz = csr_cpu.nnz();
    double csr_bytes = nnz * sizeof(float) + nnz * sizeof(int32_t) +
                       (num_rows + 1) * sizeof(int32_t);
    double reduction_pct = (1.0 - csr_bytes / dense_bytes) * 100.0;
    float ratio = paramserver::CompressionRatio(csr_cpu);

    // --- GPU compression benchmark ---
    float *d_dense = nullptr;
    CheckCuda(cudaMalloc(&d_dense, total * sizeof(float)), "alloc dense");
    CheckCuda(cudaMemcpy(d_dense, dense.data(), total * sizeof(float),
                         cudaMemcpyHostToDevice),
              "copy dense");

    // Warmup
    auto warmup =
        paramserver::cuda::CompressToCSRDevice(d_dense, num_rows, num_cols);
    warmup.Free();
    CheckCuda(cudaDeviceSynchronize(), "warmup sync");

    // Timed
    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "create start");
    CheckCuda(cudaEventCreate(&stop), "create stop");

    CheckCuda(cudaEventRecord(start), "record start");
    auto dcsr =
        paramserver::cuda::CompressToCSRDevice(d_dense, num_rows, num_cols);
    CheckCuda(cudaEventRecord(stop), "record stop");
    CheckCuda(cudaEventSynchronize(stop), "sync");

    float gpu_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&gpu_ms, start, stop), "elapsed");

    // Verify round-trip correctness.
    float *d_reconstructed = nullptr;
    CheckCuda(cudaMalloc(&d_reconstructed, total * sizeof(float)),
              "alloc reconstructed");
    paramserver::cuda::DecompressFromCSRDevice(dcsr, d_reconstructed);
    CheckCuda(cudaDeviceSynchronize(), "decompress sync");

    std::vector<float> reconstructed(total);
    CheckCuda(cudaMemcpy(reconstructed.data(), d_reconstructed,
                         total * sizeof(float), cudaMemcpyDeviceToHost),
              "copy back");

    double max_err = 0.0;
    for (int32_t i = 0; i < total; ++i) {
      max_err = std::max(
          max_err, static_cast<double>(std::fabs(dense[i] - reconstructed[i])));
    }

    double speedup = cpu_ms / gpu_ms;

    std::cout << std::setw(11) << (sp * 100) << "%" << std::setw(10) << nnz
              << std::setw(15) << (dense_bytes / 1024.0) << std::setw(15)
              << (csr_bytes / 1024.0) << std::setw(14) << reduction_pct << "%"
              << std::setw(12) << ratio << std::setw(15) << cpu_ms
              << std::setw(15) << gpu_ms << std::setw(11) << speedup << "x"
              << std::endl;

    if (max_err > 1e-6) {
      std::cout << "  *** WARNING: round-trip error = " << max_err << std::endl;
    }

    dcsr.Free();
    cudaFree(d_dense);
    cudaFree(d_reconstructed);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  // Large-scale summary
  std::cout
      << "\n=== Large Scale CSR Compression (10M elements, 90% sparse) ==="
      << std::endl;
  {
    int32_t lr = 10000, lc = 1000;
    int32_t lt = lr * lc;
    auto dense = GenerateSparse(lr, lc, 0.90f, 123);
    double dense_bytes = lt * sizeof(float);

    auto csr = paramserver::CompressToCSR(dense, lr, lc, 1e-8f);
    double csr_bytes = csr.nnz() * sizeof(float) + csr.nnz() * sizeof(int32_t) +
                       (lr + 1) * sizeof(int32_t);

    std::cout << "  Dense payload:  " << (dense_bytes / (1024.0 * 1024.0))
              << " MB" << std::endl;
    std::cout << "  CSR payload:    " << (csr_bytes / (1024.0 * 1024.0))
              << " MB" << std::endl;
    std::cout << "  NNZ:            " << csr.nnz() << " / " << lt << std::endl;
    std::cout << "  Payload reduction: "
              << ((1.0 - csr_bytes / dense_bytes) * 100.0) << "%" << std::endl;
    std::cout << "  Compression ratio: " << paramserver::CompressionRatio(csr)
              << "x" << std::endl;
  }

  return 0;
}
