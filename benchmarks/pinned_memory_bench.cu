// =============================================================================
// Benchmark 3: Pinned vs Pageable Memory Transfer Latency
//
// Measures:
//   - H→D and D→H transfer times for pageable (std::vector) vs pinned memory
//   - Latency difference in ms
//   - Effective PCIe bandwidth (GB/s)
//
// This directly populates the resume line:
//   "cutting communication latency by [X]ms"
// =============================================================================

#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cuda/gpu_sparse_ops.cuh"

namespace {

void CheckCuda(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(err));
  }
}

struct TransferResult {
  double h2d_ms;
  double d2h_ms;
  double h2d_bw_gbs;
  double d2h_bw_gbs;
};

// Benchmark pageable (std::vector) memory transfers.
TransferResult BenchPageable(int32_t num_elements, int iterations) {
  size_t bytes = num_elements * sizeof(float);
  std::vector<float> host_data(num_elements, 1.0f);

  float *d_buf = nullptr;
  CheckCuda(cudaMalloc(&d_buf, bytes), "alloc device");

  // Warmup
  cudaMemcpy(d_buf, host_data.data(), bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(host_data.data(), d_buf, bytes, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // H→D benchmark
  cudaEvent_t s1, e1;
  CheckCuda(cudaEventCreate(&s1), "");
  CheckCuda(cudaEventCreate(&e1), "");
  CheckCuda(cudaEventRecord(s1), "");
  for (int i = 0; i < iterations; ++i) {
    cudaMemcpy(d_buf, host_data.data(), bytes, cudaMemcpyHostToDevice);
  }
  CheckCuda(cudaEventRecord(e1), "");
  CheckCuda(cudaEventSynchronize(e1), "");
  float h2d_ms = 0;
  cudaEventElapsedTime(&h2d_ms, s1, e1);

  // D→H benchmark
  cudaEvent_t s2, e2;
  CheckCuda(cudaEventCreate(&s2), "");
  CheckCuda(cudaEventCreate(&e2), "");
  CheckCuda(cudaEventRecord(s2), "");
  for (int i = 0; i < iterations; ++i) {
    cudaMemcpy(host_data.data(), d_buf, bytes, cudaMemcpyDeviceToHost);
  }
  CheckCuda(cudaEventRecord(e2), "");
  CheckCuda(cudaEventSynchronize(e2), "");
  float d2h_ms = 0;
  cudaEventElapsedTime(&d2h_ms, s2, e2);

  cudaFree(d_buf);
  cudaEventDestroy(s1);
  cudaEventDestroy(e1);
  cudaEventDestroy(s2);
  cudaEventDestroy(e2);

  double total_bytes_d = bytes * static_cast<double>(iterations);
  return {h2d_ms, d2h_ms, (total_bytes_d / 1e9) / (h2d_ms / 1e3),
          (total_bytes_d / 1e9) / (d2h_ms / 1e3)};
}

// Benchmark pinned (page-locked) memory transfers.
TransferResult BenchPinned(int32_t num_elements, int iterations) {
  size_t bytes = num_elements * sizeof(float);

  float *h_pinned = nullptr;
  CheckCuda(cudaMallocHost(&h_pinned, bytes), "alloc pinned");
  for (int32_t i = 0; i < num_elements; ++i) {
    h_pinned[i] = 1.0f;
  }

  float *d_buf = nullptr;
  CheckCuda(cudaMalloc(&d_buf, bytes), "alloc device");

  cudaStream_t stream;
  CheckCuda(cudaStreamCreate(&stream), "create stream");

  // Warmup
  cudaMemcpyAsync(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  // H→D benchmark
  cudaEvent_t s1, e1;
  CheckCuda(cudaEventCreate(&s1), "");
  CheckCuda(cudaEventCreate(&e1), "");
  CheckCuda(cudaEventRecord(s1, stream), "");
  for (int i = 0; i < iterations; ++i) {
    cudaMemcpyAsync(d_buf, h_pinned, bytes, cudaMemcpyHostToDevice, stream);
  }
  CheckCuda(cudaEventRecord(e1, stream), "");
  CheckCuda(cudaEventSynchronize(e1), "");
  float h2d_ms = 0;
  cudaEventElapsedTime(&h2d_ms, s1, e1);

  // D→H benchmark
  cudaEvent_t s2, e2;
  CheckCuda(cudaEventCreate(&s2), "");
  CheckCuda(cudaEventCreate(&e2), "");
  CheckCuda(cudaEventRecord(s2, stream), "");
  for (int i = 0; i < iterations; ++i) {
    cudaMemcpyAsync(h_pinned, d_buf, bytes, cudaMemcpyDeviceToHost, stream);
  }
  CheckCuda(cudaEventRecord(e2, stream), "");
  CheckCuda(cudaEventSynchronize(e2), "");
  float d2h_ms = 0;
  cudaEventElapsedTime(&d2h_ms, s2, e2);

  cudaStreamDestroy(stream);
  cudaFree(d_buf);
  cudaFreeHost(h_pinned);
  cudaEventDestroy(s1);
  cudaEventDestroy(e1);
  cudaEventDestroy(s2);
  cudaEventDestroy(e2);

  double total_bytes_d = bytes * static_cast<double>(iterations);
  return {h2d_ms, d2h_ms, (total_bytes_d / 1e9) / (h2d_ms / 1e3),
          (total_bytes_d / 1e9) / (d2h_ms / 1e3)};
}

// Benchmark CSR transfer: DeviceCSR → HostCSRPinned vs DeviceCSR → std::vector
TransferResult BenchCSRTransfer(int32_t num_rows, int32_t num_cols,
                                float sparsity, int iterations) {
  int32_t total = num_rows * num_cols;

  // Generate sparse data and upload to GPU.
  std::vector<float> dense(total);
  unsigned int seed = 42;
  for (int32_t i = 0; i < total; ++i) {
    unsigned int h = (seed * 2654435761u) ^ (i * 2246822519u);
    h = ((h >> 16) ^ h) * 0x45d9f3b;
    h = ((h >> 16) ^ h);
    float r = static_cast<float>(h & 0xFFFF) / 65535.0f;
    dense[i] =
        (r < sparsity)
            ? 0.0f
            : (static_cast<float>((h >> 8) & 0xFFFF) / 65535.0f - 0.5f) * 0.02f;
  }

  float *d_dense = nullptr;
  CheckCuda(cudaMalloc(&d_dense, total * sizeof(float)), "alloc");
  CheckCuda(cudaMemcpy(d_dense, dense.data(), total * sizeof(float),
                       cudaMemcpyHostToDevice),
            "copy");

  auto dcsr =
      paramserver::cuda::CompressToCSRDevice(d_dense, num_rows, num_cols);
  int32_t nnz = dcsr.nnz;
  size_t csr_bytes = nnz * sizeof(float) + nnz * sizeof(int32_t) +
                     (num_rows + 1) * sizeof(int32_t);

  cudaStream_t stream;
  CheckCuda(cudaStreamCreate(&stream), "create stream");

  // Warmup
  {
    auto tmp = paramserver::cuda::DeviceCSRToHostPinned(dcsr, stream);
    cudaStreamSynchronize(stream);
  }

  // Pinned D→H benchmark
  cudaEvent_t s1, e1;
  CheckCuda(cudaEventCreate(&s1), "");
  CheckCuda(cudaEventCreate(&e1), "");
  CheckCuda(cudaEventRecord(s1, stream), "");
  for (int i = 0; i < iterations; ++i) {
    auto pinned = paramserver::cuda::DeviceCSRToHostPinned(dcsr, stream);
    cudaStreamSynchronize(stream);
  }
  CheckCuda(cudaEventRecord(e1, stream), "");
  CheckCuda(cudaEventSynchronize(e1), "");
  float pinned_ms = 0;
  cudaEventElapsedTime(&pinned_ms, s1, e1);

  // Pageable D→H benchmark
  cudaEvent_t s2, e2;
  CheckCuda(cudaEventCreate(&s2), "");
  CheckCuda(cudaEventCreate(&e2), "");
  CheckCuda(cudaEventRecord(s2), "");
  for (int i = 0; i < iterations; ++i) {
    std::vector<float> vals;
    std::vector<int32_t> cols, rows;
    paramserver::cuda::DeviceCSRToHost(dcsr, vals, cols, rows);
  }
  CheckCuda(cudaEventRecord(e2), "");
  CheckCuda(cudaEventSynchronize(e2), "");
  float pageable_ms = 0;
  cudaEventElapsedTime(&pageable_ms, s2, e2);

  dcsr.Free();
  cudaFree(d_dense);
  cudaStreamDestroy(stream);
  cudaEventDestroy(s1);
  cudaEventDestroy(e1);
  cudaEventDestroy(s2);
  cudaEventDestroy(e2);

  double total_bytes_d = csr_bytes * static_cast<double>(iterations);
  return {pageable_ms, pinned_ms, (total_bytes_d / 1e9) / (pageable_ms / 1e3),
          (total_bytes_d / 1e9) / (pinned_ms / 1e3)};
}

} // namespace

int main() {
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::cerr << "No CUDA device found." << std::endl;
    return 1;
  }

  cudaDeviceProp prop;
  CheckCuda(cudaGetDeviceProperties(&prop, 0), "get props");
  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "PCIe bandwidth (theoretical): ~12-16 GB/s (Gen3 x16)\n"
            << std::endl;

  std::cout << std::fixed << std::setprecision(2);

  // --- Dense Transfer Benchmark ---
  struct Config {
    int32_t elements;
    int iterations;
    const char *label;
  };

  Config configs[] = {
      {100'000, 500, "100K (400 KB)"},
      {1'000'000, 200, "1M (4 MB)"},
      {10'000'000, 100, "10M (40 MB)"},
      {50'000'000, 50, "50M (200 MB)"},
  };

  std::cout << "=== Dense Transfer: Pageable vs Pinned Memory ===" << std::endl;
  std::cout << std::setw(20) << "Size" << std::setw(15) << "Page H2D(ms)"
            << std::setw(15) << "Pin H2D(ms)" << std::setw(15) << "Pin BW"
            << std::setw(15) << "Page D2H(ms)" << std::setw(15) << "Pin D2H(ms)"
            << std::setw(15) << "Latency Cut" << std::endl;
  std::cout << std::string(110, '-') << std::endl;

  for (const auto &cfg : configs) {
    auto page = BenchPageable(cfg.elements, cfg.iterations);
    auto pin = BenchPinned(cfg.elements, cfg.iterations);

    double h2d_cut_ms = (page.h2d_ms - pin.h2d_ms) / cfg.iterations;
    double d2h_cut_ms = (page.d2h_ms - pin.d2h_ms) / cfg.iterations;

    std::cout << std::setw(20) << cfg.label << std::setw(15)
              << (page.h2d_ms / cfg.iterations) << std::setw(15)
              << (pin.h2d_ms / cfg.iterations) << std::setw(14)
              << pin.h2d_bw_gbs << "G" << std::setw(15)
              << (page.d2h_ms / cfg.iterations) << std::setw(15)
              << (pin.d2h_ms / cfg.iterations) << std::setw(14)
              << (h2d_cut_ms + d2h_cut_ms) << "ms" << std::endl;
  }

  // --- CSR Transfer Benchmark ---
  std::cout
      << "\n=== CSR Transfer: Pinned vs Pageable (1000x1000, 90% sparse) ==="
      << std::endl;
  {
    auto result = BenchCSRTransfer(1000, 1000, 0.90f, 200);
    double per_iter_pageable = result.h2d_ms / 200.0;
    double per_iter_pinned = result.d2h_ms / 200.0;
    double cut_ms = per_iter_pageable - per_iter_pinned;

    std::cout << "  Pageable D2H:  " << per_iter_pageable << " ms/transfer"
              << std::endl;
    std::cout << "  Pinned D2H:    " << per_iter_pinned << " ms/transfer"
              << std::endl;
    std::cout << "  Latency cut:   " << cut_ms << " ms per transfer"
              << std::endl;
    std::cout << "  Pageable BW:   " << result.h2d_bw_gbs << " GB/s"
              << std::endl;
    std::cout << "  Pinned BW:     " << result.d2h_bw_gbs << " GB/s"
              << std::endl;
  }

  std::cout << "\n=== CSR Transfer: Pinned vs Pageable (10Kx1K, 90% sparse) ==="
            << std::endl;
  {
    auto result = BenchCSRTransfer(10000, 1000, 0.90f, 50);
    double per_iter_pageable = result.h2d_ms / 50.0;
    double per_iter_pinned = result.d2h_ms / 50.0;
    double cut_ms = per_iter_pageable - per_iter_pinned;

    std::cout << "  Pageable D2H:  " << per_iter_pageable << " ms/transfer"
              << std::endl;
    std::cout << "  Pinned D2H:    " << per_iter_pinned << " ms/transfer"
              << std::endl;
    std::cout << "  Latency cut:   " << cut_ms << " ms per transfer"
              << std::endl;
    std::cout << "  Pageable BW:   " << result.h2d_bw_gbs << " GB/s"
              << std::endl;
    std::cout << "  Pinned BW:     " << result.d2h_bw_gbs << " GB/s"
              << std::endl;
  }

  return 0;
}
