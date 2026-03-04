// =============================================================================
// Benchmark 1: GPU Atomics Throughput & CPU vs GPU Speedup
//
// Measures:
//   - GPU HBM throughput (GB/s) for atomicAdd-based parameter updates
//   - CPU baseline throughput for the same operation
//   - Speedup ratio (GPU / CPU)
//
// This directly populates the resume line:
//   "achieving [X] GB/s memory throughput and a [X]x speedup over CPU-bound
//    baselines"
// =============================================================================

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cuda/gpu_parameter_store.cuh"

namespace {

void CheckCuda(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(err));
  }
}

// CPU baseline: serial SGD update  params[i] -= lr * grad[i]
void CpuApplyGradient(float *params, const float *grad, float lr,
                      int32_t count) {
  for (int32_t i = 0; i < count; ++i) {
    params[i] -= lr * grad[i];
  }
}

// Initialize a vector with pseudo-random gradient values.
void FillGradient(std::vector<float> &grad) {
  for (size_t i = 0; i < grad.size(); ++i) {
    grad[i] = static_cast<float>((i * 7 + 13) % 1000) / 10000.0f - 0.05f;
  }
}

struct BenchResult {
  double throughput_gbs;
  double elapsed_ms;
  int32_t num_params;
  int iterations;
};

// Benchmark GPU atomic updates.
BenchResult BenchGPU(int32_t num_params, int iterations) {
  paramserver::cuda::GPUParameterStore store(num_params);

  // Create gradient on device.
  std::vector<float> h_grad(num_params);
  FillGradient(h_grad);

  float *d_grad = nullptr;
  CheckCuda(cudaMalloc(&d_grad, num_params * sizeof(float)), "alloc grad");
  CheckCuda(cudaMemcpy(d_grad, h_grad.data(), num_params * sizeof(float),
                       cudaMemcpyHostToDevice),
            "copy grad");

  cudaStream_t stream;
  CheckCuda(cudaStreamCreate(&stream), "create stream");

  // Warmup
  for (int i = 0; i < 10; ++i) {
    store.ApplyGradientAsync(d_grad, 0.001f, num_params, stream);
  }
  CheckCuda(cudaStreamSynchronize(stream), "warmup sync");

  // Timed run
  cudaEvent_t start, stop;
  CheckCuda(cudaEventCreate(&start), "create start event");
  CheckCuda(cudaEventCreate(&stop), "create stop event");

  CheckCuda(cudaEventRecord(start, stream), "record start");
  for (int i = 0; i < iterations; ++i) {
    store.ApplyGradientAsync(d_grad, 0.001f, num_params, stream);
  }
  CheckCuda(cudaEventRecord(stop, stream), "record stop");
  CheckCuda(cudaEventSynchronize(stop), "event sync");

  float elapsed_ms = 0.0f;
  CheckCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed time");

  // Each iteration reads params + grad, writes params → 3 * num_params * 4
  // bytes
  double total_bytes =
      3.0 * num_params * sizeof(float) * static_cast<double>(iterations);
  double throughput_gbs = (total_bytes / 1e9) / (elapsed_ms / 1e3);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);
  cudaFree(d_grad);

  return {throughput_gbs, elapsed_ms, num_params, iterations};
}

// Benchmark CPU baseline.
BenchResult BenchCPU(int32_t num_params, int iterations) {
  std::vector<float> params(num_params, 0.0f);
  std::vector<float> grad(num_params);
  FillGradient(grad);

  // Warmup
  for (int i = 0; i < 3; ++i) {
    CpuApplyGradient(params.data(), grad.data(), 0.001f, num_params);
  }

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < iterations; ++i) {
    CpuApplyGradient(params.data(), grad.data(), 0.001f, num_params);
  }
  auto end = std::chrono::high_resolution_clock::now();

  double elapsed_ms =
      std::chrono::duration<double, std::milli>(end - start).count();
  double total_bytes =
      3.0 * num_params * sizeof(float) * static_cast<double>(iterations);
  double throughput_gbs = (total_bytes / 1e9) / (elapsed_ms / 1e3);

  return {throughput_gbs, elapsed_ms, num_params, iterations};
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
  CheckCuda(cudaGetDeviceProperties(&prop, 0), "get device props");
  std::cout << "GPU: " << prop.name << std::endl;
  std::cout << "Memory Bus Width: " << prop.memoryBusWidth << " bits\n"
            << std::endl;

  // Test multiple sizes to show scaling.
  struct Config {
    int32_t num_params;
    int iterations;
    const char *label;
  };

  Config configs[] = {
      {100'000, 1000, "100K params"},
      {1'000'000, 500, "1M params"},
      {10'000'000, 200, "10M params"},
      {50'000'000, 100, "50M params"},
  };

  std::cout << std::fixed << std::setprecision(2);
  std::cout << "=== Dense SGD Update Benchmark: GPU atomicAdd vs CPU ==="
            << std::endl;
  std::cout << std::setw(15) << "Size" << std::setw(15) << "GPU (GB/s)"
            << std::setw(15) << "GPU (ms)" << std::setw(15) << "CPU (GB/s)"
            << std::setw(15) << "CPU (ms)" << std::setw(12) << "Speedup"
            << std::endl;
  std::cout << std::string(87, '-') << std::endl;

  for (const auto &cfg : configs) {
    auto gpu = BenchGPU(cfg.num_params, cfg.iterations);
    auto cpu = BenchCPU(cfg.num_params, cfg.iterations);
    double speedup = cpu.elapsed_ms / gpu.elapsed_ms;

    std::cout << std::setw(15) << cfg.label << std::setw(15)
              << gpu.throughput_gbs << std::setw(15) << gpu.elapsed_ms
              << std::setw(15) << cpu.throughput_gbs << std::setw(15)
              << cpu.elapsed_ms << std::setw(11) << speedup << "x" << std::endl;
  }

  std::cout << "\n=== Concurrent Multi-Stream Benchmark ===" << std::endl;
  std::cout << "Simulating " << 4 << " workers pushing gradients concurrently\n"
            << std::endl;

  {
    int32_t num_params = 10'000'000;
    int iterations = 200;
    int num_streams = 4;

    paramserver::cuda::GPUParameterStore store(num_params);

    std::vector<float> h_grad(num_params);
    FillGradient(h_grad);

    // Allocate per-stream device gradients.
    std::vector<float *> d_grads(num_streams);
    std::vector<cudaStream_t> streams(num_streams);
    for (int s = 0; s < num_streams; ++s) {
      CheckCuda(cudaMalloc(&d_grads[s], num_params * sizeof(float)),
                "alloc grad");
      CheckCuda(cudaMemcpy(d_grads[s], h_grad.data(),
                           num_params * sizeof(float), cudaMemcpyHostToDevice),
                "copy grad");
      CheckCuda(cudaStreamCreate(&streams[s]), "create stream");
    }

    // Warmup
    for (int s = 0; s < num_streams; ++s) {
      store.ApplyGradientAsync(d_grads[s], 0.001f, num_params, streams[s]);
    }
    for (int s = 0; s < num_streams; ++s) {
      cudaStreamSynchronize(streams[s]);
    }

    cudaEvent_t start, stop;
    CheckCuda(cudaEventCreate(&start), "create start");
    CheckCuda(cudaEventCreate(&stop), "create stop");

    CheckCuda(cudaEventRecord(start, nullptr), "record start");
    for (int i = 0; i < iterations; ++i) {
      for (int s = 0; s < num_streams; ++s) {
        store.ApplyGradientAsync(d_grads[s], 0.001f, num_params, streams[s]);
      }
    }
    for (int s = 0; s < num_streams; ++s) {
      cudaStreamSynchronize(streams[s]);
    }
    CheckCuda(cudaEventRecord(stop, nullptr), "record stop");
    CheckCuda(cudaEventSynchronize(stop), "sync");

    float elapsed_ms = 0.0f;
    CheckCuda(cudaEventElapsedTime(&elapsed_ms, start, stop), "elapsed");

    int total_updates = iterations * num_streams;
    double total_bytes =
        3.0 * num_params * sizeof(float) * static_cast<double>(total_updates);
    double throughput_gbs = (total_bytes / 1e9) / (elapsed_ms / 1e3);
    double updates_per_sec = total_updates / (elapsed_ms / 1e3);

    std::cout << "  Total updates:     " << total_updates << std::endl;
    std::cout << "  Elapsed time:      " << elapsed_ms << " ms" << std::endl;
    std::cout << "  Aggregate throughput: " << throughput_gbs << " GB/s"
              << std::endl;
    std::cout << "  Updates/sec:       " << updates_per_sec << std::endl;

    for (int s = 0; s < num_streams; ++s) {
      cudaFree(d_grads[s]);
      cudaStreamDestroy(streams[s]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  return 0;
}
