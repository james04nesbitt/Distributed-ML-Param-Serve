#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <cuda_runtime.h>
#include <grpcpp/grpcpp.h>

#include "proto/parameter_server.grpc.pb.h"
#include "src/cuda/gpu_sparse_ops.cuh"
#include "src/worker/worker_client.h"
#include "src/worker/worker_cuda_ops.cuh"

namespace {

void CheckCuda(cudaError_t err, const char *context) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(context) + ": " +
                             cudaGetErrorString(err));
  }
}

} // namespace

int main(int argc, char **argv) {
  std::string server_address = "localhost:50051";
  int32_t worker_id = 0;
  int32_t num_iterations = 100;
  int32_t num_rows = 1000; // Gradient matrix dimensions
  int32_t num_cols = 1000;
  float sparsity = 0.9f; // 90% zeros

  if (argc > 1)
    server_address = argv[1];
  if (argc > 2)
    worker_id = std::atoi(argv[2]);
  if (argc > 3)
    num_iterations = std::atoi(argv[3]);
  if (argc > 4)
    num_rows = std::atoi(argv[4]);
  if (argc > 5)
    num_cols = std::atoi(argv[5]);

  // Verify CUDA device is available.
  int device_count = 0;
  CheckCuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
  if (device_count == 0) {
    std::cerr << "No CUDA device found. Exiting." << std::endl;
    return 1;
  }

  // Connect to the parameter server.
  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  paramserver::WorkerClient client(channel);

  if (!client.Register(worker_id, "localhost:0")) {
    std::cerr << "Failed to register with server" << std::endl;
    return 1;
  }

  int32_t total = num_rows * num_cols;

  // Allocate dense gradient buffer on GPU.
  float *d_gradient = paramserver::worker::AllocateGradientDevice(total);

  // Create a CUDA stream for async operations.
  cudaStream_t stream;
  CheckCuda(cudaStreamCreate(&stream), "create worker stream");

  std::cout << "[Worker " << worker_id
            << "] Starting training loop: " << num_iterations << " iterations, "
            << num_rows << "x" << num_cols << " gradient matrix, "
            << (sparsity * 100) << "% sparsity" << std::endl;

  double total_push_ms = 0.0;
  double total_pull_ms = 0.0;

  for (int32_t iter = 0; iter < num_iterations; ++iter) {
    auto iter_start = std::chrono::high_resolution_clock::now();

    // --- Step 1: Generate sparse random gradient on GPU ---
    paramserver::worker::GenerateSparseGradient(
        d_gradient, total, sparsity, static_cast<unsigned int>(iter + 1));

    // --- Step 2: Compress to CSR on GPU (no D→H copy) ---
    paramserver::cuda::DeviceCSR dcsr =
        paramserver::cuda::CompressToCSRDevice(d_gradient, num_rows, num_cols);

    // --- Step 3: Transfer CSR to host via pinned memory ---
    paramserver::cuda::HostCSRPinned host_csr =
        paramserver::cuda::DeviceCSRToHostPinned(dcsr, stream);
    CheckCuda(cudaStreamSynchronize(stream), "sync after D2H");

    // --- Step 4: Package into PushRequest protobuf ---
    paramserver::PushRequest push_request;
    auto *update = push_request.add_updates();
    update->set_layer_name("global");
    update->set_learning_rate(0.001f);
    update->set_worker_id(worker_id);
    update->set_iteration(iter);

    auto *csr_msg = update->mutable_gradients();
    csr_msg->set_num_rows(num_rows);
    csr_msg->set_num_cols(num_cols);

    // Copy from pinned buffers into protobuf.
    csr_msg->mutable_values()->Reserve(host_csr.nnz);
    for (int32_t i = 0; i < host_csr.nnz; ++i) {
      csr_msg->add_values(host_csr.values.data()[i]);
    }
    csr_msg->mutable_col_indices()->Reserve(host_csr.nnz);
    for (int32_t i = 0; i < host_csr.nnz; ++i) {
      csr_msg->add_col_indices(host_csr.col_indices.data()[i]);
    }
    csr_msg->mutable_row_offsets()->Reserve(num_rows + 1);
    for (int32_t i = 0; i < num_rows + 1; ++i) {
      csr_msg->add_row_offsets(host_csr.row_offsets.data()[i]);
    }

    // --- Step 5: Push gradients to server ---
    auto push_start = std::chrono::high_resolution_clock::now();
    int64_t server_iter = 0;
    bool push_ok = client.PushGradients(push_request, &server_iter);
    auto push_end = std::chrono::high_resolution_clock::now();

    double push_ms =
        std::chrono::duration<double, std::milli>(push_end - push_start)
            .count();
    total_push_ms += push_ms;

    // --- Step 6: Pull parameters from server ---
    auto pull_start = std::chrono::high_resolution_clock::now();
    paramserver::PullResponse pull_response;
    bool pull_ok = client.PullParameters(worker_id, {"global"}, &pull_response);
    auto pull_end = std::chrono::high_resolution_clock::now();

    double pull_ms =
        std::chrono::duration<double, std::milli>(pull_end - pull_start)
            .count();
    total_pull_ms += pull_ms;

    auto iter_end = std::chrono::high_resolution_clock::now();
    double iter_ms =
        std::chrono::duration<double, std::milli>(iter_end - iter_start)
            .count();

    // Free device CSR for this iteration.
    dcsr.Free();

    if ((iter + 1) % 10 == 0 || iter == 0) {
      std::cout << "[Worker " << worker_id << "] iter=" << (iter + 1) << "/"
                << num_iterations << " | nnz=" << host_csr.nnz
                << " | push=" << push_ms << "ms"
                << " | pull=" << pull_ms << "ms"
                << " | total=" << iter_ms << "ms"
                << " | server_iter=" << server_iter
                << (push_ok && pull_ok ? "" : " [FAILED]") << std::endl;
    }
  }

  // --- Summary ---
  double avg_push = total_push_ms / num_iterations;
  double avg_pull = total_pull_ms / num_iterations;
  double total_data_mb = static_cast<double>(total * sizeof(float)) *
                         num_iterations / (1024.0 * 1024.0);
  double total_time_s = (total_push_ms + total_pull_ms) / 1000.0;

  std::cout << "\n[Worker " << worker_id << "] === Training Complete ==="
            << "\n  Iterations:        " << num_iterations
            << "\n  Avg push latency:  " << avg_push << " ms"
            << "\n  Avg pull latency:  " << avg_pull << " ms"
            << "\n  Avg round-trip:    " << (avg_push + avg_pull) << " ms"
            << "\n  Data throughput:   " << (total_data_mb / total_time_s)
            << " MB/s" << std::endl;

  // Cleanup.
  cudaFree(d_gradient);
  cudaStreamDestroy(stream);

  return 0;
}
