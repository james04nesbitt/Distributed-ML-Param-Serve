#pragma once

#include <memory>
#include <mutex>
#include <string>

#include <grpcpp/grpcpp.h>

#include "proto/parameter_server.grpc.pb.h"
#include "src/core/parameter_store.h"
#include "src/server/server_cuda_bridge.h"

namespace paramserver {

// gRPC service implementation for the parameter server.
// Routes gradient updates through a GPU-resident parameter store,
// using CUDA atomics for lock-free Async SGD.
class ParameterServerImpl final : public paramserver::ParameterServer::Service {
public:
  // total_params: size of the flat GPU parameter array.
  // num_streams: number of CUDA streams in the pool for concurrent requests.
  ParameterServerImpl(std::shared_ptr<ParameterStore> store,
                      int32_t total_params, int num_streams = 8);
  ~ParameterServerImpl();

  grpc::Status PushGradients(grpc::ServerContext *context,
                             const PushRequest *request,
                             PushResponse *response) override;

  grpc::Status PullParameters(grpc::ServerContext *context,
                              const PullRequest *request,
                              PullResponse *response) override;

  grpc::Status RegisterWorker(grpc::ServerContext *context,
                              const RegisterRequest *request,
                              RegisterResponse *response) override;

private:
  // Acquire a CUDA stream from the round-robin pool.
  server_cuda::StreamHandle AcquireStream();

  std::shared_ptr<ParameterStore> store_;
  server_cuda::GPUStoreHandle *gpu_store_ = nullptr;

  // Stream pool for concurrent per-request execution.
  server_cuda::StreamHandle *stream_pool_ = nullptr;
  int num_streams_ = 0;
  std::mutex stream_mu_;
  int next_stream_ = 0;
};

} // namespace paramserver
