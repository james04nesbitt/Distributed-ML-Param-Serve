#include "src/server/parameter_server_impl.h"

#include <cstring>
#include <iostream>

namespace paramserver {

ParameterServerImpl::ParameterServerImpl(std::shared_ptr<ParameterStore> store,
                                         int32_t total_params, int num_streams)
    : store_(std::move(store)), num_streams_(num_streams) {
  gpu_store_ = server_cuda::CreateGPUStore(total_params);
  stream_pool_ = server_cuda::CreateStreamPool(num_streams);

  std::cout << "[Server] GPU parameter store initialized: " << total_params
            << " params, " << num_streams << " CUDA streams" << std::endl;
}

ParameterServerImpl::~ParameterServerImpl() {
  if (stream_pool_) {
    server_cuda::DestroyStreamPool(stream_pool_, num_streams_);
  }
  if (gpu_store_) {
    server_cuda::DestroyGPUStore(gpu_store_);
  }
}

server_cuda::StreamHandle ParameterServerImpl::AcquireStream() {
  std::lock_guard<std::mutex> lock(stream_mu_);
  auto s = stream_pool_[next_stream_];
  next_stream_ = (next_stream_ + 1) % num_streams_;
  return s;
}

grpc::Status ParameterServerImpl::PushGradients(grpc::ServerContext *context,
                                                const PushRequest *request,
                                                PushResponse *response) {
  auto stream = AcquireStream();

  for (int u = 0; u < request->updates_size(); ++u) {
    const auto &update = request->updates(u);
    const auto &csr_proto = update.gradients();
    float lr = update.learning_rate();

    int32_t nnz = csr_proto.values_size();
    int32_t num_rows = csr_proto.num_rows();
    int32_t num_cols = csr_proto.num_cols();

    if (nnz == 0)
      continue;

    // Allocate pinned host buffers and copy from protobuf.
    auto *h_values =
        static_cast<float *>(server_cuda::AllocPinned(nnz * sizeof(float)));
    auto *h_col_indices =
        static_cast<int32_t *>(server_cuda::AllocPinned(nnz * sizeof(int32_t)));
    auto *h_row_offsets = static_cast<int32_t *>(
        server_cuda::AllocPinned((num_rows + 1) * sizeof(int32_t)));

    std::memcpy(h_values, csr_proto.values().data(), nnz * sizeof(float));
    std::memcpy(h_col_indices, csr_proto.col_indices().data(),
                nnz * sizeof(int32_t));
    std::memcpy(h_row_offsets, csr_proto.row_offsets().data(),
                (num_rows + 1) * sizeof(int32_t));

    // Async H→D + apply with lock-free atomics (synchronizes internally).
    server_cuda::ApplyCSRGradient(gpu_store_, h_values, h_col_indices,
                                  h_row_offsets, nnz, num_rows, num_cols, lr,
                                  stream);

    server_cuda::FreePinned(h_values);
    server_cuda::FreePinned(h_col_indices);
    server_cuda::FreePinned(h_row_offsets);
  }

  int64_t iter = store_->IncrementIteration();
  response->set_success(true);
  response->set_server_iteration(iter);
  response->set_message("Gradients applied via GPU atomics");

  std::cout << "[Server] PushGradients: applied " << request->updates_size()
            << " updates, iteration=" << iter << std::endl;

  return grpc::Status::OK;
}

grpc::Status ParameterServerImpl::PullParameters(grpc::ServerContext *context,
                                                 const PullRequest *request,
                                                 PullResponse *response) {
  auto stream = AcquireStream();
  int32_t total = server_cuda::GetStoreSize(gpu_store_);

  // Allocate pinned host buffer for async D→H transfer.
  auto *h_params =
      static_cast<float *>(server_cuda::AllocPinned(total * sizeof(float)));

  server_cuda::CopyParamsToHost(gpu_store_, h_params, total, stream);

  // Serialize into protobuf response as a single dense layer.
  auto *param_data = response->add_parameters();
  param_data->set_layer_name("global");
  param_data->mutable_values()->Reserve(total);
  for (int32_t i = 0; i < total; ++i) {
    param_data->add_values(h_params[i]);
  }
  param_data->add_shape(total);

  server_cuda::FreePinned(h_params);

  response->set_server_iteration(store_->GetIteration());

  std::cout << "[Server] PullParameters: worker_id=" << request->worker_id()
            << ", sent " << total << " params" << std::endl;

  return grpc::Status::OK;
}

grpc::Status ParameterServerImpl::RegisterWorker(grpc::ServerContext *context,
                                                 const RegisterRequest *request,
                                                 RegisterResponse *response) {
  response->set_success(true);
  response->set_assigned_id(request->worker_id());
  response->set_message("Worker registered successfully");

  std::cout << "[Server] RegisterWorker: id=" << request->worker_id()
            << ", address=" << request->worker_address() << std::endl;

  return grpc::Status::OK;
}

} // namespace paramserver
