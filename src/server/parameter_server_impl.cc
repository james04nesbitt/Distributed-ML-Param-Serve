#include "src/server/parameter_server_impl.h"

#include <iostream>

namespace paramserver {

ParameterServerImpl::ParameterServerImpl(std::shared_ptr<ParameterStore> store)
    : store_(std::move(store)) {}

grpc::Status ParameterServerImpl::PushGradients(grpc::ServerContext *context,
                                                const PushRequest *request,
                                                PushResponse *response) {
  // TODO(milestone-2): Implement gradient application from CSR format.
  int64_t iter = store_->IncrementIteration();
  response->set_success(true);
  response->set_server_iteration(iter);
  response->set_message("Gradients received (stub)");

  std::cout << "[Server] PushGradients: received " << request->updates_size()
            << " updates, iteration=" << iter << std::endl;

  return grpc::Status::OK;
}

grpc::Status ParameterServerImpl::PullParameters(grpc::ServerContext *context,
                                                 const PullRequest *request,
                                                 PullResponse *response) {
  // TODO(milestone-2): Return actual parameters from store.
  response->set_server_iteration(store_->GetIteration());

  std::cout << "[Server] PullParameters: worker_id=" << request->worker_id()
            << ", requested " << request->layer_names_size() << " layers"
            << std::endl;

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
