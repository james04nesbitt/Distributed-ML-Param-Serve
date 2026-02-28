#include "src/worker/worker_client.h"

#include <iostream>

namespace paramserver {

WorkerClient::WorkerClient(std::shared_ptr<grpc::Channel> channel)
    : stub_(ParameterServer::NewStub(channel)) {}

bool WorkerClient::Register(int32_t worker_id, const std::string &address) {
  RegisterRequest request;
  request.set_worker_id(worker_id);
  request.set_worker_address(address);

  RegisterResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->RegisterWorker(&context, request, &response);
  if (status.ok() && response.success()) {
    std::cout << "[Worker " << worker_id
              << "] Registered: " << response.message() << std::endl;
    return true;
  }

  std::cerr << "[Worker " << worker_id
            << "] Registration failed: " << status.error_message() << std::endl;
  return false;
}

bool WorkerClient::PushGradients(const PushRequest &request,
                                 int64_t *server_iteration) {
  PushResponse response;
  grpc::ClientContext context;

  grpc::Status status = stub_->PushGradients(&context, request, &response);
  if (status.ok() && response.success()) {
    if (server_iteration) {
      *server_iteration = response.server_iteration();
    }
    return true;
  }

  std::cerr << "[Worker] PushGradients failed: " << status.error_message()
            << std::endl;
  return false;
}

bool WorkerClient::PullParameters(int32_t worker_id,
                                  const std::vector<std::string> &layer_names,
                                  PullResponse *response) {
  PullRequest request;
  request.set_worker_id(worker_id);
  for (const auto &name : layer_names) {
    request.add_layer_names(name);
  }

  grpc::ClientContext context;

  grpc::Status status = stub_->PullParameters(&context, request, response);
  if (!status.ok()) {
    std::cerr << "[Worker " << worker_id
              << "] PullParameters failed: " << status.error_message()
              << std::endl;
    return false;
  }

  return true;
}

} // namespace paramserver
