#pragma once

#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "proto/parameter_server.grpc.pb.h"
#include "src/core/parameter_store.h"

namespace paramserver {

// gRPC service implementation for the parameter server.
class ParameterServerImpl final : public paramserver::ParameterServer::Service {
public:
  explicit ParameterServerImpl(std::shared_ptr<ParameterStore> store);

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
  std::shared_ptr<ParameterStore> store_;
};

} // namespace paramserver
