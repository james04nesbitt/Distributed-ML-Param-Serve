#pragma once

#include <memory>
#include <string>
#include <vector>

#include <grpcpp/grpcpp.h>

#include "proto/parameter_server.grpc.pb.h"

namespace paramserver {

// Client wrapper for communicating with the parameter server.
class WorkerClient {
public:
  explicit WorkerClient(std::shared_ptr<grpc::Channel> channel);

  // Register this worker with the server.
  bool Register(int32_t worker_id, const std::string &address);

  // Push gradient updates to the server.
  bool PushGradients(const PushRequest &request, int64_t *server_iteration);

  // Pull current parameters from the server.
  bool PullParameters(int32_t worker_id,
                      const std::vector<std::string> &layer_names,
                      PullResponse *response);

private:
  std::unique_ptr<ParameterServer::Stub> stub_;
};

} // namespace paramserver
