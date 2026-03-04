#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "src/core/parameter_store.h"
#include "src/server/parameter_server_impl.h"

int main(int argc, char **argv) {
  std::string server_address = "0.0.0.0:50051";
  int32_t total_params = 1000000; // 1M parameters (configurable)
  int num_streams = 8;

  if (argc > 1) {
    server_address = argv[1];
  }
  if (argc > 2) {
    total_params = std::atoi(argv[2]);
  }
  if (argc > 3) {
    num_streams = std::atoi(argv[3]);
  }

  auto store = std::make_shared<paramserver::ParameterStore>();
  paramserver::ParameterServerImpl service(store, total_params, num_streams);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Parameter server listening on " << server_address << " ("
            << total_params << " params, " << num_streams << " CUDA streams)"
            << std::endl;
  server->Wait();

  return 0;
}
