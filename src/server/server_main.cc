#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "src/core/parameter_store.h"
#include "src/server/parameter_server_impl.h"

int main(int argc, char **argv) {
  std::string server_address = "0.0.0.0:50051";

  auto store = std::make_shared<paramserver::ParameterStore>();
  paramserver::ParameterServerImpl service(store);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  std::cout << "Parameter server listening on " << server_address << std::endl;
  server->Wait();

  return 0;
}
