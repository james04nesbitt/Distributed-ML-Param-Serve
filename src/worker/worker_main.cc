#include <iostream>
#include <memory>
#include <string>

#include <grpcpp/grpcpp.h>

#include "src/worker/worker_client.h"

int main(int argc, char **argv) {
  std::string server_address = "localhost:50051";
  int32_t worker_id = 0;

  if (argc > 1) {
    server_address = argv[1];
  }
  if (argc > 2) {
    worker_id = std::atoi(argv[2]);
  }

  auto channel =
      grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  paramserver::WorkerClient client(channel);

  // Register with the server
  if (!client.Register(worker_id, "localhost:0")) {
    std::cerr << "Failed to register with server" << std::endl;
    return 1;
  }

  // TODO(milestone-2): Implement training loop
  //   1. Pull parameters
  //   2. Compute gradients on local data
  //   3. Compress gradients to CSR
  //   4. Push gradients
  //   5. Repeat

  std::cout << "[Worker " << worker_id << "] Connected to " << server_address
            << ". Training loop not yet implemented." << std::endl;

  return 0;
}
