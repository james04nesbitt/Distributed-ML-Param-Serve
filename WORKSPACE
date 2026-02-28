workspace(name = "distributed_ml_param_serve")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# ============================================================================
# gRPC (includes Protobuf as a transitive dependency)
# ============================================================================
http_archive(
    name = "com_github_grpc_grpc",
    sha256 = "",  # TODO: fill in after first successful fetch
    strip_prefix = "grpc-1.70.1",
    urls = ["https://github.com/grpc/grpc/archive/refs/tags/v1.70.1.tar.gz"],
)

load("@com_github_grpc_grpc//bazel:grpc_deps.bzl", "grpc_deps")

grpc_deps()

load("@com_github_grpc_grpc//bazel:grpc_extra_deps.bzl", "grpc_extra_deps")

grpc_extra_deps()

# ============================================================================
# Google Test
# ============================================================================
http_archive(
    name = "com_google_googletest",
    strip_prefix = "googletest-1.15.2",
    urls = ["https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz"],
)
