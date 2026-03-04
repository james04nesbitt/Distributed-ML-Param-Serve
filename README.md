# Distributed GPU-Accelerated Parameter Server

**C++17 · CUDA · gRPC · Protobuf · Bazel**

A high-performance distributed parameter server for sparse machine learning models, built in C++ with GPU-accelerated gradient synchronization via CUDA. Workers push compressed sparse gradients over gRPC; the server applies them using lock-free CUDA atomic operations on GPU-resident parameters — enabling fully asynchronous, barrier-free distributed SGD.

## Performance Highlights

Benchmarked on **NVIDIA RTX 5070** (Blackwell, PCIe Gen4):

| Metric | Result |
|--------|--------|
| GPU HBM throughput (atomicAdd SGD) | **585 GB/s** |
| Speedup over CPU-bound baseline | **87×** (50M params) — up to **205×** at 1M |
| Aggregate throughput (4 concurrent streams) | **592 GB/s** |
| CSR payload reduction (90% sparse gradients) | **80%** smaller |
| CSR payload reduction (99% sparse gradients) | **98%** smaller |
| Pinned memory PCIe bandwidth | **28.7 GB/s** effective |
| Communication latency cut (40 MB payload) | **5.67 ms** per transfer |
| GPU CSR compression vs CPU | **7–28×** faster |

## Architecture

```
┌───────────────┐   gRPC    ┌─────────────────────────────┐   gRPC    ┌───────────────┐
│   Worker 0    │◄─────────►│                             │◄─────────►│   Worker 2    │
│  (GPU + CSR)  │           │      Parameter Server       │           │  (GPU + CSR)  │
└───────────────┘           │                             │           └───────────────┘
                            │  ┌───────────────────────┐  │
┌───────────────┐   gRPC    │  │  GPU Parameter Store   │  │   gRPC    ┌───────────────┐
│   Worker 1    │◄─────────►│  │  (lock-free atomics,   │  │◄─────────►│   Worker 3    │
│  (GPU + CSR)  │           │  │   8-stream pool)       │  │           │  (GPU + CSR)  │
└───────────────┘           │  └───────────────────────┘  │           └───────────────┘
                            └─────────────────────────────┘
```

**Data flow per iteration:**
1. Worker generates sparse gradient on GPU
2. GPU-native CSR compression (Thrust prefix-scan, no D→H copies)
3. Async transfer to host via pinned (page-locked) memory
4. gRPC `PushGradients` with CSR-serialized protobuf payload
5. Server unpacks into pinned buffers → async H→D → `atomicAdd` kernel
6. Worker pulls updated parameters via `PullParameters`

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design decisions and component deep-dives.

## Project Structure

```
Distributed-ML-Param-Serve/
├── proto/                              # Protobuf/gRPC service definitions
│   └── parameter_server.proto
├── src/
│   ├── core/                           # CPU-side libraries
│   │   ├── parameter_store.{h,cc}      # Weight matrix storage (CPU baseline)
│   │   └── sparse_format.{h,cc}        # CSR compression utilities
│   ├── cuda/                           # CUDA kernels & GPU data structures
│   │   ├── gpu_parameter_store.{cuh,cu}  # GPU weight store + atomics
│   │   └── gpu_sparse_ops.{cuh,cu}       # GPU CSR compress/decompress + pinned memory
│   ├── server/                         # Parameter server
│   │   ├── server_main.cc              # Server entry point
│   │   ├── parameter_server_impl.{h,cc}  # gRPC service implementation
│   │   └── server_cuda_bridge.{h,cu}     # CUDA bridge (NVCC isolation)
│   └── worker/                         # Training workers
│       ├── worker_main.cc              # Worker entry point + training loop
│       ├── worker_client.{h,cc}        # gRPC client wrapper
│       └── worker_cuda_ops.{cuh,cu}    # Worker-side CUDA kernels
├── benchmarks/                         # Performance benchmarks
│   ├── gpu_throughput_bench.cu         # GPU HBM throughput + CPU speedup
│   ├── csr_compression_bench.cu        # CSR payload reduction
│   └── pinned_memory_bench.cu          # Pinned vs pageable latency
├── tests/                              # Unit tests (Google Test)
├── MODULE.bazel                        # Bazel module (bzlmod)
└── ARCHITECTURE.md                     # Detailed design document
```

## Building & Running

### Prerequisites
- [Bazel](https://bazel.build/) 9.x
- C++17 compatible compiler (MSVC 2022, GCC 11+, or Clang 14+)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 12.x+ with an NVIDIA GPU

### Build Everything
```bash
bazel build //src/server:parameter_server //src/worker:worker //benchmarks/...
```

### Run the System

**Terminal 1 — Parameter Server:**
```bash
# Args: [address] [total_params] [num_cuda_streams]
bazel-bin/src/server/parameter_server.exe 0.0.0.0:50051 1000000 8
```

**Terminal 2 — Worker:**
```bash
# Args: [server_address] [worker_id] [iterations] [grad_rows] [grad_cols]
bazel-bin/src/worker/worker.exe localhost:50051 0 100 1000 1000
```

### Run Benchmarks
```bash
# GPU throughput & CPU vs GPU speedup
bazel-bin/benchmarks/gpu_throughput_bench.exe

# CSR compression ratio at various sparsity levels
bazel-bin/benchmarks/csr_compression_bench.exe

# Pinned vs pageable memory transfer latency
bazel-bin/benchmarks/pinned_memory_bench.exe
```

### Run Tests
```bash
bazel test //tests/...
```

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | C++17 | Core implementation |
| GPU Compute | CUDA + Thrust | Kernels, atomics, prefix-scan |
| RPC | gRPC + Protobuf | Worker ↔ server communication |
| Build | Bazel 9 (bzlmod) | Hermetic, reproducible builds |
| Testing | Google Test | Unit + GPU test framework |

## License

MIT
