#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <vector>

#include "src/cuda/gpu_sparse_ops.cuh"

namespace paramserver {
namespace cuda {
namespace {

class GPUSparseOpsTest : public ::testing::Test {
protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available, skipping GPU tests";
    }
  }
};

TEST_F(GPUSparseOpsTest, HostToDeviceCSRRoundTrip) {
  // Create a CSR on host, transfer to device, transfer back, verify.
  std::vector<float> values = {1.0f, 2.0f, 3.0f};
  std::vector<int32_t> col_indices = {0, 2, 1};
  std::vector<int32_t> row_offsets = {0, 1, 1, 3}; // 3 rows

  DeviceCSR dcsr = HostToDeviceCSR(values, col_indices, row_offsets, 3, 3);

  EXPECT_EQ(dcsr.nnz, 3);
  EXPECT_EQ(dcsr.num_rows, 3);
  EXPECT_EQ(dcsr.num_cols, 3);

  // Transfer back
  std::vector<float> out_values;
  std::vector<int32_t> out_col_indices;
  std::vector<int32_t> out_row_offsets;
  DeviceCSRToHost(dcsr, out_values, out_col_indices, out_row_offsets);

  ASSERT_EQ(out_values.size(), 3);
  EXPECT_FLOAT_EQ(out_values[0], 1.0f);
  EXPECT_FLOAT_EQ(out_values[1], 2.0f);
  EXPECT_FLOAT_EQ(out_values[2], 3.0f);

  ASSERT_EQ(out_col_indices.size(), 3);
  EXPECT_EQ(out_col_indices[0], 0);
  EXPECT_EQ(out_col_indices[1], 2);
  EXPECT_EQ(out_col_indices[2], 1);

  ASSERT_EQ(out_row_offsets.size(), 4);
  EXPECT_EQ(out_row_offsets[0], 0);
  EXPECT_EQ(out_row_offsets[1], 1);
  EXPECT_EQ(out_row_offsets[2], 1);
  EXPECT_EQ(out_row_offsets[3], 3);

  dcsr.Free();
}

TEST_F(GPUSparseOpsTest, DecompressCSROnGPU) {
  // CSR for 3x4 matrix:
  // [[1, 0, 0, 2],
  //  [0, 0, 0, 0],
  //  [3, 4, 0, 5]]
  std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<int32_t> col_indices = {0, 3, 0, 1, 3};
  std::vector<int32_t> row_offsets = {0, 2, 2, 5};

  DeviceCSR dcsr = HostToDeviceCSR(values, col_indices, row_offsets, 3, 4);

  // Allocate dense output on device
  float *d_dense = nullptr;
  cudaMalloc(&d_dense, 3 * 4 * sizeof(float));

  DecompressFromCSRDevice(dcsr, d_dense);
  cudaDeviceSynchronize();

  // Copy back and verify
  std::vector<float> host_dense(12);
  cudaMemcpy(host_dense.data(), d_dense, 12 * sizeof(float),
             cudaMemcpyDeviceToHost);

  // Row 0
  EXPECT_FLOAT_EQ(host_dense[0], 1.0f);
  EXPECT_FLOAT_EQ(host_dense[1], 0.0f);
  EXPECT_FLOAT_EQ(host_dense[2], 0.0f);
  EXPECT_FLOAT_EQ(host_dense[3], 2.0f);
  // Row 1 (all zeros)
  EXPECT_FLOAT_EQ(host_dense[4], 0.0f);
  EXPECT_FLOAT_EQ(host_dense[5], 0.0f);
  EXPECT_FLOAT_EQ(host_dense[6], 0.0f);
  EXPECT_FLOAT_EQ(host_dense[7], 0.0f);
  // Row 2
  EXPECT_FLOAT_EQ(host_dense[8], 3.0f);
  EXPECT_FLOAT_EQ(host_dense[9], 4.0f);
  EXPECT_FLOAT_EQ(host_dense[10], 0.0f);
  EXPECT_FLOAT_EQ(host_dense[11], 5.0f);

  cudaFree(d_dense);
  dcsr.Free();
}

TEST_F(GPUSparseOpsTest, CompressToCSRDeviceRoundTrip) {
  // Upload a dense matrix to GPU, compress to CSR, then decompress and verify.
  std::vector<float> dense = {
      1.0f, 0.0f, 0.0f, // row 0
      0.0f, 2.0f, 0.0f, // row 1
      0.0f, 0.0f, 3.0f, // row 2 (diagonal matrix)
  };

  float *d_dense = nullptr;
  cudaMalloc(&d_dense, 9 * sizeof(float));
  cudaMemcpy(d_dense, dense.data(), 9 * sizeof(float), cudaMemcpyHostToDevice);

  DeviceCSR dcsr = CompressToCSRDevice(d_dense, 3, 3);

  EXPECT_EQ(dcsr.nnz, 3);
  EXPECT_EQ(dcsr.num_rows, 3);
  EXPECT_EQ(dcsr.num_cols, 3);

  // Decompress back
  cudaMemset(d_dense, 0, 9 * sizeof(float));
  DecompressFromCSRDevice(dcsr, d_dense);
  cudaDeviceSynchronize();

  std::vector<float> recovered(9);
  cudaMemcpy(recovered.data(), d_dense, 9 * sizeof(float),
             cudaMemcpyDeviceToHost);

  for (size_t i = 0; i < dense.size(); ++i) {
    EXPECT_FLOAT_EQ(recovered[i], dense[i]) << "mismatch at index " << i;
  }

  cudaFree(d_dense);
  dcsr.Free();
}

TEST_F(GPUSparseOpsTest, EmptyCSR) {
  // All-zero matrix should produce empty CSR
  std::vector<float> zeros(12, 0.0f);

  float *d_dense = nullptr;
  cudaMalloc(&d_dense, 12 * sizeof(float));
  cudaMemcpy(d_dense, zeros.data(), 12 * sizeof(float), cudaMemcpyHostToDevice);

  DeviceCSR dcsr = CompressToCSRDevice(d_dense, 3, 4);
  EXPECT_EQ(dcsr.nnz, 0);

  cudaFree(d_dense);
  dcsr.Free();
}

TEST_F(GPUSparseOpsTest, DeviceCSRFreeIsIdempotent) {
  DeviceCSR dcsr;
  dcsr.Free(); // Should not crash
  dcsr.Free(); // Double-free should be safe
}

} // namespace
} // namespace cuda
} // namespace paramserver
