#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <cmath>
#include <numeric>
#include <vector>

#include "src/cuda/gpu_parameter_store.cuh"

namespace paramserver {
namespace cuda {
namespace {

class GPUParameterStoreTest : public ::testing::Test {
protected:
  void SetUp() override {
    // Skip all tests if no CUDA device is available.
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available, skipping GPU tests";
    }
  }
};

TEST_F(GPUParameterStoreTest, InitializesWithZeros) {
  constexpr int kSize = 1024;
  GPUParameterStore store(kSize);

  std::vector<float> host(kSize);
  store.CopyToHost(host.data(), kSize);

  for (int i = 0; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(host[i], 0.0f) << "at index " << i;
  }
}

TEST_F(GPUParameterStoreTest, CopyFromHostRoundTrip) {
  constexpr int kSize = 256;
  std::vector<float> input(kSize);
  for (int i = 0; i < kSize; ++i) {
    input[i] = static_cast<float>(i) * 0.1f;
  }

  GPUParameterStore store(kSize);
  store.CopyFromHost(input.data(), kSize);

  std::vector<float> output(kSize);
  store.CopyToHost(output.data(), kSize);

  for (int i = 0; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(output[i], input[i]) << "at index " << i;
  }
}

TEST_F(GPUParameterStoreTest, ApplyDenseGradient) {
  // params = [1, 2, 3, 4], gradient = [0.1, 0.2, 0.3, 0.4], lr = 1.0
  // result = [1 - 0.1, 2 - 0.2, 3 - 0.3, 4 - 0.4] = [0.9, 1.8, 2.7, 3.6]
  constexpr int kSize = 4;
  std::vector<float> params = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> gradient = {0.1f, 0.2f, 0.3f, 0.4f};

  GPUParameterStore store(kSize);
  store.CopyFromHost(params.data(), kSize);

  // Upload gradient to device
  float *d_gradient = nullptr;
  cudaMalloc(&d_gradient, kSize * sizeof(float));
  cudaMemcpy(d_gradient, gradient.data(), kSize * sizeof(float),
             cudaMemcpyHostToDevice);

  store.ApplyGradientAsync(d_gradient, 1.0f, kSize);
  cudaDeviceSynchronize();

  std::vector<float> result(kSize);
  store.CopyToHost(result.data(), kSize);

  EXPECT_NEAR(result[0], 0.9f, 1e-5f);
  EXPECT_NEAR(result[1], 1.8f, 1e-5f);
  EXPECT_NEAR(result[2], 2.7f, 1e-5f);
  EXPECT_NEAR(result[3], 3.6f, 1e-5f);

  cudaFree(d_gradient);
}

TEST_F(GPUParameterStoreTest, ApplyDenseGradientWithLearningRate) {
  // params = [0, 0, 0, 0], gradient = [1, 2, 3, 4], lr = 0.5
  // result = [0 - 0.5, 0 - 1.0, 0 - 1.5, 0 - 2.0]
  constexpr int kSize = 4;
  std::vector<float> gradient = {1.0f, 2.0f, 3.0f, 4.0f};

  GPUParameterStore store(kSize);

  float *d_gradient = nullptr;
  cudaMalloc(&d_gradient, kSize * sizeof(float));
  cudaMemcpy(d_gradient, gradient.data(), kSize * sizeof(float),
             cudaMemcpyHostToDevice);

  store.ApplyGradientAsync(d_gradient, 0.5f, kSize);
  cudaDeviceSynchronize();

  std::vector<float> result(kSize);
  store.CopyToHost(result.data(), kSize);

  EXPECT_NEAR(result[0], -0.5f, 1e-5f);
  EXPECT_NEAR(result[1], -1.0f, 1e-5f);
  EXPECT_NEAR(result[2], -1.5f, 1e-5f);
  EXPECT_NEAR(result[3], -2.0f, 1e-5f);

  cudaFree(d_gradient);
}

TEST_F(GPUParameterStoreTest, ConcurrentStreamsApplyGradients) {
  // Simulate two workers on separate CUDA streams applying gradients
  // concurrently to the same parameters. Both push gradient = [1,1,1,1]
  // with lr=1.0, so result should be [0-1-1, 0-1-1, ...] = [-2, -2, -2, -2]
  constexpr int kSize = 4;
  std::vector<float> gradient(kSize, 1.0f);

  GPUParameterStore store(kSize);

  // Create two streams
  cudaStream_t stream0, stream1;
  cudaStreamCreate(&stream0);
  cudaStreamCreate(&stream1);

  // Upload gradient to device
  float *d_gradient = nullptr;
  cudaMalloc(&d_gradient, kSize * sizeof(float));
  cudaMemcpy(d_gradient, gradient.data(), kSize * sizeof(float),
             cudaMemcpyHostToDevice);

  // Both streams apply the same gradient concurrently
  store.ApplyGradientAsync(d_gradient, 1.0f, kSize, stream0);
  store.ApplyGradientAsync(d_gradient, 1.0f, kSize, stream1);

  cudaStreamSynchronize(stream0);
  cudaStreamSynchronize(stream1);

  std::vector<float> result(kSize);
  store.CopyToHost(result.data(), kSize);

  for (int i = 0; i < kSize; ++i) {
    EXPECT_NEAR(result[i], -2.0f, 1e-5f) << "at index " << i;
  }

  cudaFree(d_gradient);
  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);
}

TEST_F(GPUParameterStoreTest, ApplyCSRGradient) {
  // 2x3 parameter matrix, applying a sparse gradient:
  // gradient = [[0, 0.5, 0], [0.3, 0, 0]]  (CSR: 2 non-zeros)
  // params start at zero, lr = 1.0
  // result = [[0, -0.5, 0], [-0.3, 0, 0]]
  constexpr int kRows = 2;
  constexpr int kCols = 3;
  constexpr int kSize = kRows * kCols;

  GPUParameterStore store(kSize);

  // CSR representation of the sparse gradient
  std::vector<float> values = {0.5f, 0.3f};
  std::vector<int32_t> col_indices = {1, 0};
  std::vector<int32_t> row_offsets = {0, 1, 2};

  // Upload CSR to device
  float *d_values = nullptr;
  int32_t *d_col_indices = nullptr;
  int32_t *d_row_offsets = nullptr;

  cudaMalloc(&d_values, values.size() * sizeof(float));
  cudaMalloc(&d_col_indices, col_indices.size() * sizeof(int32_t));
  cudaMalloc(&d_row_offsets, row_offsets.size() * sizeof(int32_t));

  cudaMemcpy(d_values, values.data(), values.size() * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_col_indices, col_indices.data(),
             col_indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_offsets, row_offsets.data(),
             row_offsets.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

  store.ApplyCSRGradientAsync(d_values, d_col_indices, d_row_offsets, kRows,
                              kCols, 1.0f);
  cudaDeviceSynchronize();

  std::vector<float> result(kSize);
  store.CopyToHost(result.data(), kSize);

  // Row 0: [0, -0.5, 0]
  EXPECT_NEAR(result[0], 0.0f, 1e-5f);
  EXPECT_NEAR(result[1], -0.5f, 1e-5f);
  EXPECT_NEAR(result[2], 0.0f, 1e-5f);
  // Row 1: [-0.3, 0, 0]
  EXPECT_NEAR(result[3], -0.3f, 1e-5f);
  EXPECT_NEAR(result[4], 0.0f, 1e-5f);
  EXPECT_NEAR(result[5], 0.0f, 1e-5f);

  cudaFree(d_values);
  cudaFree(d_col_indices);
  cudaFree(d_row_offsets);
}

TEST_F(GPUParameterStoreTest, MoveSemantics) {
  constexpr int kSize = 16;
  std::vector<float> data(kSize, 42.0f);

  GPUParameterStore store1(kSize);
  store1.CopyFromHost(data.data(), kSize);

  // Move construct
  GPUParameterStore store2(std::move(store1));
  EXPECT_EQ(store2.size(), kSize);

  std::vector<float> result(kSize);
  store2.CopyToHost(result.data(), kSize);
  for (int i = 0; i < kSize; ++i) {
    EXPECT_FLOAT_EQ(result[i], 42.0f);
  }
}

} // namespace
} // namespace cuda
} // namespace paramserver
