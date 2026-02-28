#include "src/core/sparse_format.h"

#include <cmath>

#include <gtest/gtest.h>

namespace paramserver {
namespace {

TEST(SparseFormatTest, CompressDecompressIdentity) {
  // 3x4 matrix with some zeros
  std::vector<float> dense = {
      1.0f, 0.0f, 0.0f, 2.0f, // row 0: sparse
      0.0f, 0.0f, 0.0f, 0.0f, // row 1: all zero
      3.0f, 4.0f, 0.0f, 5.0f, // row 2: partially sparse
  };

  auto csr = CompressToCSR(dense, 3, 4);

  EXPECT_EQ(csr.num_rows, 3);
  EXPECT_EQ(csr.num_cols, 4);
  EXPECT_EQ(csr.nnz(), 5); // 5 non-zero values: 1,2,3,4,5

  // Decompress and verify round-trip
  auto recovered = DecompressFromCSR(csr);
  ASSERT_EQ(recovered.size(), dense.size());
  for (size_t i = 0; i < dense.size(); ++i) {
    EXPECT_FLOAT_EQ(recovered[i], dense[i]);
  }
}

TEST(SparseFormatTest, CompressionRatio) {
  // Mostly-zero matrix should have good compression
  std::vector<float> sparse(100, 0.0f);
  sparse[0] = 1.0f;
  sparse[50] = 2.0f;

  auto csr = CompressToCSR(sparse, 10, 10);
  float ratio = CompressionRatio(csr);

  // 100 floats dense vs ~2 values + 2 col_indices + 11 row_offsets in CSR
  EXPECT_GT(ratio, 1.0f); // CSR should be smaller
}

TEST(SparseFormatTest, AllZeroMatrix) {
  std::vector<float> zeros(12, 0.0f);
  auto csr = CompressToCSR(zeros, 3, 4);

  EXPECT_EQ(csr.nnz(), 0);

  auto recovered = DecompressFromCSR(csr);
  for (float v : recovered) {
    EXPECT_FLOAT_EQ(v, 0.0f);
  }
}

TEST(SparseFormatTest, DenseMatrix) {
  std::vector<float> dense = {1.0f, 2.0f, 3.0f, 4.0f};
  auto csr = CompressToCSR(dense, 2, 2);

  EXPECT_EQ(csr.nnz(), 4); // All non-zero

  auto recovered = DecompressFromCSR(csr);
  ASSERT_EQ(recovered.size(), 4);
  for (size_t i = 0; i < dense.size(); ++i) {
    EXPECT_FLOAT_EQ(recovered[i], dense[i]);
  }
}

} // namespace
} // namespace paramserver
