#pragma once

#include <cstdint>
#include <vector>

namespace paramserver {

// Compressed Sparse Row (CSR) matrix format.
// Stores only non-zero elements, making it efficient for sparse gradients.
//
// For an m×n matrix with nnz non-zero elements:
//   values:      [nnz]    - the non-zero values
//   col_indices: [nnz]    - column index for each non-zero value
//   row_offsets: [m + 1]  - index into values[] where each row starts
//                           row_offsets[m] == nnz
struct CSRMatrix {
  std::vector<float> values;
  std::vector<int32_t> col_indices;
  std::vector<int32_t> row_offsets;
  int32_t num_rows = 0;
  int32_t num_cols = 0;

  // Number of non-zero elements.
  int32_t nnz() const { return static_cast<int32_t>(values.size()); }
};

// Compress a dense matrix (row-major) into CSR format.
// Elements with absolute value <= threshold are treated as zero.
CSRMatrix CompressToCSR(const std::vector<float> &dense, int32_t num_rows,
                        int32_t num_cols, float threshold = 1e-8f);

// Decompress a CSR matrix back into dense row-major format.
std::vector<float> DecompressFromCSR(const CSRMatrix &csr);

// Compute the compression ratio: dense_size / csr_size.
// Returns how many times smaller the CSR representation is.
float CompressionRatio(const CSRMatrix &csr);

} // namespace paramserver
