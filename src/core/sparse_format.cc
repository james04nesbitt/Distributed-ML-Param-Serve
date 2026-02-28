#include "src/core/sparse_format.h"

#include <cmath>
#include <stdexcept>

namespace paramserver {

CSRMatrix CompressToCSR(const std::vector<float> &dense, int32_t num_rows,
                        int32_t num_cols, float threshold) {
  if (static_cast<int32_t>(dense.size()) != num_rows * num_cols) {
    throw std::runtime_error("Dense matrix size does not match dimensions");
  }

  CSRMatrix csr;
  csr.num_rows = num_rows;
  csr.num_cols = num_cols;
  csr.row_offsets.reserve(num_rows + 1);

  for (int32_t row = 0; row < num_rows; ++row) {
    csr.row_offsets.push_back(csr.nnz());
    for (int32_t col = 0; col < num_cols; ++col) {
      float val = dense[row * num_cols + col];
      if (std::fabs(val) > threshold) {
        csr.values.push_back(val);
        csr.col_indices.push_back(col);
      }
    }
  }
  csr.row_offsets.push_back(csr.nnz()); // sentinel

  return csr;
}

std::vector<float> DecompressFromCSR(const CSRMatrix &csr) {
  std::vector<float> dense(csr.num_rows * csr.num_cols, 0.0f);

  for (int32_t row = 0; row < csr.num_rows; ++row) {
    for (int32_t idx = csr.row_offsets[row]; idx < csr.row_offsets[row + 1];
         ++idx) {
      dense[row * csr.num_cols + csr.col_indices[idx]] = csr.values[idx];
    }
  }

  return dense;
}

float CompressionRatio(const CSRMatrix &csr) {
  // Dense size: num_rows * num_cols floats
  float dense_bytes =
      static_cast<float>(csr.num_rows * csr.num_cols) * sizeof(float);

  // CSR size: values (float) + col_indices (int32) + row_offsets (int32)
  float csr_bytes = static_cast<float>(csr.nnz()) * sizeof(float) +
                    static_cast<float>(csr.nnz()) * sizeof(int32_t) +
                    static_cast<float>(csr.num_rows + 1) * sizeof(int32_t);

  if (csr_bytes == 0.0f)
    return 0.0f;
  return dense_bytes / csr_bytes;
}

} // namespace paramserver
