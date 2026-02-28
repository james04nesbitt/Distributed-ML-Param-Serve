#include "src/core/parameter_store.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace paramserver {

void ParameterStore::InitializeLayer(const std::string &layer_name,
                                     const std::vector<int32_t> &shape) {
  std::lock_guard<std::mutex> lock(mu_);

  int32_t total_size = 1;
  for (int32_t dim : shape) {
    total_size *= dim;
  }

  LayerData data;
  data.values.resize(total_size, 0.0f);
  data.shape = shape;
  layers_[layer_name] = std::move(data);
}

void ParameterStore::ApplyGradient(const std::string &layer_name,
                                   const std::vector<float> &gradient,
                                   float learning_rate) {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = layers_.find(layer_name);
  if (it == layers_.end()) {
    throw std::runtime_error("Layer not found: " + layer_name);
  }

  auto &values = it->second.values;
  if (gradient.size() != values.size()) {
    throw std::runtime_error("Gradient size mismatch for layer: " + layer_name);
  }

  for (size_t i = 0; i < values.size(); ++i) {
    values[i] -= learning_rate * gradient[i];
  }
}

std::vector<float>
ParameterStore::GetParameters(const std::string &layer_name) const {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = layers_.find(layer_name);
  if (it == layers_.end()) {
    throw std::runtime_error("Layer not found: " + layer_name);
  }

  return it->second.values;
}

std::vector<int32_t>
ParameterStore::GetShape(const std::string &layer_name) const {
  std::lock_guard<std::mutex> lock(mu_);

  auto it = layers_.find(layer_name);
  if (it == layers_.end()) {
    throw std::runtime_error("Layer not found: " + layer_name);
  }

  return it->second.shape;
}

bool ParameterStore::HasLayer(const std::string &layer_name) const {
  std::lock_guard<std::mutex> lock(mu_);
  return layers_.count(layer_name) > 0;
}

int64_t ParameterStore::GetIteration() const {
  std::lock_guard<std::mutex> lock(mu_);
  return iteration_;
}

int64_t ParameterStore::IncrementIteration() {
  std::lock_guard<std::mutex> lock(mu_);
  return ++iteration_;
}

} // namespace paramserver
