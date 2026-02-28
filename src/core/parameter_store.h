#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace paramserver {

// In-memory storage for model parameters.
// Currently uses mutex-based locking; will be replaced with lock-free
// atomics in Milestone 3 (Async SGD).
class ParameterStore {
 public:
  ParameterStore() = default;

  // Initialize a parameter layer with the given shape and zero values.
  void InitializeLayer(const std::string& layer_name,
                       const std::vector<int32_t>& shape);

  // Apply a dense gradient update to a layer.
  // parameters[i] -= learning_rate * gradient[i]
  void ApplyGradient(const std::string& layer_name,
                     const std::vector<float>& gradient,
                     float learning_rate);

  // Get the current parameter values for a layer.
  std::vector<float> GetParameters(const std::string& layer_name) const;

  // Get the shape of a layer.
  std::vector<int32_t> GetShape(const std::string& layer_name) const;

  // Check if a layer exists.
  bool HasLayer(const std::string& layer_name) const;

  // Get the current global iteration count.
  int64_t GetIteration() const;

  // Increment and return the global iteration count.
  int64_t IncrementIteration();

 private:
  struct LayerData {
    std::vector<float> values;
    std::vector<int32_t> shape;
  };

  mutable std::mutex mu_;
  std::unordered_map<std::string, LayerData> layers_;
  int64_t iteration_ = 0;
};

}  // namespace paramserver
