#include "src/core/parameter_store.h"

#include <gtest/gtest.h>

namespace paramserver {
namespace {

TEST(ParameterStoreTest, InitializeAndRetrieve) {
  ParameterStore store;
  store.InitializeLayer("dense_1", {3, 4});

  ASSERT_TRUE(store.HasLayer("dense_1"));
  EXPECT_FALSE(store.HasLayer("nonexistent"));

  auto params = store.GetParameters("dense_1");
  EXPECT_EQ(params.size(), 12);
  for (float v : params) {
    EXPECT_FLOAT_EQ(v, 0.0f);
  }

  auto shape = store.GetShape("dense_1");
  ASSERT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 3);
  EXPECT_EQ(shape[1], 4);
}

TEST(ParameterStoreTest, ApplyGradient) {
  ParameterStore store;
  store.InitializeLayer("weights", {2, 2});

  // gradient = [1, 2, 3, 4], lr = 0.1
  // result = [0, 0, 0, 0] - 0.1 * [1, 2, 3, 4] = [-0.1, -0.2, -0.3, -0.4]
  store.ApplyGradient("weights", {1.0f, 2.0f, 3.0f, 4.0f}, 0.1f);

  auto params = store.GetParameters("weights");
  EXPECT_FLOAT_EQ(params[0], -0.1f);
  EXPECT_FLOAT_EQ(params[1], -0.2f);
  EXPECT_FLOAT_EQ(params[2], -0.3f);
  EXPECT_FLOAT_EQ(params[3], -0.4f);
}

TEST(ParameterStoreTest, IterationCounter) {
  ParameterStore store;
  EXPECT_EQ(store.GetIteration(), 0);
  EXPECT_EQ(store.IncrementIteration(), 1);
  EXPECT_EQ(store.IncrementIteration(), 2);
  EXPECT_EQ(store.GetIteration(), 2);
}

} // namespace
} // namespace paramserver
