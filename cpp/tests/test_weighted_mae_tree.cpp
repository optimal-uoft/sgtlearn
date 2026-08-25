/**
 * @file test_weighted_mae_tree.cpp
 * @brief Correctness tests for ``WeightedMAETree`` vs ``Criterion::absoluteError``.
 */

#include <Criterion.h>
#include <algorithms/WeightedMAETree.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

Criterion::AbsoluteErrorStats brute(const std::vector<float> &ys,
                                    const std::vector<float> &ws) {
  return Criterion::absoluteError(ys, ws);
}

} // namespace

TEST_CASE("WeightedMAETree median/mae match Criterion on random batches",
          "[weighted_mae_tree]") {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> yDist(-50.0F, 50.0F);
  std::uniform_real_distribution<float> wDist(0.1F, 3.0F);

  for (int n : {1, 2, 3, 10, 64, 257}) {
    std::vector<float> ys(static_cast<size_t>(n));
    std::vector<float> ws(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      ys[static_cast<size_t>(i)] = yDist(rng);
      ws[static_cast<size_t>(i)] = wDist(rng);
    }

    WeightedMAETree tree;
    tree.insert_batch(ys, ws);
    const auto ref = brute(ys, ws);
    const auto got = tree.medianAndMae();
    REQUIRE_THAT(got.first, WithinAbs(ref.median, 1e-5));
    REQUIRE_THAT(got.second, WithinAbs(ref.mae, 1e-5));
    REQUIRE_THAT(tree.totalWeight(), WithinAbs(ref.totalWeight, 1e-6));
  }
}

TEST_CASE("WeightedMAETree supports remove_batch round-trip",
          "[weighted_mae_tree]") {
  std::vector<float> ys = {1.F, 2.F, 3.F, 4.F, 5.F, 2.F};
  std::vector<float> ws = {1.F, 1.F, 2.F, 1.F, 1.F, 0.5F};
  std::vector<float> dropY = {2.F, 4.F};
  std::vector<float> dropW = {1.F, 1.F};

  WeightedMAETree tree;
  tree.insert_batch(ys, ws);
  tree.remove_batch(dropY, dropW);

  std::vector<float> remainY = {1.F, 3.F, 5.F, 2.F};
  std::vector<float> remainW = {1.F, 2.F, 1.F, 0.5F};
  const auto ref = brute(remainY, remainW);
  const auto got = tree.medianAndMae();
  REQUIRE_THAT(got.first, WithinAbs(ref.median, 1e-6));
  REQUIRE_THAT(got.second, WithinAbs(ref.mae, 1e-6));
}

TEST_CASE("WeightedMAETree half-tie median matches Criterion",
          "[weighted_mae_tree]") {
  // Equal total weight on each side of the cut → average of adjacent keys.
  std::vector<float> ys = {1.F, 3.F};
  std::vector<float> ws = {1.F, 1.F};
  WeightedMAETree tree;
  tree.insert_batch(ys, ws);
  const auto ref = brute(ys, ws);
  REQUIRE_THAT(tree.median(), WithinAbs(ref.median, 1e-12));
  REQUIRE_THAT(tree.mae(), WithinAbs(ref.mae, 1e-12));
  REQUIRE_THAT(tree.median(), WithinAbs(2.0, 1e-12));
}

TEST_CASE("WeightedMAETree duplicate keys", "[weighted_mae_tree]") {
  std::vector<float> ys(100, 4.5F);
  std::vector<float> ws(100, 0.25F);
  ys[0] = -10.F;
  ws[0] = 1.F;

  WeightedMAETree tree;
  tree.insert_batch(ys, ws);
  const auto ref = brute(ys, ws);
  REQUIRE_THAT(tree.median(), WithinAbs(ref.median, 1e-6));
  REQUIRE_THAT(tree.mae(), WithinAbs(ref.mae, 1e-6));
}
