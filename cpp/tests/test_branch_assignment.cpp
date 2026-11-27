/**
 * @file test_branch_assignment.cpp
 * @brief Catch2 tests for branch-assignment objectives and coordinate descent.
 */

#include <BranchAssignmentObjectives/AbsoluteErrorBranchAssignment.h>
#include <BranchAssignmentObjectives/BranchAssignmentFactory.h>
#include <BranchAssignmentObjectives/LeafAggregationBranchAssignment.h>
#include <algorithms/CoordinateDescent.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

constexpr double kEps = 1e-6;

void assert_coordinate_descent_finds_split(BranchAssignment &obj,
                                           size_t numPartitions,
                                           size_t samplesPerBin,
                                           double tol = kEps) {
  std::mt19937_64 rng(42);
  const double initial = obj.objective();
  const double returned =
      coordinateDescent(numPartitions, obj, rng, /*maxIters=*/30,
                        /*patience=*/8, /*hasNanRoutingBin=*/false);
  const double from_state = obj.objective();
  REQUIRE(from_state < initial - tol);
  REQUIRE_THAT(from_state, WithinAbs(returned, tol));
  const std::vector<size_t> forward = {0, 1};
  const std::vector<size_t> reverse = {1, 0};
  REQUIRE((obj.assignments == forward || obj.assignments == reverse));
  REQUIRE(obj.partitionSampleCounts() ==
          std::vector<size_t>{samplesPerBin, samplesPerBin});
}

} // namespace

TEST_CASE("Entropy branch assignment coordinate descent",
          "[branch_assignment][coordinate_descent]") {
  constexpr size_t kParts = 2;
  std::vector<size_t> assignments = {0, 0};
  std::vector<std::vector<std::vector<double>>> stats = {{{10, 0}}, {{0, 10}}};
  std::vector<double> leafWeights = {10, 10};
  std::vector<size_t> leafSampleCounts = {10, 10};
  LeafAggregationBranchAssignment obj(LearningCriterion::Entropy, assignments,
                                      kParts, stats, leafWeights,
                                      leafSampleCounts, 1);
  assert_coordinate_descent_finds_split(obj, kParts, 10);
}

TEST_CASE("Gini branch assignment coordinate descent",
          "[branch_assignment][coordinate_descent]") {
  constexpr size_t kParts = 2;
  std::vector<size_t> assignments = {0, 0};
  std::vector<std::vector<std::vector<double>>> stats = {{{10, 0}}, {{0, 10}}};
  std::vector<double> leafWeights = {10, 10};
  std::vector<size_t> leafSampleCounts = {10, 10};
  LeafAggregationBranchAssignment obj(LearningCriterion::Gini, assignments,
                                      kParts, stats, leafWeights,
                                      leafSampleCounts, 1);
  assert_coordinate_descent_finds_split(obj, kParts, 10);
}

TEST_CASE("SquaredError branch assignment coordinate descent",
          "[branch_assignment][coordinate_descent]") {
  constexpr size_t kParts = 2;
  std::vector<size_t> assignments = {0, 0};
  std::vector<std::vector<std::vector<double>>> stats = {
      {{0.0, 0.0}}, {{30.0, 300.0}}};
  std::vector<double> leafWeights = {3, 3};
  std::vector<size_t> leafSampleCounts = {3, 3};
  LeafAggregationBranchAssignment obj(LearningCriterion::SquaredError,
                                      assignments, kParts, stats, leafWeights,
                                      leafSampleCounts, 1);
  assert_coordinate_descent_finds_split(obj, kParts, 3);
}

TEST_CASE("AbsoluteErrorBranchAssignment coordinate descent",
          "[branch_assignment][coordinate_descent]") {
  constexpr size_t kParts = 2;
  std::vector<size_t> assignments = {0, 0};
  std::vector<std::vector<std::vector<float>>> leafYs = {
      {{0.F, 0.F, 0.F}},
      {{10.F, 10.F, 10.F}},
  };
  std::vector<std::vector<float>> leafWs = {
      {1.F, 1.F, 1.F},
      {1.F, 1.F, 1.F},
  };
  std::vector<double> leafWeights = {3, 3};
  std::vector<size_t> leafSampleCounts = {3, 3};
  AbsoluteErrorBranchAssignment obj(assignments, kParts, leafYs, leafWs,
                                    leafWeights, leafSampleCounts);
  assert_coordinate_descent_finds_split(obj, kParts, 3);
}

TEST_CASE("BranchAssignment tracks partition sample counts",
          "[branch_assignment][partition_counts]") {
  std::vector<size_t> assignments = {0, 1, 0};
  std::vector<std::vector<std::vector<double>>> stats = {
      {{4, 0}}, {{0, 2}}, {{1, 1}}};
  std::vector<double> leafWeights = {4, 2, 2};
  std::vector<size_t> leafSampleCounts = {4, 2, 2};
  LeafAggregationBranchAssignment obj(LearningCriterion::Entropy, assignments,
                                      2, stats, leafWeights, leafSampleCounts,
                                      1);
  REQUIRE(obj.partitionSampleCounts() == std::vector<size_t>{6, 2});
  REQUIRE(obj.partitionCountsMeetMinLeaf(2));
  REQUIRE_FALSE(obj.partitionCountsMeetMinLeaf(3));

  obj.removeLeaf(1);
  REQUIRE(obj.partitionSampleCounts() == std::vector<size_t>{6, 0});
  REQUIRE_FALSE(obj.partitionCountsMeetMinLeaf(1));

  obj.addLeaf(1, 1);
  REQUIRE(obj.partitionSampleCounts() == std::vector<size_t>{6, 2});
}
