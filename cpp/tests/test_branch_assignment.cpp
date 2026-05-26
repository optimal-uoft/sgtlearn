/**
 * @file test_branch_assignment.cpp
 * @brief Catch2 tests for branch-assignment objectives and coordinate descent.
 */

#include <algorithms/CoordinateDescent.h>
#include <BranchAssignmentObjectives/BranchAssignmentFactory.h>
#include <BranchAssignmentObjectives/BranchAssignmentVariants.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

constexpr double kEps = 1e-6;

void assert_coordinate_descent_non_worsening(BranchAssignment &obj,
                                             size_t numPartitions,
                                             double tol = kEps) {
  std::mt19937_64 rng(42);
  const double initial = obj.objective();
  const double returned =
      coordinateDescent(numPartitions, obj, rng, /*maxIters=*/30,
                        /*patience=*/8);
  const double from_state = obj.objective();
  REQUIRE(from_state <= initial + tol);
  REQUIRE_THAT(from_state, WithinAbs(returned, tol));
}

} // namespace

TEST_CASE("EntropyBranchAssignment coordinate descent",
          "[branch_assignment][coordinate_descent]") {
  constexpr size_t kClasses = 2;
  constexpr size_t kParts = 2;
  std::vector<size_t> assignments = {0, 0};
  std::vector<std::vector<double>> stats = {{10, 0}, {0, 10}};
  std::vector<double> leafWeights = {10, 10};
  EntropyBranchAssignment obj(assignments, kParts, stats, leafWeights, kClasses);
  assert_coordinate_descent_non_worsening(obj, kParts);
}

TEST_CASE("GiniBranchAssignment coordinate descent",
          "[branch_assignment][coordinate_descent]") {
  constexpr size_t kClasses = 2;
  constexpr size_t kParts = 2;
  std::vector<size_t> assignments = {0, 0};
  std::vector<std::vector<double>> stats = {{10, 0}, {0, 10}};
  std::vector<double> leafWeights = {10, 10};
  GiniBranchAssignment obj(assignments, kParts, stats, leafWeights, kClasses);
  assert_coordinate_descent_non_worsening(obj, kParts);
}

TEST_CASE("SquaredErrorBranchAssignment coordinate descent",
          "[branch_assignment][coordinate_descent]") {
  constexpr size_t kParts = 2;
  std::vector<size_t> assignments = {0, 0};
  std::vector<std::vector<double>> stats = {{0.0, 0.0}, {30.0, 300.0}};
  std::vector<double> leafWeights = {3, 3};
  SquaredErrorBranchAssignment obj(assignments, kParts, stats, leafWeights);
  assert_coordinate_descent_non_worsening(obj, kParts);
}

TEST_CASE("GainHessianBranchAssignment coordinate descent",
          "[branch_assignment][coordinate_descent]") {
  constexpr size_t kParts = 2;
  std::vector<size_t> assignments = {0, 0};
  std::vector<std::vector<float>> stats = {{3.F, 1.F}, {1.F, 2.F}};
  std::vector<double> leafWeights = {3, 3};
  GainHessianBranchAssignment obj(assignments, kParts, stats, leafWeights, 1.0);
  assert_coordinate_descent_non_worsening(obj, kParts);
}

TEST_CASE("AbsoluteErrorBranchAssignment coordinate descent",
          "[branch_assignment][coordinate_descent]") {
  constexpr size_t kParts = 2;
  std::vector<size_t> assignments = {0, 0};
  std::vector<std::vector<float>> leafYs = {
      {0.F, 0.F, 0.F},
      {10.F, 10.F, 10.F},
  };
  std::vector<std::vector<float>> leafWs = {
      {1.F, 1.F, 1.F},
      {1.F, 1.F, 1.F},
  };
  std::vector<double> leafWeights = {3, 3};
  AbsoluteErrorBranchAssignment obj(assignments, kParts, leafYs, leafWs,
                                    leafWeights);
  assert_coordinate_descent_non_worsening(obj, kParts);
}

TEST_CASE("makeClassificationBranchAssignment entropy matches manual",
          "[branch_assignment][factory]") {
  std::vector<size_t> assignments = {0, 1};
  std::vector<std::vector<double>> stats = {{5, 0}, {0, 5}};
  std::vector<double> leafWeights = {5, 5};
  auto ptr = makeClassificationBranchAssignment(
      LearningCriterion::Entropy, assignments, 2, stats, leafWeights, 2);
  EntropyBranchAssignment direct(assignments, 2, stats, leafWeights, 2);
  REQUIRE_THAT(ptr->objective(), WithinAbs(direct.objective(), kEps));
}

TEST_CASE("makeRegressionBranchAssignment squared error matches manual",
          "[branch_assignment][factory]") {
  std::vector<size_t> assignments = {0, 1};
  std::vector<std::vector<double>> stats = {{0.0, 0.0}, {20.0, 200.0}};
  std::vector<double> leafWeights = {2, 2};
  auto ptr = makeRegressionBranchAssignment(LearningCriterion::SquaredError,
                                            assignments, 2, stats, leafWeights);
  SquaredErrorBranchAssignment direct(assignments, 2, stats, leafWeights);
  REQUIRE_THAT(ptr->objective(), WithinAbs(direct.objective(), kEps));
}

TEST_CASE("makeRegressionBranchAssignment gain hessian matches manual",
          "[branch_assignment][factory]") {
  std::vector<size_t> assignments = {0, 1};
  std::vector<std::vector<double>> stats = {{1.0, 1.0}, {2.0, 2.0}};
  std::vector<double> leafWeights = {1, 1};
  const double lambda = 0.5;
  auto ptr = makeRegressionBranchAssignment(LearningCriterion::GainHessian,
                                            assignments, 2, stats, leafWeights,
                                            lambda);
  std::vector<std::vector<float>> ghStats = {{1.F, 1.F}, {2.F, 2.F}};
  GainHessianBranchAssignment direct(assignments, 2, ghStats, leafWeights,
                                     lambda);
  REQUIRE_THAT(ptr->objective(), WithinAbs(direct.objective(), kEps));
}

TEST_CASE("makeRegressionBranchAssignment absolute error matches manual",
          "[branch_assignment][factory]") {
  std::vector<size_t> assignments = {0, 1};
  std::vector<std::vector<float>> leafYs = {{1.F, 2.F}, {10.F}};
  std::vector<std::vector<float>> leafWs = {{1.F, 1.F}, {1.F}};
  std::vector<double> leafWeights = {2, 1};
  std::vector<std::vector<double>> unusedStats;
  auto ptr = makeRegressionBranchAssignment(
      LearningCriterion::AbsoluteError, assignments, 2, unusedStats,
      leafWeights, 1.0, &leafYs, &leafWs);
  AbsoluteErrorBranchAssignment direct(assignments, 2, leafYs, leafWs,
                                       leafWeights);
  REQUIRE_THAT(ptr->objective(), WithinAbs(direct.objective(), kEps));
}

TEST_CASE("makeClassificationBranchAssignment gini matches manual",
          "[branch_assignment][factory]") {
  std::vector<size_t> assignments = {0, 1};
  std::vector<std::vector<double>> stats = {{3, 0}, {0, 2}};
  std::vector<double> leafWeights = {3, 2};
  auto ptr = makeClassificationBranchAssignment(LearningCriterion::Gini,
                                                assignments, 2, stats,
                                                leafWeights, 2);
  GiniBranchAssignment direct(assignments, 2, stats, leafWeights, 2);
  REQUIRE_THAT(ptr->objective(), WithinAbs(direct.objective(), kEps));
}
