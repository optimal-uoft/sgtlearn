/**
 * @file test_tao_objective.cpp
 * @brief Catch2 tests for TAO per-node scoring (bin relabel, threshold scan).
 */

#include <Discretizers/ClassificationDiscretizer.h>
#include <Discretizers/factories/DiscretizerFactories.h>
#include <algorithms/TAO/TaoObjective.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;

TEST_CASE("TaoObjective relabels a bin by full-credit care reward", "[tao]") {
  // One bin: 10 care samples are correct in children {0, 1}, 8 only in child
  // 2. Split router weights give the histogram (5, 5, 8), whose argmax is
  // child 2 (8 correct); full credit gives (10, 10, 8) -> child 0 (10 correct).
  tao::NodeCareSet care;
  std::vector<arma::uword> xCols;
  std::vector<size_t> yRows;
  std::vector<float> wRows;
  for (arma::uword i = 0; i < 18; ++i) {
    const bool shared = i < 10;
    care.careCols.push_back(i);
    care.careRewards.push_back(shared ? std::vector<double>{1, 1, 0}
                                      : std::vector<double>{0, 0, 1});
    care.careWeights.push_back(1.0);
    for (size_t child : shared ? std::vector<size_t>{0, 1}
                               : std::vector<size_t>{2}) {
      xCols.push_back(i);
      yRows.push_back(child);
      wRows.push_back(shared ? 0.5f : 1.0f);
    }
  }
  // Distinct from the bin's best child, so the unused NaN bin's map shows.
  care.dummyChild = 1;
  const arma::fmat X(1, 18, arma::fill::zeros); // constant feature: one bin
  care.Xexp = X.cols(arma::uvec(xCols));
  care.yexp = arma::Row<size_t>(yRows);
  care.wexp = arma::Row<float>(wRows);

  auto disc = makeClassificationDiscretizer(LearningCriterion::Gini,
                                            DiscretizerInputKind::Numeric);
  arma::uvec feature = {0};
  disc->Train(care.Xexp, feature, care.yexp, 3, 1, 0.0, 0, 0, care.wexp);
  REQUIRE(disc->numLeaves() == 1);
  REQUIRE(disc->leafStats()[0][0] == std::vector<double>{5, 5, 8});

  const tao::TaoObjective objective(care, X, /*lambda=*/0.0,
                                    /*nodeSampleCount=*/18);
  std::vector<size_t> binToPartition;
  const double score = objective.scoreDiscretizer(*disc, binToPartition, 1.0);

  REQUIRE(binToPartition == std::vector<size_t>{0, 1});
  REQUIRE_THAT(score, WithinAbs(10.0 / 18.0, 1e-12));
}

TEST_CASE("TaoObjective finds the exact best threshold for two children",
          "[tao]") {
  const float nan = std::numeric_limits<float>::quiet_NaN();
  const arma::fmat X = {{4, 1, 3, 2, nan}};
  // x = 1, 2 are correct in child 1; x = 3, 4 and NaN in child 0.
  tao::NodeCareSet care;
  care.careCols = {0, 1, 2, 3, 4};
  care.careRewards = {{1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}};
  care.dummyChild = 1;
  const tao::TaoObjective objective(care, X, 0.0, 5);

  const auto cut = objective.bestThresholdCut(0, 1);
  REQUIRE(cut.has_value());
  REQUIRE(cut->leftMax == 2.0f);
  REQUIRE(cut->leftChild == 1);
  REQUIRE(cut->nanChild == 0);
  REQUIRE_THAT(cut->rewardSum, WithinAbs(5.0, 1e-12));

  // Three finite care samples per side are needed, and only four exist.
  REQUIRE_FALSE(objective.bestThresholdCut(0, 3).has_value());
}
