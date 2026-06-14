/**
 * @file test_missing_values.cpp
 * @brief Catch2 tests for non-finite feature handling in discretizers and routing.
 */

#include <Estimators/ClassificationShapeGeneralizedTree.h>
#include <Estimators/RegressionShapeGeneralizedTree.h>
#include <algorithms/missing_values.h>

#include <catch2/catch_test_macros.hpp>

#include <armadillo>
#include <cmath>
#include <limits>

TEST_CASE("sort_index_finite_first places NaN at tail") {
  arma::frowvec row{{3.F, std::numeric_limits<float>::quiet_NaN(), 1.F, 2.F}};
  const arma::uvec order = missing_values::sort_index_finite_first(row);

  REQUIRE(order.n_elem == 4);
  REQUIRE(row(order(0)) == 1.F);
  REQUIRE(row(order(1)) == 2.F);
  REQUIRE(row(order(2)) == 3.F);
  REQUIRE(std::isnan(row(order(3))));
}

TEST_CASE("ClassificationShapeGeneralizedTree fits and predicts with NaN in X") {
  arma::fmat X(2, 6);
  X(0, 0) = 1.F;
  X(0, 1) = 2.F;
  X(0, 2) = std::numeric_limits<float>::quiet_NaN();
  X(0, 3) = 4.F;
  X(0, 4) = 5.F;
  X(0, 5) = 6.F;
  X(1, 0) = 0.F;
  X(1, 1) = 1.F;
  X(1, 2) = 0.F;
  X(1, 3) = 1.F;
  X(1, 4) = 0.F;
  X(1, 5) = 1.F;

  arma::Row<size_t> y(6);
  y(0) = 0;
  y(1) = 0;
  y(2) = 1;
  y(3) = 1;
  y(4) = 0;
  y(5) = 1;

  ClassificationShapeGeneralizedTree tree(
      LearningCriterion::Gini, 2, 2,
      TreeBuildingParams{.minLeafSize = 1,
                         .minGainSplit = 0.0,
                         .maxDepth = 3,
                         .maxLeafNodes = 0},
      TreeBuildingParams{.minLeafSize = 1,
                         .minGainSplit = 0.0,
                         .maxDepth = 2,
                         .maxLeafNodes = 4},
      CoordinateDescentParams{}, 42);

  arma::Row<float> sampleWeights(6);
  sampleWeights.ones();
  REQUIRE_NOTHROW(tree.fit(X, y, sampleWeights));

  arma::fmat X_pred = X;
  X_pred(1, 0) = std::numeric_limits<float>::quiet_NaN();
  const arma::Row<size_t> preds = tree.predict(X_pred);
  REQUIRE(preds.n_elem == X_pred.n_cols);
}

TEST_CASE("RegressionShapeGeneralizedTree fits with scattered NaN in X") {
  arma::fmat X(2, 5);
  X(0, 0) = 1.F;
  X(0, 1) = std::numeric_limits<float>::quiet_NaN();
  X(0, 2) = 3.F;
  X(0, 3) = 4.F;
  X(0, 4) = 5.F;
  X(1, 0) = 0.F;
  X(1, 1) = 1.F;
  X(1, 2) = 0.F;
  X(1, 3) = std::numeric_limits<float>::quiet_NaN();
  X(1, 4) = 0.F;

  arma::Row<float> y(5);
  y(0) = 1.F;
  y(1) = 2.F;
  y(2) = 3.F;
  y(3) = 4.F;
  y(4) = 5.F;

  RegressionShapeGeneralizedTree tree(
      LearningCriterion::SquaredError, 2,
      TreeBuildingParams{.minLeafSize = 1,
                         .minGainSplit = 0.0,
                         .maxDepth = 3,
                         .maxLeafNodes = 0},
      TreeBuildingParams{.minLeafSize = 1,
                         .minGainSplit = 0.0,
                         .maxDepth = 2,
                         .maxLeafNodes = 4},
      CoordinateDescentParams{}, 42);

  arma::Row<float> sampleWeights(5);
  sampleWeights.ones();
  REQUIRE_NOTHROW(tree.fit(X, y, sampleWeights));
  const arma::Row<double> preds = tree.predict(X);
  REQUIRE(preds.n_elem == X.n_cols);
}
