/**
 * @file test_splitters.cpp
 * @brief Catch2 tests for univariate splitters (Gini, entropy, MSE, MAE, gain/hessian).
 */

#include <Splitters/AbsoluteErrorSplitter.h>
#include <Criterion.h>
#include <Splitters/EntropySplitter.h>
#include <Splitters/GainHessianSplitter.h>
#include <Splitters/GiniSplitter.h>
#include <Splitters/SquaredErrorSplitter.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <armadillo>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

constexpr double kEps = 1e-6;

arma::frowvec unitWeights(size_t n) {
  arma::frowvec w(n);
  w.ones();
  return w;
}

double brute_mae(const std::vector<float> &vals) {
  if (vals.empty()) {
    return 0.0;
  }
  std::vector<double> sorted(vals.begin(), vals.end());
  std::sort(sorted.begin(), sorted.end());
  const int n = static_cast<int>(sorted.size());
  double med;
  if (n % 2 == 1) {
    med = sorted[static_cast<size_t>(n / 2)];
  } else {
    med = 0.5 * (sorted[static_cast<size_t>(n / 2 - 1)] +
                 sorted[static_cast<size_t>(n / 2)]);
  }
  double s = 0.0;
  for (double x : sorted) {
    s += std::fabs(x - med);
  }
  return s / static_cast<double>(n);
}

} // namespace

TEST_CASE("GiniSplitter score and makeRoot / predict") {
  arma::frowvec X{{1.F, 2.F, 3.F}};
  arma::Mat<size_t> y(1, 3);
  y(0, 0) = 0;
  y(0, 1) = 1;
  y(0, 2) = 1;
  arma::frowvec w = unitWeights(3);

  GiniSplitter splitter(X, w, y, 2);
  SplitCandidate root = splitter.makeRoot();
  REQUIRE(root.start == 0);
  REQUIRE(root.end == 2);
  REQUIRE(root.numSamples == 3);
  REQUIRE_THAT(root.nodeWeight, WithinAbs(3.0, kEps));
  REQUIRE_THAT(root.score, WithinAbs(1.0 - (1.0 / 9.0 + 4.0 / 9.0), kEps));
  REQUIRE(splitter.predict(root) == 1);

  std::vector<double> pure{{5, 0}};
  REQUIRE_THAT(splitter.score(pure, 0, 4), WithinAbs(0.0, kEps));

  std::vector<double> fifty_fifty{{2, 2}};
  REQUIRE_THAT(splitter.score(fifty_fifty, 0, 3), WithinAbs(0.5, kEps));
}

TEST_CASE("GiniSplitter respects sample_weights in aggregates") {
  arma::frowvec X{{1.F, 2.F, 3.F}};
  arma::Mat<size_t> y(1, 3);
  y.fill(0);
  y(0, 2) = 1;
  arma::frowvec w{{2.F, 1.F, 1.F}};

  GiniSplitter splitter(X, w, y, 2);
  SplitCandidate root = splitter.makeRoot();
  REQUIRE_THAT(root.nodeWeight, WithinAbs(4.0, kEps));
  const auto &stats = splitter.getStats(root);
  REQUIRE_THAT(stats[0], WithinAbs(3.0, kEps));
  REQUIRE_THAT(stats[1], WithinAbs(1.0, kEps));
}

TEST_CASE("EntropySplitter score") {
  arma::frowvec X(1);
  X(0) = 1.F;
  arma::Mat<size_t> y(1, 1);
  y(0, 0) = 0;
  arma::frowvec w = unitWeights(1);

  EntropySplitter splitter(X, w, y, 2);
  std::vector<double> uniform2{{2, 2}};
  REQUIRE_THAT(splitter.score(uniform2, 0, 3), WithinAbs(1.0, kEps));

  std::vector<double> pure{{4, 0}};
  REQUIRE_THAT(splitter.score(pure, 0, 3), WithinAbs(0.0, kEps));

  std::vector<double> three_way{{1, 1, 1}};
  const double expected = std::log2(3.0);
  REQUIRE_THAT(splitter.score(three_way, 0, 2), WithinAbs(expected, 1e-5));
}

TEST_CASE("SquaredErrorSplitter makeRoot predict and score") {
  arma::frowvec X{{1.F, 2.F, 3.F}};
  arma::Mat<float> y(1, 3);
  y(0, 0) = 1.F;
  y(0, 1) = 2.F;
  y(0, 2) = 3.F;
  arma::frowvec w = unitWeights(3);

  SquaredErrorSplitter splitter(X, w, y);
  SplitCandidate root = splitter.makeRoot();
  REQUIRE_THAT(splitter.predict(root), WithinAbs(2.F, 1e-5f));
  REQUIRE_THAT(root.score, WithinAbs(2.0, kEps));
  REQUIRE(root.numSamples == 3);
  REQUIRE_THAT(root.nodeWeight, WithinAbs(3.0, kEps));

  std::vector<float> stats{{6.F, 14.F}};
  REQUIRE_THAT(splitter.score(stats, 0, 2), WithinAbs(2.0, kEps));
  std::vector<float> one_point{{3.F, 9.F}};
  REQUIRE_THAT(splitter.score(one_point, 0, 0), WithinAbs(0.0, kEps));
}

TEST_CASE("SquaredErrorSplitter weighted MSE") {
  arma::frowvec X{{1.F, 2.F}};
  arma::Mat<float> y(1, 2);
  y(0, 0) = 0.F;
  y(0, 1) = 2.F;
  arma::frowvec w{{1.F, 3.F}};

  SquaredErrorSplitter splitter(X, w, y);
  SplitCandidate root = splitter.makeRoot();
  REQUIRE_THAT(root.nodeWeight, WithinAbs(4.0, kEps));
  REQUIRE_THAT(splitter.predict(root), WithinAbs(1.5F, 1e-5f));
}

TEST_CASE("GainHessianSplitter makeRoot predict score and validates y rows") {
  arma::frowvec X{{1.F, 2.F}};
  arma::Mat<float> y(2, 2);
  y(0, 0) = 1.F;
  y(1, 0) = 2.F;
  y(0, 1) = 3.F;
  y(1, 1) = 4.F;
  arma::frowvec w = unitWeights(2);

  const double lambda = 1.0;
  GainHessianSplitter splitter(X, w, y, lambda);
  SplitCandidate root = splitter.makeRoot();
  const double g = 4.0;
  const double h = 6.0;
  const double expected = g * g / (h + lambda);
  REQUIRE_THAT(root.score, WithinAbs(expected, kEps));
  REQUIRE_THAT(static_cast<double>(splitter.predict(root)), WithinAbs(expected, kEps));

  arma::Mat<float> y_bad(1, 2);
  y_bad.fill(1.F);
  REQUIRE_THROWS_AS((GainHessianSplitter(X, w, y_bad, lambda)),
                    std::invalid_argument);
}

TEST_CASE("AbsoluteErrorSplitter predict and score match brute-force MAE") {
  arma::frowvec X{{0.F, 1.F, 2.F, 3.F, 4.F}};
  arma::Mat<float> y(1, 5);
  y.row(0) = arma::Row<float>{{3.F, 1.F, 4.F, 1.F, 5.F}};
  arma::frowvec w = unitWeights(5);

  AbsoluteErrorSplitter splitter(X, w, y);
  SplitCandidate root = splitter.makeRoot();
  std::vector<float> vals{{3.F, 1.F, 4.F, 1.F, 5.F}};
  const double expected_mae = brute_mae(vals);
  REQUIRE_THAT(static_cast<double>(splitter.predict(root)), WithinAbs(3.0, 1e-4));
  REQUIRE_THAT(splitter.score({}, 0, 4), WithinAbs(expected_mae, 1e-4));
  REQUIRE(root.numSamples == 5);
}

TEST_CASE("AbsoluteErrorSplitter weighted score matches Criterion::absoluteError") {
  arma::frowvec X{{0.F, 1.F, 2.F, 3.F}};
  arma::Mat<float> y(1, 4);
  y.row(0) = arma::Row<float>{{1.F, 5.F, 2.F, 8.F}};
  arma::frowvec w{{1.F, 2.F, 3.F, 4.F}};

  AbsoluteErrorSplitter splitter(X, w, y);
  SplitCandidate root = splitter.makeRoot();
  std::vector<float> ys{{1.F, 5.F, 2.F, 8.F}};
  std::vector<float> ws{{1.F, 2.F, 3.F, 4.F}};
  const auto ref = Criterion::absoluteError(ys, ws);
  REQUIRE_THAT(splitter.score({}, 0, 3), WithinAbs(ref.mae, 1e-5));
  REQUIRE_THAT(static_cast<double>(splitter.predict(root)),
               WithinAbs(ref.median, 1e-5));
}

TEST_CASE("GiniSplitter findBestSplit separates two pure class blocks") {
  arma::frowvec X{{1.F, 2.F, 3.F, 10.F, 11.F, 12.F}};
  arma::Mat<size_t> y(1, 6);
  for (int i = 0; i < 3; ++i) {
    y(0, i) = 0;
  }
  for (int i = 3; i < 6; ++i) {
    y(0, i) = 1;
  }
  arma::frowvec w = unitWeights(6);

  GiniSplitter splitter(X, w, y, 2);
  SplitCandidate root = splitter.makeRoot();
  const bool found = splitter.findBestSplit(root, 2);
  REQUIRE(found);
  REQUIRE(root.leftStart == 0);
  REQUIRE(root.leftEnd == 2);
  REQUIRE(root.rightStart == 3);
  REQUIRE(root.rightEnd == 5);
  REQUIRE(root.leftNumSamples == 3);
  REQUIRE(root.rightNumSamples == 3);
  REQUIRE_THAT(root.leftWeight, WithinAbs(3.0, kEps));
  REQUIRE_THAT(root.rightWeight, WithinAbs(3.0, kEps));
  REQUIRE(root.informationGain > 0.0);

  auto children = splitter.makeChildren(root);
  REQUIRE(children.size() == 2);
  SplitCandidate left = children[1];
  SplitCandidate right = children[0];
  REQUIRE(splitter.predict(left) == 0);
  REQUIRE(splitter.predict(right) == 1);
}

TEST_CASE("Splitter findBestSplit returns false when leaf too small") {
  arma::frowvec X{{1.F, 2.F, 3.F}};
  arma::Mat<size_t> y(1, 3);
  y.fill(0);
  arma::frowvec w = unitWeights(3);

  GiniSplitter splitter(X, w, y, 1);
  SplitCandidate root = splitter.makeRoot();
  REQUIRE_FALSE(splitter.findBestSplit(root, 2));
}
