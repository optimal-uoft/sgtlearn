/**
 * @file test_wavelet_tree_mae.cpp
 * @brief Catch2 tests for wavelet-tree median and MAE aggregates.
 */

#include <Criterion.h>
#include <algorithms/WaveletTreeMAE.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <armadillo>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

double brute_mae_to_reference(const std::vector<double> &sorted_sub,
                              double reference) {
    double s = 0.0;
    for (double x : sorted_sub) {
        s += std::fabs(x - reference);
    }
    return s / static_cast<double>(sorted_sub.size());
}

/** Odd: middle element; even: average of two middles (sorted sub, 0-based). */
double conventional_median(const std::vector<double> &sorted_sub) {
    const int N = static_cast<int>(sorted_sub.size());
    if (N % 2 == 1) {
        return sorted_sub[static_cast<size_t>(N / 2)];
    }
    return 0.5 * (sorted_sub[static_cast<size_t>(N / 2 - 1)] +
                  sorted_sub[static_cast<size_t>(N / 2)]);
}

std::vector<double> interval_slice(const arma::Row<float> &arr, int l, int r) {
    std::vector<double> sub;
    sub.reserve(static_cast<size_t>(r - l + 1));
    for (int i = l; i <= r; ++i) {
        sub.push_back(
            static_cast<double>(arr(static_cast<arma::uword>(i))));
    }
    return sub;
}

/** 0-based order index of lower central observation (matches QuantileStats::k). */
int lower_middle_index_0based(int len) {
    return (len % 2 == 1) ? (len / 2) : (len / 2 - 1);
}

arma::Row<float> unitWeights(arma::uword n) {
    arma::Row<float> w(n);
    w.ones();
    return w;
}

} // namespace

TEST_CASE("WaveletTreeMAE median quantileStats match brute force (odd and even lengths)") {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1000.0F, 1000.0F);

    constexpr std::array<int, 3> sizes{{10, 100, 1000}};
    constexpr int intervals_per_size = 5;
    constexpr double tol = 1e-5;
    constexpr double sum_tol = 1e-4;

    for (int n : sizes) {
        arma::Row<float> arr(static_cast<arma::uword>(n));
        for (arma::uword i = 0; i < arr.n_elem; ++i) {
            arr(i) = dist(rng);
        }

        WaveletTreeMAE wt(arr, unitWeights(arr.n_elem));

        for (int t = 0; t < intervals_per_size; ++t) {
            std::uniform_int_distribution<int> ldist(0, n - 1);
            const int l = ldist(rng);
            std::uniform_int_distribution<int> rdist(l, n - 1);
            const int r = rdist(rng);
            const int len = r - l + 1;

            auto sub = interval_slice(arr, l, r);
            std::vector<float> ys(sub.size());
            std::vector<float> ws(sub.size(), 1.f);
            for (size_t i = 0; i < sub.size(); ++i)
                ys[i] = static_cast<float>(sub[i]);
            const auto ref = Criterion::absoluteError(ys, ws);

            QuantileStats st = wt.quantileStatsForMedian(l, r);
            const double sum_sub =
                std::accumulate(sub.begin(), sub.end(), 0.0);

            REQUIRE(st.N == len);
            REQUIRE_THAT(st.sum_all, WithinAbs(sum_sub, sum_tol));
            REQUIRE_THAT(st.median_val, WithinAbs(ref.median, tol));
            REQUIRE_THAT(st.mae(), WithinAbs(ref.mae, tol));
        }
    }
}

TEST_CASE("quantileStatsForMedian odd vs even length") {
    arma::Row<float> a1{{3.F, 1.F, 4.F, 1.F, 5.F}};
    WaveletTreeMAE wt1(a1, unitWeights(a1.n_elem));
    const auto ref1 = Criterion::absoluteError(
        std::vector<float>{{3.F, 1.F, 4.F, 1.F, 5.F}},
        std::vector<float>(5, 1.f));
    QuantileStats q1 = wt1.quantileStatsForMedian(0, 4);
    REQUIRE(q1.N == 5);
    REQUIRE_THAT(q1.median_val, WithinAbs(ref1.median, 1e-6));
    REQUIRE_THAT(q1.mae(), WithinAbs(ref1.mae, 1e-6));

    arma::Row<float> a2{{10.F, 20.F, 30.F, 40.F}};
    WaveletTreeMAE wt2(a2, unitWeights(a2.n_elem));
    const auto ref2 = Criterion::absoluteError(
        std::vector<float>{{10.F, 20.F, 30.F, 40.F}},
        std::vector<float>(4, 1.f));
    QuantileStats q2 = wt2.quantileStatsForMedian(0, 3);
    REQUIRE(q2.N == 4);
    REQUIRE_THAT(q2.median_val, WithinAbs(ref2.median, 1e-6));
    REQUIRE_THAT(q2.mae(), WithinAbs(ref2.mae, 1e-6));
}

TEST_CASE("WaveletTreeMAE weighted median and MAE match Criterion (random weights)") {
  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1000.0F, 1000.0F);
  std::uniform_int_distribution<int> wdist(1, 5);

  constexpr std::array<int, 3> sizes{{10, 100, 1000}};
  constexpr int intervals_per_size = 5;
  constexpr double tol_median = 1e-5;
  constexpr double tol_mae = 1e-5;
  constexpr double tol_sum = 1e-4;

  for (int n : sizes) {
    arma::Row<float> arr(static_cast<arma::uword>(n));
    arma::Row<float> w(static_cast<arma::uword>(n));
    for (arma::uword i = 0; i < arr.n_elem; ++i) {
      arr(i) = dist(rng);
      w(i) = static_cast<float>(wdist(rng));
    }

    WaveletTreeMAE wt(arr, w);

    for (int t = 0; t < intervals_per_size; ++t) {
      std::uniform_int_distribution<int> ldist(0, n - 1);
      const int l = ldist(rng);
      std::uniform_int_distribution<int> rdist(l, n - 1);
      const int r = rdist(rng);
      const int len = r - l + 1;

      auto sub = interval_slice(arr, l, r);
      std::vector<float> ys(sub.size());
      std::vector<float> ws(sub.size());
      for (size_t i = 0; i < sub.size(); ++i) {
        ys[i] = static_cast<float>(sub[i]);
        ws[i] = w(static_cast<arma::uword>(l + static_cast<int>(i)));
      }

      const auto ref = Criterion::absoluteError(ys, ws);
      QuantileStats st = wt.quantileStatsForMedian(l, r);

      double wySum = 0.0;
      for (int i = l; i <= r; ++i) {
        wySum += static_cast<double>(arr(static_cast<arma::uword>(i))) *
                 static_cast<double>(w(static_cast<arma::uword>(i)));
      }

      REQUIRE(st.N == len);
      REQUIRE_THAT(st.sum_all, WithinAbs(wySum, tol_sum));
      REQUIRE_THAT(st.median_val, WithinAbs(ref.median, tol_median));
      REQUIRE_THAT(st.mae(), WithinAbs(ref.mae, tol_mae));
    }
  }
}

TEST_CASE("WaveletTreeMAE weighted median tie case (half weight split)") {
  arma::Row<float> arr{{0.F, 10.F}};
  arma::Row<float> w{{1.F, 1.F}};
  WaveletTreeMAE wt(arr, w);

  const QuantileStats st = wt.quantileStatsForMedian(0, 1);
  REQUIRE(st.N == 2);
  REQUIRE_THAT(st.median_val, WithinAbs(5.0, 1e-6));
  REQUIRE_THAT(st.mae(), WithinAbs(5.0, 1e-6));
}

TEST_CASE("WaveletTreeMAE weighted median with duplicates") {
  // Sorted (y,w): (0,1), (0,1), (10,1). Weighted median crosses half at y=0.
  arma::Row<float> arr{{0.F, 0.F, 10.F}};
  arma::Row<float> w{{1.F, 1.F, 1.F}};
  WaveletTreeMAE wt(arr, w);

  const QuantileStats st = wt.quantileStatsForMedian(0, 2);
  REQUIRE(st.N == 3);
  REQUIRE_THAT(st.median_val, WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(st.mae(), WithinAbs(10.0 / 3.0, 1e-6));
}
