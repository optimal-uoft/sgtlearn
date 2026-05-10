/**
 * @file test_wavelet_tree_mae.cpp
 * @brief Catch2 tests for wavelet-tree median and MAE aggregates.
 */

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

        WaveletTreeMAE wt(arr);

        for (int t = 0; t < intervals_per_size; ++t) {
            std::uniform_int_distribution<int> ldist(0, n - 1);
            const int l = ldist(rng);
            std::uniform_int_distribution<int> rdist(l, n - 1);
            const int r = rdist(rng);
            const int len = r - l + 1;

            auto sub = interval_slice(arr, l, r);
            std::sort(sub.begin(), sub.end());

            const int k_lower = lower_middle_index_0based(len);
            const double expected_med = conventional_median(sub);
            const double expected_mae =
                brute_mae_to_reference(sub, expected_med);

            QuantileStats st = wt.quantileStatsForMedian(l, r);
            const double sum_sub =
                std::accumulate(sub.begin(), sub.end(), 0.0);

            REQUIRE(st.k == k_lower);
            REQUIRE(st.N == len);
            REQUIRE_THAT(st.sum_all, WithinAbs(sum_sub, sum_tol));
            REQUIRE_THAT(st.median_val, WithinAbs(expected_med, tol));
            REQUIRE_THAT(st.mae(), WithinAbs(expected_mae, tol));
        }
    }
}

TEST_CASE("quantileStatsForMedian odd vs even length") {
    arma::Row<float> a1{{3.F, 1.F, 4.F, 1.F, 5.F}};
    WaveletTreeMAE wt1(a1);
    std::vector<double> s1{{1., 1., 3., 4., 5.}};
    const int k1 = lower_middle_index_0based(5);
    REQUIRE(k1 == 2);
    QuantileStats q1 = wt1.quantileStatsForMedian(0, 4);
    REQUIRE(q1.k == 2);
    REQUIRE(q1.N == 5);
    REQUIRE_THAT(q1.median_val, WithinAbs(3.0, 1e-6));
    REQUIRE_THAT(q1.mae(),
                 WithinAbs(brute_mae_to_reference(s1, 3.0), 1e-6));

    arma::Row<float> a2{{10.F, 20.F, 30.F, 40.F}};
    WaveletTreeMAE wt2(a2);
    std::vector<double> s2{{10., 20., 30., 40.}};
    const int k2 = lower_middle_index_0based(4);
    REQUIRE(k2 == 1);
    QuantileStats q2 = wt2.quantileStatsForMedian(0, 3);
    REQUIRE(q2.k == 1);
    REQUIRE(q2.N == 4);
    REQUIRE_THAT(q2.median_val, WithinAbs(25.0, 1e-6));
    REQUIRE_THAT(q2.mae(),
                 WithinAbs(brute_mae_to_reference(s2, 25.0), 1e-6));
}
