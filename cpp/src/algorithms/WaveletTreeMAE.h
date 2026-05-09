#pragma once

#include <armadillo>
#include <memory>
#include <vector>

/** k-th smallest in [l,r] (0-based k) and sum of elements ≤ that value. */
struct QuantileResult {
    double value = 0.0;
    double sum_le = 0.0;
    int k = 0;
    int n = 0;
};

struct QuantileStats {
    double median_val;
    /** Sum of elements ≤ lower central value (even N) or ≤ median (odd N). */
    double sum_less_k;
    double sum_all;
    /** 0-based order index of that lower central element in the subrange. */
    int k;
    int N;
    double mean_abs_error;

    double mae() const { return mean_abs_error; }
};

struct WaveletRangeAgg;

/**
 * Static wavelet tree on compressed ranks. Range order statistics and sum by
 * rank use O(log σ) walks (σ = alphabet size).
 *
 * Uses heap allocation: WaveletRangeAgg holds a unique_ptr root; each WNode
 * stores bitrate vectors and prefix sums plus unique_ptr children recursively.
 */
class WaveletTreeMAE {
    int alphabet_size;
    int data_size;
    std::vector<float> unique_elements;
    arma::Row<float> orig_;
    std::vector<double> orig_value_;
    std::vector<double> global_prefix_sums;
    std::unique_ptr<WaveletRangeAgg> range_agg_;

    int get_compressed_rank(float val) const;

    int last_rank_strict_lt(double m) const;
    int last_rank_le(double m) const;

    double mean_abs_error_log(int L, int R, double m) const;

    /** Inclusive 0-based indices within the built row. */
    bool range_ok(int l, int r) const;

public:
    explicit WaveletTreeMAE(const arma::Row<float> &arr);
    ~WaveletTreeMAE();

    WaveletTreeMAE(const WaveletTreeMAE &) = delete;
    WaveletTreeMAE &operator=(const WaveletTreeMAE &) = delete;
    WaveletTreeMAE(WaveletTreeMAE &&) noexcept = default;
    WaveletTreeMAE &operator=(WaveletTreeMAE &&) noexcept = default;

    /** Inclusive 0-based indices. */
    double sum_all(int l, int r) const;

    /** Inclusive 0-based [l,r], 0-based order index k in [0, N). sum_le = sum ≤ k-th. */
    QuantileResult quantile(int l, int r, int k) const;

    /**
     * Inclusive 0-based [l, r]. Odd N: median at order index N/2; even N: average of
     * order indices N/2-1 and N/2; sum_less_k is sum_le from quantile(..., N/2-1).
     */
    QuantileStats quantileStatsForMedian(int l, int r) const;

    double mae_from_quantile(int l, int r, int k) const;
};
