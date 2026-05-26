#pragma once

/**
 * @file WaveletTreeMAE.h
 * @brief Wavelet tree on discrete ranks for fast range medians and MAE contributions (used by ``AbsoluteErrorSplitter``).
 *
 * Traversal (``b``, ``countLE``, ``kth``) is by sample index/count; each node stores
 * prefix sums of sample weight and of weight × value for range aggregates.
 */

#include <armadillo>
#include <memory>
#include <vector>

/** k-th smallest in [l,r] (0-based k) and weighted sum of y for values ≤ that order statistic. */
struct QuantileResult {
    double value = 0.0;
    double sum_le = 0.0;
    int k = 0;
    int n = 0;
};

struct QuantileStats {
    double median_val;
    /** Weighted sum of y for values ≤ lower central order statistic. */
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
 * Static wavelet tree on compressed ranks. Range order statistics use O(log σ)
 * rank walks; ``kth`` / ``countLE`` traverse by sample count.
 */
class WaveletTreeMAE {
    int alphabet_size;
    int data_size;
    std::vector<float> unique_elements;
    arma::Row<float> orig_;
    std::vector<double> orig_value_;
    std::vector<double> orig_weight_;
    std::vector<double> global_prefix_wy_;
    std::vector<double> global_prefix_w_;
    std::unique_ptr<WaveletRangeAgg> range_agg_;

    int get_compressed_rank(float val) const;

    int last_rank_strict_lt(double m) const;
    int last_rank_le(double m) const;

    double mean_abs_error(int L, int R, double m) const;

    /** Smallest rank where cumulative weight in [L,R] exceeds @p half. */
    int weightedMedianRank(int L, int R, double half) const;

    double sumW(int l, int r) const;
    double sumWLE(int l, int r, int kmax) const;
    double sumWyLE(int l, int r, int kmax) const;

    bool range_ok(int l, int r) const;

public:
    /** @p weights must have the same length as @p arr. */
    explicit WaveletTreeMAE(const arma::Row<float> &arr,
                            const arma::Row<float> &weights);
    ~WaveletTreeMAE();

    WaveletTreeMAE(const WaveletTreeMAE &) = delete;
    WaveletTreeMAE &operator=(const WaveletTreeMAE &) = delete;
    WaveletTreeMAE(WaveletTreeMAE &&) noexcept = default;
    WaveletTreeMAE &operator=(WaveletTreeMAE &&) noexcept = default;

    /** Inclusive 0-based indices: ``sum(w * y)`` on ``[l,r]``. */
    double sum_all(int l, int r) const;

    /** Inclusive 0-based [l,r], 0-based order index k in [0, N). */
    QuantileResult quantile(int l, int r, int k) const;

    /** Weighted median and mean MAE on ``[l,r]`` (sklearn-style). */
    QuantileStats quantileStatsForMedian(int l, int r) const;

    double mae_from_quantile(int l, int r, int k) const;
};
