/**
 * @file WaveletTreeMAE.cpp
 * @brief Wavelet tree construction and range median / MAE utilities.
 */

#include <memory>
#include <utility>
#include <cstddef>
#include <algorithms/WaveletTreeMAE.h>

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>

struct WNode {
    int lo = 0;
    int hi = 0;
    /** Sample-count bitmap for splitting; traversal uses counts, not weights. */
    std::vector<int> b;
    /** Prefix sum of weights for items routed left (rank <= mid). */
    std::vector<double> bw_left;
    /** Prefix sums of sample weight and weight × value over this node's segment. */
    std::vector<double> pref_w;
    std::vector<double> pref_wy;
    std::unique_ptr<WNode> chL;
    std::unique_ptr<WNode> chR;

    WNode(typename std::vector<int>::iterator rb,
          typename std::vector<int>::iterator re,
          typename std::vector<double>::iterator wb,
          typename std::vector<double>::iterator wyb, int xlo, int xhi)
        : lo(xlo), hi(xhi) {
        const int n = static_cast<int>(re - rb);
        if (n <= 0 || lo > hi) {
            return;
        }
        pref_w.assign(static_cast<size_t>(n) + 1, 0.0);
        pref_wy.assign(static_cast<size_t>(n) + 1, 0.0);
        for (int i = 0; i < n; ++i) {
            pref_w[static_cast<size_t>(i + 1)] =
                pref_w[static_cast<size_t>(i)] + wb[static_cast<size_t>(i)];
            pref_wy[static_cast<size_t>(i + 1)] =
                pref_wy[static_cast<size_t>(i)] + wyb[static_cast<size_t>(i)];
        }
        if (lo == hi) {
            return;
        }
        const int mid = lo + (hi - lo) / 2;
        b.reserve(static_cast<size_t>(n) + 1);
        b.push_back(0);
        bw_left.reserve(static_cast<size_t>(n) + 1);
        bw_left.push_back(0.0);
        for (int i = 0; i < n; ++i) {
            const bool goesLeft = (rb[static_cast<size_t>(i)] <= mid);
            b.push_back(b.back() +
                         (goesLeft ? 1 : 0));
            bw_left.push_back(bw_left.back() +
                              (goesLeft ? wb[static_cast<size_t>(i)] : 0.0));
        }
        struct Item {
            int rank;
            double w;
            double wy;
        };
        std::vector<Item> items(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            items[static_cast<size_t>(i)] = {
                rb[static_cast<size_t>(i)], wb[static_cast<size_t>(i)],
                wyb[static_cast<size_t>(i)]};
        }
        const auto pivot = std::stable_partition(
            items.begin(), items.end(),
            [mid](const Item &x) { return x.rank <= mid; });
        std::vector<int> lr;
        std::vector<double> lw, lwy;
        lr.reserve(static_cast<size_t>(pivot - items.begin()));
        lw.reserve(lr.capacity());
        lwy.reserve(lr.capacity());
        for (auto it = items.begin(); it != pivot; ++it) {
            lr.push_back(it->rank);
            lw.push_back(it->w);
            lwy.push_back(it->wy);
        }
        std::vector<int> rr;
        std::vector<double> rw, rwy;
        rr.reserve(static_cast<size_t>(items.end() - pivot));
        rw.reserve(rr.capacity());
        rwy.reserve(rr.capacity());
        for (auto it = pivot; it != items.end(); ++it) {
            rr.push_back(it->rank);
            rw.push_back(it->w);
            rwy.push_back(it->wy);
        }
        chL = std::make_unique<WNode>(lr.begin(), lr.end(), lw.begin(), lwy.begin(),
                                      lo, mid);
        chR = std::make_unique<WNode>(rr.begin(), rr.end(), rw.begin(), rwy.begin(),
                                      mid + 1, hi);
    }

    int countLE(int L, int R, int kmax) const {
        if (L > R) {
            return 0;
        }
        if (kmax < lo) {
            return 0;
        }
        if (hi <= kmax) {
            return R - L + 1;
        }
        if (lo == hi) {
            return (kmax >= lo) ? (R - L + 1) : 0;
        }
        const int leftL = b[static_cast<size_t>(L)];
        const int leftR = b[static_cast<size_t>(R + 1)] - 1;
        const int rightL = L - b[static_cast<size_t>(L)];
        const int rightR = R - b[static_cast<size_t>(R + 1)];
        return chL->countLE(leftL, leftR, kmax) +
               chR->countLE(rightL, rightR, kmax);
    }

    double sumWLE(int L, int R, int kmax) const {
        if (L > R) {
            return 0.0;
        }
        if (kmax < lo) {
            return 0.0;
        }
        if (hi <= kmax) {
            return pref_w[static_cast<size_t>(R + 1)] -
                   pref_w[static_cast<size_t>(L)];
        }
        if (lo == hi) {
            return (kmax >= lo) ? (pref_w[static_cast<size_t>(R + 1)] -
                                    pref_w[static_cast<size_t>(L)])
                                : 0.0;
        }
        const int leftL = b[static_cast<size_t>(L)];
        const int leftR = b[static_cast<size_t>(R + 1)] - 1;
        const int rightL = L - b[static_cast<size_t>(L)];
        const int rightR = R - b[static_cast<size_t>(R + 1)];
        return chL->sumWLE(leftL, leftR, kmax) + chR->sumWLE(rightL, rightR, kmax);
    }

    double sumWyLE(int L, int R, int kmax) const {
        if (L > R) {
            return 0.0;
        }
        if (kmax < lo) {
            return 0.0;
        }
        if (hi <= kmax) {
            return pref_wy[static_cast<size_t>(R + 1)] -
                   pref_wy[static_cast<size_t>(L)];
        }
        if (lo == hi) {
            return (kmax >= lo) ? (pref_wy[static_cast<size_t>(R + 1)] -
                                    pref_wy[static_cast<size_t>(L)])
                                : 0.0;
        }
        const int leftL = b[static_cast<size_t>(L)];
        const int leftR = b[static_cast<size_t>(R + 1)] - 1;
        const int rightL = L - b[static_cast<size_t>(L)];
        const int rightR = R - b[static_cast<size_t>(R + 1)];
        return chL->sumWyLE(leftL, leftR, kmax) +
               chR->sumWyLE(rightL, rightR, kmax);
    }

    int kth(int L, int R, int k) const {
        if (L > R || k < 1) {
            return -1;
        }
        if (lo == hi) {
            return lo;
        }
        const int inLeft =
            b[static_cast<size_t>(R + 1)] - b[static_cast<size_t>(L)];
        if (k <= inLeft) {
            return chL->kth(b[static_cast<size_t>(L)],
                           b[static_cast<size_t>(R + 1)] - 1, k);
        }
        const int rightL = L - b[static_cast<size_t>(L)];
        const int rightR = R - b[static_cast<size_t>(R + 1)];
        return chR->kth(rightL, rightR, k - inLeft);
    }

    /**
     * Return smallest rank in [L,R] whose cumulative weight exceeds @p target.
     *
     * @p target is in [0, totalWeight); traversal uses bw_left for fast weight
     * aggregation of the left child inside the current node's subsequence.
     */
    int selectByWeight(int L, int R, double target) const {
        if (L > R) {
            return -1;
        }
        if (lo == hi) {
            return lo;
        }
        const int leftL = b[static_cast<size_t>(L)];
        const int leftR = b[static_cast<size_t>(R + 1)] - 1;
        const int rightL = L - b[static_cast<size_t>(L)];
        const int rightR = R - b[static_cast<size_t>(R + 1)];
        const double wLeft = bw_left[static_cast<size_t>(R + 1)] -
                             bw_left[static_cast<size_t>(L)];
        if (wLeft > target) {
            return chL ? chL->selectByWeight(leftL, leftR, target) : -1;
        }
        return chR ? chR->selectByWeight(rightL, rightR, target - wLeft) : -1;
    }
};

struct WaveletRangeAgg {
    std::unique_ptr<WNode> root;
    int max_rank = 0;

    WaveletRangeAgg() = default;

    WaveletRangeAgg(std::vector<int> ranks, std::vector<double> weights,
                    std::vector<double> wy) {
        if (ranks.empty()) {
            return;
        }
        max_rank = *std::max_element(ranks.begin(), ranks.end());
        root = std::make_unique<WNode>(ranks.begin(), ranks.end(), weights.begin(),
                                       wy.begin(), 0, max_rank);
    }

    int kth(int L, int R, int k) const {
        return root ? root->kth(L, R, k) : -1;
    }

    int countLE(int L, int R, int kmax) const {
        if (!root || L > R) {
            return 0;
        }
        return root->countLE(L, R, kmax);
    }

    double sumWLE(int L, int R, int kmax) const {
        if (!root || L > R) {
            return 0.0;
        }
        return root->sumWLE(L, R, kmax);
    }

    double sumWyLE(int L, int R, int kmax) const {
        if (!root || L > R) {
            return 0.0;
        }
        return root->sumWyLE(L, R, kmax);
    }

    int selectByWeight(int L, int R, double target) const {
        if (!root || L > R) {
            return -1;
        }
        return root->selectByWeight(L, R, target);
    }
};

int WaveletTreeMAE::get_compressed_rank(float val) const {
    auto it =
        std::lower_bound(unique_elements.begin(), unique_elements.end(), val);
    return static_cast<int>(
        std::distance(unique_elements.begin(), it));
}

int WaveletTreeMAE::last_rank_strict_lt(double m) const {
    if (unique_elements.empty()) {
        return -1;
    }
    const auto it = std::lower_bound(
        unique_elements.begin(), unique_elements.end(), m,
        [](float u, double mm) { return static_cast<double>(u) < mm; });
    return static_cast<int>(it - unique_elements.begin()) - 1;
}

int WaveletTreeMAE::last_rank_le(double m) const {
    if (unique_elements.empty()) {
        return -1;
    }
    const auto it = std::upper_bound(
        unique_elements.begin(), unique_elements.end(), m,
        [](double mm, float u) { return mm < static_cast<double>(u); });
    return static_cast<int>(it - unique_elements.begin()) - 1;
}

bool WaveletTreeMAE::range_ok(int l, int r) const {
    return l >= 0 && r >= 0 && l <= r && r < data_size;
}

double WaveletTreeMAE::sumW(int l, int r) const {
    if (!range_ok(l, r)) {
        return 0.0;
    }
    return global_prefix_w_[static_cast<size_t>(r + 1)] -
           global_prefix_w_[static_cast<size_t>(l)];
}

double WaveletTreeMAE::sumWLE(int l, int r, int kmax) const {
    if (!range_agg_ || !range_ok(l, r)) {
        return 0.0;
    }
    return range_agg_->sumWLE(l, r, kmax);
}

double WaveletTreeMAE::sumWyLE(int l, int r, int kmax) const {
    if (!range_agg_ || !range_ok(l, r)) {
        return 0.0;
    }
    return range_agg_->sumWyLE(l, r, kmax);
}

int WaveletTreeMAE::weightedMedianRank(int L, int R, double half) const {
    if (!range_agg_ || !range_ok(L, R)) {
        return -1;
    }
    // Find smallest rank where cumulative weight > half.
    // If exactly half sits at the end of the left subtree at some split, we
    // traverse right to get the first element beyond half, matching
    // sklearn/Criterion::absoluteError semantics.
    return range_agg_->selectByWeight(L, R, half);
}

double WaveletTreeMAE::mean_abs_error(int L, int R, double m) const {
    if (!range_agg_ || !range_ok(L, R)) {
        return 0.0;
    }
    const double W = sumW(L, R);
    if (W <= 0.0) {
        return 0.0;
    }
    const int r_lt = last_rank_strict_lt(m);
    const double w_lt = (r_lt >= 0) ? sumWLE(L, R, r_lt) : 0.0;
    const double wy_lt = (r_lt >= 0) ? sumWyLE(L, R, r_lt) : 0.0;
    const int r_le = last_rank_le(m);
    const double w_le = (r_le >= 0) ? sumWLE(L, R, r_le) : 0.0;
    const double wy_le = (r_le >= 0) ? sumWyLE(L, R, r_le) : 0.0;
    const double w_gt = W - w_le;
    const double wy_gt = sum_all(L, R) - wy_le;
    const double pinball = m * w_lt - wy_lt + wy_gt - m * w_gt;
    return pinball / W;
}

WaveletTreeMAE::WaveletTreeMAE(const arma::Row<float> &arr,
                               const arma::Row<float> &weights)
    : orig_(arr) {
    data_size = static_cast<int>(orig_.n_elem);
    if (weights.n_elem != orig_.n_elem)
        throw std::invalid_argument(
            "WaveletTreeMAE: weights length must match arr length");
    orig_value_.resize(static_cast<size_t>(data_size));
    orig_weight_.resize(static_cast<size_t>(data_size));

    for (int i = 0; i < data_size; ++i) {
        const arma::uword ui = static_cast<arma::uword>(i);
        orig_value_[static_cast<size_t>(i)] = static_cast<double>(orig_(ui));
        orig_weight_[static_cast<size_t>(i)] = static_cast<double>(weights(ui));
    }

    std::set<float> distinct_elements(orig_.begin(), orig_.end());
    unique_elements.assign(distinct_elements.begin(), distinct_elements.end());
    alphabet_size = static_cast<int>(unique_elements.size());

    global_prefix_wy_.assign(static_cast<size_t>(data_size + 1), 0.0);
    global_prefix_w_.assign(static_cast<size_t>(data_size + 1), 0.0);
    for (int i = 0; i < data_size; ++i) {
        const size_t si = static_cast<size_t>(i);
        global_prefix_wy_[si + 1] =
            global_prefix_wy_[si] +
            orig_weight_[si] * orig_value_[si];
        global_prefix_w_[si + 1] =
            global_prefix_w_[si] + orig_weight_[si];
    }

    if (data_size > 0 && alphabet_size > 0) {
        std::vector<int> ranks(static_cast<size_t>(data_size));
        std::vector<double> ws(static_cast<size_t>(data_size));
        std::vector<double> wys(static_cast<size_t>(data_size));
        for (int i = 0; i < data_size; ++i) {
            int r = get_compressed_rank(
                orig_(static_cast<arma::uword>(i)));
            if (r < 0) {
                r = 0;
            } else if (r >= alphabet_size) {
                r = alphabet_size - 1;
            }
            ranks[static_cast<size_t>(i)] = r;
            ws[static_cast<size_t>(i)] = orig_weight_[static_cast<size_t>(i)];
            wys[static_cast<size_t>(i)] =
                orig_weight_[static_cast<size_t>(i)] *
                orig_value_[static_cast<size_t>(i)];
        }
        range_agg_ = std::make_unique<WaveletRangeAgg>(
            std::move(ranks), std::move(ws), std::move(wys));
    }
}

WaveletTreeMAE::~WaveletTreeMAE() = default;

double WaveletTreeMAE::sum_all(int l, int r) const {
    if (!range_ok(l, r)) {
        return 0.0;
    }
    return global_prefix_wy_[static_cast<size_t>(r + 1)] -
           global_prefix_wy_[static_cast<size_t>(l)];
}

QuantileResult WaveletTreeMAE::quantile(int l, int r, int k) const {
    QuantileResult out;
    const int L = l;
    const int R = r;
    const int N = R - L + 1;
    if (!range_agg_ || !range_ok(l, r) || N <= 0 || k < 0 || k >= N) {
        return out;
    }
    const int rank = range_agg_->kth(L, R, k + 1);
    if (rank < 0 ||
        rank >= static_cast<int>(unique_elements.size())) {
        return out;
    }
    out.value =
        static_cast<double>(unique_elements[static_cast<size_t>(rank)]);
    out.sum_le = range_agg_->sumWyLE(L, R, rank);
    out.k = k;
    out.n = N;
    return out;
}

QuantileStats WaveletTreeMAE::quantileStatsForMedian(int l, int r) const {
    const int L = l;
    const int R = r;
    const int N = R - L + 1;
    if (!range_agg_ || !range_ok(l, r) || N <= 0 || alphabet_size <= 0) {
        return QuantileStats{};
    }
    const double W = sumW(l, r);
    if (W <= 0.0) {
        return QuantileStats{};
    }
    const double half = 0.5 * W;
    const int medRank = weightedMedianRank(L, R, half);
    if (medRank < 0 || medRank >= alphabet_size) {
        return QuantileStats{};
    }
    const double wLess = (medRank > 0) ? sumWLE(L, R, medRank - 1) : 0.0;
    double medianVal = static_cast<double>(unique_elements[static_cast<size_t>(medRank)]);
    int k_lower = 0;
    if (medRank > 0) {
        // number of elements strictly less than median rank
        const int c_lt = range_agg_->countLE(L, R, medRank - 1);
        k_lower = (c_lt > 0) ? (c_lt - 1) : 0;
        if (std::fabs(wLess - half) <= 1e-12 && c_lt > 0) {
            // predecessor (largest element with rank < medRank) by count-based kth.
            const int prevRank = range_agg_->kth(L, R, c_lt);
            if (prevRank >= 0 && prevRank < alphabet_size) {
                const double prevVal =
                    static_cast<double>(unique_elements[static_cast<size_t>(prevRank)]);
                medianVal = 0.5 * (prevVal + medianVal);
            }
        }
    }
    const int r_lt = last_rank_strict_lt(medianVal);
    const double wy_lt = (r_lt >= 0) ? sumWyLE(l, r, r_lt) : 0.0;
    const double maeVal = mean_abs_error(l, r, medianVal);
    return QuantileStats{medianVal, wy_lt, sum_all(l, r),
                         k_lower, N, maeVal};
}

double WaveletTreeMAE::mae_from_quantile(int l, int r, int k) const {
    const QuantileResult q = quantile(l, r, k);
    if (q.n <= 0) {
        return 0.0;
    }
    return mean_abs_error(l, r, q.value);
}
