/**
 * @file WaveletTreeMAE.cpp
 * @brief Wavelet tree construction and range median / MAE utilities.
 */

#include <algorithms/WaveletTreeMAE.h>

#include <algorithm>
#include <cmath>
#include <set>

struct WtItem {
    int rank;
    double val;
};

struct WNode {
    int lo = 0;
    int hi = 0;
    std::vector<int> b;
    std::vector<double> pref;
    std::unique_ptr<WNode> chL;
    std::unique_ptr<WNode> chR;

    WNode(typename std::vector<int>::iterator rb,
          typename std::vector<int>::iterator re,
          typename std::vector<double>::iterator vb, int xlo, int xhi)
        : lo(xlo), hi(xhi) {
        const int n = static_cast<int>(re - rb);
        if (n <= 0 || lo > hi) {
            return;
        }
        pref.assign(static_cast<size_t>(n) + 1, 0.0);
        for (int i = 0; i < n; ++i) {
            pref[static_cast<size_t>(i + 1)] =
                pref[static_cast<size_t>(i)] + vb[static_cast<size_t>(i)];
        }
        if (lo == hi) {
            return;
        }
        const int mid = lo + (hi - lo) / 2;
        b.reserve(static_cast<size_t>(n) + 1);
        b.push_back(0);
        for (int i = 0; i < n; ++i) {
            b.push_back(b.back() +
                         (rb[static_cast<size_t>(i)] <= mid ? 1 : 0));
        }
        std::vector<WtItem> items(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            items[static_cast<size_t>(i)] = {rb[static_cast<size_t>(i)],
                                              vb[static_cast<size_t>(i)]};
        }
        const auto pivot = std::stable_partition(
            items.begin(), items.end(),
            [mid](const WtItem &x) { return x.rank <= mid; });
        std::vector<int> lr;
        std::vector<double> lv;
        lr.reserve(static_cast<size_t>(pivot - items.begin()));
        lv.reserve(lr.capacity());
        for (auto it = items.begin(); it != pivot; ++it) {
            lr.push_back(it->rank);
            lv.push_back(it->val);
        }
        std::vector<int> rr;
        std::vector<double> rv;
        rr.reserve(static_cast<size_t>(items.end() - pivot));
        rv.reserve(rr.capacity());
        for (auto it = pivot; it != items.end(); ++it) {
            rr.push_back(it->rank);
            rv.push_back(it->val);
        }
        chL = std::make_unique<WNode>(lr.begin(), lr.end(), lv.begin(), lo, mid);
        chR = std::make_unique<WNode>(rr.begin(), rr.end(), rv.begin(), mid + 1,
                                      hi);
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

    double sumLE(int L, int R, int kmax) const {
        if (L > R) {
            return 0.0;
        }
        if (kmax < lo) {
            return 0.0;
        }
        if (hi <= kmax) {
            return pref[static_cast<size_t>(R + 1)] -
                   pref[static_cast<size_t>(L)];
        }
        if (lo == hi) {
            return (kmax >= lo) ? (pref[static_cast<size_t>(R + 1)] -
                                    pref[static_cast<size_t>(L)])
                                : 0.0;
        }
        const int leftL = b[static_cast<size_t>(L)];
        const int leftR = b[static_cast<size_t>(R + 1)] - 1;
        const int rightL = L - b[static_cast<size_t>(L)];
        const int rightR = R - b[static_cast<size_t>(R + 1)];
        return chL->sumLE(leftL, leftR, kmax) + chR->sumLE(rightL, rightR, kmax);
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
};

struct WaveletRangeAgg {
    std::unique_ptr<WNode> root;
    int max_rank = 0;

    WaveletRangeAgg() = default;

    WaveletRangeAgg(std::vector<int> ranks, std::vector<double> values) {
        if (ranks.empty()) {
            return;
        }
        max_rank = *std::max_element(ranks.begin(), ranks.end());
        root = std::make_unique<WNode>(ranks.begin(), ranks.end(), values.begin(),
                                       0, max_rank);
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

    double sumLE(int L, int R, int kmax) const {
        if (!root || L > R) {
            return 0.0;
        }
        return root->sumLE(L, R, kmax);
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

double WaveletTreeMAE::mean_abs_error_log(int L, int R, double m) const {
    if (!range_agg_ || !range_ok(L, R)) {
        return 0.0;
    }
    const int N = R - L + 1;
    if (N <= 0) {
        return 0.0;
    }
    const int r_lt = last_rank_strict_lt(m);
    const int c_lt =
        (r_lt >= 0) ? range_agg_->countLE(L, R, r_lt) : 0;
    const double s_lt =
        (r_lt >= 0) ? range_agg_->sumLE(L, R, r_lt) : 0.0;
    const int r_le = last_rank_le(m);
    const int c_le =
        (r_le >= 0) ? range_agg_->countLE(L, R, r_le) : 0;
    const double s_le =
        (r_le >= 0) ? range_agg_->sumLE(L, R, r_le) : 0.0;
    const int c_gt = N - c_le;
    const double total = global_prefix_sums[static_cast<size_t>(R + 1)] -
                         global_prefix_sums[static_cast<size_t>(L)];
    const double s_gt = total - s_le;
    const double sum_abs = m * static_cast<double>(c_lt) - s_lt + s_gt -
                           m * static_cast<double>(c_gt);
    return sum_abs / static_cast<double>(N);
}

WaveletTreeMAE::WaveletTreeMAE(const arma::Row<float> &arr) : orig_(arr) {
    data_size = static_cast<int>(orig_.n_elem);
    orig_value_.resize(static_cast<size_t>(data_size));
    for (int i = 0; i < data_size; ++i) {
        orig_value_[static_cast<size_t>(i)] =
            static_cast<double>(orig_(static_cast<arma::uword>(i)));
    }

    std::set<float> distinct_elements(orig_.begin(), orig_.end());
    unique_elements.assign(distinct_elements.begin(), distinct_elements.end());
    alphabet_size = static_cast<int>(unique_elements.size());

    global_prefix_sums.assign(static_cast<size_t>(data_size + 1), 0.0);
    for (int i = 0; i < data_size; ++i) {
        global_prefix_sums[static_cast<size_t>(i + 1)] =
            global_prefix_sums[static_cast<size_t>(i)] +
            orig_value_[static_cast<size_t>(i)];
    }

    if (data_size > 0 && alphabet_size > 0) {
        std::vector<int> ranks(static_cast<size_t>(data_size));
        for (int i = 0; i < data_size; ++i) {
            int r = get_compressed_rank(
                orig_(static_cast<arma::uword>(i)));
            if (r < 0) {
                r = 0;
            } else if (r >= alphabet_size) {
                r = alphabet_size - 1;
            }
            ranks[static_cast<size_t>(i)] = r;
        }
        std::vector<double> vals = orig_value_;
        range_agg_ =
            std::make_unique<WaveletRangeAgg>(std::move(ranks), std::move(vals));
    }
}

WaveletTreeMAE::~WaveletTreeMAE() = default;

double WaveletTreeMAE::sum_all(int l, int r) const {
    if (!range_ok(l, r)) {
        return 0.0;
    }
    return global_prefix_sums[static_cast<size_t>(r + 1)] -
           global_prefix_sums[static_cast<size_t>(l)];
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
    out.sum_le = range_agg_->sumLE(L, R, rank);
    out.k = k;
    out.n = N;
    return out;
}

QuantileStats WaveletTreeMAE::quantileStatsForMedian(int l, int r) const {
    const int L = l;
    const int R = r;
    const int N = R - L + 1;
    if (!range_agg_ || !range_ok(l, r) || N <= 0) {
        return QuantileStats{};
    }
    const double sum_all_val = sum_all(l, r);
    // 0-based order index of lower central element (matches sum_less_k).
    const int k_lower = (N % 2 == 1) ? (N / 2) : (N / 2 - 1);

    if (N % 2 == 1) {
        const QuantileResult q = quantile(l, r, N / 2);
        const double mae_val = mean_abs_error_log(L, R, q.value);
        return QuantileStats{
            q.value, q.sum_le, sum_all_val, k_lower, N, mae_val,
        };
    }

    const QuantileResult q_lo = quantile(l, r, N / 2 - 1);
    const QuantileResult q_hi = quantile(l, r, N / 2);
    const double median_val = 0.5 * (q_lo.value + q_hi.value);
    const double mae_val = mean_abs_error_log(L, R, median_val);
    return QuantileStats{
        median_val, q_lo.sum_le, sum_all_val, k_lower, N, mae_val,
    };
}

double WaveletTreeMAE::mae_from_quantile(int l, int r, int k) const {
    const QuantileResult q = quantile(l, r, k);
    if (q.n <= 0) {
        return 0.0;
    }
    return mean_abs_error_log(l, r, q.value);
}
