#pragma once

/**
 * @file algorithms/missing_values.h
 * @brief Non-finite feature handling for discretizers and tree routing.
 */

#include <armadillo>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace missing_values {

inline bool is_finite(float value) {
  return std::isfinite(static_cast<double>(value));
}

/** Sorted sample indices (finite ascending, then non-finite tail) plus tail start. */
struct FiniteFirstSort {
  arma::uvec order;
  /** Index in ``order`` of the first non-finite sample; ``order.n_elem`` if none. */
  size_t first_non_finite_index = 0;
};

/** Sort sample indices: ascending finite values, then non-finite tail. */
inline FiniteFirstSort sort_index_finite_first(const arma::frowvec &row) {
  const arma::uword n = row.n_elem;
  std::vector<arma::uword> finite_idxs;
  std::vector<arma::uword> missing_idxs;
  finite_idxs.reserve(static_cast<size_t>(n));
  missing_idxs.reserve(static_cast<size_t>(n));

  for (arma::uword i = 0; i < n; ++i) {
    if (is_finite(row(i)))
      finite_idxs.push_back(i);
    else
      missing_idxs.push_back(i);
  }

  std::sort(finite_idxs.begin(), finite_idxs.end(),
            [&](arma::uword a, arma::uword b) { return row(a) < row(b); });

  FiniteFirstSort out;
  out.first_non_finite_index = finite_idxs.size();
  out.order.set_size(n);
  arma::uword pos = 0;
  for (arma::uword idx : finite_idxs)
    out.order(pos++) = idx;
  for (arma::uword idx : missing_idxs)
    out.order(pos++) = idx;
  return out;
}

/** Partition with the largest count; smallest index on ties. */
inline size_t partition_with_max_count_min_index_tie(
    const std::vector<size_t> &counts) {
  if (counts.empty())
    return 0;
  size_t best = 0;
  for (size_t p = 1; p < counts.size(); ++p) {
    if (counts[p] > counts[best])
      best = p;
  }
  return best;
}

/** Pick the candidate with the lowest score; smallest index on ties. */
inline size_t pick_lowest_score_min_index_tie(const std::vector<double> &scores,
                                              double eps = 1e-12) {
  if (scores.empty())
    return 0;
  size_t best = 0;
  for (size_t p = 1; p < scores.size(); ++p) {
    if (scores[p] < scores[best] - eps ||
        (std::abs(scores[p] - scores[best]) <= eps && p < best))
      best = p;
  }
  return best;
}

} // namespace missing_values
