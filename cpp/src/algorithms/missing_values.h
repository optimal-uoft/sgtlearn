#pragma once

/**
 * @file algorithms/missing_values.h
 * @brief Non-finite feature handling for discretizers and tree routing.
 *
 * Training sorts finite values first and leaves NaN/inf at the tail so
 * Armadillo never sees NaN in ``sort_index``. Inference routes non-finite
 * values to the last bin (the missing tail).
 */

#include <armadillo>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iterator>
#include <vector>

namespace missing_values {

inline bool is_finite(float value) {
  return std::isfinite(static_cast<double>(value));
}

/** Sort sample indices: ascending finite values, then non-finite tail. */
inline arma::uvec sort_index_finite_first(const arma::frowvec &row) {
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

  arma::uvec order(n);
  arma::uword pos = 0;
  for (arma::uword idx : finite_idxs)
    order(pos++) = idx;
  for (arma::uword idx : missing_idxs)
    order(pos++) = idx;
  return order;
}

/** Bin index for ``value`` against sorted cut points (``lower_bound``). */
template <typename ThresholdIter>
size_t discretizer_bin(float value, ThresholdIter begin, ThresholdIter end) {
  if (!is_finite(value))
    return static_cast<size_t>(std::distance(begin, end));
  const auto it = std::lower_bound(begin, end, value);
  return static_cast<size_t>(std::distance(begin, it));
}

} // namespace missing_values
