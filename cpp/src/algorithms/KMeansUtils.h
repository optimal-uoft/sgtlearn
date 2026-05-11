#pragma once

#include <armadillo>
#include <algorithm>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

namespace algorithms {

/**
 * Weighted k-means assignments for row vectors.
 *
 * @param X           (nPoints, dim) each row is a point.
 * @param w           (nPoints,) non-negative weights (used in centroid updates).
 * @param k           number of clusters.
 * @param seed        RNG seed for empty-cluster reseeding.
 * @param assignments output labels in [0, k), length nPoints.
 */
inline void initAssignmentsWeightedKMeans(const arma::mat &X, const arma::vec &w,
                                          size_t k, uint64_t seed,
                                          std::vector<size_t> &assignments) {
  const size_t B = X.n_rows;
  const size_t C = X.n_cols;
  assignments.resize(B);
  if (B == 0 || k == 0)
    return;
  if (B < k) {
    for (size_t b = 0; b < B; ++b)
      assignments[b] = b % k;
    return;
  }

  arma::mat centroids(k, C);
  for (size_t j = 0; j < k; ++j) {
    const size_t idx = (j * B) / k;
    centroids.row(j) = X.row(idx);
  }

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> pickBin(0, B - 1);

  for (size_t iter = 0; iter < 50; ++iter) {
    bool changed = false;
    for (size_t b = 0; b < B; ++b) {
      double bestD = std::numeric_limits<double>::infinity();
      size_t bestJ = 0;
      for (size_t j = 0; j < k; ++j) {
        double d = 0.0;
        for (size_t c = 0; c < C; ++c) {
          const double diff = X(b, c) - centroids(j, c);
          d += diff * diff;
        }
        if (d < bestD) {
          bestD = d;
          bestJ = j;
        }
      }
      if (assignments[b] != bestJ) {
        assignments[b] = bestJ;
        changed = true;
      }
    }

    centroids.zeros();
    arma::vec sumW(k, arma::fill::zeros);
    for (size_t b = 0; b < B; ++b) {
      const size_t j = assignments[b];
      for (size_t c = 0; c < C; ++c)
        centroids(j, c) += w(b) * X(b, c);
      sumW(j) += w(b);
    }
    for (size_t j = 0; j < k; ++j) {
      if (sumW(j) > 1e-12)
        centroids.row(j) /= sumW(j);
      else
        centroids.row(j) = X.row(pickBin(rng));
    }

    if (!changed && iter > 0)
      break;
  }
}

} // namespace algorithms
