#pragma once

/**
 * @file BinPartitionAssignments.h
 * @brief Seed mappings from inner-tree bins to outer partitions (round-robin,
 *        identity, k-means) and lightweight feasibility checks.
 */

#include "algorithms/KMeansUtils.h"

#include <armadillo>
#include <algorithm>
#include <cstddef>
#include <random>
#include <vector>

namespace algorithms {

/** ``assignments[b] = b % k`` for ``b in [0, numBins)``. */
inline void roundRobinBinAssignments(size_t numBins, size_t k,
                                     std::vector<size_t> &assignments) {
  assignments.resize(numBins);
  for (size_t b = 0; b < numBins; ++b)
    assignments[b] = b % k;
}

/** One bin per partition: ``assignments[b] = b`` (``k == numBins``). */
inline void identityBinAssignments(size_t numBins,
                                   std::vector<size_t> &assignments) {
  assignments.resize(numBins);
  for (size_t b = 0; b < numBins; ++b)
    assignments[b] = b;
}

/**
 * K-means seed for bin->partition assignment from per-bin class histograms.
 *
 * Each bin is represented by its normalized class-count vector; weighted
 * k-means assigns bins to ``k`` partitions.
 */
inline void seedBinAssignmentsKMeans(
    size_t k, size_t numBins, size_t numClasses,
    const std::vector<std::vector<double>> &binClassCounts,
    const std::vector<size_t> &binSizes, const std::vector<double> &binWeights,
    std::mt19937_64 &rng, std::vector<size_t> &assignments) {
  arma::mat Xk(numBins, numClasses);
  arma::vec wk(numBins);
  for (size_t b = 0; b < numBins; ++b) {
    wk(b) = b < binWeights.size()
                ? std::max(1.0, binWeights[b])
                : std::max(1.0, static_cast<double>(binSizes[b]));
    double sum = 0.0;
    for (size_t c = 0; c < numClasses; ++c)
      sum += binClassCounts[b][c];
    if (sum <= 0.0) {
      Xk.row(b).fill(1.0 / static_cast<double>(numClasses));
    } else {
      for (size_t c = 0; c < numClasses; ++c)
        Xk(b, c) = static_cast<double>(binClassCounts[b][c]) / sum;
    }
  }
  initAssignmentsWeightedKMeans(Xk, wk, k, rng, assignments);
}

/** score: ``impurity + branchingPenalty * (k - 1)``. */
inline double penalizedBranchingScore(double childImpurity, size_t k,
                                      double branchingPenalty) {
  return childImpurity +
         branchingPenalty * static_cast<double>(k > 0 ? k - 1 : 0);
}

} // namespace algorithms
