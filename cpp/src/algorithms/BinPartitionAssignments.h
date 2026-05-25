#pragma once

/**
 * @file BinPartitionAssignments.h
 * @brief Seed mappings from inner-tree bins to outer partitions (round-robin,
 *        identity) and lightweight feasibility checks.
 */

#include <cstddef>
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

/** Every partition in ``[0, k)`` must hold at least ``minLeafSize`` samples. */
inline bool partitionCountsMeetMinLeaf(const std::vector<size_t> &assignments,
                                     const std::vector<size_t> &binSizes,
                                     size_t k, size_t minLeafSize) {
  std::vector<size_t> wt(k, 0);
  for (size_t b = 0; b < assignments.size(); ++b) {
    const size_t p = assignments[b];
    if (p >= k)
      return false;
    wt[p] += binSizes[b];
  }
  for (size_t p = 0; p < k; ++p) {
    if (wt[p] < minLeafSize)
      return false;
  }
  return true;
}

/** Python ``construct_mapping`` score: ``impurity + branchingPenalty * (k - 1)``. */
inline double penalizedBranchingScore(double childImpurity, size_t k,
                                      double branchingPenalty) {
  return childImpurity +
         branchingPenalty * static_cast<double>(k > 0 ? k - 1 : 0);
}

} // namespace algorithms
