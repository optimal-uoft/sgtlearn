#pragma once

/**
 * @file ShapeBranchingTypes.h
 * @brief Result struct for inner branching (axis + thresholds + bin-to-partition map).
 */

#include <cstddef>
#include <vector>

/**
 * Payload from one inner branch attempt at a node: chosen feature, inner
 * thresholds, bin-to-partition map after coordinate descent, and training
 * routing metadata.
 *
 * @tparam LeafStat  Per-sample or per-bin scalar type stored in ``leafStats``
 *                   rows (e.g. ``size_t`` for class counts, ``float`` for
 *                   regression ``[sum y, sum y^2]`` aggregates).
 */
template <typename LeafStat> struct ShapeBranchingResult {
  size_t featureIndex = 0;
  std::vector<float> innerThresholds;
  std::vector<size_t> binToPartition;
  /** parentImpurity - weightedChildImpurity */
  double impurityDecrease = 0.0;
  /**
   * For each column of the subsampled X matrix (same order as the node's
   * sampleIndices): inner discretizer bin index. Matches
   * UnivariateDiscretizer::getInSampleDiscretizations.
   */
  std::vector<size_t> sampleBins;
  /**
   * Per inner discretizer bin: leaf-side statistics (same row count as
   * ``binToPartition``). Classification uses class counts; regression squared
   * error uses ``[sum y, sum y^2]`` per bin.
   */
  std::vector<std::vector<LeafStat>> leafStats;
  /** Per inner bin: unweighted sample count ``N`` (for ``min_samples_leaf`` checks). */
  std::vector<size_t> leafNumSamples;
  /** Fan-out chosen for this split (``2`` .. ``numPartitions`` cap). */
  size_t numPartitionsUsed = 0;
  /** Unweighted sample count per child partition after coordinate descent. */
  std::vector<size_t> partitionSampleCounts;
};
