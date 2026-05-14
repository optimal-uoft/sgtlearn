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
 * routing metadata. Built inside ``ClassificationShapeGeneralizedTree::fit``.
 */
struct ShapeBranchingResult {
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
   * Per inner discretizer bin: class counts (same as
   * UnivariateDiscretizer::getLeafStats). Bin order matches ``binToPartition``
   * indices. Used to build child ``leafClassCounts`` without rescanning ``y``.
   */
  std::vector<std::vector<size_t>> leafStats;
};
