#pragma once

/**
 * @file ShapeBranchingTypes.h
 * @brief Result struct for inner branching (axis + thresholds + bin-to-partition map).
 */

#include <cstddef>
#include <vector>

/**
 * Result of one inner "shape branching" fit: pick an axis, thresholds, and a
 * bin -> child-partition map (Python BranchingTree / construct_mapping for the
 * univariate case).
 */
struct ShapeBranchingResult {
  size_t featureIndex = 0;
  std::vector<float> innerThresholds;
  std::vector<size_t> binToPartition;
  /** parentImpurity - weightedChildImpurity */
  double impurityDecrease = 0.0;
};
