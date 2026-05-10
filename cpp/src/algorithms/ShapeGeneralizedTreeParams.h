#pragma once

/**
 * @file ShapeGeneralizedTreeParams.h
 * @brief Hyperparameter structs shared by shape-generalized outer and inner tree trainers.
 */

#include <cstddef>

/**
 * Hyperparameters and small POD bundles for shape-generalized trees.
 *
 * Kept separate from task-specific trainers so classification and future
 * regression variants can share the same tuning surface without pulling in
 * label-type-specific code.
 */

/**
 * Standard tree-building hyperparameters. Shared by the outer routing tree
 * and the inner per-feature univariate discretizer.
 *
 * - minLeafSize:  minimum samples in a node for it to remain split-eligible.
 * - minGainSplit: minimum impurity reduction required to commit a split.
 * - maxDepth:     0 = unlimited; otherwise expansion stops at this depth.
 * - maxLeafNodes: 0 = depth-first / unlimited; otherwise best-first growth up
 *                 to this many leaves (Heap frontier in TreeBuilder).
 */
struct TreeBuildingParams {
  size_t minLeafSize = 1;
  double minGainSplit = 1e-7;
  size_t maxDepth = 0;
  size_t maxLeafNodes = 0;
  /** Added to effective child impurity when ranking splits (Python branching_penalty * (k-1)). */
  double branchingPenalty = 0.0;
};

/**
 * Coordinate-descent hyperparameters for the bin->partition assignment loop
 * (see algorithms/CoordinateDescent.h).
 */
struct CoordinateDescentParams {
  size_t maxIters = 10;
  size_t patience = 5;
  size_t seed = 42;
  /**
   * If true (default), initialize bin->partition assignments with weighted
   * k-means (k = numPartitions) on per-bin normalized class counts before
   * coordinate descent; otherwise use round-robin.
   */
  bool smartInit = true;
};
