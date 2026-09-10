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
 * - minGainSplit: constant outer growth cost; inner CART retains its threshold.
 * - maxDepth:     0 = unlimited; otherwise expansion stops at this depth.
 * - maxLeafNodes: 0 = unlimited (outer growth remains best-first); inner CART
 *                 uses depth-first without a finite leaf bound.
 */
struct TreeBuildingParams {
  size_t minLeafSize = 1;
  double minGainSplit = 1e-7;
  size_t maxDepth = 0;
  size_t maxLeafNodes = 0;
  /** Constant outer cost for each occupied child beyond binary: lambda*(k-2). */
  double branchingPenalty = 0.0;
};

/**
 * Coordinate-descent hyperparameters for the bin->partition assignment loop
 * (see algorithms/CoordinateDescent.h).
 */
struct CoordinateDescentParams {
  size_t maxIters = 10;
  size_t patience = 5;
};
