#pragma once

#include "Domain/LearningCriterion.h"

#include <armadillo>
#include <cstddef>
#include <optional>
#include <vector>

struct TreeBuildingParams;
struct CoordinateDescentParams;

/**
 * Output of the inner "branching" step (Python `BranchingTree` / construct_mapping
 * for the univariate case): chosen axis, thresholds, and bin -> child partition.
 */
struct ShapeBranchingResult {
  size_t featureIndex = 0;
  std::vector<float> innerThresholds;
  std::vector<size_t> binToPartition;
  /** parentImpurity - weightedChildImpurity */
  double impurityDecrease = 0.0;
};

/**
 * Try all candidate features: discretizer + coordinate descent per feature;
 * return the best split that satisfies outer min leaf per partition and min
 * impurity decrease. `branchingPenalty` is added to the child impurity score when
 * comparing candidates (higher penalty => prefer simpler routing), matching
 * Python's score = impurity + penalty * (k - 1) for fixed k as a constant offset.
 */
std::optional<ShapeBranchingResult> fitShapeBranch(
    LearningCriterion criterion, size_t numClasses, size_t numPartitions,
    const TreeBuildingParams &innerParams,
    const CoordinateDescentParams &cdParams, double branchingPenalty,
    const arma::fmat &Xsub, const arma::Row<size_t> &ysub,
    const arma::uvec &features, double parentImpurity,
    double outerMinImpurityDecrease, size_t outerMinSamplesLeaf,
    double comparatorEps);
