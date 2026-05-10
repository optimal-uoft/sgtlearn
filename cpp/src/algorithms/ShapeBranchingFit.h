#pragma once

/**
 * @file ShapeBranchingFit.h
 * @brief Entry point: score every candidate routing feature via discretizer + coordinate descent.
 */

#include "Domain/LearningCriterion.h"
#include "algorithms/ShapeBranchingTypes.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"

#include <armadillo>
#include <cstddef>
#include <optional>
#include <vector>

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
