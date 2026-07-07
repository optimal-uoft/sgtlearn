#pragma once

/**
 * @file Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h
 * @brief Shared helpers for per-node shape-function split search.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "Discretizers/InnerDiscretizerBase.h"
#include "algorithms/ShapeBranchingTypes.h"

#include <armadillo>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <vector>

/** Minimum objective improvement required to keep a coordinate-descent result. */
inline constexpr double kShapeFunctionCdImprovementEps = 1e-10;

struct ShapeBranchAssignmentSearchResult {
  double bestFeatureScore = std::numeric_limits<double>::infinity();
  size_t chosenK = 0;
  std::vector<size_t> assignments;
  std::vector<size_t> partitionSampleCounts;
  std::vector<std::vector<double>> partitionClassCounts;
  std::vector<double> partitionWeights;
  double impurityDecrease = 0.0;
  bool found = false;
};

struct ShapeBestBranchingState {
  double penalizedChildScore = std::numeric_limits<double>::infinity();
  ShapeBranchingResult<double> branching;
  std::vector<double> binWeights;
  std::vector<std::vector<double>> partitionClassCounts;
  std::vector<double> partitionWeights;
  std::shared_ptr<const InnerDiscretizerBase<double>> winningDiscretizer;
};

void markShapeFunctionNodeAsLeaf(ShapeFunctionNode &node);

void fillSampleBinsFromDiscretizer(
    size_t xSubCols, const std::vector<std::vector<size_t>> &perBinCols,
    std::vector<size_t> &sampleBins);

arma::Row<float> subSampleWeights(const arma::Row<float> &weights,
                                  const arma::uvec &subIdx);

void applySharedShapeBranchingFields(
    ShapeBestBranchingState &best, const ShapeBranchAssignmentSearchResult &search,
    size_t featureIndex, size_t xSubCols,
    const std::vector<std::vector<size_t>> &perBinCols);

bool featureHasBetterShapeBranching(
    const ShapeBranchAssignmentSearchResult &search,
    ShapeBestBranchingState &best, size_t featureIndex, size_t xSubCols,
    std::unique_ptr<InnerDiscretizerBase<double>> disc, double scoreEpsilon,
    const std::function<void(ShapeBestBranchingState &,
                             const ShapeBranchAssignmentSearchResult &,
                             const std::vector<std::vector<double>> &)> &
        applyTaskFields);

std::vector<std::vector<size_t>>
routeSamplesToPartitions(const ShapeFunctionNode &parent, const arma::fmat &X);

std::vector<ShapeFunctionNode>
makeRoutedChildNodes(const ShapeFunctionNode &parent,
                     const std::vector<std::vector<size_t>> &buckets,
                     size_t treeNumPartitions);
