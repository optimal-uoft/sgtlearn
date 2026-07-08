#pragma once

/**
 * @file Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h
 * @brief Shared helpers for per-node shape-function split search.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "Discretizers/InnerDiscretizerBase.h"
#include "Domain/LearningCriterion.h"
#include "algorithms/ShapeBranchingTypes.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"

#include <armadillo>
#include <cstddef>
#include <functional>
#include <limits>
#include <memory>
#include <random>
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
  /** Column indices into ``X`` used for routing at inference. */
  arma::uvec routingColumnIndices;
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
    const arma::uvec &routingColumnIndices,
    std::unique_ptr<InnerDiscretizerBase<double>> disc, double scoreEpsilon,
    const std::function<void(ShapeBestBranchingState &,
                             const ShapeBranchAssignmentSearchResult &,
                             const std::vector<std::vector<double>> &)> &
        applyTaskFields);

/**
 * Search partition counts k in [2, min(numBins, treeNumPartitions)] on a trained
 * inner discretizer and return the best penalized branch assignment.
 *
 * For AbsoluteError, pass @p ysub and @p wsub (node-local targets and weights)
 * so per-bin raw samples can be built for MAE branch assignment.
 *
 * @param useKMeansSeed  when true and smartInit is enabled, seed assignments
 *                       with k-means on bin stats (classification); otherwise
 *                       round-robin.
 * @param numClasses     required when @p useKMeansSeed is true.
 */
ShapeBranchAssignmentSearchResult searchShapeBranchAssignmentFromDiscretizer(
    InnerDiscretizerBase<double> &disc, LearningCriterion criterion,
    double parentImp, size_t treeNumPartitions,
    const TreeBuildingParams &outerParams,
    const CoordinateDescentParams &cdParams, double scoreEpsilon,
    std::mt19937_64 &rng, bool useKMeansSeed = false, size_t numClasses = 0,
    const arma::Row<float> *ysub = nullptr, const arma::Row<float> *wsub = nullptr,
    size_t xSubCols = 0);

std::vector<std::vector<size_t>>
routeSamplesToPartitions(const ShapeFunctionNode &parent, const arma::fmat &X);

std::vector<ShapeFunctionNode>
makeRoutedChildNodes(const ShapeFunctionNode &parent,
                     const std::vector<std::vector<size_t>> &buckets,
                     size_t treeNumPartitions);
