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
  /** Classification: nested ``[partition][output][class]``. */
  std::vector<std::vector<std::vector<double>>> partitionClassCounts;
  /** Regression MSE: nested ``[partition][output][Σw·y, Σw·y²]``. */
  std::vector<std::vector<std::vector<double>>> partitionAggStats;
  std::vector<double> partitionWeights;
  double impurityDecrease = 0.0;
  bool found = false;
};

struct ShapeBestBranchingState {
  double penalizedChildScore = std::numeric_limits<double>::infinity();
  ShapeBranchingResult<std::vector<double>> branching;
  std::vector<double> binWeights;
  std::vector<std::vector<std::vector<double>>> partitionClassCounts;
  std::vector<std::vector<std::vector<double>>> nestedLeafStats;
  std::vector<double> partitionWeights;
  std::shared_ptr<const InnerDiscretizerBase> winningDiscretizer;
  /** Column indices into ``X`` used for routing at inference. */
  arma::uvec routingColumnIndices;
  std::vector<size_t> logicalFeatureIndices;
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
    std::unique_ptr<InnerDiscretizer<std::vector<double>>> disc,
    double scoreEpsilon,
    const std::function<void(
        ShapeBestBranchingState &, const ShapeBranchAssignmentSearchResult &,
        const std::vector<std::vector<std::vector<double>>> &)> &
        applyTaskFields);

/**
 * Search partition counts k in [2, min(numBins, treeNumPartitions)] on a trained
 * inner discretizer and return the best penalized branch assignment.
 *
 * Leaf stats are nested ``[bin][output][*]`` (class counts or MSE moments).
 */
ShapeBranchAssignmentSearchResult searchShapeBranchAssignmentFromDiscretizer(
    InnerDiscretizer<std::vector<double>> &disc, LearningCriterion criterion,
    double parentImp, size_t treeNumPartitions,
    const TreeBuildingParams &outerParams,
    const CoordinateDescentParams &cdParams, double scoreEpsilon,
    std::mt19937_64 &rng, bool useKMeansSeed = false,
    const std::vector<size_t> &classesPerOutput = {}, size_t nOutputs = 1,
    const arma::Mat<float> *ysub = nullptr, const arma::Row<float> *wsub = nullptr,
    size_t xSubCols = 0, bool hasNanRoutingBin = true);

std::vector<std::vector<size_t>>
routeSamplesToPartitions(const ShapeFunctionNode &parent, const arma::fmat &X);

std::vector<ShapeFunctionNode>
makeRoutedChildNodes(const ShapeFunctionNode &parent,
                     const std::vector<std::vector<size_t>> &buckets,
                     size_t treeNumPartitions);
