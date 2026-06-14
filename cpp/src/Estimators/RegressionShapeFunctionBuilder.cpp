/**
 * @file Estimators/RegressionShapeFunctionBuilder.cpp
 * @brief Per-node regression shape-function split search.
 */

#include "Estimators/RegressionShapeFunctionBuilder.h"

#include "Estimators/RegressionShapeGeneralizedTree.h"
#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "Discretizers/RegressionDiscretizer.h"
#include "algorithms/BinPartitionAssignments.h"
#include "algorithms/CoordinateDescent.h"
#include "algorithms/NanPartitionRouting.h"
#include "algorithms/missing_values.h"

#include <armadillo>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

RegressionShapeFunctionBuilder::RegressionShapeFunctionBuilder(
    RegressionShapeGeneralizedTree &tree, const arma::fmat &X,
    const arma::Row<float> &y, const arma::uvec &featureCandidates)
    : tree_(tree), X_(X), y_(y), featureCandidates_(featureCandidates) {}

RegressionShapeFunctionBuilder::BranchAssignmentSearchResult
RegressionShapeFunctionBuilder::searchBestBranchAssignment(
    size_t numBins, double parentImp,
    std::vector<std::vector<double>> &stats, const std::vector<size_t> &sizes,
    std::vector<double> &binWeights,
    std::vector<std::vector<float>> *maeLeafYs,
    std::vector<std::vector<float>> *maeLeafWs) {
  BranchAssignmentSearchResult result;
  const size_t kMax = std::min(numBins, tree_.numPartitions_);

  for (size_t k = 2; k <= kMax; ++k) {
    std::vector<size_t> trialAssignments;
    if (k == numBins)
      algorithms::identityBinAssignments(numBins, trialAssignments);
    else
      algorithms::roundRobinBinAssignments(numBins, k, trialAssignments);

    std::unique_ptr<BranchAssignment> branchObj;
    if (tree_.criterion_ == LearningCriterion::SquaredError) {
      branchObj = makeRegressionBranchAssignment(
          LearningCriterion::SquaredError, trialAssignments, k, stats,
          binWeights, sizes);
      if (k < numBins && tree_.cdParams_.smartInit) {
        const std::vector<size_t> snapshot = trialAssignments;
        const double objBeforeCd = branchObj->objective();
        coordinateDescent(k, *branchObj, tree_.rng_, tree_.cdParams_.maxIters,
                          tree_.cdParams_.patience);
        const double objAfterCd = branchObj->objective();
          const bool keepCd = std::isfinite(objAfterCd) &&
                              objAfterCd <
                                  objBeforeCd -
                                  ShapeFunctionBuilder::kCdObjectiveImprovementEps;
        if (!keepCd) {
          trialAssignments = snapshot;
          branchObj = makeRegressionBranchAssignment(
              LearningCriterion::SquaredError, trialAssignments, k, stats,
              binWeights, sizes);
        }
      }
    } else {
      if (!maeLeafYs || !maeLeafWs)
        continue;
      branchObj = makeRegressionBranchAssignment(
          LearningCriterion::AbsoluteError, trialAssignments, k, stats,
          binWeights, sizes, 1.0, maeLeafYs, maeLeafWs);
    }

    if (!branchObj->partitionCountsMeetMinLeaf(tree_.outerParams_.minLeafSize))
      continue;

    const double childImp = branchObj->objective();
    const double gain = parentImp - childImp;
    if (gain < tree_.outerParams_.minGainSplit - tree_.outerTreeBuilder_.eps)
      continue;

    const double score = algorithms::penalizedBranchingScore(
        childImp, k, tree_.outerParams_.branchingPenalty);
    if (score < result.bestFeatureScore - tree_.outerTreeBuilder_.eps) {
      result.bestFeatureScore = score;
      result.chosenK = k;
      result.assignments = std::move(trialAssignments);
      result.partitionSampleCounts = branchObj->partitionSampleCounts();
      result.impurityDecrease = gain;
      result.found = true;
    }
  }

  return result;
}

bool RegressionShapeFunctionBuilder::adoptFeatureBranchIfBetter(
    double bestFeatureScore, double &bestPenalizedChild,
    ShapeBranchingResult<double> &brBest, std::vector<size_t> &binSizesForBest,
    std::vector<double> &binWeightsForBest, size_t featureIndex,
    const BranchAssignmentSearchResult &featureBest, size_t numBins,
    size_t xSubCols, RegressionDiscretizer &disc,
    const std::vector<std::vector<double>> &stats,
    const std::vector<size_t> &sizesCopy,
    const std::vector<double> &binWeights) {
  if (bestFeatureScore >= bestPenalizedChild - tree_.outerTreeBuilder_.eps)
    return false;

  bestPenalizedChild = bestFeatureScore;
  brBest.featureIndex = featureIndex;
  const auto &dth = disc.thresholds();
  brBest.innerThresholds.resize(dth.size());
  for (size_t t = 0; t < dth.size(); ++t)
    brBest.innerThresholds[t] = static_cast<float>(dth[t]);
  brBest.binToPartition = featureBest.assignments;
  brBest.impurityDecrease = featureBest.impurityDecrease;
  brBest.numPartitionsUsed = featureBest.chosenK;
  brBest.partitionSampleCounts = featureBest.partitionSampleCounts;
  fillSampleBinsFromDiscretizer(xSubCols, disc.inSampleDiscretizations(),
                                brBest.sampleBins);
  binSizesForBest.assign(sizesCopy.begin(), sizesCopy.end());
  binWeightsForBest.assign(binWeights.begin(), binWeights.end());

  if (tree_.criterion_ == LearningCriterion::SquaredError) {
    brBest.leafStats = stats;
  } else {
    brBest.leafStats.clear();
    brBest.leafStats.resize(numBins);
  }
  return true;
}

size_t RegressionShapeFunctionBuilder::chooseNanPredictionPartition(
    const ShapeFunctionNode &node,
    const std::vector<size_t> &partitionSampleCounts, const arma::fmat &Xsub,
    const arma::Row<float> &ysub, const arma::uvec &subIdx) const {
  const arma::Row<float> wsub = subSampleWeights(tree_.fitSampleWeights_, subIdx);
  const arma::frowvec featRow = Xsub.row(node.routingFeature);
  const auto missingCols =
      nan_partition_routing::missing_column_indices(featRow);
  if (missingCols.empty()) {
    return missing_values::partition_with_max_count_min_index_tie(
        partitionSampleCounts);
  }
  if (tree_.criterion_ == LearningCriterion::SquaredError) {
    return nan_partition_routing::choose_nan_partition_squared_error(
        node.numPartitions, node.binToPartition, node.splitLeafStats,
        node.splitBinWeights, missingCols, ysub, wsub);
  }
  return nan_partition_routing::choose_nan_partition_absolute_error(
      node.numPartitions, featRow, node.sampleBins, node.binToPartition,
      missingCols, ysub, wsub);
}

bool RegressionShapeFunctionBuilder::findBestSplit(ShapeFunctionNode &node,
                                                   size_t minLeaf) {
  const size_t ns = node.sampleIndices.n_elem;
  node.score = tree_.impurityAtNode(y_, node);

  if (ns < 2 * minLeaf) {
    markLeafNoSplit(node);
    return false;
  }

  const double parentImp = node.score;
  if (parentImp <= tree_.outerTreeBuilder_.eps) {
    markLeafNoSplit(node);
    return false;
  }

  const arma::uvec &subIdx = node.sampleIndices;
  const arma::fmat Xsub = X_.cols(subIdx);
  const arma::Row<float> ysub = y_.cols(subIdx);

  std::vector<size_t> featurePool(
      static_cast<size_t>(featureCandidates_.n_elem));
  for (arma::uword i = 0; i < featureCandidates_.n_elem; ++i)
    featurePool[static_cast<size_t>(i)] =
        static_cast<size_t>(featureCandidates_(i));
  const std::vector<size_t> featureSubset = tree_.featureBagging_(
      std::span<const size_t>(featurePool.data(), featurePool.size()),
      tree_.rng_);

  const size_t xSubCols = static_cast<size_t>(Xsub.n_cols);
  double bestPenalizedChild = std::numeric_limits<double>::infinity();
  ShapeBranchingResult<double> brBest{};
  std::vector<size_t> binSizesForBest;
  std::vector<double> binWeightsForBest;
  arma::uvec featOne(1);

  for (size_t fi = 0; fi < featureSubset.size(); ++fi) {
    const size_t f = featureSubset[fi];
    if (f >= Xsub.n_rows)
      throw std::invalid_argument(
          "RegressionShapeFunctionBuilder::findBestSplit: candidate feature "
          "index >= X.n_rows");
    featOne(0) = static_cast<arma::uword>(f);

    const arma::Row<float> wsub =
        subSampleWeights(tree_.fitSampleWeights_, subIdx);

    auto disc = makeRegressionDiscretizer(tree_.criterion_);
    disc->Train(Xsub, featOne, ysub, tree_.innerParams_.minLeafSize,
                tree_.innerParams_.minGainSplit, tree_.innerParams_.maxDepth,
                tree_.innerParams_.maxLeafNodes, wsub);
    const size_t B = disc->numLeaves();
    if (B < 2)
      continue;

    auto &stats = disc->leafStats();
    auto &sizes = disc->leafNumSamples();
    auto &binWeights = disc->leafNodeWeights();
    const auto sizes_copy = sizes;
    const auto &perBinCols = disc->inSampleDiscretizations();

    std::vector<std::vector<float>> maeLeafYsStorage;
    std::vector<std::vector<float>> maeLeafWsStorage;
    std::vector<std::vector<float>> *maeLeafYsPtr = nullptr;
    std::vector<std::vector<float>> *maeLeafWsPtr = nullptr;
    if (tree_.criterion_ == LearningCriterion::AbsoluteError) {
      maeLeafYsStorage.resize(B);
      maeLeafWsStorage.resize(B);
      for (size_t b = 0; b < B; ++b) {
        for (size_t colIdx : perBinCols[b]) {
          if (colIdx >= xSubCols)
            throw std::runtime_error("RegressionShapeFunctionBuilder: "
                                     "discretizer sample index "
                                     ">= Xsub columns");
          maeLeafYsStorage[b].push_back(ysub(colIdx));
          maeLeafWsStorage[b].push_back(wsub(colIdx));
        }
      }
      maeLeafYsPtr = &maeLeafYsStorage;
      maeLeafWsPtr = &maeLeafWsStorage;
    }

    const BranchAssignmentSearchResult featureBest =
        searchBestBranchAssignment(B, parentImp, stats, sizes, binWeights,
                                   maeLeafYsPtr, maeLeafWsPtr);
    if (!featureBest.found)
      continue;

    adoptFeatureBranchIfBetter(featureBest.bestFeatureScore, bestPenalizedChild,
                               brBest, binSizesForBest, binWeightsForBest, f,
                               featureBest, B, xSubCols, *disc, stats,
                               sizes_copy, binWeights);
  }

  if (!std::isfinite(bestPenalizedChild) ||
      bestPenalizedChild >= std::numeric_limits<double>::infinity() ||
      brBest.impurityDecrease <= tree_.outerTreeBuilder_.eps) {
    markLeafNoSplit(node);
    return false;
  }

  node.isLeaf = false;
  node.routingFeature = brBest.featureIndex;
  node.innerThresholds = std::move(brBest.innerThresholds);
  node.binToPartition = std::move(brBest.binToPartition);
  node.sampleBins = std::move(brBest.sampleBins);
  node.numPartitions = brBest.numPartitionsUsed;
  node.informationGain = brBest.impurityDecrease;

  if (tree_.criterion_ == LearningCriterion::SquaredError)
    node.splitLeafStats = std::move(brBest.leafStats);
  else
    node.splitLeafStats.clear();
  node.splitBinWeights = std::move(binWeightsForBest);
  node.binSampleCounts = std::move(binSizesForBest);
  node.nanPredictionPartition = chooseNanPredictionPartition(
      node, brBest.partitionSampleCounts, Xsub, ysub, subIdx);

  return true;
}
