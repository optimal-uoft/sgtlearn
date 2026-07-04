/**
 * @file Estimators/ShapeFunctions/RegressionShapeFunctionBuilder.cpp
 * @brief Per-node regression shape-function split search.
 */

#include "Estimators/ShapeFunctions/RegressionShapeFunctionBuilder.h"

#include "Estimators/RegressionShapeGeneralizedTree.h"
#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "Criterion.h"
#include "Discretizers/RegressionDiscretizer.h"
#include "algorithms/BinPartitionAssignments.h"
#include "algorithms/CoordinateDescent.h"
#include "Estimators/ShapeFunctions/NanPartitionRouting.h"
#include "algorithms/missing_values.h"

#include <armadillo>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

double meanAbsoluteDeviationFromMedian(const std::vector<float> &ys,
                                       const std::vector<float> &ws) {
  return Criterion::absoluteError(ys, ws).mae;
}

double weightedMeanFromAggregates(double sumWY, double sumW) {
  return sumW > 0.0 ? sumWY / sumW : 0.0;
}

struct PartitionMoments {
  double sumWY = 0.0;
  double sumWY2 = 0.0;
  double sumW = 0.0;
};

PartitionMoments aggregatePartitionFromBins(
    const std::vector<std::vector<double>> &binStats,
    const std::vector<double> &binWeights,
    const std::vector<size_t> &binToPartition, size_t partition) {
  PartitionMoments out;
  for (size_t b = 0; b < binStats.size(); ++b) {
    if (binToPartition[b] != partition)
      continue;
    if (binStats[b].size() >= 2) {
      out.sumWY += binStats[b][0];
      out.sumWY2 += binStats[b][1];
    }
    if (b < binWeights.size())
      out.sumW += binWeights[b];
  }
  return out;
}

} // namespace

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

void RegressionShapeFunctionBuilder::applyTaskBranchingFields(
    BestBranchingState &best, const BranchAssignmentSearchResult &search,
    const std::vector<std::vector<double>> &leafStats) {
  (void)search;
  if (tree_.criterion_ == LearningCriterion::SquaredError) {
    best.branching.leafStats = leafStats;
  } else {
    best.branching.leafStats.clear();
    best.branching.leafStats.resize(leafStats.size());
  }
}

void RegressionShapeFunctionBuilder::assignNanPredictionPartition(
    ShapeFunctionNode &node,
    const std::vector<size_t> &partitionSampleCounts, bool nanSeen,
    const std::vector<double> &nanStats, double nanWeight,
    const std::vector<size_t> &nanInSampleIndices, const arma::fmat &Xsub,
    const arma::Row<float> &ysub, const arma::uvec &subIdx) const {
  node.splitMissingStats.clear();
  node.splitMissingWeight = 0.0;

  // No NaN observed for this feature during training: route NaN at inference to
  // the largest child partition (smallest index on ties).
  if (!nanSeen) {
    node.nanPredictionPartition =
        missing_values::partition_with_max_count_min_index_tie(
            partitionSampleCounts);
    return;
  }

  if (tree_.criterion_ == LearningCriterion::SquaredError) {
    const double missingSumWY = nanStats.size() > 0 ? nanStats[0] : 0.0;
    const double missingSumWY2 = nanStats.size() > 1 ? nanStats[1] : 0.0;
    node.nanPredictionPartition =
        nan_partition_routing::choose_nan_partition_squared_error_from_moments(
            node.numPartitions, node.binToPartition, node.splitLeafStats,
            node.splitBinWeights, missingSumWY, missingSumWY2, nanWeight);
    node.splitMissingStats = {missingSumWY, missingSumWY2};
    node.splitMissingWeight = nanWeight;
    return;
  }

  // Absolute error: the NaN bucket needs the raw targets to recompute medians.
  const arma::Row<float> wsub = subSampleWeights(tree_.fitSampleWeights_, subIdx);
  const arma::frowvec featRow = Xsub.row(node.routingFeatures.front());
  node.nanPredictionPartition =
      nan_partition_routing::choose_nan_partition_absolute_error(
          node.numPartitions, featRow, node.sampleBins, node.binToPartition,
          nanInSampleIndices, ysub, wsub);
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
  BestBranchingState best{};
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

    featureHasBetterBranching(featureBest, best, f, xSubCols, std::move(disc),
                              tree_.outerTreeBuilder_.eps);
  }

  if (!std::isfinite(best.penalizedChildScore) ||
      best.penalizedChildScore >= std::numeric_limits<double>::infinity() ||
      best.branching.impurityDecrease <= tree_.outerTreeBuilder_.eps) {
    markLeafNoSplit(node);
    return false;
  }

  node.isLeaf = false;
  node.routingFeatures = {best.branching.featureIndex};
  node.innerDiscretizer = best.winningDiscretizer;
  node.binToPartition = std::move(best.branching.binToPartition);
  node.sampleBins = std::move(best.branching.sampleBins);
  node.numPartitions = best.branching.numPartitionsUsed;
  node.informationGain = best.branching.impurityDecrease;

  if (tree_.criterion_ == LearningCriterion::SquaredError)
    node.splitLeafStats = std::move(best.branching.leafStats);
  else
    node.splitLeafStats.clear();
  node.splitBinWeights = std::move(best.binWeights);
  node.binSampleCounts = std::move(best.branching.leafNumSamples);
  assignNanPredictionPartition(node, best.branching.partitionSampleCounts,
                               best.nanSeen, best.nanStats, best.nanNodeWeight,
                               best.nanInSampleIndices, Xsub, ysub, subIdx);

  return true;
}

std::vector<ShapeFunctionNode> RegressionShapeFunctionBuilder::makeChildren(
    const ShapeFunctionNode &parent) {
  const auto buckets = routeSamplesToPartitions(parent, X_);
  auto children = makeRoutedChildNodes(parent, buckets, tree_.numPartitions_);

  if (tree_.criterion_ == LearningCriterion::SquaredError) {
    if (parent.splitLeafStats.size() != parent.splitBinWeights.size())
      throw std::runtime_error(
          "RegressionShapeFunctionBuilder::makeChildren: splitBinWeights / "
          "splitLeafStats size mismatch");

    for (size_t p = 0; p < children.size(); ++p) {
      PartitionMoments moments = aggregatePartitionFromBins(
          parent.splitLeafStats, parent.splitBinWeights,
          parent.binToPartition, p);
      if (p == parent.nanPredictionPartition &&
          parent.splitMissingStats.size() >= 2) {
        moments.sumWY += parent.splitMissingStats[0];
        moments.sumWY2 += parent.splitMissingStats[1];
        moments.sumW += parent.splitMissingWeight;
      }
      const std::vector<double> agg{moments.sumWY, moments.sumWY2};
      children[p].score = Criterion::squaredError(agg, moments.sumW);
      const std::vector<float> aggF{static_cast<float>(moments.sumWY),
                                    static_cast<float>(moments.sumWY2)};
      children[p].isLeaf = true;
      tree_.leafRegressionStats.push_back(aggF);
      tree_.leafNumSamples.push_back(children[p].sampleIndices.n_elem);
      tree_.leafPredictions_.push_back(
          weightedMeanFromAggregates(moments.sumWY, moments.sumW));
    }
  } else {
    for (size_t p = 0; p < children.size(); ++p) {
      std::vector<float> ys;
      std::vector<float> ws;
      ys.reserve(children[p].sampleIndices.n_elem);
      ws.reserve(children[p].sampleIndices.n_elem);
      for (arma::uword j = 0; j < children[p].sampleIndices.n_elem; ++j) {
        const size_t si = static_cast<size_t>(children[p].sampleIndices(j));
        ys.push_back(y_(static_cast<arma::uword>(si)));
        ws.push_back(static_cast<float>(tree_.fitSampleWeights_(si)));
      }
      children[p].score = meanAbsoluteDeviationFromMedian(ys, ws);
      children[p].isLeaf = true;
      tree_.leafRegressionStats.push_back({});
      tree_.leafNumSamples.push_back(children[p].sampleIndices.n_elem);
      tree_.leafPredictions_.push_back(
          static_cast<float>(Criterion::absoluteError(ys, ws).median));
    }
  }

  return children;
}
