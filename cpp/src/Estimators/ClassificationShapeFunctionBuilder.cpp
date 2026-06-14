/**
 * @file Estimators/ClassificationShapeFunctionBuilder.cpp
 * @brief Per-node classification shape-function split search.
 */

#include "Estimators/ClassificationShapeFunctionBuilder.h"

#include "Estimators/ClassificationShapeGeneralizedTree.h"
#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "BranchAssignmentObjectives/LeafAggregationBranchAssignment.h"
#include "Criterion.h"
#include "Discretizers/ClassificationDiscretizer.h"
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

ClassificationShapeFunctionBuilder::ClassificationShapeFunctionBuilder(
    ClassificationShapeGeneralizedTree &tree, const arma::fmat &X,
    const arma::Row<size_t> &y, const arma::uvec &featureCandidates)
    : tree_(tree), X_(X), y_(y), featureCandidates_(featureCandidates) {}

ClassificationShapeFunctionBuilder::BranchAssignmentSearchResult
ClassificationShapeFunctionBuilder::searchBestBranchAssignment(
    size_t numBins, double parentImp,
    std::vector<std::vector<double>> &stats, const std::vector<size_t> &sizes,
    std::vector<double> &weights) {
  BranchAssignmentSearchResult result;
  const size_t kMax = std::min(numBins, tree_.numPartitions_);

  for (size_t k = 2; k <= kMax; ++k) {
    std::vector<size_t> trialAssignments;
    if (k == numBins) {
      algorithms::identityBinAssignments(numBins, trialAssignments);
    } else if (!tree_.cdParams_.smartInit || k < 2 || numBins < k) {
      algorithms::roundRobinBinAssignments(numBins, k, trialAssignments);
    } else {
      algorithms::seedBinAssignmentsKMeans(k, numBins, tree_.numClasses_, stats,
                                           sizes, weights, tree_.rng_,
                                           trialAssignments);
    }

    auto branchObj = makeClassificationBranchAssignment(
        tree_.criterion_, trialAssignments, k, stats, weights, sizes,
        tree_.numClasses_);

    if (k < numBins) {
      const std::vector<size_t> snapshot = trialAssignments;
      const double objBeforeCd = branchObj->objective();
      coordinateDescent(k, *branchObj, tree_.rng_, tree_.cdParams_.maxIters,
                        tree_.cdParams_.patience);
      const double objAfterCd = branchObj->objective();
      if (!std::isfinite(objAfterCd) ||
          objAfterCd > objBeforeCd + ShapeFunctionBuilder::kCdObjectiveImprovementEps) {
        trialAssignments = snapshot;
        branchObj = makeClassificationBranchAssignment(
            tree_.criterion_, trialAssignments, k, stats, weights, sizes,
            tree_.numClasses_);
      }
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
      const auto *leafAgg =
          dynamic_cast<leaf_aggregate::LeafAggregationBranchAssignment<double> *>(
              branchObj.get());
      if (!leafAgg)
        throw std::runtime_error(
            "ClassificationShapeFunctionBuilder: classification branch "
            "assignment must expose partition aggregates");
      result.partitionClassCounts = leafAgg->aggregatedPartitionStats();
      result.partitionWeights = leafAgg->aggregatedPartitionWeights();
      result.impurityDecrease = gain;
      result.found = true;
    }
  }

  return result;
}

bool ClassificationShapeFunctionBuilder::adoptFeatureBranchIfBetter(
    double bestFeatureScore, double &bestPenalizedChild,
    ShapeBranchingResult<double> &brBest,
    std::vector<double> &binWeightsForBest,
    std::vector<std::vector<double>> &partitionClassCountsForBest,
    std::vector<double> &partitionWeightsForBest, size_t featureIndex,
    const BranchAssignmentSearchResult &featureBest, size_t xSubCols,
    ClassificationDiscretizer &disc,
    const std::vector<std::vector<double>> &stats,
    const std::vector<size_t> &sizes, const std::vector<double> &weights) {
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
  brBest.leafStats = stats;
  brBest.leafNumSamples = sizes;
  binWeightsForBest.assign(weights.begin(), weights.end());
  partitionClassCountsForBest = featureBest.partitionClassCounts;
  partitionWeightsForBest = featureBest.partitionWeights;
  return true;
}

size_t ClassificationShapeFunctionBuilder::chooseNanPredictionPartition(
    size_t numPartitions, size_t routingFeature,
    const std::vector<size_t> &partitionSampleCounts,
    const std::vector<std::vector<double>> &partitionClassCounts,
    const std::vector<double> &partitionWeights, const arma::fmat &Xsub,
    const arma::Row<size_t> &ysub, const arma::uvec &subIdx) const {
  const arma::Row<float> wsub = subSampleWeights(tree_.fitSampleWeights_, subIdx);
  const arma::frowvec featRow = Xsub.row(routingFeature);
  const auto missingCols =
      nan_partition_routing::missing_column_indices(featRow);
  if (missingCols.empty()) {
    return missing_values::partition_with_max_count_min_index_tie(
        partitionSampleCounts);
  }

  const size_t numClasses = tree_.numClasses_;
  std::vector<double> missingCounts(numClasses, 0.0);
  double missingWeight = 0.0;
  for (size_t col : missingCols) {
    if (col >= static_cast<size_t>(ysub.n_elem))
      continue;
    const size_t lab = ysub(col);
    if (lab >= numClasses)
      continue;
    const double w = static_cast<double>(wsub(col));
    missingCounts[lab] += w;
    missingWeight += w;
  }

  double baseWeightedLoss = 0.0;
  double totalWeight = missingWeight;
  for (size_t p = 0; p < numPartitions; ++p) {
    const double pw = p < partitionWeights.size() ? partitionWeights[p] : 0.0;
    totalWeight += pw;
    if (pw > 0.0 && p < partitionClassCounts.size()) {
      if (tree_.criterion_ == LearningCriterion::Gini)
        baseWeightedLoss +=
            pw * Criterion::gini(partitionClassCounts[p], pw);
      else
        baseWeightedLoss +=
            pw * Criterion::entropy(partitionClassCounts[p], pw);
    }
  }
  if (totalWeight <= 0.0)
    return 0;

  std::vector<double> trialScores(numPartitions,
                                  std::numeric_limits<double>::infinity());
  for (size_t p = 0; p < numPartitions; ++p) {
    const double pw = p < partitionWeights.size() ? partitionWeights[p] : 0.0;
    std::vector<double> trialCounts =
        p < partitionClassCounts.size() ? partitionClassCounts[p]
                                      : std::vector<double>(numClasses, 0.0);
    for (size_t c = 0; c < numClasses; ++c)
      trialCounts[c] += missingCounts[c];
    const double trialWeight = pw + missingWeight;
    double trialLoss = 0.0;
    if (trialWeight > 0.0) {
      if (tree_.criterion_ == LearningCriterion::Gini)
        trialLoss = Criterion::gini(trialCounts, trialWeight);
      else
        trialLoss = Criterion::entropy(trialCounts, trialWeight);
    }
    double basePartLoss = 0.0;
    if (pw > 0.0 && p < partitionClassCounts.size()) {
      if (tree_.criterion_ == LearningCriterion::Gini)
        basePartLoss = Criterion::gini(partitionClassCounts[p], pw);
      else
        basePartLoss = Criterion::entropy(partitionClassCounts[p], pw);
    }
    const double weightedLoss =
        baseWeightedLoss - pw * basePartLoss + trialWeight * trialLoss;
    trialScores[p] = weightedLoss / totalWeight;
  }
  return missing_values::pick_lowest_score_min_index_tie(trialScores);
}

bool ClassificationShapeFunctionBuilder::findBestSplit(ShapeFunctionNode &node,
                                                       size_t minLeaf) {
  const size_t ns = node.sampleIndices.n_elem;
  node.score =
      tree_.impurityForClassCounts(tree_.classCounts[node.nodeIndex]);

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
  const arma::Row<size_t> ysub = y_.cols(subIdx);

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
  std::vector<double> binWeightsForBest;
  std::vector<std::vector<double>> partitionClassCountsForBest;
  std::vector<double> partitionWeightsForBest;
  arma::uvec featOne(1);

  for (size_t fi = 0; fi < featureSubset.size(); ++fi) {
    const size_t f = featureSubset[fi];
    if (f >= Xsub.n_rows)
      throw std::invalid_argument(
          "ClassificationShapeFunctionBuilder::findBestSplit: candidate "
          "feature index >= X.n_rows");
    featOne(0) = static_cast<arma::uword>(f);

    const arma::Row<float> wsub =
        subSampleWeights(tree_.fitSampleWeights_, subIdx);

    auto disc = makeClassificationDiscretizer(tree_.criterion_);
    disc->Train(Xsub, featOne, ysub, tree_.numClasses_,
                tree_.innerParams_.minLeafSize,
                tree_.innerParams_.minGainSplit, tree_.innerParams_.maxDepth,
                tree_.innerParams_.maxLeafNodes, wsub);
    const size_t B = disc->numLeaves();
    if (B < 2)
      continue;

    auto &stats = disc->leafStats();
    auto &sizes = disc->leafNumSamples();
    auto &weights = disc->leafNodeWeights();

    const BranchAssignmentSearchResult featureBest =
        searchBestBranchAssignment(B, parentImp, stats, sizes, weights);
    if (!featureBest.found)
      continue;

    adoptFeatureBranchIfBetter(featureBest.bestFeatureScore, bestPenalizedChild,
                               brBest, binWeightsForBest,
                               partitionClassCountsForBest,
                               partitionWeightsForBest, f, featureBest, xSubCols,
                               *disc, stats, sizes, weights);
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
  node.splitLeafStats = std::move(brBest.leafStats);
  node.splitBinWeights = std::move(binWeightsForBest);
  node.binSampleCounts = std::move(brBest.leafNumSamples);
  node.numPartitions = brBest.numPartitionsUsed;
  node.informationGain = brBest.impurityDecrease;
  node.nanPredictionPartition = chooseNanPredictionPartition(
      brBest.numPartitionsUsed, brBest.featureIndex,
      brBest.partitionSampleCounts, partitionClassCountsForBest,
      partitionWeightsForBest, Xsub, ysub, subIdx);

  return true;
}
