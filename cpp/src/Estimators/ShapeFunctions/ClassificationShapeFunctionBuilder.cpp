/**
 * @file Estimators/ShapeFunctions/ClassificationShapeFunctionBuilder.cpp
 * @brief Per-node classification shape-function split search.
 */

#include "Estimators/ShapeFunctions/ClassificationShapeFunctionBuilder.h"

#include "Estimators/ClassificationShapeGeneralizedTree.h"
#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "BranchAssignmentObjectives/LeafAggregationBranchAssignment.h"
#include "Criterion.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "algorithms/BinPartitionAssignments.h"
#include "algorithms/CoordinateDescent.h"
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
    : partitionImpurity_(partitionImpurityFnFor(tree.criterion_)), tree_(tree),
      X_(X), y_(y), featureCandidates_(featureCandidates) {}

ClassificationShapeFunctionBuilder::PartitionImpurityFn
ClassificationShapeFunctionBuilder::partitionImpurityFnFor(
    LearningCriterion criterion) {
  if (criterion == LearningCriterion::Gini)
    return &Criterion::gini;
  if (criterion == LearningCriterion::Entropy)
    return &Criterion::entropy;
  throw std::invalid_argument(
      "ClassificationShapeFunctionBuilder: criterion must be Entropy or Gini");
}

double ClassificationShapeFunctionBuilder::partitionImpurity(
    const std::vector<double> &classCounts, double totalWeight) const {
  if (!partitionImpurity_)
    throw std::runtime_error(
        "ClassificationShapeFunctionBuilder::partitionImpurity: "
        "impurity function not set");
  return partitionImpurity_(classCounts, totalWeight);
}

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

void ClassificationShapeFunctionBuilder::applyTaskBranchingFields(
    BestBranchingState &best, const BranchAssignmentSearchResult &search,
    const std::vector<std::vector<double>> &leafStats) {
  best.branching.leafStats = leafStats;
  best.partitionClassCounts = search.partitionClassCounts;
  best.partitionWeights = search.partitionWeights;
}

void ClassificationShapeFunctionBuilder::assignNanPredictionPartition(
    ShapeFunctionNode &node,
    const std::vector<size_t> &partitionSampleCounts,
    const std::vector<std::vector<double>> &partitionClassCounts,
    const std::vector<double> &partitionWeights, bool nanSeen,
    const std::vector<double> &nanStats, double nanWeight) const {
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

  const size_t numClasses = tree_.numClasses_;
  const size_t numPartitions = node.numPartitions;
  node.splitMissingStats.assign(numClasses, 0.0);
  for (size_t c = 0; c < numClasses; ++c)
    node.splitMissingStats[c] = c < nanStats.size() ? nanStats[c] : 0.0;
  node.splitMissingWeight = nanWeight;

  double baseWeightedLoss = 0.0;
  double totalWeight = node.splitMissingWeight;
  for (size_t p = 0; p < numPartitions; ++p) {
    const double pw = p < partitionWeights.size() ? partitionWeights[p] : 0.0;
    totalWeight += pw;
    if (pw > 0.0 && p < partitionClassCounts.size())
      baseWeightedLoss +=
          pw * partitionImpurity(partitionClassCounts[p], pw);
  }
  if (totalWeight <= 0.0) {
    node.nanPredictionPartition = 0;
    return;
  }

  std::vector<double> trialScores(numPartitions,
                                  std::numeric_limits<double>::infinity());
  for (size_t p = 0; p < numPartitions; ++p) {
    const double pw = p < partitionWeights.size() ? partitionWeights[p] : 0.0;
    std::vector<double> trialCounts =
        p < partitionClassCounts.size() ? partitionClassCounts[p]
                                      : std::vector<double>(numClasses, 0.0);
    for (size_t c = 0; c < numClasses; ++c)
      trialCounts[c] += node.splitMissingStats[c];
    const double trialWeight = pw + node.splitMissingWeight;
    double trialLoss = 0.0;
    if (trialWeight > 0.0)
      trialLoss = partitionImpurity(trialCounts, trialWeight);
    double basePartLoss = 0.0;
    if (pw > 0.0 && p < partitionClassCounts.size())
      basePartLoss = partitionImpurity(partitionClassCounts[p], pw);
    const double weightedLoss =
        baseWeightedLoss - pw * basePartLoss + trialWeight * trialLoss;
    trialScores[p] = weightedLoss / totalWeight;
  }
  node.nanPredictionPartition =
      missing_values::pick_lowest_score_min_index_tie(trialScores);
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
  BestBranchingState best{};
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
  node.splitLeafStats = std::move(best.branching.leafStats);
  node.splitBinWeights = std::move(best.binWeights);
  node.binSampleCounts = std::move(best.branching.leafNumSamples);
  node.numPartitions = best.branching.numPartitionsUsed;
  node.informationGain = best.branching.impurityDecrease;
  assignNanPredictionPartition(
      node, best.branching.partitionSampleCounts, best.partitionClassCounts,
      best.partitionWeights, best.nanSeen, best.nanStats, best.nanNodeWeight);

  return true;
}

std::vector<ShapeFunctionNode> ClassificationShapeFunctionBuilder::makeChildren(
    const ShapeFunctionNode &parent) {
  const auto buckets = routeSamplesToPartitions(parent, X_);
  const auto &binStats = parent.splitLeafStats;
  if (binStats.size() != parent.binToPartition.size())
    throw std::runtime_error(
        "ClassificationShapeFunctionBuilder::makeChildren: splitLeafStats / "
        "binToPartition size mismatch");

  auto children = makeRoutedChildNodes(parent, buckets, tree_.numPartitions_);
  for (size_t p = 0; p < children.size(); ++p) {
    std::vector<double> childClassCounts(tree_.numClasses_, 0.0);
    for (size_t b = 0; b < binStats.size(); ++b) {
      if (parent.binToPartition[b] != p)
        continue;
      const auto &sb = binStats[b];
      for (size_t c = 0; c < tree_.numClasses_; ++c)
        childClassCounts[c] +=
            (c < sb.size()) ? static_cast<double>(sb[c]) : 0.0;
    }
    if (p == parent.nanPredictionPartition && !parent.splitMissingStats.empty()) {
      for (size_t c = 0; c < tree_.numClasses_; ++c)
        childClassCounts[c] +=
            (c < parent.splitMissingStats.size())
                ? parent.splitMissingStats[c]
                : 0.0;
    }
    children[p].score = tree_.impurityForClassCounts(childClassCounts);
    children[p].isLeaf = true;
    tree_.classCounts.push_back(childClassCounts);
  }
  return children;
}
