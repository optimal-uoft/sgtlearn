/**
 * @file Estimators/ShapeFunctions/ClassificationShapeFunctionBuilder.cpp
 * @brief Per-node classification shape-function split search.
 */

#include "Estimators/ShapeFunctions/ClassificationShapeFunctionBuilder.h"

#include "Estimators/ClassificationShapeGeneralizedTree.h"
#include "Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h"
#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "BranchAssignmentObjectives/LeafAggregationBranchAssignment.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "Discretizers/DiscretizerFactories.h"
#include "algorithms/BinPartitionAssignments.h"
#include "algorithms/CoordinateDescent.h"

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

ShapeBranchAssignmentSearchResult
ClassificationShapeFunctionBuilder::searchBestBranchAssignment(
    size_t numRoutingBins, double parentImp,
    std::vector<std::vector<double>> &stats, const std::vector<size_t> &sizes,
    std::vector<double> &weights) {
  ShapeBranchAssignmentSearchResult result;
  const size_t kMax = std::min(numRoutingBins, tree_.numPartitions_);

  for (size_t k = 2; k <= kMax; ++k) {
    std::vector<size_t> trialAssignments;
    if (k == numRoutingBins) {
      algorithms::identityBinAssignments(numRoutingBins, trialAssignments);
    } else if (!tree_.cdParams_.smartInit || k < 2 || numRoutingBins < k) {
      algorithms::roundRobinBinAssignments(numRoutingBins, k, trialAssignments);
    } else {
      algorithms::seedBinAssignmentsKMeans(k, numRoutingBins, tree_.numClasses_,
                                         stats, sizes, weights, tree_.rng_,
                                         trialAssignments);
    }

    auto branchObj = makeClassificationBranchAssignment(
        tree_.criterion_, trialAssignments, k, stats, weights, sizes,
        tree_.numClasses_);

    if (k < numRoutingBins) {
      const std::vector<size_t> snapshot = branchObj->assignments;
      const double objBeforeCd = branchObj->objective();
      coordinateDescent(k, *branchObj, tree_.rng_, tree_.cdParams_.maxIters,
                        tree_.cdParams_.patience);
      const double objAfterCd = branchObj->objective();
      if (!std::isfinite(objAfterCd) ||
          objAfterCd > objBeforeCd + kShapeFunctionCdImprovementEps) {
        std::vector<size_t> rollback = snapshot;
        branchObj = makeClassificationBranchAssignment(
            tree_.criterion_, rollback, k, stats, weights, sizes,
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
      result.assignments = trialAssignments;
      result.partitionSampleCounts = branchObj->partitionSampleCounts();
      const auto *leafAgg =
          dynamic_cast<leaf_aggregate::LeafAggregationBranchAssignment<double> *>(
              branchObj.get());
      if (!leafAgg)
        throw std::runtime_error(
            "ClassificationShapeFunctionBuilder: branch assignment must "
            "expose partition aggregates");
      result.partitionClassCounts = leafAgg->aggregatedPartitionStats();
      result.partitionWeights = leafAgg->aggregatedPartitionWeights();
      result.impurityDecrease = gain;
      result.found = true;
    }
  }

  return result;
}

bool ClassificationShapeFunctionBuilder::findBestSplit(ShapeFunctionNode &node,
                                                       size_t minLeaf) {
  const size_t ns = node.sampleIndices.n_elem;
  node.score =
      tree_.impurityForClassCounts(tree_.classCounts[node.nodeIndex]);

  if (ns < 2 * minLeaf) {
    markShapeFunctionLeafNoSplit(node);
    return false;
  }

  const double parentImp = node.score;
  if (parentImp <= tree_.outerTreeBuilder_.eps) {
    markShapeFunctionLeafNoSplit(node);
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
  ShapeBestBranchingState best{};
  arma::uvec featOne(1);

  const auto applyTaskFields =
      [](ShapeBestBranchingState &state,
         const ShapeBranchAssignmentSearchResult &search,
         const std::vector<std::vector<double>> &leafStats) {
        state.branching.leafStats = leafStats;
        state.partitionClassCounts = search.partitionClassCounts;
        state.partitionWeights = search.partitionWeights;
      };

  for (size_t fi = 0; fi < featureSubset.size(); ++fi) {
    const size_t f = featureSubset[fi];
    if (f >= Xsub.n_rows)
      throw std::invalid_argument(
          "ClassificationShapeFunctionBuilder::findBestSplit: feature index "
          ">= X.n_rows");
    featOne(0) = static_cast<arma::uword>(f);

    const arma::Row<float> wsub =
        subSampleWeights(tree_.fitSampleWeights_, subIdx);

    auto disc = makeClassificationDiscretizer(tree_.criterion_,
                                              DiscretizerInputKind::Numeric);
    disc->Train(Xsub, featOne, ysub, tree_.numClasses_,
                tree_.innerParams_.minLeafSize,
                tree_.innerParams_.minGainSplit, tree_.innerParams_.maxDepth,
                tree_.innerParams_.maxLeafNodes, wsub);
    if (disc->numLeaves() < 2)
      continue;

    auto &stats = disc->leafStats();
    auto &sizes = disc->leafNumSamples();
    auto &weights = disc->leafNodeWeights();
    const size_t numRoutingBins = stats.size();

    const ShapeBranchAssignmentSearchResult featureBest =
        searchBestBranchAssignment(numRoutingBins, parentImp, stats, sizes,
                                   weights);
    if (!featureBest.found)
      continue;

    featureHasBetterShapeBranching(
        featureBest, best, f, xSubCols,
        std::unique_ptr<InnerDiscretizerBase<double>>(std::move(disc)),
        tree_.outerTreeBuilder_.eps, applyTaskFields);
  }

  if (!std::isfinite(best.penalizedChildScore) ||
      best.penalizedChildScore >= std::numeric_limits<double>::infinity() ||
      best.branching.impurityDecrease <= tree_.outerTreeBuilder_.eps) {
    markShapeFunctionLeafNoSplit(node);
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
    children[p].score = tree_.impurityForClassCounts(childClassCounts);
    children[p].isLeaf = true;
    tree_.classCounts.push_back(childClassCounts);
  }
  return children;
}
