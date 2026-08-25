/**
 * @file Estimators/ShapeFunctions/ShapeFunctionSplitSearch.cpp
 */

#include <memory>
#include <utility>
#include <functional>
#include <cstddef>
#include "Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h"

#include "BranchAssignmentObjectives/BranchAssignment.h"
#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "BranchAssignmentObjectives/LeafAggregationBranchAssignment.h"
#include "BranchAssignmentObjectives/MaeBranchConfig.h"
#include "algorithms/BinPartitionAssignments.h"
#include "algorithms/CoordinateDescent.h"

#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace {

void refineShapeBranchAssignmentNested(
    std::unique_ptr<BranchAssignment> &branchObj, size_t k,
    size_t numRoutingBins, LearningCriterion criterion,
    const CoordinateDescentParams &cdParams, std::mt19937_64 &rng,
    std::vector<std::vector<std::vector<double>>> &stats,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts,
    const std::vector<size_t> &classesPerOutput, size_t nOutputs) {
  if (k >= numRoutingBins ||
      criterion == LearningCriterion::AbsoluteError)
    return;

  const std::vector<size_t> snapshot = branchObj->assignments;
  const double objBeforeCd = branchObj->objective();
  coordinateDescent(k, *branchObj, rng, cdParams.maxIters, cdParams.patience);
  const double objAfterCd = branchObj->objective();
  if (std::isfinite(objAfterCd) &&
      objAfterCd <= objBeforeCd + kShapeFunctionCdImprovementEps)
    return;

  std::vector<size_t> rollback = snapshot;
  branchObj = makeBranchAssignment(criterion, rollback, k, stats, leafWeights,
                                   leafSampleCounts, classesPerOutput, nOutputs);
}

void refineShapeBranchAssignmentAbsoluteError(
    std::unique_ptr<BranchAssignment> &branchObj, size_t k,
    size_t numRoutingBins, const CoordinateDescentParams &cdParams,
    std::mt19937_64 &rng,
    std::vector<std::vector<std::vector<float>>> &maeLeafYs,
    std::vector<std::vector<float>> &maeLeafWs,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts) {
  if (k >= numRoutingBins || !mae_branch_config::coordinateDescentEnabled())
    return;

  const std::vector<size_t> snapshot = branchObj->assignments;
  const double objBeforeCd = branchObj->objective();
  coordinateDescent(k, *branchObj, rng, cdParams.maxIters, cdParams.patience);
  const double objAfterCd = branchObj->objective();
  if (std::isfinite(objAfterCd) &&
      objAfterCd <= objBeforeCd + kShapeFunctionCdImprovementEps)
    return;

  std::vector<size_t> rollback = snapshot;
  std::vector<std::vector<double>> dummyLeafStats(maeLeafYs.size());
  branchObj = makeBranchAssignment(
      LearningCriterion::AbsoluteError, rollback, k, dummyLeafStats, leafWeights,
      leafSampleCounts, &maeLeafYs, &maeLeafWs);
}

void seedTrialBinAssignments(size_t k, size_t numRoutingBins,
                             const std::vector<std::vector<double>> &stats,
                             const std::vector<size_t> &sizes,
                             std::vector<double> &leafWeights, bool useKMeansSeed,
                             bool smartInit, size_t kmeansDim,
                             std::mt19937_64 &rng,
                             std::vector<size_t> &trialAssignments) {
  if (k == numRoutingBins) {
    algorithms::identityBinAssignments(numRoutingBins, trialAssignments);
  } else if (useKMeansSeed && smartInit && k >= 2 && numRoutingBins >= k) {
    algorithms::seedBinAssignmentsKMeans(k, numRoutingBins, kmeansDim, stats,
                                       sizes, leafWeights, rng,
                                       trialAssignments);
  } else {
    algorithms::roundRobinBinAssignments(numRoutingBins, k, trialAssignments);
  }
}

std::vector<std::vector<double>>
flattenNestedBinStats(const std::vector<std::vector<std::vector<double>>> &stats,
                      size_t &kmeansDimOut) {
  std::vector<std::vector<double>> flat(stats.size());
  kmeansDimOut = 0;
  for (size_t b = 0; b < stats.size(); ++b) {
    for (const auto &hist : stats[b])
      flat[b].insert(flat[b].end(), hist.begin(), hist.end());
    kmeansDimOut = std::max(kmeansDimOut, flat[b].size());
  }
  for (auto &row : flat) {
    if (row.size() < kmeansDimOut)
      row.resize(kmeansDimOut, 0.0);
  }
  return flat;
}

} // namespace

ShapeBranchAssignmentSearchResult searchShapeBranchAssignmentFromDiscretizer(
    InnerDiscretizer<std::vector<double>> &disc, LearningCriterion criterion,
    double parentImp, size_t treeNumPartitions,
    const TreeBuildingParams &outerParams,
    const CoordinateDescentParams &cdParams, double scoreEpsilon,
    std::mt19937_64 &rng, bool useKMeansSeed,
    const std::vector<size_t> &classesPerOutput, size_t nOutputs,
    const arma::Mat<float> *ysub, const arma::Row<float> *wsub,
    size_t xSubCols) {
  auto &stats = disc.leafStats();
  auto &sizes = disc.leafNumSamples();
  auto &leafWeights = disc.leafNodeWeights();
  const size_t numRoutingBins = stats.size();

  size_t kmeansDim = 0;
  const std::vector<std::vector<double>> flatStats =
      flattenNestedBinStats(stats, kmeansDim);

  std::vector<std::vector<std::vector<float>>> maeLeafYsStorage;
  std::vector<std::vector<float>> maeLeafWsStorage;
  std::vector<std::vector<std::vector<float>>> *maeLeafYs = nullptr;
  std::vector<std::vector<float>> *maeLeafWs = nullptr;
  std::vector<std::vector<double>> dummyLeafStats;
  if (criterion == LearningCriterion::AbsoluteError) {
    if (!ysub || !wsub)
      throw std::invalid_argument(
          "searchShapeBranchAssignmentFromDiscretizer(AbsoluteError): ysub "
          "and wsub required");
    const size_t maeOutputs = std::max<size_t>(ysub->n_rows, 1);
    const auto &perBinCols = disc.inSampleDiscretizations();
    maeLeafYsStorage.assign(
        numRoutingBins,
        std::vector<std::vector<float>>(maeOutputs, std::vector<float>{}));
    maeLeafWsStorage.resize(numRoutingBins);
    for (size_t b = 0; b < numRoutingBins; ++b) {
      for (size_t colIdx : perBinCols[b]) {
        if (colIdx >= xSubCols)
          throw std::runtime_error(
              "searchShapeBranchAssignmentFromDiscretizer: discretizer sample "
              "index >= Xsub columns");
        for (size_t o = 0; o < maeOutputs; ++o)
          maeLeafYsStorage[b][o].push_back(
              (*ysub)(o, static_cast<arma::uword>(colIdx)));
        maeLeafWsStorage[b].push_back((*wsub)(colIdx));
      }
    }
    maeLeafYs = &maeLeafYsStorage;
    maeLeafWs = &maeLeafWsStorage;
    dummyLeafStats.resize(numRoutingBins);
  }

  ShapeBranchAssignmentSearchResult result;
  const size_t kMax = std::min(numRoutingBins, treeNumPartitions);

  for (size_t k = 2; k <= kMax; ++k) {
    std::vector<size_t> trialAssignments;
    seedTrialBinAssignments(k, numRoutingBins, flatStats, sizes, leafWeights,
                            useKMeansSeed, cdParams.smartInit, kmeansDim, rng,
                            trialAssignments);

    std::unique_ptr<BranchAssignment> branchObj;
    if (criterion == LearningCriterion::AbsoluteError) {
      branchObj = makeBranchAssignment(criterion, trialAssignments, k,
                                       dummyLeafStats, leafWeights, sizes,
                                       maeLeafYs, maeLeafWs);
      refineShapeBranchAssignmentAbsoluteError(
          branchObj, k, numRoutingBins, cdParams, rng, maeLeafYsStorage,
          maeLeafWsStorage, leafWeights, sizes);
    } else {
      branchObj = makeBranchAssignment(criterion, trialAssignments, k, stats,
                                       leafWeights, sizes, classesPerOutput,
                                       nOutputs);
      refineShapeBranchAssignmentNested(
          branchObj, k, numRoutingBins, criterion, cdParams, rng, stats,
          leafWeights, sizes, classesPerOutput, nOutputs);
    }

    if (!branchObj->partitionCountsMeetMinLeaf(outerParams.minLeafSize))
      continue;

    const double childImp = branchObj->objective();
    const double gain = parentImp - childImp;
    if (gain < outerParams.minGainSplit - scoreEpsilon)
      continue;

    const double score = algorithms::penalizedBranchingScore(
        childImp, k, outerParams.branchingPenalty);
    if (score < result.bestFeatureScore - scoreEpsilon) {
      result.bestFeatureScore = score;
      result.chosenK = k;
      result.assignments = trialAssignments;
      result.partitionSampleCounts = branchObj->partitionSampleCounts();
      if (const auto *leafAgg =
              dynamic_cast<leaf_aggregate::LeafAggregationBranchAssignment<
                  std::vector<double>> *>(branchObj.get())) {
        const auto &aggStats = leafAgg->aggregatedPartitionStats();
        if (criterion == LearningCriterion::SquaredError)
          result.partitionAggStats = aggStats;
        else
          result.partitionClassCounts = aggStats;
        result.partitionWeights = leafAgg->aggregatedPartitionWeights();
      }
      result.impurityDecrease = gain;
      result.found = true;
    }
  }

  return result;
}

void markShapeFunctionNodeAsLeaf(ShapeFunctionNode &node) {
  node.isLeaf = true;
  node.informationGain = 0.0;
  node.splitFeatureIndex = 0;
  node.innerDiscretizer.reset();
  node.sampleBins.clear();
  node.splitLeafStats.clear();
  node.splitClassCounts.clear();
  node.splitBinWeights.clear();
  node.binSampleCounts.clear();
}

void fillSampleBinsFromDiscretizer(
    size_t xSubCols, const std::vector<std::vector<size_t>> &perBinCols,
    std::vector<size_t> &sampleBins) {
  sampleBins.assign(xSubCols, 0);
  for (size_t b = 0; b < perBinCols.size(); ++b) {
    for (size_t colIdx : perBinCols[b]) {
      if (colIdx >= xSubCols)
        throw std::runtime_error(
            "fillSampleBinsFromDiscretizer: sample index >= Xsub columns");
      sampleBins[colIdx] = b;
    }
  }
}

arma::Row<float> subSampleWeights(const arma::Row<float> &weights,
                                  const arma::uvec &subIdx) {
  arma::Row<float> wsub(subIdx.n_elem);
  for (arma::uword j = 0; j < subIdx.n_elem; ++j)
    wsub(j) = weights(subIdx(j));
  return wsub;
}

void applySharedShapeBranchingFields(
    ShapeBestBranchingState &best, const ShapeBranchAssignmentSearchResult &search,
    size_t featureIndex, size_t xSubCols,
    const std::vector<std::vector<size_t>> &perBinCols) {
  best.penalizedChildScore = search.bestFeatureScore;
  best.branching.featureIndex = featureIndex;
  best.branching.binToPartition = search.assignments;
  best.branching.impurityDecrease = search.impurityDecrease;
  best.branching.numPartitionsUsed = search.chosenK;
  best.branching.partitionSampleCounts = search.partitionSampleCounts;
  fillSampleBinsFromDiscretizer(xSubCols, perBinCols, best.branching.sampleBins);
}

bool featureHasBetterShapeBranching(
    const ShapeBranchAssignmentSearchResult &search,
    ShapeBestBranchingState &best, size_t featureIndex, size_t xSubCols,
    const arma::uvec &routingColumnIndices,
    std::unique_ptr<InnerDiscretizer<std::vector<double>>> disc,
    double scoreEpsilon,
    const std::function<void(
        ShapeBestBranchingState &, const ShapeBranchAssignmentSearchResult &,
        const std::vector<std::vector<std::vector<double>>> &)> &
        applyTaskFields) {
  if (!search.found || search.assignments.empty())
    return false;
  if (search.bestFeatureScore >= best.penalizedChildScore - scoreEpsilon)
    return false;

  applySharedShapeBranchingFields(best, search, featureIndex, xSubCols,
                                  disc->inSampleDiscretizations());
  best.branching.leafNumSamples = disc->leafNumSamples();
  best.binWeights.assign(disc->leafNodeWeights().begin(),
                         disc->leafNodeWeights().end());
  applyTaskFields(best, search, disc->leafStats());
  best.routingColumnIndices = routingColumnIndices;
  best.winningDiscretizer =
      std::shared_ptr<const InnerDiscretizerBase>(std::move(disc));
  return true;
}

std::vector<std::vector<size_t>>
routeSamplesToPartitions(const ShapeFunctionNode &parent, const arma::fmat &X) {
  (void)X;
  if (parent.sampleBins.size() != parent.sampleIndices.n_elem)
    throw std::runtime_error(
        "routeSamplesToPartitions: sampleBins length mismatch");

  const size_t numChildPartitions = parent.numPartitions;
  std::vector<std::vector<size_t>> buckets(numChildPartitions);
  for (arma::uword i = 0; i < parent.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(parent.sampleIndices(i));
    const size_t bin = parent.sampleBins[static_cast<size_t>(i)];
    if (bin >= parent.binToPartition.size())
      throw std::runtime_error("routeSamplesToPartitions: bin id out of range");
    const size_t p = parent.binToPartition[bin];
    if (p >= numChildPartitions)
      throw std::runtime_error(
          "routeSamplesToPartitions: partition out of range");
    buckets[p].push_back(si);
  }
  return buckets;
}

std::vector<ShapeFunctionNode>
makeRoutedChildNodes(const ShapeFunctionNode &parent,
                     const std::vector<std::vector<size_t>> &buckets,
                     size_t treeNumPartitions) {
  std::vector<ShapeFunctionNode> children;
  children.reserve(buckets.size());
  for (const auto &bucket : buckets) {
    ShapeFunctionNode ch;
    ch.height = parent.height + 1;
    ch.sampleIndices = arma::conv_to<arma::uvec>::from(bucket);
    ch.numPartitions = treeNumPartitions;
    children.push_back(std::move(ch));
  }
  return children;
}
