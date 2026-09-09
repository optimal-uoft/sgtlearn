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
#include "Discretizers/univariate/UnivariateDiscretizer.h"

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
    const std::vector<size_t> &classesPerOutput, size_t nOutputs,
    bool hasNanRoutingBin) {
  if (k >= numRoutingBins ||
      criterion == LearningCriterion::AbsoluteError)
    return;

  const std::vector<size_t> snapshot = branchObj->assignments;
  const double objBeforeCd = branchObj->objective();
  coordinateDescent(k, *branchObj, rng, cdParams.maxIters, cdParams.patience,
                    hasNanRoutingBin);
  const double objAfterCd = branchObj->objective();
  if (std::isfinite(objAfterCd) &&
      objAfterCd <= objBeforeCd + kShapeFunctionCdImprovementEps)
    return;

  auto &assignments = branchObj->assignments;
  assignments = snapshot;
  branchObj = makeBranchAssignment(criterion, assignments, k, stats, leafWeights,
                                   leafSampleCounts, classesPerOutput, nOutputs);
}

void refineShapeBranchAssignmentAbsoluteError(
    std::unique_ptr<BranchAssignment> &branchObj, size_t k,
    size_t numRoutingBins, const CoordinateDescentParams &cdParams,
    std::mt19937_64 &rng,
    std::vector<std::vector<std::vector<float>>> &maeLeafYs,
    std::vector<std::vector<float>> &maeLeafWs,
    std::vector<double> &leafWeights,
    const std::vector<size_t> &leafSampleCounts, bool hasNanRoutingBin) {
  if (k >= numRoutingBins || !mae_branch_config::coordinateDescentEnabled())
    return;

  const std::vector<size_t> snapshot = branchObj->assignments;
  const double objBeforeCd = branchObj->objective();
  coordinateDescent(k, *branchObj, rng, cdParams.maxIters, cdParams.patience,
                    hasNanRoutingBin);
  const double objAfterCd = branchObj->objective();
  if (std::isfinite(objAfterCd) &&
      objAfterCd <= objBeforeCd + kShapeFunctionCdImprovementEps)
    return;

  auto &assignments = branchObj->assignments;
  assignments = snapshot;
  std::vector<std::vector<double>> dummyLeafStats(maeLeafYs.size());
  branchObj = makeBranchAssignment(
      LearningCriterion::AbsoluteError, assignments, k, dummyLeafStats, leafWeights,
      leafSampleCounts, &maeLeafYs, &maeLeafWs);
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

ShapeBranchAssignmentSearchResult searchClassificationAssignments(
    InnerDiscretizer<std::vector<double>> &disc, LearningCriterion criterion,
    double parentImp, size_t treeNumPartitions,
    const TreeBuildingParams &outerParams,
    const CoordinateDescentParams &cdParams, double scoreEpsilon,
    std::mt19937_64 &rng, const std::vector<size_t> &classesPerOutput,
    size_t nOutputs, bool hasNanRoutingBin) {
  auto &stats = disc.leafStats();
  auto &weights = disc.leafNodeWeights();
  const auto &sizes = disc.leafNumSamples();
  const size_t numBins = stats.size();
  ShapeBranchAssignmentSearchResult result;
  const auto makeObjective = [&](std::vector<size_t> &labels, size_t k) {
    return makeBranchAssignment(criterion, labels, k, stats, weights, sizes,
                                classesPerOutput, nOutputs);
  };
  const auto occupiedCount = [&](const BranchAssignment &objective) {
    size_t occupied = 0;
    for (size_t count : objective.partitionSampleCounts()) {
      if (count == 0)
        continue;
      if (count < outerParams.minLeafSize)
        return size_t{0};
      ++occupied;
    }
    return occupied;
  };
  const auto consider = [&](BranchAssignment &objective) {
    const size_t occupied = occupiedCount(objective);
    if (occupied < 2)
      return;
    const double impurity = objective.objective();
    const double gain = parentImp - impurity;
    if (!std::isfinite(impurity) || gain <= scoreEpsilon ||
        gain < outerParams.minGainSplit - scoreEpsilon)
      return;
    const double score = algorithms::penalizedBranchingScore(
        impurity, occupied, outerParams.branchingPenalty);
    if (score >= result.bestFeatureScore - scoreEpsilon)
      return;
    result.found = true;
    result.bestFeatureScore = score;
    result.chosenK = occupied;
    result.assignments = objective.assignments;
    result.partitionSampleCounts = objective.partitionSampleCounts();
    result.impurityDecrease = gain;
  };

  // Check full-data completions without perturbing the live finite-bin search.
  const auto observe = [&](BranchAssignment &objective) {
    if (!hasNanRoutingBin || objective.assignments.back() < objective.numPartitions) {
      consider(objective);
      return;
    }
    // ponytail: rebuild per trial; reuse a scratch accumulator if large-bin profiling warrants it.
    auto labels = objective.assignments;
    labels.back() = 0;
    auto complete = makeObjective(labels, objective.numPartitions);
    for (size_t p = 0; p < objective.numPartitions; ++p) {
      complete->removeLeaf(numBins - 1);
      complete->addLeaf(numBins - 1, p);
      consider(*complete);
    }
  };

  std::vector<size_t> root;
  double rootImpurity = std::numeric_limits<double>::infinity();
  for (size_t missingBranch = 0; missingBranch < 2; ++missingBranch) {
    auto labels = disc.rootBinAssignments(missingBranch);
    if (labels.empty())
      continue;
    auto objective = makeObjective(labels, 2);
    const double impurity = objective->objective();
    if (impurity < rootImpurity) {
      rootImpurity = impurity;
      root = labels;
    }
    result.rootFeasible |= occupiedCount(*objective) >= 2;
    consider(*objective);
  }

  // Only numeric bins have an ordering suitable for a secondary threshold scan.
  if (!result.rootFeasible && !numericInnerThresholds(disc).empty()) {
    const size_t finiteBins = numBins - 1;
    for (size_t cut = 1; cut < finiteBins; ++cut) {
      std::vector<size_t> labels(numBins, 1);
      std::fill(labels.begin(), labels.begin() + cut, 0);
      for (size_t missingBranch = 0; missingBranch < 2; ++missingBranch) {
        labels.back() = missingBranch;
        auto objective = makeObjective(labels, 2);
        consider(*objective);
      }
    }
  }

  size_t dim = 0;
  const auto flat = flattenNestedBinStats(stats, dim);
  for (size_t k = 2; k <= std::min(numBins, treeNumPartitions); ++k) {
    std::vector<size_t> labels;
    algorithms::seedBinAssignmentsKMeans(k, numBins, dim, flat, sizes,
                                         weights, rng, labels);
    auto objective = makeObjective(labels, k);
    observe(*objective); // Retain the feasible k-means seed even if root initializes CD.
    if (!root.empty() && rootImpurity <= objective->objective()) {
      labels = root;
      objective = makeObjective(labels, k);
    }
    coordinateDescent(k, *objective, rng, cdParams.maxIters, cdParams.patience,
                       hasNanRoutingBin, observe);
  }

  if (result.found) {
    // Compact all result metadata together. Empty routing bins still need a label.
    std::vector<size_t> compact(result.partitionSampleCounts.size(), 0);
    size_t next = 0;
    for (size_t p = 0; p < compact.size(); ++p)
      if (result.partitionSampleCounts[p] > 0)
        compact[p] = next++;
    for (auto &label : result.assignments)
      label = compact[label];
    auto objective = makeObjective(result.assignments, result.chosenK);
    result.partitionSampleCounts = objective->partitionSampleCounts();
    const auto &aggregate = dynamic_cast<const
        leaf_aggregate::LeafAggregationBranchAssignment<std::vector<double>> &>(
            *objective);
    result.partitionClassCounts = aggregate.aggregatedPartitionStats();
    result.partitionWeights = aggregate.aggregatedPartitionWeights();
    result.impurityDecrease = parentImp - objective->objective();
    result.bestFeatureScore = algorithms::penalizedBranchingScore(
        objective->objective(), result.chosenK, outerParams.branchingPenalty);
  }
  return result;
}

} // namespace

ShapeBranchAssignmentSearchResult searchShapeBranchAssignmentFromDiscretizer(
    InnerDiscretizer<std::vector<double>> &disc, LearningCriterion criterion,
    double parentImp, size_t treeNumPartitions,
    const TreeBuildingParams &outerParams,
    const CoordinateDescentParams &cdParams, double scoreEpsilon,
    std::mt19937_64 &rng,
    const std::vector<size_t> &classesPerOutput, size_t nOutputs,
    const arma::Mat<float> *ysub, const arma::Row<float> *wsub,
    size_t xSubCols, bool hasNanRoutingBin) {
  if (criterion == LearningCriterion::Gini ||
      criterion == LearningCriterion::Entropy)
    return searchClassificationAssignments(
        disc, criterion, parentImp, treeNumPartitions, outerParams, cdParams,
        scoreEpsilon, rng, classesPerOutput, nOutputs, hasNanRoutingBin);
  auto &stats = disc.leafStats();
  auto &sizes = disc.leafNumSamples();
  auto &leafWeights = disc.leafNodeWeights();
  const size_t numRoutingBins = stats.size();

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
    algorithms::roundRobinBinAssignments(numRoutingBins, k, trialAssignments);

    std::unique_ptr<BranchAssignment> branchObj;
    if (criterion == LearningCriterion::AbsoluteError) {
      branchObj = makeBranchAssignment(criterion, trialAssignments, k,
                                       dummyLeafStats, leafWeights, sizes,
                                       maeLeafYs, maeLeafWs);
      refineShapeBranchAssignmentAbsoluteError(
          branchObj, k, numRoutingBins, cdParams, rng, maeLeafYsStorage,
          maeLeafWsStorage, leafWeights, sizes, hasNanRoutingBin);
    } else {
      branchObj = makeBranchAssignment(criterion, trialAssignments, k, stats,
                                       leafWeights, sizes, classesPerOutput,
                                       nOutputs);
      refineShapeBranchAssignmentNested(
          branchObj, k, numRoutingBins, criterion, cdParams, rng, stats,
          leafWeights, sizes, classesPerOutput, nOutputs, hasNanRoutingBin);
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
      // Coordinate descent updates the live branch object, not the seed vector.
      // Persist that refined routing so the objective and eventual child buckets
      // describe the same split.
      result.assignments = branchObj->assignments;
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
  node.logicalFeatureIndices.clear();
  node.retainedPairCandidates.clear();
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
  best.logicalFeatureIndices = {featureIndex};
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
