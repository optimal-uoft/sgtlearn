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
#include <numeric>

namespace {

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
    const CoordinateDescentParams &cdParams,
    std::mt19937_64 &rng,
    const std::vector<size_t> &classesPerOutput, size_t nOutputs,
    const arma::Mat<float> *ysub, const arma::Row<float> *wsub,
    size_t xSubCols, bool hasNanRoutingBin) {
  auto &stats = disc.leafStats();
  auto &weights = disc.leafNodeWeights();
  const auto &sizes = disc.leafNumSamples();
  const size_t numBins = stats.size();
  const double totalWeight = std::accumulate(weights.begin(), weights.end(), 0.0);
  constexpr double eps = std::numeric_limits<double>::epsilon();
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
        numBins,
        std::vector<std::vector<float>>(maeOutputs, std::vector<float>{}));
    maeLeafWsStorage.resize(numBins);
    for (size_t b = 0; b < numBins; ++b) {
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
    dummyLeafStats.resize(numBins);
  }

  ShapeBranchAssignmentSearchResult result;
  const auto makeObjective = [&](std::vector<size_t> &labels, size_t k) {
    if (criterion == LearningCriterion::AbsoluteError)
      return makeBranchAssignment(criterion, labels, k, dummyLeafStats, weights,
                                  sizes, maeLeafYs, maeLeafWs);
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
    const double impurity = objective.objective() / nOutputs;
    const double gain = totalWeight * (parentImp - impurity);
    const double score = gain - outerParams.minGainSplit -
        outerParams.branchingPenalty * static_cast<double>(occupied - 2);
    if (!std::isfinite(score) || score <= eps)
      return;
    if (score < result.regularizedGain ||
        (score == result.regularizedGain && occupied >= result.chosenK))
      return;
    result.found = true;
    result.regularizedGain = score;
    result.chosenK = occupied;
    result.assignments = objective.assignments;
    result.partitionSampleCounts = objective.partitionSampleCounts();
    result.impurityDecrease = gain;
    result.childImpurity = impurity;
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
  std::vector<std::vector<double>> flat;
  arma::mat means;
  arma::vec binWeights;
  if (criterion == LearningCriterion::SquaredError) {
    means.zeros(numBins, nOutputs);
    binWeights = arma::vec(weights);
    // Within-bin SSE is constant: clustering weighted means minimizes merged SSE.
    for (size_t b = 0; b < numBins; ++b)
      if (weights[b] > 0.0)
        for (size_t o = 0; o < nOutputs; ++o)
          means(b, o) = stats[b][o][0] / weights[b];
  } else if (criterion != LearningCriterion::AbsoluteError) {
    flat = flattenNestedBinStats(stats, dim);
  }
  for (size_t k = 2; k <= std::min(numBins, treeNumPartitions); ++k) {
    std::vector<size_t> labels;
    if (criterion == LearningCriterion::SquaredError)
      algorithms::initAssignmentsWeightedKMeans(means, binWeights, k, rng, labels);
    else if (criterion == LearningCriterion::AbsoluteError)
      algorithms::roundRobinBinAssignments(numBins, k, labels);
    else
      algorithms::seedBinAssignmentsKMeans(k, numBins, dim, flat, sizes,
                                           weights, rng, labels);
    auto objective = makeObjective(labels, k);
    observe(*objective); // Retain feasible secondary seeds independently of the root.
    if (!root.empty() && (criterion == LearningCriterion::AbsoluteError ||
                          rootImpurity <= objective->objective())) {
      labels = root;
      objective = makeObjective(labels, k);
    }
    if (criterion != LearningCriterion::AbsoluteError ||
        mae_branch_config::coordinateDescentEnabled())
      coordinateDescent(k, *objective, rng, cdParams.maxIters, cdParams.patience,
                         hasNanRoutingBin, observe);
  }

  if (result.found) {
    // An unseen missing value follows the largest branch, as in finite-bin CD.
    if (hasNanRoutingBin && sizes.back() == 0)
      result.assignments.back() =
          missing_values::partition_with_max_count_min_index_tie(result.partitionSampleCounts);
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
    if (const auto *aggregate = dynamic_cast<const
            leaf_aggregate::LeafAggregationBranchAssignment<std::vector<double>> *>(
                objective.get())) {
      if (criterion == LearningCriterion::SquaredError)
        result.partitionAggStats = aggregate->aggregatedPartitionStats();
      else
        result.partitionClassCounts = aggregate->aggregatedPartitionStats();
      result.partitionWeights = aggregate->aggregatedPartitionWeights();
    }
    result.childImpurity = objective->objective() / nOutputs;
    result.impurityDecrease = totalWeight * (parentImp - result.childImpurity);
    result.regularizedGain = result.impurityDecrease - outerParams.minGainSplit -
        outerParams.branchingPenalty * static_cast<double>(result.chosenK - 2);
  }
  return result;
}

void markShapeFunctionNodeAsLeaf(ShapeFunctionNode &node) {
  node.isLeaf = true;
  node.informationGain = 0.0;
  node.regularizedGain = 0.0;
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
  best.regularizedGain = search.regularizedGain;
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
    const std::function<void(
        ShapeBestBranchingState &, const ShapeBranchAssignmentSearchResult &,
        const std::vector<std::vector<std::vector<double>>> &)> &
        applyTaskFields) {
  if (!search.found || search.assignments.empty())
    return false;
  if (!std::isfinite(search.regularizedGain) ||
      search.regularizedGain <= std::numeric_limits<double>::epsilon())
    return false;
  if (search.regularizedGain < best.regularizedGain ||
      (search.regularizedGain == best.regularizedGain &&
       search.chosenK >= best.branching.numPartitionsUsed))
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
