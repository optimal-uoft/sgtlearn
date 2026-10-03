/**
 * @file algorithms/TAO/TreeAlternatingOptimization.cpp
 * @brief Shared TAO driver: per-node routing search and bottom-up DFS loop.
 */

#include <memory>
#include <cstddef>
#include "algorithms/TAO/TreeAlternatingOptimization.h"

#include "Discretizers/ClassificationDiscretizer.h"
#include "Discretizers/pair/PairClassificationDiscretizer.h"
#include "Discretizers/factories/DiscretizerFactories.h"
#include "Discretizers/InnerDiscretizerBase.h"
#include "algorithms/TAO/TaoObjective.h"
#include "algorithms/missing_values.h"

#include <limits>
#include <optional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tao {
namespace {

std::vector<std::vector<arma::uword>>
computeNodeSamples(const std::vector<ShapeFunctionNode> &nodes,
                   const std::vector<std::vector<size_t>> &childIndices,
                   size_t rootIndex, const arma::fmat &X) {
  std::vector<std::vector<arma::uword>> nodeSamples(nodes.size());
  const arma::uword n = X.n_cols;
  nodeSamples[rootIndex].reserve(static_cast<size_t>(n));
  for (arma::uword c = 0; c < n; ++c)
    nodeSamples[rootIndex].push_back(c);

  std::queue<size_t> frontier;
  frontier.push(rootIndex);
  while (!frontier.empty()) {
    const size_t ni = frontier.front();
    frontier.pop();
    const ShapeFunctionNode &node = nodes[ni];
    if (node.isLeaf)
      continue;
    const auto &children = childIndices[ni];
    for (arma::uword col : nodeSamples[ni]) {
      const size_t part = node.routeSampleToPartition(X, col);
      if (part >= children.size())
        throw std::runtime_error(
            "tao::computeNodeSamples: routed partition out of range");
      nodeSamples[children[part]].push_back(col);
    }
    for (size_t childIdx : children)
      frontier.push(childIdx);
  }
  return nodeSamples;
}

/**
 * Numeric router for ``cut`` on raw ``feature``: finite ``x <= cut.leftMax``
 * -> bin 0, larger -> bin 1, non-finite -> NaN bin 2. Trained as a depth-1
 * discretizer on the finite care samples labelled by side; that split is pure,
 * so the splitter recovers it with its own threshold convention. Writes the
 * bin of every column of ``X`` to ``bins``.
 */
std::unique_ptr<ClassificationDiscretizer>
materializeThresholdCut(LearningCriterion criterion, const NodeCareSet &care,
                        const arma::fmat &X, size_t feature,
                        const ThresholdCut &cut, arma::Row<size_t> &bins) {
  const auto side = [&](arma::uword col) -> size_t {
    const float value = X(static_cast<arma::uword>(feature), col);
    if (!missing_values::is_finite(value))
      return 2;
    return value > cut.leftMax ? 1 : 0;
  };
  std::vector<arma::uword> cols;
  std::vector<size_t> labels;
  for (arma::uword col : care.careCols) {
    const size_t s = side(col);
    if (s == 2)
      continue;
    cols.push_back(col);
    labels.push_back(s);
  }

  arma::uvec featOne = {static_cast<arma::uword>(feature)};
  auto disc =
      makeClassificationDiscretizer(criterion, DiscretizerInputKind::Numeric);
  disc->Train(X.cols(arma::uvec(cols)), featOne, arma::Row<size_t>(labels), 2,
              /*minLeafSize=*/1, /*minGainSplit=*/0.0, /*maxDepth=*/1,
              /*maxLeafNodes=*/2);
  disc->transform(X, bins);

  bool reproduced = disc->numLeaves() == 2;
  for (size_t i = 0; reproduced && i < care.size(); ++i)
    reproduced = bins(care.careCols[i]) == side(care.careCols[i]);
  if (!reproduced)
    throw std::runtime_error(
        "tao: threshold router does not reproduce the exact cut on feature " +
        std::to_string(feature));
  return disc;
}

} // namespace

bool optimizeNodeInPlace(
    TaoAdapter &adapter,
    const std::vector<std::vector<arma::uword>> &nodeSamples, size_t nodeIdx,
    double lambda, double taoPairScale) {
  auto &nodes = adapter.nodes();
  auto &childIndices = adapter.childIndices();
  const arma::fmat &X = adapter.X();
  const size_t numFeatures = adapter.numFeatures();
  const LearningCriterion routerCriterion = adapter.routerCriterion();
  const TreeBuildingParams innerParams = adapter.innerParams();

  ShapeFunctionNode &node = nodes[nodeIdx];
  if (node.isLeaf)
    return false;

  const std::vector<arma::uword> &samples = nodeSamples[nodeIdx];
  if (samples.empty())
    return false;
  const std::vector<size_t> &children = childIndices[nodeIdx];
  const size_t k = children.size();
  if (k < 2)
    return false;

  const NodeCareSet care = adapter.buildCareSet(samples, children);
  if (care.empty())
    return false;

  TaoObjective objective(care, X, lambda,
                         static_cast<double>(samples.size()));
  const double currentScale = node.logicalFeatureIndices.empty()
                                  ? 0.0
                                  : (node.logicalFeatureIndices.size() == 2
                                         ? taoPairScale
                                         : 1.0);
  const double currScore = objective.scoreCurrent(node, currentScale);
  const double dummyScore = objective.scoreDummy();

  double bestSingleScore = -std::numeric_limits<double>::infinity();
  bool haveSingle = false;
  size_t bestFeature = 0;
  std::vector<size_t> bestBinToPartition;
  std::shared_ptr<const InnerDiscretizerBase> bestDiscretizer;
  arma::uvec featOne(1);

  for (size_t f = 0; f < numFeatures; ++f) {
    featOne(0) = static_cast<arma::uword>(f);
    auto disc = makeClassificationDiscretizer(routerCriterion,
                                              DiscretizerInputKind::Numeric);
    disc->Train(care.Xexp, featOne, care.yexp, k, innerParams.minLeafSize,
                innerParams.minGainSplit, innerParams.maxDepth,
                innerParams.maxLeafNodes, care.wexp);

    std::vector<size_t> binToPartition;
    const double score = objective.scoreDiscretizer(
        *disc, binToPartition, /*complexityScale=*/1.0);
    if (score > bestSingleScore) {
      bestSingleScore = score;
      bestFeature = f;
      bestBinToPartition = std::move(binToPartition);
      bestDiscretizer =
          std::shared_ptr<const InnerDiscretizerBase>(std::move(disc));
      haveSingle = true;
    }
  }

  // Two children: also try the exact best threshold on each feature, which
  // the inner-tree fit above only approximates.
  if (k == 2) {
    std::optional<ThresholdCut> bestCut;
    size_t cutFeature = 0;
    for (size_t f = 0; f < numFeatures; ++f) {
      const auto cut = objective.bestThresholdCut(f, innerParams.minLeafSize);
      if (cut && (!bestCut || cut->rewardSum > bestCut->rewardSum)) {
        bestCut = cut;
        cutFeature = f;
      }
    }
    if (bestCut) {
      arma::Row<size_t> bins;
      auto disc = materializeThresholdCut(routerCriterion, care, X, cutFeature,
                                          *bestCut, bins);
      std::vector<size_t> binToPartition = {
          bestCut->leftChild, 1 - bestCut->leftChild, bestCut->nanChild};
      const double score = objective.scoreBinAssignment(
          bins, binToPartition, /*complexityScale=*/1.0);
      if (score > bestSingleScore) {
        bestSingleScore = score;
        bestFeature = cutFeature;
        bestBinToPartition = std::move(binToPartition);
        bestDiscretizer =
            std::shared_ptr<const InnerDiscretizerBase>(std::move(disc));
        haveSingle = true;
      }
    }
  }

  double bestPairScore = -std::numeric_limits<double>::infinity();
  bool havePair = false;
  std::array<size_t, 2> bestPairLogical{};
  std::vector<size_t> bestPairRoutingFeatures;
  std::vector<size_t> bestPairBinToPartition;
  std::shared_ptr<const InnerDiscretizerBase> bestPairDiscretizer;
  for (const RetainedPairCandidate &pair : node.retainedPairCandidates) {
    arma::uvec rawFeatures = arma::join_cols(pair.features[0].indices,
                                            pair.features[1].indices);
    auto disc = std::make_unique<PairClassificationDiscretizer>(
        routerCriterion, pair.features[0], pair.features[1]);
    disc->Train(care.Xexp, rawFeatures, care.yexp, std::vector<size_t>{k},
                innerParams.minLeafSize, innerParams.minGainSplit,
                innerParams.maxDepth, innerParams.maxLeafNodes, care.wexp);
    std::vector<size_t> binToPartition;
    const double score = objective.scoreDiscretizer(
        *disc, binToPartition, taoPairScale);
    if (!havePair || score > bestPairScore ||
        (score == bestPairScore &&
         pair.logicalFeatureIndices < bestPairLogical)) {
      bestPairScore = score;
      bestPairLogical = pair.logicalFeatureIndices;
      bestPairRoutingFeatures.assign(rawFeatures.begin(), rawFeatures.end());
      bestPairBinToPartition = std::move(binToPartition);
      bestPairDiscretizer =
          std::shared_ptr<const InnerDiscretizerBase>(std::move(disc));
      havePair = true;
    }
  }

  if (dummyScore >= currScore && dummyScore >= bestSingleScore &&
      dummyScore >= bestPairScore) {
    node.isLeaf = false;
    node.logicalFeatureIndices.clear();
    node.routingFeatures = {0};
    featOne(0) = 0;
    auto disc = makeClassificationDiscretizer(routerCriterion,
                                              DiscretizerInputKind::Numeric);
    disc->Train(care.Xexp, featOne, care.yexp, k, innerParams.minLeafSize,
                innerParams.minGainSplit, innerParams.maxDepth, 1, care.wexp);
    node.innerDiscretizer =
        std::shared_ptr<const InnerDiscretizerBase>(std::move(disc));
    node.binToPartition = {objective.dummyChild(), objective.dummyChild()};
    node.numPartitions = k;
    node.informationGain = 0.0;
    adapter.refreshNodeBinMetadata(node, samples);
    return true;
  }
  if (haveSingle && bestSingleScore > currScore &&
      bestSingleScore > dummyScore && bestSingleScore >= bestPairScore) {
    node.isLeaf = false;
    node.splitFeatureIndex = bestFeature;
    node.logicalFeatureIndices = {bestFeature};
    node.routingFeatures = {bestFeature};
    node.innerDiscretizer = std::move(bestDiscretizer);
    node.binToPartition = std::move(bestBinToPartition);
    node.numPartitions = k;
    adapter.refreshNodeBinMetadata(node, samples);
    return true;
  }
  if (havePair && bestPairScore > currScore &&
      bestPairScore > dummyScore && bestPairScore > bestSingleScore) {
    node.isLeaf = false;
    node.splitFeatureIndex = bestPairLogical[0];
    node.logicalFeatureIndices.assign(bestPairLogical.begin(),
                                      bestPairLogical.end());
    node.routingFeatures = std::move(bestPairRoutingFeatures);
    node.innerDiscretizer = std::move(bestPairDiscretizer);
    node.binToPartition = std::move(bestPairBinToPartition);
    node.numPartitions = k;
    adapter.refreshNodeBinMetadata(node, samples);
    return true;
  }
  return false;
}

void optimize(TaoAdapter &adapter, size_t nRuns, double lambda,
              double taoPairScale) {
  auto &nodes = adapter.nodes();
  auto &childIndices = adapter.childIndices();
  const size_t rootIndex = adapter.rootIndex();
  const arma::fmat &X = adapter.X();

  if (nodes.empty())
    return;

  for (size_t run = 0; run < nRuns; ++run) {
    bool changed = false;
    std::vector<std::vector<arma::uword>> nodeSamples =
        computeNodeSamples(nodes, childIndices, rootIndex, X);

    std::vector<std::pair<size_t, bool>> stack;
    stack.emplace_back(rootIndex, false);
    while (!stack.empty()) {
      const auto [nodeIdx, expanded] = stack.back();
      stack.pop_back();
      if (nodes[nodeIdx].isLeaf)
        continue;
      if (!expanded) {
        stack.emplace_back(nodeIdx, true);
        for (size_t child : childIndices[nodeIdx])
          stack.emplace_back(child, false);
        continue;
      }
      if (optimizeNodeInPlace(adapter, nodeSamples, nodeIdx, lambda,
                              taoPairScale)) {
        changed = true;
        nodeSamples =
            computeNodeSamples(nodes, childIndices, rootIndex, X);
        adapter.recomputeLeafStats(nodeSamples);
      }
    }

    if (!changed)
      break;
  }
}

} // namespace tao
