/**
 * @file algorithms/TAO/TreeAlternatingOptimization.cpp
 * @brief Shared TAO driver: per-node routing search and bottom-up DFS loop.
 */

#include "algorithms/TAO/TreeAlternatingOptimization.h"

#include "Discretizers/ClassificationDiscretizer.h"
#include "algorithms/TAO/TaoObjective.h"

#include <limits>
#include <queue>
#include <stdexcept>
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
      const float v = X(node.routingFeature, col);
      const size_t part = node.routeFeatureValueToPartition(v);
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

double totalSampleWeight(const arma::Row<float> &sampleWeights,
                         arma::uword numSamples) {
  if (sampleWeights.is_empty())
    return static_cast<double>(numSamples);
  double sum = 0.0;
  for (arma::uword i = 0; i < sampleWeights.n_elem; ++i)
    sum += static_cast<double>(sampleWeights(i));
  if (sum <= 0.0)
    return static_cast<double>(numSamples);
  return sum;
}

} // namespace

bool optimizeNodeInPlace(
    TaoAdapter &adapter,
    const std::vector<std::vector<arma::uword>> &nodeSamples, size_t nodeIdx,
    double lambda) {
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

  const double nTotal =
      totalSampleWeight(adapter.sampleWeights(), X.n_cols);
  TaoObjective objective(care, X, lambda, nTotal);
  const double currScore = objective.scoreCurrent(node);
  const double dummyScore = objective.scoreDummy();

  double bestSingleScore = -std::numeric_limits<double>::infinity();
  bool haveSingle = false;
  size_t bestFeature = 0;
  std::vector<float> bestThresholds;
  std::vector<size_t> bestBinToPartition;
  arma::uvec featOne(1);

  for (size_t f = 0; f < numFeatures; ++f) {
    featOne(0) = static_cast<arma::uword>(f);
    auto disc = makeClassificationDiscretizer(routerCriterion);
    disc->Train(care.Xexp, featOne, care.yexp, k, innerParams.minLeafSize,
                innerParams.minGainSplit, innerParams.maxDepth,
                innerParams.maxLeafNodes, care.wexp);

    std::vector<float> thresholds;
    std::vector<size_t> binToPartition;
    const double score =
        objective.scoreDiscretizer(f, *disc, thresholds, binToPartition);
    if (score > bestSingleScore) {
      bestSingleScore = score;
      bestFeature = f;
      bestThresholds = std::move(thresholds);
      bestBinToPartition = std::move(binToPartition);
      haveSingle = true;
    }
  }

  if (dummyScore >= currScore && dummyScore >= bestSingleScore) {
    node.isLeaf = false;
    node.routingFeature = 0;
    node.innerThresholds = {std::numeric_limits<float>::infinity()};
    node.binToPartition = {objective.dummyChild()};
    node.nanPredictionPartition = objective.dummyChild();
    node.numPartitions = k;
    return true;
  }
  if (haveSingle && bestSingleScore > currScore &&
      bestSingleScore > dummyScore) {
    node.isLeaf = false;
    node.routingFeature = bestFeature;
    node.innerThresholds = std::move(bestThresholds);
    node.binToPartition = std::move(bestBinToPartition);
    node.nanPredictionPartition = objective.dummyChild();
    node.numPartitions = k;
    return true;
  }
  return false;
}

void optimize(TaoAdapter &adapter, size_t nRuns, double lambda) {
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
      if (optimizeNodeInPlace(adapter, nodeSamples, nodeIdx, lambda)) {
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
