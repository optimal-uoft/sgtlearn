#pragma once

/**
 * @file algorithms/TreeAlternatingOptimization.h
 * @brief Task-agnostic Tree-Alternating Optimization (TAO) for fitted
 *        shape-generalized trees.
 *
 * TAO refines an already-fitted tree without changing its topology: it sweeps
 * the internal nodes bottom-up and, at each node, replaces the routing rule
 * with the one that maximizes a per-task *reward* over the samples whose
 * routing actually matters (the "care set"). Leaf statistics are refreshed
 * whenever routing changes.
 *
 * Everything except two things is identical across learning tasks, so the
 * generic driver ``optimizeImpl`` is parameterized on a small *task policy*:
 *
 *  1. **The care set + pseudolabels.** Captured by ``childRewards``: given the
 *     leaf each child subtree would route a sample to, the policy returns a
 *     reward per child. A sample is in the care set when those rewards are not
 *     all equal; the pseudolabels are the reward-maximizing children.
 *       - Classification: reward is 1 if that child's leaf predicts the true
 *         class, else 0. Care set = "some children right, some wrong";
 *         pseudolabels = every correct child.
 *       - Regression: reward is the negative training loss of that child's leaf
 *         prediction -- squared error for the squared-error criterion, absolute
 *         error (|.|) for the MAE criterion. Care set = children disagree in
 *         error; pseudolabel = the child with minimal error.
 *
 *  2. **Leaf-statistic refresh.** Captured by ``recomputeLeafStats``:
 *     classification rebuilds per-node class histograms; regression rebuilds
 *     per-leaf constant predictions (mean for squared error, median for
 *     absolute error).
 *
 * The router that each node fits is *always a classifier* predicting a child
 * index, regardless of the tree's task, so a classification discretizer is
 * used for the routing refit in both cases. Because candidate rules (the
 * current rule, a constant router, and the best single-feature fit) are scored
 * by their summed real reward and the current rule is always a candidate, each
 * node step is non-worsening; with the leaf refresh that follows, a full sweep
 * never increases training loss.
 */

#include "Estimators/ClassificationShapeGeneralizedTree.h"
#include "Estimators/RegressionShapeGeneralizedTree.h"
#include "Estimators/ShapeGeneralizedTree.h"
#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"

#include "Criterion.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "Domain/LearningCriterion.h"
#include "algorithms/missing_values.h"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tao {

namespace detail {

inline size_t argMax(const std::vector<double> &counts) {
  return static_cast<size_t>(std::distance(
      counts.begin(), std::max_element(counts.begin(), counts.end())));
}

inline size_t routeValue(float value, const std::vector<float> &thresholds,
                         const std::vector<size_t> &binToPartition,
                         size_t nanPartition) {
  if (binToPartition.empty())
    return nanPartition;
  if (!missing_values::is_finite(value))
    return nanPartition;
  const auto it =
      std::lower_bound(thresholds.begin(), thresholds.end(), value);
  size_t bin = static_cast<size_t>(it - thresholds.begin());
  if (bin >= binToPartition.size())
    bin = binToPartition.size() - 1;
  return binToPartition[bin];
}

/**
 * Recompute, for every node, the set of training columns that reach it under
 * the tree's current routing rules (Python ``point_idxs``). Sample index sets
 * are cleared after ``fit``, so TAO rebuilds them by routing the full design
 * matrix from the root each time the topology's routing changes.
 *
 * Routing is top-down, so a BFS (parents before children) is the natural
 * order here: a node's column set must be complete before its children are
 * routed.
 */
inline std::vector<std::vector<arma::uword>>
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

/**
 * Walk from ``startNode`` down the current routing rules to the leaf that
 * sample ``col`` would reach, and return that leaf's node index. The per-task
 * prediction is read from the leaf afterwards by the task policy.
 */
inline size_t walkToLeaf(const std::vector<ShapeFunctionNode> &nodes,
                         const std::vector<std::vector<size_t>> &childIndices,
                         size_t startNode, const arma::fmat &X,
                         arma::uword col) {
  size_t idx = startNode;
  while (!nodes[idx].isLeaf) {
    const ShapeFunctionNode &node = nodes[idx];
    const float v = X(node.routingFeature, col);
    const size_t part = node.routeFeatureValueToPartition(v);
    const auto &children = childIndices[idx];
    if (part >= children.size())
      throw std::runtime_error("tao::walkToLeaf: routed partition out of range");
    idx = children[part];
  }
  return idx;
}

/** Recompute per-node weighted class histograms from current sample sets. */
inline void recomputeClassCounts(
    std::vector<std::vector<double>> &classCounts,
    const std::vector<std::vector<arma::uword>> &nodeSamples,
    const arma::Row<size_t> &y, const arma::Row<float> &sampleWeights,
    size_t numClasses) {
  for (size_t ni = 0; ni < nodeSamples.size(); ++ni) {
    std::vector<double> counts(numClasses, 0.0);
    for (arma::uword col : nodeSamples[ni])
      counts[y(col)] += static_cast<double>(sampleWeights(col));
    classCounts[ni] = std::move(counts);
  }
}

} // namespace detail

/**
 * Compile-time contract for a TAO task policy. Anything passed to
 * ``optimizeImpl`` must expose the fitted tree's mutable routing structure plus
 * the two task-specific hooks; this concept makes a mismatch a clear error at
 * the call site instead of a deep template failure.
 *
 *  - ``nodes`` / ``childIndices``: mutable views of the fitted topology TAO
 *    rewrites in place.
 *  - ``rootIndex`` / ``X`` / ``numFeatures``: routing entry point and design
 *    matrix (column-major, one sample per column).
 *  - ``routerCriterion`` / ``innerParams``: the *classification* criterion and
 *    discretizer params for the per-node routing refit (the router is always a
 *    classifier over child ids, regardless of the tree's task).
 *  - ``childRewards``: fills a per-child reward for one sample given the leaf
 *    each child subtree routes it to (the care set + pseudolabels fall out of
 *    these rewards).
 *  - ``recomputeLeafStats``: refreshes per-leaf statistics from current sample
 *    sets after routing changes.
 */
template <typename T>
concept TaoTask = requires(
    T t, const std::vector<size_t> &childLeaves, arma::uword col,
    std::vector<double> &reward,
    const std::vector<std::vector<arma::uword>> &nodeSamples) {
  { t.nodes() } -> std::same_as<std::vector<ShapeFunctionNode> &>;
  { t.childIndices() } -> std::same_as<std::vector<std::vector<size_t>> &>;
  { t.rootIndex() } -> std::convertible_to<size_t>;
  { t.X() } -> std::same_as<const arma::fmat &>;
  { t.numFeatures() } -> std::convertible_to<size_t>;
  { t.routerCriterion() } -> std::same_as<LearningCriterion>;
  { t.innerParams() } -> std::convertible_to<const TreeBuildingParams &>;
  { t.childRewards(childLeaves, col, reward) } -> std::same_as<void>;
  { t.recomputeLeafStats(nodeSamples) } -> std::same_as<void>;
};

/**
 * Optimize a single internal node's routing rule in place against the current
 * sample sets, returning whether the rule changed. Candidate rules (keep the
 * current rule, a constant router, or the best single-feature classifier over
 * child ids) are scored by their summed per-sample reward; since the current
 * rule is always a candidate, the chosen rule never lowers the node's reward.
 */
template <TaoTask Task>
bool optimizeNodeInPlace(
    Task &task, const std::vector<std::vector<arma::uword>> &nodeSamples,
    size_t nodeIdx, double lambda) {
  auto &nodes = task.nodes();
  auto &childIndices = task.childIndices();
  const arma::fmat &X = task.X();
  const size_t numFeatures = task.numFeatures();
  const LearningCriterion routerCriterion = task.routerCriterion();
  const TreeBuildingParams innerParams = task.innerParams();

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

  // --- Care set: per-sample reward over the k children. A sample cares only
  // if some child is strictly better; its pseudolabels are the reward-
  // maximizing children. The reward row is kept for scoring. ---
  std::vector<arma::uword> careCols;
  std::vector<std::vector<double>> careRewards; // per care row, length k
  std::vector<std::vector<size_t>> careGoodChildren;
  std::vector<size_t> childLeaves(k);
  std::vector<double> reward(k);
  for (arma::uword col : samples) {
    for (size_t c = 0; c < k; ++c)
      childLeaves[c] =
          detail::walkToLeaf(nodes, childIndices, children[c], X, col);
    task.childRewards(childLeaves, col, reward);

    double best = reward[0];
    double worst = reward[0];
    for (size_t c = 1; c < k; ++c) {
      best = std::max(best, reward[c]);
      worst = std::min(worst, reward[c]);
    }
    const double tol = 1e-9 * (1.0 + std::fabs(best));
    if (best - worst <= tol)
      continue; // routing cannot change this sample's outcome

    std::vector<size_t> good;
    for (size_t c = 0; c < k; ++c)
      if (best - reward[c] <= tol)
        good.push_back(c);

    careCols.push_back(col);
    careRewards.push_back(reward);
    careGoodChildren.push_back(std::move(good));
  }

  const size_t nCare = careCols.size();
  if (nCare == 0)
    return false;

  // --- Flatten the (possibly multi-)pseudolabel care set to single-label
  // (multi_to_single): one duplicate per reward-maximizing child, target =
  // that child index. The router is a plain classifier over child ids. ---
  size_t nExpanded = 0;
  for (const auto &good : careGoodChildren)
    nExpanded += good.size();
  arma::fmat Xexp(numFeatures, nExpanded);
  arma::Row<size_t> yexp(nExpanded);
  std::vector<double> childCounts(k, 0.0);
  size_t pos = 0;
  for (size_t i = 0; i < nCare; ++i) {
    for (size_t child : careGoodChildren[i]) {
      Xexp.col(pos) = X.col(careCols[i]);
      yexp(pos) = child;
      childCounts[child] += 1.0;
      ++pos;
    }
  }
  const arma::Row<float> wexp(nExpanded, arma::fill::ones);

  // Constant-router target: most frequently reward-maximizing child. Also the
  // fallback partition for non-finite values of any candidate feature.
  const size_t dummyChild = detail::argMax(childCounts);

  // --- Reward of the node's current routing rule. ---
  double currReward = 0.0;
  for (size_t i = 0; i < nCare; ++i) {
    const float v = X(node.routingFeature, careCols[i]);
    const size_t child = node.routeFeatureValueToPartition(v);
    currReward += careRewards[i][child];
  }
  const double currScore = currReward / static_cast<double>(nCare) - lambda;

  // --- Reward of the constant (dummy) router. ---
  double dummyReward = 0.0;
  for (size_t i = 0; i < nCare; ++i)
    dummyReward += careRewards[i][dummyChild];
  const double dummyScore = dummyReward / static_cast<double>(nCare);

  // --- Best single-feature classifier predicting the child. ---
  double bestSingleScore = -std::numeric_limits<double>::infinity();
  bool haveSingle = false;
  size_t bestFeature = 0;
  std::vector<float> bestThresholds;
  std::vector<size_t> bestBinToPartition;
  arma::uvec featOne(1);

  for (size_t f = 0; f < numFeatures; ++f) {
    featOne(0) = static_cast<arma::uword>(f);
    auto disc = makeClassificationDiscretizer(routerCriterion);
    disc->Train(Xexp, featOne, yexp, k, innerParams.minLeafSize,
                innerParams.minGainSplit, innerParams.maxDepth,
                innerParams.maxLeafNodes, wexp);
    const size_t numBins = disc->numLeaves();
    if (numBins < 1)
      continue;

    const std::vector<double> &thr = disc->thresholds();
    const std::vector<std::vector<double>> &leafStats = disc->leafStats();
    std::vector<float> thresholds(thr.size());
    for (size_t b = 0; b < thr.size(); ++b)
      thresholds[b] = static_cast<float>(thr[b]);
    std::vector<size_t> binToPartition(leafStats.size());
    for (size_t b = 0; b < leafStats.size(); ++b)
      binToPartition[b] = detail::argMax(leafStats[b]);

    double rewardSum = 0.0;
    for (size_t i = 0; i < nCare; ++i) {
      const float v = X(f, careCols[i]);
      const size_t child =
          detail::routeValue(v, thresholds, binToPartition, dummyChild);
      rewardSum += careRewards[i][child];
    }
    const double score = rewardSum / static_cast<double>(nCare) - lambda;
    if (score > bestSingleScore) {
      bestSingleScore = score;
      bestFeature = f;
      bestThresholds = std::move(thresholds);
      bestBinToPartition = std::move(binToPartition);
      haveSingle = true;
    }
  }

  // --- Pick the winner, mirroring the original tie-breaking order. ---
  if (dummyScore >= currScore && dummyScore >= bestSingleScore) {
    node.isLeaf = false;
    node.routingFeature = 0;
    node.innerThresholds = {std::numeric_limits<float>::infinity()};
    node.binToPartition = {dummyChild};
    node.nanPredictionPartition = dummyChild;
    node.numPartitions = k;
    return true;
  }
  if (haveSingle && bestSingleScore > currScore &&
      bestSingleScore > dummyScore) {
    node.isLeaf = false;
    node.routingFeature = bestFeature;
    node.innerThresholds = std::move(bestThresholds);
    node.binToPartition = std::move(bestBinToPartition);
    node.nanPredictionPartition = dummyChild;
    node.numPartitions = k;
    return true;
  }
  return false;
}

/**
 * Generic TAO driver. ``Task`` is a small policy object (see
 * ``ClassificationTask`` / ``RegressionTask``) satisfying ``TaoTask``.
 *
 * The bottom-up sweep is a direct post-order DFS: a node is optimized only
 * after its whole subtree is final, which the recursion gives for free without
 * a separate ordering pass. Whenever a node's rule changes, the sample-to-node
 * assignment and leaf statistics are refreshed so later (shallower) nodes route
 * children down to current leaves when computing pseudolabels.
 *
 * @param task    task policy bound to the fitted tree, design matrix, targets,
 *                and sample weights.
 * @param nRuns   maximum bottom-up sweeps; stops early when a full sweep
 *                changes nothing.
 * @param lambda  per-split complexity penalty subtracted from a routing rule's
 *                normalized reward; 0 disables it.
 */
template <TaoTask Task>
void optimizeImpl(Task &task, size_t nRuns, double lambda) {
  auto &nodes = task.nodes();
  auto &childIndices = task.childIndices();
  const size_t rootIndex = task.rootIndex();
  const arma::fmat &X = task.X();

  if (nodes.empty())
    return;

  for (size_t run = 0; run < nRuns; ++run) {
    bool changed = false;
    std::vector<std::vector<arma::uword>> nodeSamples =
        detail::computeNodeSamples(nodes, childIndices, rootIndex, X);

    // Post-order DFS via an explicit self-recursive lambda: recurse into every
    // child, then optimize this node. No precomputed node ordering is needed.
    const auto visit = [&](auto &&self, size_t nodeIdx) -> void {
      if (nodes[nodeIdx].isLeaf)
        return;
      for (size_t child : childIndices[nodeIdx])
        self(self, child);
      if (optimizeNodeInPlace(task, nodeSamples, nodeIdx, lambda)) {
        changed = true;
        // Routing changed: refresh sample partitions + leaf statistics so the
        // ancestors visited next see current state.
        nodeSamples =
            detail::computeNodeSamples(nodes, childIndices, rootIndex, X);
        task.recomputeLeafStats(nodeSamples);
      }
    };
    visit(visit, rootIndex);

    if (!changed)
      break;
  }
}

/**
 * Classification task policy: rewards are 0/1 correctness of each child leaf's
 * argmax prediction, and the leaf refresh rebuilds per-node class histograms.
 */
class ClassificationTask {
public:
  ClassificationTask(ClassificationShapeGeneralizedTree &tree,
                     const arma::fmat &X, const arma::Row<size_t> &y,
                     const arma::Row<float> &sampleWeights)
      : tree_(tree), X_(X), y_(y), w_(sampleWeights) {}

  std::vector<ShapeFunctionNode> &nodes() {
    return const_cast<std::vector<ShapeFunctionNode> &>(tree_.nodes());
  }
  std::vector<std::vector<size_t>> &childIndices() {
    return const_cast<std::vector<std::vector<size_t>> &>(tree_.childIndices());
  }
  size_t rootIndex() const { return tree_.rootIndex(); }
  const arma::fmat &X() const { return X_; }
  size_t numFeatures() const { return static_cast<size_t>(X_.n_rows); }
  LearningCriterion routerCriterion() const { return tree_.criterion(); }
  const TreeBuildingParams &innerParams() const { return tree_.innerParams(); }

  void childRewards(const std::vector<size_t> &childLeaves, arma::uword col,
                    std::vector<double> &reward) const {
    const size_t label = y_(col);
    for (size_t c = 0; c < childLeaves.size(); ++c)
      reward[c] = (detail::argMax(tree_.classCounts[childLeaves[c]]) == label)
                      ? 1.0
                      : 0.0;
  }

  void recomputeLeafStats(
      const std::vector<std::vector<arma::uword>> &nodeSamples) {
    detail::recomputeClassCounts(tree_.classCounts, nodeSamples, y_, w_,
                                 tree_.numClasses());
  }

private:
  ClassificationShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<size_t> &y_;
  const arma::Row<float> &w_;
};

/**
 * Regression task policy. The per-child reward is the negative training loss of
 * that child leaf's constant prediction, matched to the tree's criterion so a
 * sweep is non-worsening in that loss:
 *   - squared error: reward = -(pred - y)^2, leaf refresh = weighted mean;
 *   - absolute error (MAE): reward = -|pred - y|, leaf refresh = weighted
 *     median.
 * Either way the reward-maximizing child is the minimal-error one.
 *
 * The router itself is still a classifier over child ids, so a Gini
 * classification discretizer is used for the routing refit regardless of the
 * regression criterion.
 */
class RegressionTask {
public:
  RegressionTask(RegressionShapeGeneralizedTree &tree, const arma::fmat &X,
                 const arma::Row<float> &y, const arma::Row<float> &sampleWeights)
      : tree_(tree), X_(X), y_(y), w_(sampleWeights),
        squared_(tree.criterion() == LearningCriterion::SquaredError) {}

  std::vector<ShapeFunctionNode> &nodes() {
    return const_cast<std::vector<ShapeFunctionNode> &>(tree_.nodes());
  }
  std::vector<std::vector<size_t>> &childIndices() {
    return const_cast<std::vector<std::vector<size_t>> &>(tree_.childIndices());
  }
  size_t rootIndex() const { return tree_.rootIndex(); }
  const arma::fmat &X() const { return X_; }
  size_t numFeatures() const { return static_cast<size_t>(X_.n_rows); }
  LearningCriterion routerCriterion() const { return LearningCriterion::Gini; }
  const TreeBuildingParams &innerParams() const { return tree_.innerParams(); }

  void childRewards(const std::vector<size_t> &childLeaves, arma::uword col,
                    std::vector<double> &reward) const {
    const auto &leafPred = tree_.leafPredictions();
    const double target = static_cast<double>(y_(col));
    for (size_t c = 0; c < childLeaves.size(); ++c) {
      const double d = leafPred[childLeaves[c]] - target;
      // Negative training loss so the reward-maximizing child is the
      // minimal-error one; match the tree's loss so a sweep is non-worsening.
      reward[c] = squared_ ? -(d * d) : -std::fabs(d);
    }
  }

  void recomputeLeafStats(
      const std::vector<std::vector<arma::uword>> &nodeSamples) {
    auto &nodes = const_cast<std::vector<ShapeFunctionNode> &>(tree_.nodes());
    auto &leafPred =
        const_cast<std::vector<double> &>(tree_.leafPredictions());

    for (size_t ni = 0; ni < nodes.size(); ++ni) {
      if (!nodes[ni].isLeaf)
        continue;
      const std::vector<arma::uword> &cols = nodeSamples[ni];

      if (squared_) {
        double sumWY = 0.0;
        double sumWY2 = 0.0;
        double sumW = 0.0;
        for (arma::uword col : cols) {
          const double wi = static_cast<double>(w_(col));
          const double v = static_cast<double>(y_(col));
          sumWY += wi * v;
          sumWY2 += wi * v * v;
          sumW += wi;
        }
        leafPred[ni] = sumW > 0.0 ? sumWY / sumW : 0.0;
        if (ni < tree_.leafRegressionStats.size())
          tree_.leafRegressionStats[ni] = {static_cast<float>(sumWY),
                                           static_cast<float>(sumWY2)};
      } else {
        std::vector<float> ys;
        std::vector<float> ws;
        ys.reserve(cols.size());
        ws.reserve(cols.size());
        for (arma::uword col : cols) {
          ys.push_back(y_(col));
          ws.push_back(w_(col));
        }
        leafPred[ni] = Criterion::absoluteError(ys, ws).median;
        if (ni < tree_.leafRegressionStats.size())
          tree_.leafRegressionStats[ni].clear();
      }
      if (ni < tree_.leafNumSamples.size())
        tree_.leafNumSamples[ni] = cols.size();
    }
  }

private:
  RegressionShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<float> &y_;
  const arma::Row<float> &w_;
  bool squared_;
};

/**
 * Run TAO on a fitted classification shape-generalized tree, in place.
 *
 * @param tree           fitted estimator; routing rules + leaf histograms are
 *                       mutated, topology is preserved.
 * @param X              (numFeatures, numSamples) training design matrix.
 * @param y              (numSamples,) integer class labels in [0, numClasses).
 * @param sampleWeights  (numSamples,) per-sample weights for leaf histograms.
 * @param nRuns          maximum bottom-up sweeps.
 * @param lambda         per-split complexity penalty.
 */
inline void optimizeClassification(ClassificationShapeGeneralizedTree &tree,
                                   const arma::fmat &X,
                                   const arma::Row<size_t> &y,
                                   const arma::Row<float> &sampleWeights,
                                   size_t nRuns = 10, double lambda = 0.0) {
  if (!tree.isFitted())
    throw std::runtime_error("tao::optimizeClassification: tree is not fitted");
  if (X.n_cols != y.n_elem)
    throw std::invalid_argument(
        "tao::optimizeClassification: X.n_cols must match y.n_elem");
  if (sampleWeights.n_elem != y.n_elem)
    throw std::invalid_argument(
        "tao::optimizeClassification: sampleWeights length must match y.n_elem");

  ClassificationTask task(tree, X, y, sampleWeights);
  optimizeImpl(task, nRuns, lambda);
}

/**
 * Run TAO on a fitted regression shape-generalized tree, in place.
 *
 * @param tree           fitted estimator; routing rules + leaf predictions are
 *                       mutated, topology is preserved.
 * @param X              (numFeatures, numSamples) training design matrix.
 * @param y              (numSamples,) real-valued targets.
 * @param sampleWeights  (numSamples,) per-sample weights for leaf predictions.
 * @param nRuns          maximum bottom-up sweeps.
 * @param lambda         per-split complexity penalty.
 */
inline void optimizeRegression(RegressionShapeGeneralizedTree &tree,
                               const arma::fmat &X, const arma::Row<float> &y,
                               const arma::Row<float> &sampleWeights,
                               size_t nRuns = 10, double lambda = 0.0) {
  if (!tree.isFitted())
    throw std::runtime_error("tao::optimizeRegression: tree is not fitted");
  if (X.n_cols != y.n_elem)
    throw std::invalid_argument(
        "tao::optimizeRegression: X.n_cols must match y.n_elem");
  if (sampleWeights.n_elem != y.n_elem)
    throw std::invalid_argument(
        "tao::optimizeRegression: sampleWeights length must match y.n_elem");

  RegressionTask task(tree, X, y, sampleWeights);
  optimizeImpl(task, nRuns, lambda);
}

} // namespace tao
