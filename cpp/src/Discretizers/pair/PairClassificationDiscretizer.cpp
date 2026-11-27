#include "Discretizers/pair/PairClassificationDiscretizer.h"

#include "Criterion.h"
#include "algorithms/missing_values.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

PairClassificationDiscretizer::PairClassificationDiscretizer(
    LearningCriterion criterion, FeatureInfo first, FeatureInfo second)
    : PairDiscretizerMixin(criterion, std::move(first), std::move(second)) {}

double PairClassificationDiscretizer::impurity(
    const std::vector<std::vector<double>> &stats) const {
  if (criterion_ == LearningCriterion::Gini)
    return Criterion::gini(stats);
  if (criterion_ == LearningCriterion::Entropy)
    return Criterion::entropy(stats);
  throw std::invalid_argument("pair CART requires a classification criterion");
}

PairSplit PairClassificationDiscretizer::bestSplit(
    const PairBuildNode &node, const arma::fmat &X, const arma::Mat<size_t> &y,
    const std::vector<size_t> &classes, const arma::Row<float> &weights,
    size_t minLeafSize, double totalWeight) const {
  PairSplit best;
  double bestChildScore = std::numeric_limits<double>::infinity();

  const auto makeStats = [&y, &classes, &weights](
                             const std::vector<size_t> &samples) {
    std::vector<std::vector<double>> stats(classes.size());
    for (size_t o = 0; o < classes.size(); ++o)
      stats[o].assign(classes[o], 0.0);
    for (size_t sample : samples)
      for (size_t o = 0; o < classes.size(); ++o)
        stats[o][y(o, sample)] += weights(sample);
    return stats;
  };
  const auto sumWeight = [&weights](const std::vector<size_t> &samples) {
    double total = 0.0;
    for (size_t sample : samples)
      total += weights(sample);
    return total;
  };

  for (size_t axis = 0; axis < 2; ++axis) {
    std::vector<size_t> valid;
    std::vector<size_t> missing;
    for (size_t sample : node.samples)
      (axisMissing(axis, X, sample) ? missing : valid).push_back(sample);

    const auto missingStats = makeStats(missing);
    const double missingWeight = sumWeight(missing);
    const auto consider = [&](size_t rawFeature, double threshold,
                              std::vector<size_t> leftSamples,
                              std::vector<size_t> rightSamples,
                              const std::vector<std::vector<double>> &leftStats,
                              const std::vector<std::vector<double>> &rightStats,
                              double leftWeight, double rightWeight) {
      if (leftSamples.size() < minLeafSize ||
          rightSamples.size() < minLeafSize)
        return;
      const double childScore =
          node.weight > 0.0
              ? (leftWeight * impurity(leftStats) +
                 rightWeight * impurity(rightStats) +
                 missingWeight * impurity(missingStats)) /
                    node.weight
              : 0.0;
      if (childScore >=
          bestChildScore - std::numeric_limits<double>::epsilon())
        return;
      best.found = true;
      bestChildScore = childScore;
      best.featurePosition = axis;
      best.rawFeature = rawFeature;
      best.featureType = axes_[axis].type;
      best.threshold = threshold;
      best.left = std::move(leftSamples);
      best.right = std::move(rightSamples);
      best.missing = missing;
      best.gain = totalWeight > 0.0
                      ? (node.weight / totalWeight) *
                            (node.impurity - childScore)
                      : 0.0;
    };

    if (axes_[axis].type == FeatureType::Categorical) {
      for (size_t rawFeature : axes_[axis].indices) {
        std::vector<size_t> left;
        std::vector<size_t> right;
        for (size_t sample : valid)
          (X(rawFeature, sample) >= 0.5F ? right : left).push_back(sample);
        consider(rawFeature, 0.5, left, right, makeStats(left),
                 makeStats(right), sumWeight(left), sumWeight(right));
      }
      continue;
    }

    const size_t rawFeature = axes_[axis].indices(0);
    std::vector<size_t> order = std::move(valid);
    std::stable_sort(order.begin(), order.end(),
                     [&X, rawFeature](size_t a, size_t b) {
                       return X(rawFeature, a) < X(rawFeature, b);
                     });

    std::vector<std::vector<double>> left(classes.size());
    std::vector<std::vector<double>> right = makeStats(order);
    for (size_t o = 0; o < classes.size(); ++o)
      left[o].assign(classes[o], 0.0);
    double leftWeight = 0.0;
    const double validWeight = node.weight - missingWeight;

    for (size_t i = 1; i < order.size(); ++i) {
      const size_t moved = order[i - 1];
      const double w = weights(moved);
      leftWeight += w;
      for (size_t o = 0; o < classes.size(); ++o) {
        left[o][y(o, moved)] += w;
        right[o][y(o, moved)] -= w;
      }
      if (i < minLeafSize)
        continue;
      if (order.size() - i < minLeafSize)
        break;

      const float previous = X(rawFeature, order[i - 1]);
      const float current = X(rawFeature, order[i]);
      if (current <= previous + 1e-7F)
        continue;
      double threshold = static_cast<double>(previous) / 2.0 +
                         static_cast<double>(current) / 2.0;
      if (!std::isfinite(threshold) || threshold == current)
        threshold = previous;
      consider(
          rawFeature, threshold,
          std::vector<size_t>(order.begin(), order.begin() + i),
          std::vector<size_t>(order.begin() + i, order.end()), left, right,
          leftWeight, validWeight - leftWeight);
    }
  }
  return best;
}

void PairClassificationDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Mat<size_t> &y,
    const std::vector<size_t> &classes, size_t minLeafSize, double minGainSplit,
    size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  this->resetTrainedOutputs();
  if (y.n_cols != X.n_cols || classes.size() != y.n_rows)
    throw std::invalid_argument("invalid pair CART training shapes");
  const arma::Row<float> weights = preparePairWeights(X, features, sampleWeights);

  const auto makeNode = [&y, &classes, &weights,
                         this](std::vector<size_t> samples, size_t depth) {
    PairBuildNode node;
    node.samples = std::move(samples);
    node.depth = depth;
    node.stats.resize(classes.size());
    for (size_t o = 0; o < classes.size(); ++o)
      node.stats[o].assign(classes[o], 0.0);
    for (size_t sample : node.samples) {
      const double w = weights(sample);
      node.weight += w;
      for (size_t o = 0; o < classes.size(); ++o)
        node.stats[o][y(o, sample)] += w;
    }
    node.impurity = impurity(node.stats);
    return node;
  };

  buildPairTree(
      X.n_cols, minGainSplit, maxDepth, maxLeafNodes, makeNode,
      [&](const PairBuildNode &node, double totalWeight) {
        return bestSplit(node, X, y, classes, weights, minLeafSize,
                         totalWeight);
      },
      this->inSampleDiscretizations_, this->leafStats_, this->leafNumSamples_,
      this->leafNodeWeights_, this->numLeaves_);
  this->markTrained();
}
