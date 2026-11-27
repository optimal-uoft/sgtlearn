#include "Discretizers/pair/PairRegressionDiscretizer.h"

#include "Criterion.h"
#include "algorithms/missing_values.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

PairRegressionDiscretizer::PairRegressionDiscretizer(
    LearningCriterion criterion, FeatureInfo first, FeatureInfo second)
    : PairDiscretizerMixin(criterion, std::move(first), std::move(second)) {
  if (criterion_ != LearningCriterion::SquaredError &&
      criterion_ != LearningCriterion::AbsoluteError)
    throw std::invalid_argument("pair CART requires a regression criterion");
}

double PairRegressionDiscretizer::impurity(
    const std::vector<size_t> &samples, const arma::Mat<float> &y,
    const arma::Row<float> &weights) const {
  if (samples.empty())
    return 0.0;
  if (criterion_ == LearningCriterion::SquaredError) {
    std::vector<std::vector<double>> stats(
        y.n_rows, std::vector<double>(2, 0.0));
    double totalWeight = 0.0;
    for (size_t sample : samples) {
      const double w = weights(sample);
      totalWeight += w;
      for (arma::uword o = 0; o < y.n_rows; ++o) {
        const double value = y(o, sample);
        stats[o][0] += w * value;
        stats[o][1] += w * value * value;
      }
    }
    return Criterion::squaredError(stats, totalWeight);
  }

  double total = 0.0;
  for (arma::uword o = 0; o < y.n_rows; ++o) {
    std::vector<float> values;
    std::vector<float> sampleWeights;
    values.reserve(samples.size());
    sampleWeights.reserve(samples.size());
    for (size_t sample : samples) {
      values.push_back(y(o, sample));
      sampleWeights.push_back(weights(sample));
    }
    total += Criterion::absoluteError(values, sampleWeights).mae;
  }
  return total;
}

PairSplit PairRegressionDiscretizer::bestSplit(
    const PairBuildNode &node, const arma::fmat &X, const arma::Mat<float> &y,
    const arma::Row<float> &weights, size_t minLeafSize,
    double totalWeight) const {
  PairSplit best;
  double bestChildScore = std::numeric_limits<double>::infinity();
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
    const double missingWeight = sumWeight(missing);
    const double missingWeightedImp =
        missingWeight * impurity(missing, y, weights);

    const auto consider = [&](size_t rawFeature, double threshold,
                              std::vector<size_t> left,
                              std::vector<size_t> right) {
      if (left.size() < minLeafSize || right.size() < minLeafSize)
        return;
      const double leftWeight = sumWeight(left);
      const double rightWeight = sumWeight(right);
      const double childScore =
          node.weight > 0.0
              ? (leftWeight * impurity(left, y, weights) +
                 rightWeight * impurity(right, y, weights) +
                 missingWeightedImp) /
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
      best.left = std::move(left);
      best.right = std::move(right);
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
        consider(rawFeature, 0.5, std::move(left), std::move(right));
      }
      continue;
    }

    const size_t rawFeature = axes_[axis].indices(0);
    std::vector<size_t> order = std::move(valid);
    std::stable_sort(order.begin(), order.end(),
                     [&X, rawFeature](size_t a, size_t b) {
                       return X(rawFeature, a) < X(rawFeature, b);
                     });
    if (criterion_ == LearningCriterion::SquaredError) {
      const size_t nOutputs = static_cast<size_t>(y.n_rows);
      std::vector<std::vector<double>> leftStats(nOutputs,
                                                 std::vector<double>(2, 0.0));
      std::vector<std::vector<double>> rightStats(nOutputs,
                                                  std::vector<double>(2, 0.0));
      double leftWeight = 0.0;
      double rightWeight = 0.0;
      for (size_t sample : order) {
        const double w = weights(sample);
        rightWeight += w;
        for (size_t o = 0; o < nOutputs; ++o) {
          const double v = y(static_cast<arma::uword>(o), sample);
          rightStats[o][0] += w * v;
          rightStats[o][1] += w * v * v;
        }
      }
      size_t bestCut = 0;
      double bestCutThreshold = 0.0;
      for (size_t i = 0; i + 1 < order.size(); ++i) {
        const size_t sample = order[i];
        const double w = weights(sample);
        leftWeight += w;
        rightWeight -= w;
        for (size_t o = 0; o < nOutputs; ++o) {
          const double v = y(static_cast<arma::uword>(o), sample);
          const double wy = w * v;
          const double wy2 = wy * v;
          leftStats[o][0] += wy;
          leftStats[o][1] += wy2;
          rightStats[o][0] -= wy;
          rightStats[o][1] -= wy2;
        }
        const size_t leftCount = i + 1;
        if (leftCount < minLeafSize || order.size() - leftCount < minLeafSize)
          continue;
        const float previous = X(rawFeature, sample);
        const float current = X(rawFeature, order[i + 1]);
        if (current <= previous + 1e-7F)
          continue;
        double threshold = static_cast<double>(previous) / 2.0 +
                           static_cast<double>(current) / 2.0;
        if (!std::isfinite(threshold) || threshold == current)
          threshold = previous;
        const double childScore =
            node.weight > 0.0
                ? (leftWeight * Criterion::squaredError(leftStats, leftWeight) +
                   rightWeight *
                       Criterion::squaredError(rightStats, rightWeight) +
                   missingWeightedImp) /
                      node.weight
                : 0.0;
        if (childScore >=
            bestChildScore - std::numeric_limits<double>::epsilon())
          continue;
        bestChildScore = childScore;
        bestCut = leftCount;
        bestCutThreshold = threshold;
      }
      if (bestCut > 0) {
        best.found = true;
        best.featurePosition = axis;
        best.rawFeature = rawFeature;
        best.featureType = FeatureType::Continuous;
        best.threshold = bestCutThreshold;
        best.left.assign(order.begin(), order.begin() + bestCut);
        best.right.assign(order.begin() + bestCut, order.end());
        best.missing = missing;
        best.gain = totalWeight > 0.0
                        ? (node.weight / totalWeight) *
                              (node.impurity - bestChildScore)
                        : 0.0;
      }
      continue;
    }
    for (size_t i = minLeafSize; i + minLeafSize <= order.size(); ++i) {
      const float previous = X(rawFeature, order[i - 1]);
      const float current = X(rawFeature, order[i]);
      if (current <= previous + 1e-7F)
        continue;
      double threshold = static_cast<double>(previous) / 2.0 +
                         static_cast<double>(current) / 2.0;
      if (!std::isfinite(threshold) || threshold == current)
        threshold = previous;
      consider(rawFeature, threshold,
               std::vector<size_t>(order.begin(), order.begin() + i),
               std::vector<size_t>(order.begin() + i, order.end()));
    }
  }
  return best;
}

void PairRegressionDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Mat<float> &y,
    size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes, const arma::Row<float> &sampleWeights) {
  this->resetTrainedOutputs();
  if (y.n_cols != X.n_cols)
    throw std::invalid_argument("invalid pair CART training shapes");
  const arma::Row<float> weights = preparePairWeights(X, features, sampleWeights);

  const auto makeNode = [&y, &weights, this](std::vector<size_t> samples,
                                             size_t depth) {
    PairBuildNode node;
    node.samples = std::move(samples);
    node.depth = depth;
    const bool keepStats = (criterion_ == LearningCriterion::SquaredError);
    if (keepStats)
      node.stats.assign(y.n_rows, std::vector<double>(2, 0.0));
    for (size_t sample : node.samples) {
      const double w = weights(sample);
      node.weight += w;
      if (keepStats) {
        for (arma::uword o = 0; o < y.n_rows; ++o) {
          const double value = y(o, sample);
          node.stats[o][0] += w * value;
          node.stats[o][1] += w * value * value;
        }
      }
    }
    node.impurity = impurity(node.samples, y, weights);
    return node;
  };

  buildPairTree(
      X.n_cols, minGainSplit, maxDepth, maxLeafNodes, makeNode,
      [&](const PairBuildNode &node, double totalWeight) {
        return bestSplit(node, X, y, weights, minLeafSize, totalWeight);
      },
      this->inSampleDiscretizations_, this->leafStats_, this->leafNumSamples_,
      this->leafNodeWeights_, this->numLeaves_);
  this->markTrained();
}
