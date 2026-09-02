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
    : criterion_(criterion), axes_{std::move(first), std::move(second)} {
  axisOffsets_[1] = axes_[0].indices.n_elem;
  axisOffsets_[2] = axisOffsets_[1] + axes_[1].indices.n_elem;
  if (axisOffsets_[1] == 0 || axisOffsets_[2] == axisOffsets_[1])
    throw std::invalid_argument("pair CART logical features cannot be empty");
  for (const FeatureInfo &axis : axes_)
    if (axis.type == FeatureType::Continuous && axis.indices.n_elem != 1)
      throw std::invalid_argument(
          "pair CART continuous logical features require one column");
  routingFeatures_ = arma::join_cols(axes_[0].indices, axes_[1].indices);
  if (criterion_ != LearningCriterion::SquaredError &&
      criterion_ != LearningCriterion::AbsoluteError)
    throw std::invalid_argument("pair CART requires a regression criterion");
}

bool PairRegressionDiscretizer::axisMissing(
    size_t axis, const arma::fmat &X, size_t sample) const {
  if (axes_[axis].type == FeatureType::Continuous)
    return !missing_values::is_finite(X(axes_[axis].indices(0), sample));
  for (size_t raw : axes_[axis].indices)
    if (X(raw, sample) >= 0.5F)
      return false;
  return true;
}

bool PairRegressionDiscretizer::axisMissing(
    size_t axis, const std::vector<float> &values) const {
  if (axes_[axis].type == FeatureType::Continuous)
    return !missing_values::is_finite(values[axisOffsets_[axis]]);
  for (size_t pos = axisOffsets_[axis]; pos < axisOffsets_[axis + 1]; ++pos)
    if (values[pos] >= 0.5F)
      return false;
  return true;
}

size_t PairRegressionDiscretizer::routingPosition(size_t rawFeature) const {
  const auto it =
      std::find(routingFeatures_.begin(), routingFeatures_.end(), rawFeature);
  if (it == routingFeatures_.end())
    throw std::runtime_error("pair CART routing feature not found");
  return static_cast<size_t>(it - routingFeatures_.begin());
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

PairRegressionDiscretizer::Split PairRegressionDiscretizer::bestSplit(
    const BuildNode &node, const arma::fmat &X, const arma::Mat<float> &y,
    const arma::Row<float> &weights, size_t minLeafSize,
    double totalWeight) const {
  Split best;
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
                 missingWeight * impurity(missing, y, weights)) /
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
  tree_.clear();
  if (features.n_elem != routingFeatures_.n_elem || y.n_cols != X.n_cols)
    throw std::invalid_argument("invalid pair CART training shapes");

  arma::Row<float> weights = sampleWeights;
  if (weights.n_elem == 0)
    weights.ones(X.n_cols);
  if (weights.n_elem != X.n_cols)
    throw std::invalid_argument("pair CART sample weight length mismatch");
  for (size_t rawFeature : routingFeatures_)
    if (rawFeature >= X.n_rows)
      throw std::invalid_argument("pair CART feature out of range");

  const auto makeNode = [&y, &weights, this](std::vector<size_t> samples,
                                             size_t depth) {
    BuildNode node;
    node.samples = std::move(samples);
    node.depth = depth;
    node.stats.assign(y.n_rows, std::vector<double>(2, 0.0));
    for (size_t sample : node.samples) {
      const double w = weights(sample);
      node.weight += w;
      for (arma::uword o = 0; o < y.n_rows; ++o) {
        const double value = y(o, sample);
        node.stats[o][0] += w * value;
        node.stats[o][1] += w * value * value;
      }
    }
    node.impurity = impurity(node.samples, y, weights);
    return node;
  };

  std::vector<size_t> all(X.n_cols);
  std::iota(all.begin(), all.end(), 0);
  std::vector<BuildNode> nodes;
  nodes.push_back(makeNode(std::move(all), 0));
  const double totalWeight = nodes.front().weight;
  size_t finiteLeafCount = 1;

  const auto splitNode = [&](size_t index, Split split) {
    const size_t depth = nodes[index].depth;
    const bool missingGoesLeft = split.left.size() >= split.right.size();
    nodes[index].routing.isLeaf = false;
    nodes[index].routing.featurePosition = split.featurePosition;
    nodes[index].routing.rawFeature = split.rawFeature;
    nodes[index].routing.featureType = split.featureType;
    nodes[index].routing.threshold = split.threshold;
    nodes[index].routing.left = nodes.size();
    nodes.push_back(makeNode(std::move(split.left), depth + 1));
    nodes[index].routing.right = nodes.size();
    nodes.push_back(makeNode(std::move(split.right), depth + 1));
    if (split.missing.empty()) {
      nodes[index].routing.missing = missingGoesLeft
                                         ? nodes[index].routing.left
                                         : nodes[index].routing.right;
    } else {
      nodes[index].routing.missing = nodes.size();
      nodes.push_back(makeNode(std::move(split.missing), depth + 1));
    }
    ++finiteLeafCount;
  };

  if (maxLeafNodes == 0) {
    const auto grow = [&](auto &&self, size_t index) -> void {
      if (maxDepth != 0 && nodes[index].depth >= maxDepth)
        return;
      Split split =
          bestSplit(nodes[index], X, y, weights, minLeafSize, totalWeight);
      if (!split.found ||
          split.gain + std::numeric_limits<double>::epsilon() < minGainSplit)
        return;
      splitNode(index, std::move(split));
      self(self, nodes[index].routing.left);
      self(self, nodes[index].routing.right);
      if (nodes[index].routing.missing != nodes[index].routing.left)
        self(self, nodes[index].routing.missing);
    };
    grow(grow, 0);
  } else {
    while (finiteLeafCount < maxLeafNodes) {
      size_t bestIndex = nodes.size();
      Split best;
      for (size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i].routing.isLeaf ||
            (maxDepth != 0 && nodes[i].depth >= maxDepth))
          continue;
        Split candidate =
            bestSplit(nodes[i], X, y, weights, minLeafSize, totalWeight);
        if (!candidate.found ||
            candidate.gain + std::numeric_limits<double>::epsilon() <
                minGainSplit)
          continue;
        if (bestIndex == nodes.size() || candidate.gain > best.gain) {
          bestIndex = i;
          best = std::move(candidate);
        }
      }
      if (bestIndex == nodes.size())
        break;
      splitNode(bestIndex, std::move(best));
    }
  }

  for (BuildNode &node : nodes) {
    if (!node.routing.isLeaf)
      continue;
    node.routing.bin = this->inSampleDiscretizations_.size();
    this->inSampleDiscretizations_.push_back(std::move(node.samples));
    this->leafStats_.push_back(
        criterion_ == LearningCriterion::SquaredError
            ? std::move(node.stats)
            : std::vector<std::vector<double>>{});
    this->leafNumSamples_.push_back(
        this->inSampleDiscretizations_.back().size());
    this->leafNodeWeights_.push_back(node.weight);
  }
  tree_.reserve(nodes.size());
  for (const BuildNode &node : nodes)
    tree_.push_back(node.routing);
  this->numLeaves_ = this->leafStats_.size();
  this->markTrained();
}

size_t PairRegressionDiscretizer::routeValues(
    const std::vector<float> &values) const {
  if (values.size() != routingFeatures_.n_elem)
    throw std::invalid_argument(
        "pair router values do not match routing features");
  size_t index = 0;
  while (!tree_[index].isLeaf) {
    const PairRoutingTreeNode &node = tree_[index];
    if (axisMissing(node.featurePosition, values)) {
      index = node.missing;
      continue;
    }
    const float value = values[routingPosition(node.rawFeature)];
    index = node.featureType == FeatureType::Categorical
                ? (value >= 0.5F ? node.right : node.left)
                : (value <= node.threshold ? node.left : node.right);
  }
  return tree_[index].bin;
}

size_t PairRegressionDiscretizer::routeToBin(
    const std::vector<float> &featureValues) const {
  this->ensureTrained();
  return routeValues(featureValues);
}

void PairRegressionDiscretizer::transform(const arma::fmat &X,
                                          arma::Row<size_t> &binLoc) const {
  this->ensureTrained();
  binLoc.set_size(X.n_cols);
  std::vector<float> values(routingFeatures_.n_elem);
  for (arma::uword i = 0; i < X.n_cols; ++i) {
    for (size_t j = 0; j < routingFeatures_.n_elem; ++j)
      values[j] = X(routingFeatures_(j), i);
    binLoc(i) = routeValues(values);
  }
}
