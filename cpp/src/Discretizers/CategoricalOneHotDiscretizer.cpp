/**
 * @file Discretizers/CategoricalOneHotDiscretizer.cpp
 */

#include "Discretizers/CategoricalOneHotDiscretizer.h"

#include "Criterion.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

double totalWeight(const arma::Row<float> &w, const std::vector<size_t> &samples) {
  double sum = 0.0;
  for (size_t i : samples)
    sum += static_cast<double>(w(i));
  return sum;
}

std::vector<double> classCountsForSamples(const arma::Row<size_t> &y,
                                          const arma::Row<float> &w,
                                          const std::vector<size_t> &samples,
                                          size_t numClasses) {
  std::vector<double> counts(numClasses, 0.0);
  for (size_t i : samples)
    counts[y(i)] += static_cast<double>(w(i));
  return counts;
}

std::vector<double> regressionStatsForSamples(const arma::Row<float> &y,
                                            const arma::Row<float> &w,
                                            const std::vector<size_t> &samples) {
  std::vector<double> stats(2, 0.0);
  for (size_t i : samples) {
    const double v = static_cast<double>(y(i));
    const double wi = static_cast<double>(w(i));
    stats[0] += wi * v;
    stats[1] += wi * v * v;
  }
  return stats;
}

std::vector<float> ysForSamples(const arma::Row<float> &y,
                                const std::vector<size_t> &samples) {
  std::vector<float> out;
  out.reserve(samples.size());
  for (size_t i : samples)
    out.push_back(y(i));
  return out;
}

std::vector<float> wsForSamples(const arma::Row<float> &w,
                                const std::vector<size_t> &samples) {
  std::vector<float> out;
  out.reserve(samples.size());
  for (size_t i : samples)
    out.push_back(w(i));
  return out;
}

size_t majorityClass(const std::vector<double> &counts) {
  return static_cast<size_t>(std::distance(
      counts.begin(), std::max_element(counts.begin(), counts.end())));
}

bool isActiveValue(float v) { return v >= 0.5f; }

size_t dominantActiveCategory(const arma::fmat &X,
                              const std::vector<size_t> &featureIndices,
                              const std::vector<size_t> &samples) {
  for (size_t feat : featureIndices) {
    bool allActive = true;
    for (size_t i : samples) {
      if (!isActiveValue(X(feat, i))) {
        allActive = false;
        break;
      }
    }
    if (allActive)
      return feat;
  }
  return SIZE_MAX;
}

struct FrontierItem {
  double informationGain;
  size_t nodeId;
  size_t height;
  std::vector<size_t> samples;

  bool operator<(const FrontierItem &o) const {
    return informationGain < o.informationGain;
  }
};

} // namespace

OneHotDiscretizerCriterion
parseOneHotDiscretizerCriterion(const std::string &raw, bool classification) {
  std::string s = raw;
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (classification) {
    if (s == "gini")
      return OneHotDiscretizerCriterion::Gini;
    if (s == "entropy" || s == "log_loss")
      return OneHotDiscretizerCriterion::Entropy;
    throw std::invalid_argument(
        "classification criterion must be 'gini' or 'entropy'");
  }
  if (s == "squared_error" || s == "mse")
    return OneHotDiscretizerCriterion::SquaredError;
  if (s == "absolute_error" || s == "mae")
    return OneHotDiscretizerCriterion::AbsoluteError;
  throw std::invalid_argument(
      "regression criterion must be 'squared_error'/'mse' or "
      "'absolute_error'/'mae'");
}

bool CategoricalOneHotDiscretizerBase::isActive(float v) { return v >= 0.5f; }

void CategoricalOneHotDiscretizerBase::trainClassification(
    const arma::fmat &X, const arma::uvec &featureIndices,
    const arma::Row<size_t> &y, size_t numClasses, size_t minLeafSize,
    double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  if (numClasses < 2)
    throw std::invalid_argument("numClasses must be >= 2");
  isClassification_ = true;
  numClasses_ = numClasses;
  featureIndices_.assign(featureIndices.begin(), featureIndices.end());
  yClass_ = y;
  yReg_.clear();
  if (sampleWeights.n_elem == 0) {
    sampleWeights_.set_size(X.n_cols);
    sampleWeights_.ones();
  } else {
    if (sampleWeights.n_elem != X.n_cols)
      throw std::invalid_argument("sample_weights length must match X.n_cols");
    sampleWeights_ = sampleWeights;
  }
  for (size_t f : featureIndices_) {
    if (f >= X.n_rows)
      throw std::invalid_argument("feature index out of range for X");
  }
  buildTree(X, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
}

void CategoricalOneHotDiscretizerBase::trainRegression(
    const arma::fmat &X, const arma::uvec &featureIndices,
    const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
    size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  isClassification_ = false;
  numClasses_ = 0;
  featureIndices_.assign(featureIndices.begin(), featureIndices.end());
  yReg_ = y;
  yClass_.clear();
  if (sampleWeights.n_elem == 0) {
    sampleWeights_.set_size(X.n_cols);
    sampleWeights_.ones();
  } else {
    if (sampleWeights.n_elem != X.n_cols)
      throw std::invalid_argument("sample_weights length must match X.n_cols");
    sampleWeights_ = sampleWeights;
  }
  for (size_t f : featureIndices_) {
    if (f >= X.n_rows)
      throw std::invalid_argument("feature index out of range for X");
  }
  buildTree(X, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
}

void CategoricalOneHotDiscretizerBase::setCriterion(OneHotDiscretizerCriterion c) {
  criterion_ = c;
}

double CategoricalOneHotDiscretizerBase::impurity(
    const std::vector<size_t> &samples) const {
  const double wTot = totalWeight(sampleWeights_, samples);
  if (wTot <= 0.0)
    return 0.0;
  if (isClassification_) {
    const auto counts =
        classCountsForSamples(yClass_, sampleWeights_, samples, numClasses_);
    if (criterion_ == OneHotDiscretizerCriterion::Gini)
      return Criterion::gini(counts, wTot);
    return Criterion::entropy(counts, wTot);
  }
  if (criterion_ == OneHotDiscretizerCriterion::SquaredError) {
    const auto stats = regressionStatsForSamples(yReg_, sampleWeights_, samples);
    return Criterion::squaredError(stats, wTot);
  }
  const auto ys = ysForSamples(yReg_, samples);
  const auto ws = wsForSamples(sampleWeights_, samples);
  return Criterion::absoluteError(ys, ws).mae;
}

bool CategoricalOneHotDiscretizerBase::bestSplit(
    const arma::fmat &X, const std::vector<size_t> &samples, double minGainSplit,
    size_t minLeafSize, double &gainOut, size_t &featureOut,
    std::vector<size_t> &activeOut, std::vector<size_t> &inactiveOut) const {
  const double parentImp = impurity(samples);
  const double wTot = totalWeight(sampleWeights_, samples);
  if (wTot <= 0.0)
    return false;

  double bestGain = -std::numeric_limits<double>::infinity();
  bool found = false;

  for (size_t feat : featureIndices_) {
    std::vector<size_t> active;
    std::vector<size_t> inactive;
    active.reserve(samples.size());
    inactive.reserve(samples.size());
    for (size_t i : samples) {
      if (isActive(X(feat, i)))
        active.push_back(i);
      else
        inactive.push_back(i);
    }
    if (active.size() < minLeafSize || inactive.size() < minLeafSize)
      continue;
    const double childImp =
        (totalWeight(sampleWeights_, inactive) * impurity(inactive) +
         totalWeight(sampleWeights_, active) * impurity(active)) /
        wTot;
    const double gain = parentImp - childImp;
    if (gain > bestGain) {
      bestGain = gain;
      featureOut = feat;
      activeOut = std::move(active);
      inactiveOut = std::move(inactive);
      found = true;
    }
  }
  if (!found || bestGain + 1e-15 < minGainSplit)
    return false;
  gainOut = bestGain;
  return true;
}

size_t CategoricalOneHotDiscretizerBase::appendLeaf(const std::vector<size_t> &samples,
                                                size_t categoryFeature) {
  LeafRecord leaf;
  leaf.samples = samples;
  leaf.categoryFeature = categoryFeature;
  leaves_.push_back(std::move(leaf));
  return leaves_.size() - 1;
}

void CategoricalOneHotDiscretizerBase::finalizeLeafOutputs() {
  inSampleDiscretizations_.clear();
  leafNumSamples_.clear();
  leafNodeWeights_.clear();
  leafStats_.clear();
  binPredictionsClass_.clear();
  binPredictionsReg_.clear();

  inSampleDiscretizations_.reserve(leaves_.size());
  leafNumSamples_.reserve(leaves_.size());
  leafNodeWeights_.reserve(leaves_.size());
  leafStats_.reserve(leaves_.size());

  for (const auto &leaf : leaves_) {
    inSampleDiscretizations_.push_back(leaf.samples);
    leafNumSamples_.push_back(leaf.samples.size());
    const double wTot = totalWeight(sampleWeights_, leaf.samples);
    leafNodeWeights_.push_back(wTot);

    if (isClassification_) {
      auto counts = classCountsForSamples(yClass_, sampleWeights_, leaf.samples,
                                          numClasses_);
      leafStats_.push_back(std::move(counts));
      binPredictionsClass_.push_back(majorityClass(leafStats_.back()));
    } else if (criterion_ == OneHotDiscretizerCriterion::SquaredError) {
      auto stats = regressionStatsForSamples(yReg_, sampleWeights_, leaf.samples);
      leafStats_.push_back(stats);
      binPredictionsReg_.push_back(
          wTot > 0.0 ? static_cast<float>(stats[0] / wTot) : 0.f);
    } else {
      const auto ys = ysForSamples(yReg_, leaf.samples);
      const auto ws = wsForSamples(sampleWeights_, leaf.samples);
      const auto maeStats = Criterion::absoluteError(ys, ws);
      leafStats_.push_back({});
      binPredictionsReg_.push_back(static_cast<float>(maeStats.median));
    }
  }
  numLeaves_ = leaves_.size();
  markTrained();
}

void CategoricalOneHotDiscretizerBase::buildTree(const arma::fmat &X,
                                             size_t minLeafSize,
                                             double minGainSplit, size_t maxDepth,
                                             size_t maxLeafNodes) {
  resetTrainedOutputs();
  routing_.clear();
  leaves_.clear();
  routing_.push_back(RoutingNode{});

  std::vector<size_t> allSamples(X.n_cols);
  for (size_t i = 0; i < X.n_cols; ++i)
    allSamples[i] = i;

  const auto makeLeafNode = [&](size_t leafBin) {
    RoutingNode n;
    n.isLeaf = true;
    n.leafBin = leafBin;
    return n;
  };

  double gain0 = 0.0;
  size_t feat0 = 0;
  std::vector<size_t> active0, inactive0;
  if (!bestSplit(X, allSamples, minGainSplit, minLeafSize, gain0, feat0, active0,
                 inactive0)) {
    const size_t cat =
        dominantActiveCategory(X, featureIndices_, allSamples);
    routing_[0] = makeLeafNode(appendLeaf(allSamples, cat));
    finalizeLeafOutputs();
    return;
  }

  const size_t activeBin = appendLeaf(active0, feat0);
  const size_t inactiveId = 1;
  routing_.push_back(RoutingNode{});
  routing_[0].isLeaf = false;
  routing_[0].splitFeature = feat0;
  routing_[0].inactiveChild = inactiveId;
  routing_[0].activeLeafBin = activeBin;

  const bool bestFirst = maxLeafNodes > 0;
  std::vector<FrontierItem> frontier;
  frontier.push_back(FrontierItem{gain0, inactiveId, 1, std::move(inactive0)});

  while (!frontier.empty()) {
    size_t pick = 0;
    if (bestFirst) {
      pick = static_cast<size_t>(std::distance(
          frontier.begin(),
          std::max_element(frontier.begin(), frontier.end())));
    } else {
      pick = frontier.size() - 1;
    }
    FrontierItem item = std::move(frontier[pick]);
    frontier.erase(frontier.begin() + static_cast<std::ptrdiff_t>(pick));

    if (maxLeafNodes > 0 && leaves_.size() >= maxLeafNodes) {
      const size_t cat =
          dominantActiveCategory(X, featureIndices_, item.samples);
      routing_[item.nodeId] = makeLeafNode(appendLeaf(item.samples, cat));
      continue;
    }
    if (maxDepth > 0 && item.height >= maxDepth) {
      const size_t cat =
          dominantActiveCategory(X, featureIndices_, item.samples);
      routing_[item.nodeId] = makeLeafNode(appendLeaf(item.samples, cat));
      continue;
    }

    double gain = 0.0;
    size_t feat = 0;
    std::vector<size_t> active, inactive;
    if (!bestSplit(X, item.samples, minGainSplit, minLeafSize, gain, feat, active,
                   inactive)) {
      const size_t cat =
          dominantActiveCategory(X, featureIndices_, item.samples);
      routing_[item.nodeId] = makeLeafNode(appendLeaf(item.samples, cat));
      continue;
    }

    const size_t actBin = appendLeaf(active, feat);
    const size_t childId = routing_.size();
    routing_.push_back(RoutingNode{});
    routing_[item.nodeId].isLeaf = false;
    routing_[item.nodeId].splitFeature = feat;
    routing_[item.nodeId].inactiveChild = childId;
    routing_[item.nodeId].activeLeafBin = actBin;
    frontier.push_back(
        FrontierItem{gain, childId, item.height + 1, std::move(inactive)});
  }

  finalizeLeafOutputs();
}

size_t CategoricalOneHotDiscretizerBase::routeOne(const arma::fmat &X,
                                              arma::uword col) const {
  ensureTrained();
  size_t nodeId = 0;
  while (!routing_[nodeId].isLeaf) {
    const RoutingNode &n = routing_[nodeId];
    if (isActive(X(n.splitFeature, col)))
      return n.activeLeafBin;
    nodeId = n.inactiveChild;
  }
  return routing_[nodeId].leafBin;
}

void CategoricalOneHotDiscretizerBase::transform(const arma::fmat &X,
                                             arma::Row<size_t> &binLoc) const {
  ensureTrained();
  binLoc.set_size(X.n_cols);
  for (arma::uword c = 0; c < X.n_cols; ++c)
    binLoc(c) = routeOne(X, c);
}

size_t CategoricalOneHotDiscretizerBase::routeToBin(
    const std::vector<float> &featureValues) const {
  if (featureValues.size() != featureIndices_.size())
    throw std::invalid_argument(
        "featureValues length must match trained featureIndices count");
  ensureTrained();
  size_t nodeId = 0;
  while (!routing_[nodeId].isLeaf) {
    const RoutingNode &n = routing_[nodeId];
    const auto it =
        std::find(featureIndices_.begin(), featureIndices_.end(), n.splitFeature);
    if (it == featureIndices_.end())
      throw std::runtime_error("routing split feature not in featureIndices");
    const size_t pos = static_cast<size_t>(it - featureIndices_.begin());
    if (isActive(featureValues[pos]))
      return n.activeLeafBin;
    nodeId = n.inactiveChild;
  }
  return routing_[nodeId].leafBin;
}

void CategoricalClassificationDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<size_t> &y,
    size_t numClasses, size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes, const arma::Row<float> &sampleWeights) {
  trainClassification(X, features, y, numClasses, minLeafSize, minGainSplit,
                      maxDepth, maxLeafNodes, sampleWeights);
}

void CategoricalRegressionDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<float> &y,
    size_t minLeafSize, double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  trainRegression(X, features, y, minLeafSize, minGainSplit, maxDepth,
                  maxLeafNodes, sampleWeights);
}
