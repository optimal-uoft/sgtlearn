/**
 * @file Estimators/RegressionShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the regression
 *        shape-generalized tree. Per-node inner fit: discretize -> search
 *        partition counts k in [2, numPartitions] -> coordinate descent (MSE)
 *        or round-robin (MAE) -> record best branch.
 */

#include "Estimators/RegressionShapeGeneralizedTree.h"

#include "Estimators/ShapeFunctions/RegressionShapeFunctionBuilder.h"
#include "Criterion.h"
#include <algorithm>
#include <armadillo>
#include <stdexcept>
#include <vector>

namespace {

double meanAbsoluteDeviationFromMedian(const std::vector<float> &ys,
                                       const std::vector<float> &ws) {
  return Criterion::absoluteError(ys, ws).mae;
}

double weightedMeanFromAggregates(double sumWY, double sumW) {
  return sumW > 0.0 ? sumWY / sumW : 0.0;
}

} // namespace

RegressionShapeGeneralizedTree::RegressionShapeGeneralizedTree(
    LearningCriterion criterion, size_t numPartitions,
    TreeBuildingParams outerParams, TreeBuildingParams innerParams,
    CoordinateDescentParams cdParams, uint64_t random_state,
    FeatureBaggingPickFn featureBagging)
    : criterion_(criterion), numPartitions_(numPartitions),
      outerParams_(outerParams), innerParams_(innerParams), cdParams_(cdParams),
      random_state_(random_state), rng_(),
      featureBagging_(featureBagging
                          ? std::move(featureBagging)
                          : FeatureBaggingPickFn(pickAllFeatureIndices)),
      outerTreeBuilder_(outerParams_.minLeafSize, outerParams_.minGainSplit,
                        outerParams_.maxDepth, outerParams_.maxLeafNodes) {
  if (criterion_ != LearningCriterion::SquaredError &&
      criterion_ != LearningCriterion::AbsoluteError)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree: criterion must be SquaredError or "
        "AbsoluteError");
  if (numPartitions_ < 2)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree: numPartitions must be >= 2");
}



double RegressionShapeGeneralizedTree::impurityAtNode(
    const arma::Row<float> &y, const ShapeFunctionNode &node) const {
  if (criterion_ == LearningCriterion::SquaredError) {
    const auto st = aggregateYSquaredStats(node, y);
    double totalWeight = 0.0;
    for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
      const size_t si = static_cast<size_t>(node.sampleIndices(i));
      totalWeight += static_cast<double>(fitSampleWeights_(si));
    }
    return Criterion::squaredError(st, totalWeight);
  }
  std::vector<float> ys;
  std::vector<float> ws;
  ys.reserve(node.sampleIndices.n_elem);
  ws.reserve(node.sampleIndices.n_elem);
  for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(node.sampleIndices(i));
    ys.push_back(y(si));
    ws.push_back(static_cast<float>(fitSampleWeights_(si)));
  }
  if (criterion_ == LearningCriterion::AbsoluteError) {
    if (ys.size() > 1) {
      std::vector<size_t> order(ys.size());
      for (size_t i = 0; i < order.size(); ++i)
        order[i] = i;
      std::sort(order.begin(), order.end(),
                [&ys](size_t a, size_t b) { return ys[a] < ys[b]; });
      std::vector<float> ysSorted;
      std::vector<float> wsSorted;
      ysSorted.reserve(ys.size());
      wsSorted.reserve(ws.size());
      for (size_t idx : order) {
        ysSorted.push_back(ys[idx]);
        wsSorted.push_back(ws[idx]);
      }
      return meanAbsoluteDeviationFromMedian(ysSorted, wsSorted);
    }
    return meanAbsoluteDeviationFromMedian(ys, ws);
  }
  throw std::invalid_argument(
      "RegressionShapeGeneralizedTree::impurityAtNode: invalid criterion");
}

std::vector<double> RegressionShapeGeneralizedTree::aggregateYSquaredStats(
    const ShapeFunctionNode &node, const arma::Row<float> &y) const {
  std::vector<double> st(2, 0.0);
  for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(node.sampleIndices(i));
    const double v = static_cast<double>(y(si));
    const double w = static_cast<double>(fitSampleWeights_(si));
    st[0] += w * v;
    st[1] += w * v * v;
  }
  return st;
}


void RegressionShapeGeneralizedTree::fit(const arma::fmat &X,
                                         const arma::Row<float> &y,
                                         const arma::Row<float> &sampleWeights) {


  if (X.n_cols != y.n_elem)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::fit: X.n_cols must match y.n_elem");
  if (X.n_rows == 0)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::fit: X must have at least one "
        "feature (row)");

  const size_t n = X.n_cols;
  if (sampleWeights.n_elem != n)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::fit: sample_weights length must "
        "match number of samples");
  fitSampleWeights_ = sampleWeights;

  rng_.seed(static_cast<std::mt19937_64::result_type>(random_state_));

  nodes_.clear();
  childIndices_.clear();
  leafRegressionStats.clear();
  leafNumSamples.clear();
  leafPredictions_.clear();

  fitted_ = false;



  ShapeFunctionNode root;
  root.height = 0;
  root.sampleIndices =
      arma::regspace<arma::uvec>(0, static_cast<arma::uword>(n - 1));
  root.nodeIndex = 0;
  root.numPartitions = numPartitions_;
  root.isLeaf = true;

  if (criterion_ == LearningCriterion::SquaredError) {
    double sumWY = 0.0;
    double sumWY2 = 0.0;
    double rootWeight = 0.0;
    for (arma::uword i = 0; i < root.sampleIndices.n_elem; ++i) {
      const size_t si = static_cast<size_t>(root.sampleIndices(i));
      const double wi = static_cast<double>(fitSampleWeights_(si));
      const double v = static_cast<double>(y(si));
      sumWY += wi * v;
      sumWY2 += wi * v * v;
      rootWeight += wi;
    }
    const std::vector<double> st{sumWY, sumWY2};
    leafRegressionStats.push_back(
        {static_cast<float>(sumWY), static_cast<float>(sumWY2)});
    leafNumSamples.push_back(n);
    leafPredictions_.push_back(weightedMeanFromAggregates(sumWY, rootWeight));
    root.score = Criterion::squaredError(st, rootWeight);
  } else {
    leafRegressionStats.push_back({});
    leafNumSamples.push_back(n);
    std::vector<float> ys;
    std::vector<float> ws;
    ys.reserve(n);
    ws.reserve(n);
    for (arma::uword i = 0; i < root.sampleIndices.n_elem; ++i) {
      const size_t si = static_cast<size_t>(root.sampleIndices(i));
      ys.push_back(y(si));
      ws.push_back(static_cast<float>(fitSampleWeights_(si)));
    }
    leafPredictions_.push_back(
        static_cast<float>(Criterion::absoluteError(ys, ws).median));
    root.score = impurityAtNode(y, root);
  }
  nodes_.push_back(root);

  childIndices_.emplace_back();
  rootIndex_ = 0;

  const arma::uvec featureCandidates =
      arma::regspace<arma::uvec>(0, X.n_rows - 1);

  RegressionShapeFunctionBuilder splitBuilder(*this, X, y, featureCandidates);

  outerTreeBuilder_.buildTree(
      nodes_[0],
      [&splitBuilder](ShapeFunctionNode &node, size_t minLeaf) {
        return splitBuilder.findBestSplit(node, minLeaf);
      },
      [&splitBuilder](ShapeFunctionNode &parent) {
        return splitBuilder.makeChildren(parent);
      },
      [this](ShapeFunctionNode &parent,
             std::vector<ShapeFunctionNode> &children) {


        const size_t pid = parent.nodeIndex;
        nodes_[pid] = parent;
        nodes_[pid].isLeaf = false;
        const size_t numChildPartitions = parent.numPartitions;
        childIndices_[pid].assign(numChildPartitions, 0);
        for (size_t p = 0; p < numChildPartitions; ++p) {
          const size_t cid = nodes_.size();
          children[p].nodeIndex = cid;
          nodes_.push_back(children[p]);
          childIndices_.push_back({});
          childIndices_[pid][p] = cid;
        }

      });



  for (auto &node : nodes_) {
    node.sampleIndices.set_size(0);
    node.sampleBins.clear();
  }

  fitted_ = true;

}


arma::Row<double>
RegressionShapeGeneralizedTree::predict(const arma::fmat &X) const {
  if (!fitted_)
    throw std::logic_error(
        "RegressionShapeGeneralizedTree::predict: model is not fitted");
  if (nodes_.empty())
    throw std::logic_error(
        "RegressionShapeGeneralizedTree::predict: empty tree");
  if (rootIndex_ >= nodes_.size())
    throw std::logic_error(
        "RegressionShapeGeneralizedTree::predict: invalid root index");
  if (childIndices_.size() != nodes_.size())
    throw std::logic_error(
        "RegressionShapeGeneralizedTree::predict: childIndices/nodes size "
        "mismatch");
  if (leafPredictions_.size() != nodes_.size())
    throw std::logic_error(
        "RegressionShapeGeneralizedTree::predict: leafPredictions/nodes size "
        "mismatch");

  const size_t n_samples = X.n_cols;
  arma::Row<double> yhat(n_samples);

  for (size_t s = 0; s < n_samples; ++s) {
    size_t idx = rootIndex_;
    while (true) {
      const auto &node = nodes_[idx];
      if (node.isLeaf) {
        if (node.nodeIndex >= leafPredictions_.size())
          throw std::runtime_error(
              "RegressionShapeGeneralizedTree::predict: leaf prediction index "
              "out of range");
        yhat(s) = leafPredictions_[node.nodeIndex];
        break;
      }
      if (node.routingFeature >= X.n_rows)
        throw std::invalid_argument(
            "RegressionShapeGeneralizedTree::predict: routing feature index "
            "out of range for X");
      const float v = X(node.routingFeature, s);
      const size_t part = node.routeFeatureValueToPartition(v);
      if (part >= childIndices_[idx].size())
        throw std::runtime_error(
            "RegressionShapeGeneralizedTree::predict: child partition out of "
            "range");
      idx = childIndices_[idx][part];
      if (idx >= nodes_.size())
        throw std::runtime_error(
            "RegressionShapeGeneralizedTree::predict: child node index out of "
            "range");
    }
  }
  return yhat;
}


