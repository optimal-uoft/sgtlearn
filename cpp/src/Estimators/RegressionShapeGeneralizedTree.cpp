/**
 * @file Estimators/RegressionShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the regression
 *        shape-generalized tree. Per-node inner fit: discretize -> search
 *        partition counts k in [2, numPartitions] -> coordinate descent (MSE)
 *        or round-robin (MAE) -> record best branch.
 */

#include <memory>
#include <utility>
#include <cstdint>
#include <cstddef>
#include "Estimators/RegressionShapeGeneralizedTree.h"

#include "Criterion.h"
#include "Discretizers/factories/DiscretizerFactories.h"
#include "Discretizers/RegressionDiscretizer.h"
#include "Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h"
#include <algorithm>
#include <armadillo>
#include <limits>
#include <numeric>
#include <span>
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

struct PartitionMoments {
  // Per-output aggregates; sumWY[o], sumWY2[o]. sumW is shared across outputs.
  std::vector<double> sumWY;
  std::vector<double> sumWY2;
  double sumW = 0.0;
};

PartitionMoments aggregatePartitionFromBins(
    const std::vector<std::vector<double>> &binStats,
    const std::vector<double> &binWeights,
    const std::vector<size_t> &binToPartition, size_t partition,
    size_t nOutputs) {
  PartitionMoments out;
  out.sumWY.assign(nOutputs, 0.0);
  out.sumWY2.assign(nOutputs, 0.0);
  for (size_t b = 0; b < binStats.size(); ++b) {
    if (binToPartition[b] != partition)
      continue;
    for (size_t o = 0; o < nOutputs; ++o) {
      if (binStats[b].size() >= 2 * (o + 1)) {
        out.sumWY[o] += binStats[b][2 * o];
        out.sumWY2[o] += binStats[b][2 * o + 1];
      }
    }
    if (b < binWeights.size())
      out.sumW += binWeights[b];
  }
  return out;
}

} // namespace

RegressionShapeGeneralizedTree::RegressionShapeGeneralizedTree(
    LearningCriterion criterion, size_t numPartitions,
    TreeBuildingParams outerParams, TreeBuildingParams innerParams,
    CoordinateDescentParams cdParams, uint64_t random_state,
    FeatureBaggingPickFn featureBagging)
    : ShapeGeneralizedTree(criterion, numPartitions, outerParams, innerParams),
      cdParams_(cdParams),
      random_state_(random_state), rng_(),
      featureBagging_(featureBagging
                          ? std::move(featureBagging)
                          : FeatureBaggingPickFn(pickAllFeatureIndices)),
      outerTreeBuilder_(outerParams_.minLeafSize, outerParams_.minGainSplit,
                        outerParams_.maxDepth, outerParams_.maxLeafNodes) {
  if (criterion != LearningCriterion::SquaredError &&
      criterion != LearningCriterion::AbsoluteError)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree: criterion must be SquaredError or "
        "AbsoluteError");
  if (numPartitions < 2)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree: numPartitions must be >= 2");
}



double RegressionShapeGeneralizedTree::impurityAtNode(
    const arma::Mat<float> &y, const ShapeFunctionNode &node) const {
  if (criterion_ == LearningCriterion::SquaredError) {
    const auto st = aggregateYSquaredStats(node, y);
    double totalWeight = 0.0;
    for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
      const size_t si = static_cast<size_t>(node.sampleIndices(i));
      totalWeight += static_cast<double>(fitSampleWeights_(si));
    }
    return Criterion::squaredErrorMulti(st, totalWeight, nOutputs_);
  }
  if (criterion_ != LearningCriterion::AbsoluteError)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::impurityAtNode: invalid criterion");

  // Absolute error: sum per-output MAE about each output's weighted median.
  double total = 0.0;
  for (size_t o = 0; o < nOutputs_; ++o) {
    std::vector<float> ys;
    std::vector<float> ws;
    ys.reserve(node.sampleIndices.n_elem);
    ws.reserve(node.sampleIndices.n_elem);
    for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
      const size_t si = static_cast<size_t>(node.sampleIndices(i));
      ys.push_back(y(static_cast<arma::uword>(o), static_cast<arma::uword>(si)));
      ws.push_back(static_cast<float>(fitSampleWeights_(si)));
    }
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
      total += meanAbsoluteDeviationFromMedian(ysSorted, wsSorted);
    } else {
      total += meanAbsoluteDeviationFromMedian(ys, ws);
    }
  }
  return total;
}

std::vector<double> RegressionShapeGeneralizedTree::aggregateYSquaredStats(
    const ShapeFunctionNode &node, const arma::Mat<float> &y) const {
  std::vector<double> st(2 * nOutputs_, 0.0);
  for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(node.sampleIndices(i));
    const double w = static_cast<double>(fitSampleWeights_(si));
    for (size_t o = 0; o < nOutputs_; ++o) {
      const double v =
          static_cast<double>(y(static_cast<arma::uword>(o), static_cast<arma::uword>(si)));
      st[2 * o] += w * v;
      st[2 * o + 1] += w * v * v;
    }
  }
  return st;
}


void RegressionShapeGeneralizedTree::fit(
    const arma::fmat &X, const arma::Mat<float> &y,
    const arma::Row<float> &sampleWeights,
    const std::vector<FeatureInfo> &features) {


  if (X.n_cols != y.n_cols)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::fit: X.n_cols must match y.n_cols");
  if (y.n_rows == 0)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::fit: y must have at least one output "
        "(row)");
  if (X.n_rows == 0)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::fit: X must have at least one "
        "feature (row)");

  nOutputs_ = static_cast<size_t>(y.n_rows);
  const size_t n = X.n_cols;
  if (sampleWeights.n_elem != n)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::fit: sample_weights length must "
        "match number of samples");
  fitSampleWeights_ = sampleWeights;
  features_ = features;

  rng_.seed(static_cast<std::mt19937_64::result_type>(random_state_));

  nodes_.clear();
  childIndices_.clear();
  leafRegressionStats.clear();
  leafNumSamples.clear();
  leafPredictions_.clear();

  fitted_ = false;
  sumOfNodeImportancesByFeature_.zeros(features.size());
  totalNodeImportanceSum_ = 0.0;
  featureImportance_.reset();

  ShapeFunctionNode root;
  root.height = 0;
  root.sampleIndices =
      arma::regspace<arma::uvec>(0, static_cast<arma::uword>(n - 1));
  root.nodeIndex = 0;
  root.numPartitions = numPartitions_;
  root.isLeaf = true;

  if (criterion_ == LearningCriterion::SquaredError) {
    const std::vector<double> st = aggregateYSquaredStats(root, y);
    double rootWeight = 0.0;
    for (arma::uword i = 0; i < root.sampleIndices.n_elem; ++i) {
      const size_t si = static_cast<size_t>(root.sampleIndices(i));
      rootWeight += static_cast<double>(fitSampleWeights_(si));
    }
    std::vector<float> statsF(st.size());
    std::vector<double> preds(nOutputs_, 0.0);
    for (size_t o = 0; o < nOutputs_; ++o) {
      statsF[2 * o] = static_cast<float>(st[2 * o]);
      statsF[2 * o + 1] = static_cast<float>(st[2 * o + 1]);
      preds[o] = weightedMeanFromAggregates(st[2 * o], rootWeight);
    }
    leafRegressionStats.push_back(std::move(statsF));
    leafNumSamples.push_back(n);
    leafPredictions_.push_back(std::move(preds));
    root.score = Criterion::squaredErrorMulti(st, rootWeight, nOutputs_);
  } else {
    leafRegressionStats.push_back({});
    leafNumSamples.push_back(n);
    std::vector<double> preds(nOutputs_, 0.0);
    for (size_t o = 0; o < nOutputs_; ++o) {
      std::vector<float> ys;
      std::vector<float> ws;
      ys.reserve(n);
      ws.reserve(n);
      for (arma::uword i = 0; i < root.sampleIndices.n_elem; ++i) {
        const size_t si = static_cast<size_t>(root.sampleIndices(i));
        ys.push_back(y(static_cast<arma::uword>(o), static_cast<arma::uword>(si)));
        ws.push_back(static_cast<float>(fitSampleWeights_(si)));
      }
      preds[o] = Criterion::absoluteError(ys, ws).median;
    }
    leafPredictions_.push_back(std::move(preds));
    root.score = impurityAtNode(y, root);
  }
  nodes_.push_back(root);

  childIndices_.emplace_back();
  rootIndex_ = 0;

  const size_t numLogicalFeatures = features_.size();

  const auto findBestSplit =
      [this, &X, &y, numLogicalFeatures](ShapeFunctionNode &node,
                                         size_t minLeaf) -> bool {
        const size_t ns = node.sampleIndices.n_elem;
        node.score = impurityAtNode(y, node);

        if (ns < 2 * minLeaf) {
          markShapeFunctionNodeAsLeaf(node);
          return false;
        }

        const double parentImp = node.score;
        if (parentImp <= outerTreeBuilder_.eps) {
          markShapeFunctionNodeAsLeaf(node);
          return false;
        }

        const arma::uvec &subIdx = node.sampleIndices;
        const arma::fmat Xsub = X.cols(subIdx);
        const arma::Mat<float> ysub = y.cols(subIdx);

        std::vector<size_t> featurePool(numLogicalFeatures);
        std::iota(featurePool.begin(), featurePool.end(), 0);
        const std::vector<size_t> featureSubset = featureBagging_(
            std::span<const size_t>(featurePool.data(), featurePool.size()),
            rng_);

        const size_t xSubCols = static_cast<size_t>(Xsub.n_cols);
        ShapeBestBranchingState best{};

        const auto applyTaskFields =
            [this](ShapeBestBranchingState &state,
                   const ShapeBranchAssignmentSearchResult &search,
                   const std::vector<std::vector<double>> &leafStats) {
              (void)search;
              if (criterion_ == LearningCriterion::SquaredError)
                state.branching.leafStats = leafStats;
              else {
                state.branching.leafStats.clear();
                state.branching.leafStats.resize(leafStats.size());
              }
            };

        for (size_t fi = 0; fi < featureSubset.size(); ++fi) {
          const size_t logicalIdx = featureSubset[fi];
          const FeatureInfo &feature = features_[logicalIdx];

          const arma::Row<float> wsub =
              subSampleWeights(fitSampleWeights_, subIdx);

          auto disc = makeRegressionDiscretizer(criterion_, feature);
          trainRegressionDiscretizer(
              *disc, feature, Xsub, ysub, innerParams_.minLeafSize,
              innerParams_.minGainSplit, innerParams_.maxDepth,
              innerParams_.maxLeafNodes, wsub);
          if (disc->numLeaves() < 2)
            continue;

          const ShapeBranchAssignmentSearchResult featureBest =
              searchShapeBranchAssignmentFromDiscretizer(
                  *disc, criterion_, parentImp, numPartitions_, outerParams_,
                  cdParams_, outerTreeBuilder_.eps, rng_,
                  /*useKMeansSeed=*/false, /*classesPerOutput=*/{}, nOutputs_,
                  criterion_ == LearningCriterion::AbsoluteError ? &ysub
                                                                 : nullptr,
                  criterion_ == LearningCriterion::AbsoluteError ? &wsub
                                                                   : nullptr,
                  xSubCols);
          if (!featureBest.found)
            continue;

          featureHasBetterShapeBranching(
              featureBest, best, logicalIdx, xSubCols, feature.indices,
              std::unique_ptr<InnerDiscretizerBase<double>>(std::move(disc)),
              outerTreeBuilder_.eps, applyTaskFields);
        }

        if (!std::isfinite(best.penalizedChildScore) ||
            best.penalizedChildScore >= std::numeric_limits<double>::infinity() ||
            best.branching.impurityDecrease <= outerTreeBuilder_.eps) {
          markShapeFunctionNodeAsLeaf(node);
          return false;
        }

        node.isLeaf = false;
        node.splitFeatureIndex = best.branching.featureIndex;
        node.routingFeatures.assign(best.routingColumnIndices.begin(),
                                    best.routingColumnIndices.end());
        node.innerDiscretizer = best.winningDiscretizer;
        node.binToPartition = std::move(best.branching.binToPartition);
        node.sampleBins = std::move(best.branching.sampleBins);
        node.numPartitions = best.branching.numPartitionsUsed;
        node.informationGain = best.branching.impurityDecrease;

        if (criterion_ == LearningCriterion::SquaredError)
          node.splitLeafStats = std::move(best.branching.leafStats);
        else
          node.splitLeafStats.clear();
        node.splitBinWeights = std::move(best.binWeights);
        node.binSampleCounts = std::move(best.branching.leafNumSamples);

        return true;
      };

  const auto makeChildren =
      [this, &X, &y](const ShapeFunctionNode &parent)
          -> std::vector<ShapeFunctionNode> {
        const auto buckets = routeSamplesToPartitions(parent, X);
        auto children =
            makeRoutedChildNodes(parent, buckets, numPartitions_);

        if (criterion_ == LearningCriterion::SquaredError) {
          if (parent.splitLeafStats.size() != parent.splitBinWeights.size())
            throw std::runtime_error(
                "RegressionShapeGeneralizedTree::fit: splitBinWeights / "
                "splitLeafStats size mismatch");

          for (size_t p = 0; p < children.size(); ++p) {
            const PartitionMoments moments = aggregatePartitionFromBins(
                parent.splitLeafStats, parent.splitBinWeights,
                parent.binToPartition, p, nOutputs_);
            std::vector<double> agg(2 * nOutputs_, 0.0);
            std::vector<float> aggF(2 * nOutputs_, 0.0f);
            std::vector<double> preds(nOutputs_, 0.0);
            for (size_t o = 0; o < nOutputs_; ++o) {
              agg[2 * o] = moments.sumWY[o];
              agg[2 * o + 1] = moments.sumWY2[o];
              aggF[2 * o] = static_cast<float>(moments.sumWY[o]);
              aggF[2 * o + 1] = static_cast<float>(moments.sumWY2[o]);
              preds[o] = weightedMeanFromAggregates(moments.sumWY[o],
                                                    moments.sumW);
            }
            children[p].score =
                Criterion::squaredErrorMulti(agg, moments.sumW, nOutputs_);
            children[p].isLeaf = true;
            leafRegressionStats.push_back(std::move(aggF));
            leafNumSamples.push_back(children[p].sampleIndices.n_elem);
            leafPredictions_.push_back(std::move(preds));
          }
        } else {
          for (size_t p = 0; p < children.size(); ++p) {
            double scoreSum = 0.0;
            std::vector<double> preds(nOutputs_, 0.0);
            for (size_t o = 0; o < nOutputs_; ++o) {
              std::vector<float> ys;
              std::vector<float> ws;
              ys.reserve(children[p].sampleIndices.n_elem);
              ws.reserve(children[p].sampleIndices.n_elem);
              for (arma::uword j = 0; j < children[p].sampleIndices.n_elem;
                   ++j) {
                const size_t si =
                    static_cast<size_t>(children[p].sampleIndices(j));
                ys.push_back(
                    y(static_cast<arma::uword>(o), static_cast<arma::uword>(si)));
                ws.push_back(static_cast<float>(fitSampleWeights_(si)));
              }
              const auto ae = Criterion::absoluteError(ys, ws);
              scoreSum += ae.mae;
              preds[o] = ae.median;
            }
            children[p].score = scoreSum;
            children[p].isLeaf = true;
            leafRegressionStats.push_back({});
            leafNumSamples.push_back(children[p].sampleIndices.n_elem);
            leafPredictions_.push_back(std::move(preds));
          }
        }

        return children;
      };

  outerTreeBuilder_.buildTree(
      nodes_[0], findBestSplit, makeChildren,
      [this](ShapeFunctionNode &parent,
             std::vector<ShapeFunctionNode> &children) {
        sumOfNodeImportancesByFeature_(parent.splitFeatureIndex) +=
            parent.informationGain;
        totalNodeImportanceSum_ += parent.informationGain;
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

  featureImportance_.zeros(sumOfNodeImportancesByFeature_.n_elem);
  if (totalNodeImportanceSum_ > 0.0)
    featureImportance_ =
        sumOfNodeImportancesByFeature_ / totalNodeImportanceSum_;
  fitted_ = true;
}


arma::Mat<double>
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
  arma::Mat<double> yhat(nOutputs_, n_samples);

  for (size_t s = 0; s < n_samples; ++s) {
    size_t idx = rootIndex_;
    while (true) {
      const auto &node = nodes_[idx];
      if (node.isLeaf) {
        if (node.nodeIndex >= leafPredictions_.size())
          throw std::runtime_error(
              "RegressionShapeGeneralizedTree::predict: leaf prediction index "
              "out of range");
        const std::vector<double> &pred = leafPredictions_[node.nodeIndex];
        for (size_t o = 0; o < nOutputs_; ++o)
          yhat(static_cast<arma::uword>(o), static_cast<arma::uword>(s)) =
              o < pred.size() ? pred[o] : 0.0;
        break;
      }
      const size_t part = node.routeSampleToPartition(X, static_cast<arma::uword>(s));
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


