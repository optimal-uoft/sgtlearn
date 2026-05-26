/**
 * @file Estimators/RegressionShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the regression
 *        shape-generalized tree. Per-node inner fit: discretize -> search
 *        partition counts k in [2, numPartitions] -> coordinate descent (MSE)
 *        or round-robin (MAE) -> record best branch.
 */

#include "Estimators/RegressionShapeGeneralizedTree.h"

#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "Criterion.h"
#include "Discretizers/RegressionDiscretizer.h"
#include "algorithms/BinPartitionAssignments.h"
#include "algorithms/CoordinateDescent.h"
#include "algorithms/ShapeBranchingTypes.h"
#include <algorithm>
#include <armadillo>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

double meanAbsoluteDeviationFromMedian(const std::vector<float> &ys,
                                       const std::vector<float> &ws) {
  return Criterion::absoluteError(ys, ws).mae;
}

/** Regression: keep CD only if branch objective drops by more than this vs seed. */
constexpr double kCdRegressionKeepCdEps = 1e-10;

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

namespace {

arma::Row<float> subSampleWeights(const arma::Row<float> &weights,
                                  const arma::uvec &subIdx) {
  arma::Row<float> wsub(subIdx.n_elem);
  for (arma::uword j = 0; j < subIdx.n_elem; ++j)
    wsub(j) = weights(subIdx(j));
  return wsub;
}

} // namespace

namespace {

double weightedMeanFromAggregates(double sumWY, double sumW) {
  return sumW > 0.0 ? sumWY / sumW : 0.0;
}

struct PartitionMoments {
  double sumWY = 0.0;
  double sumWY2 = 0.0;
  double sumW = 0.0;
};

PartitionMoments aggregatePartitionFromBins(
    const std::vector<std::vector<double>> &binStats,
    const std::vector<double> &binWeights,
    const std::vector<size_t> &binToPartition, size_t partition) {
  PartitionMoments out;
  for (size_t b = 0; b < binStats.size(); ++b) {
    if (binToPartition[b] != partition)
      continue;
    if (binStats[b].size() >= 2) {
      out.sumWY += binStats[b][0];
      out.sumWY2 += binStats[b][1];
    }
    if (b < binWeights.size())
      out.sumW += binWeights[b];
  }
  return out;
}

} // namespace

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

  outerTreeBuilder_.buildTree(
      nodes_[0],
      [this, &X, &y,
       &featureCandidates](ShapeFunctionNode &node,
                           size_t minLeaf) { // find best shape function
        const size_t ns = node.sampleIndices.n_elem;
        node.score = impurityAtNode(y, node);

        if (ns < 2 * minLeaf) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          node.splitBinWeights.clear();
          node.binSampleCounts.clear();
          return false;
        }

        const double parentImp = node.score;
        if (parentImp <= outerTreeBuilder_.eps) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          node.splitBinWeights.clear();
          node.binSampleCounts.clear();
          return false;
        }

        const arma::uvec &subIdx = node.sampleIndices;
        const arma::fmat Xsub = X.cols(subIdx);
        const arma::Row<float> ysub = y.cols(subIdx);

        std::vector<size_t> featurePool(
            static_cast<size_t>(featureCandidates.n_elem));
        for (arma::uword i = 0; i < featureCandidates.n_elem; ++i)
          featurePool[static_cast<size_t>(i)] =
              static_cast<size_t>(featureCandidates(i));
        const std::vector<size_t> featureSubset = featureBagging_(
            std::span<const size_t>(featurePool.data(), featurePool.size()),
            rng_);

        const size_t xSubCols = static_cast<size_t>(Xsub.n_cols);
        double bestPenalizedChild = std::numeric_limits<double>::infinity();
        ShapeBranchingResult<double> brBest{};
        std::vector<size_t> binSizesForBest;
        std::vector<double> binWeightsForBest;
        arma::uvec featOne(1);

        for (size_t fi = 0; fi < featureSubset.size(); ++fi) {
          const size_t f = featureSubset[fi];
          if (f >= Xsub.n_rows)
            throw std::invalid_argument(
                "RegressionShapeGeneralizedTree::fit: candidate feature index "
                ">= X.n_rows");
          featOne(0) = static_cast<arma::uword>(f);

          const arma::Row<float> wsub =
              subSampleWeights(fitSampleWeights_, subIdx);

          auto disc = makeRegressionDiscretizer(criterion_);
          disc->Train(Xsub, featOne, ysub, innerParams_.minLeafSize,
                      innerParams_.minGainSplit, innerParams_.maxDepth,
                      innerParams_.maxLeafNodes, wsub);
          const size_t B = disc->numLeaves();
          if (B == 0)
            continue;

          auto &stats = disc->leafStats();
          auto &sizes = disc->leafNumSamples();
          auto &binWeights = disc->leafNodeWeights();
          const auto sizes_copy = sizes;
          const auto &perBinCols = disc->inSampleDiscretizations();

          std::vector<std::vector<float>> maeLeafYsStorage;
          std::vector<std::vector<float>> maeLeafWsStorage;
          std::vector<std::vector<float>> *maeLeafYsPtr = nullptr;
          std::vector<std::vector<float>> *maeLeafWsPtr = nullptr;
          if (criterion_ == LearningCriterion::AbsoluteError) {
            maeLeafYsStorage.resize(B);
            maeLeafWsStorage.resize(B);
            for (size_t b = 0; b < B; ++b) {
              for (size_t colIdx : perBinCols[b]) {
                if (colIdx >= xSubCols)
                  throw std::runtime_error("RegressionShapeGeneralizedTree: "
                                           "discretizer sample index "
                                           ">= Xsub columns");
                maeLeafYsStorage[b].push_back(ysub(colIdx));
                maeLeafWsStorage[b].push_back(wsub(colIdx));
              }
            }
            maeLeafYsPtr = &maeLeafYsStorage;
            maeLeafWsPtr = &maeLeafWsStorage;
          }

          if (B < 2)
            continue;

          const size_t kMax = std::min(B, numPartitions_);
          double bestFeatureScore = std::numeric_limits<double>::infinity();
          size_t chosenK = 0;
          std::vector<size_t> assignments;
          double impurityDecrease = 0.0;
          bool foundK = false;

          for (size_t k = 2; k <= kMax; ++k) {
            std::vector<size_t> trialAssignments;
            if (k == B)
              algorithms::identityBinAssignments(B, trialAssignments);
            else
              algorithms::roundRobinBinAssignments(B, k, trialAssignments);

            std::unique_ptr<BranchAssignment> branchObj;
            if (criterion_ == LearningCriterion::SquaredError) {
              branchObj = makeRegressionBranchAssignment(
                  LearningCriterion::SquaredError, trialAssignments, k, stats,
                  binWeights);
              if (k < B && cdParams_.smartInit) {
                const std::vector<size_t> snapshot = trialAssignments;
                const double objBeforeCd = branchObj->objective();
                coordinateDescent(k, *branchObj, rng_, cdParams_.maxIters,
                                  cdParams_.patience);
                const double objAfterCd = branchObj->objective();
                const bool keepCd = std::isfinite(objAfterCd) &&
                                    objAfterCd < objBeforeCd - kCdRegressionKeepCdEps;
                if (!keepCd) {
                  trialAssignments = snapshot;
                  branchObj = makeRegressionBranchAssignment(
                      LearningCriterion::SquaredError, trialAssignments, k,
                      stats, binWeights);
                }
              }
            } else {
              if (!maeLeafYsPtr || !maeLeafWsPtr)
                continue;
              branchObj = makeRegressionBranchAssignment(
                  LearningCriterion::AbsoluteError, trialAssignments, k, stats,
                  binWeights, 1.0, maeLeafYsPtr, maeLeafWsPtr);
            }

            if (!algorithms::partitionCountsMeetMinLeaf(
                    trialAssignments, sizes, k, outerParams_.minLeafSize))
              continue;

            const double childImp = branchObj->objective();
            const double gain = parentImp - childImp;
            if (gain < outerParams_.minGainSplit - outerTreeBuilder_.eps)
              continue;

            const double score = algorithms::penalizedBranchingScore(
                childImp, k, outerParams_.branchingPenalty);
            if (score < bestFeatureScore - outerTreeBuilder_.eps) {
              bestFeatureScore = score;
              chosenK = k;
              assignments = std::move(trialAssignments);
              impurityDecrease = gain;
              foundK = true;
            }
          }

          if (!foundK)
            continue;

          if (bestFeatureScore < bestPenalizedChild - outerTreeBuilder_.eps) {
            bestPenalizedChild = bestFeatureScore;
            brBest.featureIndex = f;
            const auto &dth = disc->thresholds();
            brBest.innerThresholds.resize(dth.size());
            for (size_t t = 0; t < dth.size(); ++t)
              brBest.innerThresholds[t] = static_cast<float>(dth[t]);
            brBest.binToPartition = std::move(assignments);
            brBest.impurityDecrease = impurityDecrease;
            brBest.numPartitionsUsed = chosenK;

            brBest.sampleBins.assign(xSubCols, 0);
            for (size_t b = 0; b < perBinCols.size(); ++b) {
              for (size_t colIdx : perBinCols[b]) {
                if (colIdx >= xSubCols)
                  throw std::runtime_error("RegressionShapeGeneralizedTree: "
                                           "discretizer sample index "
                                           ">= Xsub columns");
                brBest.sampleBins[colIdx] = b;
              }
            }

            binSizesForBest.assign(sizes_copy.begin(), sizes_copy.end());
            binWeightsForBest.assign(binWeights.begin(), binWeights.end());

            if (criterion_ == LearningCriterion::SquaredError) {
              brBest.leafStats = stats;
            } else {
              brBest.leafStats.clear();
              brBest.leafStats.resize(B);
            }
          }
        }

        if (!std::isfinite(bestPenalizedChild) ||
            bestPenalizedChild >= std::numeric_limits<double>::infinity() ||
            brBest.impurityDecrease <= outerTreeBuilder_.eps) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          node.splitBinWeights.clear();
          node.binSampleCounts.clear();
          return false;
        }

        node.isLeaf = false;
        node.routingFeature = brBest.featureIndex;
        node.innerThresholds = std::move(brBest.innerThresholds);
        node.binToPartition = std::move(brBest.binToPartition);
        node.sampleBins = std::move(brBest.sampleBins);
        node.numPartitions = brBest.numPartitionsUsed;
        node.informationGain = brBest.impurityDecrease;

        if (criterion_ == LearningCriterion::SquaredError)
          node.splitLeafStats = std::move(brBest.leafStats);
        else
          node.splitLeafStats.clear();
        node.splitBinWeights = std::move(binWeightsForBest);

        node.binSampleCounts = std::move(binSizesForBest);

        return true;
      },
      [this, &y](ShapeFunctionNode &parent) {
        if (parent.sampleBins.size() != parent.sampleIndices.n_elem)
          throw std::runtime_error("RegressionShapeGeneralizedTree::fit: "
                                   "sampleBins length mismatch");

        const size_t numChildPartitions = parent.numPartitions;
        std::vector<std::vector<size_t>> buckets(numChildPartitions);
        for (arma::uword i = 0; i < parent.sampleIndices.n_elem; ++i) {
          const size_t si = static_cast<size_t>(parent.sampleIndices(i));
          const size_t bin = parent.sampleBins[static_cast<size_t>(i)];
          if (bin >= parent.binToPartition.size())
            throw std::runtime_error(
                "RegressionShapeGeneralizedTree::fit: bin id out of range");
          size_t p = parent.binToPartition[bin];
          if (p >= numChildPartitions)
            p = numChildPartitions - 1;
          buckets[p].push_back(si);
        }

        std::vector<ShapeFunctionNode> children;
        children.reserve(numChildPartitions);

        if (criterion_ == LearningCriterion::SquaredError) {
          if (parent.splitLeafStats.size() != parent.splitBinWeights.size())
            throw std::runtime_error(
                "RegressionShapeGeneralizedTree::fit: splitBinWeights / "
                "splitLeafStats size mismatch");

          for (size_t p = 0; p < numChildPartitions; ++p) {
            ShapeFunctionNode ch;
            ch.height = parent.height + 1;
            ch.sampleIndices = arma::conv_to<arma::uvec>::from(buckets[p]);
            ch.numPartitions = numPartitions_;
            const PartitionMoments moments = aggregatePartitionFromBins(
                parent.splitLeafStats, parent.splitBinWeights,
                parent.binToPartition, p);
            const std::vector<double> agg{moments.sumWY, moments.sumWY2};
            ch.score = Criterion::squaredError(agg, moments.sumW);
            const std::vector<float> aggF{
                static_cast<float>(moments.sumWY),
                static_cast<float>(moments.sumWY2)};
            ch.isLeaf = true;
            leafRegressionStats.push_back(aggF);
            leafNumSamples.push_back(ch.sampleIndices.n_elem);
            leafPredictions_.push_back(
                weightedMeanFromAggregates(moments.sumWY, moments.sumW));
            children.push_back(std::move(ch));
          }
        } else {
          for (size_t p = 0; p < numChildPartitions; ++p) {
            ShapeFunctionNode ch;
            ch.height = parent.height + 1;
            ch.sampleIndices = arma::conv_to<arma::uvec>::from(buckets[p]);
            ch.numPartitions = numPartitions_;
            std::vector<float> ys;
            std::vector<float> ws;
            ys.reserve(ch.sampleIndices.n_elem);
            ws.reserve(ch.sampleIndices.n_elem);
            for (arma::uword i = 0; i < parent.sampleIndices.n_elem; ++i) {
              const size_t si = static_cast<size_t>(parent.sampleIndices(i));
              const size_t bin = parent.sampleBins[static_cast<size_t>(i)];
              if (bin >= parent.binToPartition.size())
                throw std::runtime_error(
                    "RegressionShapeGeneralizedTree::fit: bin id out of range");
              const size_t pp = parent.binToPartition[bin] >= numChildPartitions
                                    ? numChildPartitions - 1
                                    : parent.binToPartition[bin];
              if (pp != p)
                continue;
              ys.push_back(y(si));
              ws.push_back(static_cast<float>(fitSampleWeights_(si)));
            }
            ch.score = meanAbsoluteDeviationFromMedian(ys, ws);
            ch.isLeaf = true;
            leafRegressionStats.push_back({});
            leafNumSamples.push_back(ch.sampleIndices.n_elem);
            leafPredictions_.push_back(
                static_cast<float>(Criterion::absoluteError(ys, ws).median));
            children.push_back(std::move(ch));
          }
        }
        return children;
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

size_t RegressionShapeGeneralizedTree::numLeaves() const {
  size_t c = 0;
  for (const auto &node : nodes_) {
    if (node.isLeaf)
      ++c;
  }
  return c;
}

size_t RegressionShapeGeneralizedTree::numNodes() const {
  return nodes_.size();
}

bool RegressionShapeGeneralizedTree::isFitted() const { return fitted_; }
