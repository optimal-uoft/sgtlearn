/**
 * @file Estimators/RegressionShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the regression
 *        shape-generalized tree. Per-node inner fit: discretize -> round-robin
 *        bin-to-partition seed -> coordinate descent -> record best branch.
 */

#include "Estimators/RegressionShapeGeneralizedTree.h"

#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "Criterion.h"
#include "Discretizers/RegressionDiscretizer.h"
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

double meanAbsoluteDeviationFromMedian(std::vector<float> ys) {
  if (ys.empty())
    return 0.0;
  std::sort(ys.begin(), ys.end());
  const size_t n = ys.size();
  const double med = (n % 2 == 1)
                         ? static_cast<double>(ys[n / 2])
                         : 0.5 * (static_cast<double>(ys[n / 2 - 1]) +
                                  static_cast<double>(ys[n / 2]));
  double sumAbs = 0.0;
  for (float v : ys)
    sumAbs += std::abs(static_cast<double>(v) - med);
  return sumAbs / static_cast<double>(n);
}

float medianValue(std::vector<float> ys) {
  if (ys.empty())
    return 0.f;
  std::sort(ys.begin(), ys.end());
  const size_t n = ys.size();
  if (n % 2 == 1)
    return ys[n / 2];
  return 0.5f * (ys[n / 2 - 1] + ys[n / 2]);
}

} // namespace

RegressionShapeGeneralizedTree::RegressionShapeGeneralizedTree(
    LearningCriterion criterion, size_t numPartitions,
    TreeBuildingParams outerParams, TreeBuildingParams innerParams,
    CoordinateDescentParams cdParams, uint64_t random_state,
    FeatureBaggingPickFn featureBagging)
    : criterion_(criterion), numPartitions_(numPartitions),
      outerParams_(outerParams), innerParams_(innerParams),
      cdParams_(cdParams), random_state_(random_state), rng_(),
      featureBagging_(featureBagging ? std::move(featureBagging)
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
  std::vector<float> ys;
  ys.reserve(node.sampleIndices.n_elem);
  for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(node.sampleIndices(i));
    ys.push_back(y(si));
  }
  if (criterion_ == LearningCriterion::SquaredError) {
    double sy = 0.0, sy2 = 0.0;
    for (float v : ys) {
      sy += static_cast<double>(v);
      sy2 += static_cast<double>(v) * static_cast<double>(v);
    }
    std::vector<float> st(2);
    st[0] = static_cast<float>(sy);
    st[1] = static_cast<float>(sy2);
    return Criterion::squaredError(st, ys.size());
  }
  return meanAbsoluteDeviationFromMedian(std::move(ys));
}

std::vector<float> RegressionShapeGeneralizedTree::aggregateYSquaredStats(
    const ShapeFunctionNode &node, const arma::Row<float> &y) const {
  std::vector<float> st(2, 0.f);
  for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(node.sampleIndices(i));
    const float v = y(si);
    st[0] += v;
    st[1] += v * v;
  }
  return st;
}

void RegressionShapeGeneralizedTree::fit(const arma::fmat &X,
                                         const arma::Row<float> &y) {
  if (X.n_cols != y.n_elem)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::fit: X.n_cols must match y.n_elem");
  if (X.n_rows == 0)
    throw std::invalid_argument(
        "RegressionShapeGeneralizedTree::fit: X must have at least one "
        "feature (row)");

  const size_t n = X.n_cols;

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
    const auto st = aggregateYSquaredStats(root, y);
    leafRegressionStats.push_back(st);
    leafNumSamples.push_back(n);
    leafPredictions_.push_back(
        n > 0 ? st[0] / static_cast<float>(n) : 0.f);
  } else {
    leafRegressionStats.push_back({});
    leafNumSamples.push_back(n);
    std::vector<float> ys;
    ys.reserve(n);
    for (arma::uword i = 0; i < root.sampleIndices.n_elem; ++i)
      ys.push_back(y(static_cast<size_t>(root.sampleIndices(i))));
    leafPredictions_.push_back(medianValue(std::move(ys)));
  }
  root.score = impurityAtNode(y, root);
  nodes_.push_back(root);

  childIndices_.emplace_back();
  rootIndex_ = 0;

  const arma::uvec featureCandidates =
      arma::regspace<arma::uvec>(0, X.n_rows - 1);

  outerTreeBuilder_.buildTree(
      nodes_[0],
      [this, &X, &y, &featureCandidates](ShapeFunctionNode &node,
                                         size_t minLeaf) { // find best shape function
        const size_t ns = node.sampleIndices.n_elem;
        node.score = impurityAtNode(y, node);

        if (ns < 2 * minLeaf) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          node.regressionSplitLeafStats.clear();
          return false;
        }

        const double parentImp = node.score;
        if (parentImp <= outerTreeBuilder_.eps) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          node.regressionSplitLeafStats.clear();
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
        ShapeBranchingResult<float> brBest{};
        arma::uvec featOne(1);

        for (size_t fi = 0; fi < featureSubset.size(); ++fi) {
          const size_t f = featureSubset[fi];
          if (f >= Xsub.n_rows)
            throw std::invalid_argument(
                "RegressionShapeGeneralizedTree::fit: candidate feature index "
                ">= X.n_rows");
          featOne(0) = static_cast<arma::uword>(f);

          auto disc = makeRegressionDiscretizer(criterion_);
          disc->Train(Xsub, featOne, ysub, innerParams_.minLeafSize,
                      innerParams_.minGainSplit, innerParams_.maxDepth,
                      innerParams_.maxLeafNodes);
          const size_t B = disc->numLeaves();
          if (B == 0)
            continue;

          auto &stats = disc->leafStats();
          auto &sizes = disc->leafNumSamples();
          const auto &perBinCols = disc->inSampleDiscretizations();

          std::vector<size_t> assignments(B);
          for (size_t b = 0; b < B; ++b)
            assignments[b] = b % numPartitions_;

          std::unique_ptr<BranchAssignment> branchObj;
          std::vector<std::vector<float>> maeLeafYsStorage;
          if (criterion_ == LearningCriterion::SquaredError) {
            branchObj = makeRegressionBranchAssignment(
                LearningCriterion::SquaredError, assignments, numPartitions_,
                stats, sizes);
          } else {
            maeLeafYsStorage.resize(B);
            for (size_t b = 0; b < B; ++b) {
              for (size_t colIdx : perBinCols[b]) {
                if (colIdx >= xSubCols)
                  throw std::runtime_error(
                      "RegressionShapeGeneralizedTree: discretizer sample index "
                      ">= Xsub columns");
                maeLeafYsStorage[b].push_back(ysub(colIdx));
              }
            }
            branchObj = makeRegressionBranchAssignment(
                LearningCriterion::AbsoluteError, assignments, numPartitions_,
                maeLeafYsStorage, sizes);
          }

          coordinateDescent(numPartitions_, *branchObj, rng_, cdParams_.maxIters,
                            cdParams_.patience);

          std::vector<size_t> wt(numPartitions_, 0);
          for (size_t b = 0; b < assignments.size(); ++b)
            wt[assignments[b]] += sizes[b];
          bool partitionsOk = true;
          for (size_t p = 0; p < numPartitions_; ++p) {
            if (wt[p] < outerParams_.minLeafSize) {
              partitionsOk = false;
              break;
            }
          }
          if (!partitionsOk)
            continue;

          const double childImp = branchObj->objective();
          const double penalizedChild =
              childImp + outerParams_.branchingPenalty *
                             static_cast<double>(
                                 numPartitions_ > 0 ? numPartitions_ - 1 : 0);
          const double impurityDecrease = parentImp - childImp;
          if (impurityDecrease <
              outerParams_.minGainSplit - outerTreeBuilder_.eps)
            continue;

          if (penalizedChild < bestPenalizedChild - outerTreeBuilder_.eps) {
            bestPenalizedChild = penalizedChild;
            brBest.featureIndex = f;
            const auto &dth = disc->thresholds();
            brBest.innerThresholds.resize(dth.size());
            for (size_t t = 0; t < dth.size(); ++t)
              brBest.innerThresholds[t] = static_cast<float>(dth[t]);
            brBest.binToPartition = assignments;
            brBest.impurityDecrease = impurityDecrease;

            brBest.sampleBins.assign(xSubCols, 0);
            for (size_t b = 0; b < perBinCols.size(); ++b) {
              for (size_t colIdx : perBinCols[b]) {
                if (colIdx >= xSubCols)
                  throw std::runtime_error(
                      "RegressionShapeGeneralizedTree: discretizer sample index "
                      ">= Xsub columns");
                brBest.sampleBins[colIdx] = b;
              }
            }

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
          node.regressionSplitLeafStats.clear();
          return false;
        }

        node.isLeaf = false;
        node.routingFeature = brBest.featureIndex;
        node.innerThresholds = std::move(brBest.innerThresholds);
        node.binToPartition = std::move(brBest.binToPartition);
        node.sampleBins = std::move(brBest.sampleBins);
        node.splitLeafStats.clear();
        node.numPartitions = numPartitions_;
        node.informationGain = brBest.impurityDecrease;

        if (criterion_ == LearningCriterion::SquaredError)
          node.regressionSplitLeafStats = std::move(brBest.leafStats);
        else
          node.regressionSplitLeafStats.clear();

        return true;
      },
      [this, &y](ShapeFunctionNode &parent) {
        if (parent.sampleBins.size() != parent.sampleIndices.n_elem)
          throw std::runtime_error(
              "RegressionShapeGeneralizedTree::fit: sampleBins length mismatch");

        std::vector<std::vector<float>> innerBinStats;
        if (criterion_ == LearningCriterion::SquaredError) {
          innerBinStats = parent.regressionSplitLeafStats;
          if (innerBinStats.size() != parent.binToPartition.size())
            throw std::runtime_error(
                "RegressionShapeGeneralizedTree::fit: regressionSplitLeafStats / "
                "binToPartition size mismatch");
        }

        std::vector<std::vector<size_t>> buckets(numPartitions_);
        for (arma::uword i = 0; i < parent.sampleIndices.n_elem; ++i) {
          const size_t si = static_cast<size_t>(parent.sampleIndices(i));
          const size_t bin = parent.sampleBins[static_cast<size_t>(i)];
          if (bin >= parent.binToPartition.size())
            throw std::runtime_error(
                "RegressionShapeGeneralizedTree::fit: bin id out of range");
          size_t p = parent.binToPartition[bin];
          if (p >= numPartitions_)
            p = numPartitions_ - 1;
          buckets[p].push_back(si);
        }

        std::vector<ShapeFunctionNode> children;
        children.reserve(numPartitions_);

        if (criterion_ == LearningCriterion::SquaredError) {
          const auto &binStats = innerBinStats;
          for (size_t p = 0; p < numPartitions_; ++p) {
            ShapeFunctionNode ch;
            ch.height = parent.height + 1;
            ch.sampleIndices = arma::conv_to<arma::uvec>::from(buckets[p]);
            ch.numPartitions = numPartitions_;
            std::vector<float> agg(2, 0.f);
            for (size_t b = 0; b < binStats.size(); ++b) {
              if (parent.binToPartition[b] != p)
                continue;
              const auto &sb = binStats[b];
              if (sb.size() >= 2) {
                agg[0] += sb[0];
                agg[1] += sb[1];
              }
            }
            const size_t nch = ch.sampleIndices.n_elem;
            ch.score = Criterion::squaredError(agg, nch);
            ch.isLeaf = true;
            leafRegressionStats.push_back(std::move(agg));
            leafNumSamples.push_back(nch);
            leafPredictions_.push_back(
                nch > 0 ? leafRegressionStats.back()[0] / static_cast<float>(nch)
                        : 0.f);
            children.push_back(std::move(ch));
          }
        } else {
          for (size_t p = 0; p < numPartitions_; ++p) {
            ShapeFunctionNode ch;
            ch.height = parent.height + 1;
            ch.sampleIndices = arma::conv_to<arma::uvec>::from(buckets[p]);
            ch.numPartitions = numPartitions_;
            std::vector<float> ys;
            ys.reserve(ch.sampleIndices.n_elem);
            for (arma::uword i = 0; i < parent.sampleIndices.n_elem; ++i) {
              const size_t si =
                  static_cast<size_t>(parent.sampleIndices(i));
              const size_t bin = parent.sampleBins[static_cast<size_t>(i)];
              if (bin >= parent.binToPartition.size())
                throw std::runtime_error(
                    "RegressionShapeGeneralizedTree::fit: bin id out of range");
              const size_t pp = parent.binToPartition[bin] >= numPartitions_
                                    ? numPartitions_ - 1
                                    : parent.binToPartition[bin];
              if (pp != p)
                continue;
              ys.push_back(y(si));
            }
            ch.score = meanAbsoluteDeviationFromMedian(ys);
            ch.isLeaf = true;
            leafRegressionStats.push_back({});
            leafNumSamples.push_back(ch.sampleIndices.n_elem);
            leafPredictions_.push_back(medianValue(std::move(ys)));
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
        childIndices_[pid].assign(numPartitions_, 0);
        for (size_t p = 0; p < numPartitions_; ++p) {
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
    node.splitLeafStats.clear();
    node.regressionSplitLeafStats.clear();
  }

  fitted_ = true;
}

arma::Row<float>
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
  arma::Row<float> yhat(n_samples);

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
