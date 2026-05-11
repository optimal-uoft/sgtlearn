/**
 * @file Estimators/ClassificationShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the classification
 *        shape-generalized tree. Per-node inner fit: discretize -> k-means bin
 *        init -> coordinate descent -> record best branch.
 */

#include "Estimators/ClassificationShapeGeneralizedTree.h"

#include "Discretizers/ClassificationDiscretizer.h"
#include "algorithms/CoordinateDescent.h"
#include "algorithms/KMeansUtils.h"
#include "algorithms/ShapeBranchingTypes.h"
#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "Criterion.h"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

// todo: add bootstrapping columns
// todo: validate branch assignment to discretization is happening correctly

namespace {

size_t leafArgmaxClass(const std::vector<size_t> &counts) {
  if (counts.empty())
    return 0;
  size_t best = 0;
  for (size_t c = 1; c < counts.size(); ++c) {
    if (counts[c] > counts[best])
      best = c;
  }
  return best;
}

} // namespace

ClassificationShapeGeneralizedTree::ClassificationShapeGeneralizedTree(
    LearningCriterion criterion, size_t numClasses, size_t numPartitions,
    TreeBuildingParams outerParams, TreeBuildingParams innerParams,
    CoordinateDescentParams cdParams)
    : criterion_(criterion), numClasses_(numClasses),
      numPartitions_(numPartitions), outerParams_(outerParams),
      innerParams_(innerParams), cdParams_(cdParams),
      outerTreeBuilder_(outerParams_.minLeafSize, outerParams_.minGainSplit,
                        outerParams_.maxDepth, outerParams_.maxLeafNodes) {
  if (criterion_ != LearningCriterion::Entropy &&
      criterion_ != LearningCriterion::Gini)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: criterion must be Entropy or "
        "Gini");
  if (numClasses_ < 2)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: numClasses must be >= 2");
  if (numPartitions_ < 2)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: numPartitions must be >= 2");
}

double ClassificationShapeGeneralizedTree::impurityForSampleIndices(
    const std::vector<size_t> &indices, const arma::Row<size_t> &y) const {
  if (indices.empty())
    return 0.0;
  std::vector<size_t> counts(numClasses_, 0);
  size_t n = 0;
  for (size_t si : indices) {
    const size_t lab = y(si);
    counts[lab]++;
    n++;
  }
  if (criterion_ == LearningCriterion::Gini)
    return Criterion::gini(counts, n);
  return Criterion::entropy(counts, n);
}

double ClassificationShapeGeneralizedTree::impurityForClassCounts(
    const std::vector<size_t> &classCounts) const {
  size_t n = 0;
  for (size_t c : classCounts)
    n += c;
  if (n == 0)
    return 0.0;
  if (criterion_ == LearningCriterion::Gini)
    return Criterion::gini(classCounts, n);
  return Criterion::entropy(classCounts, n);
}

void ClassificationShapeGeneralizedTree::fillLeafHistogram(
    ShapeFunctionNode &node, const arma::Row<size_t> &y) const {
  node.leafClassCounts.assign(numClasses_, 0);
  for (size_t si : node.sampleIndices) {
    const size_t lab = y(si);
    if (lab >= numClasses_)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: class label out of range");
    node.leafClassCounts[lab]++;
  }
}

void ClassificationShapeGeneralizedTree::fit(const arma::fmat &X,
                                             arma::uvec &features,
                                             const arma::Row<size_t> &y) {
  if (X.n_cols != y.n_elem)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: X.n_cols must match "
        "y.n_elem");
  if (features.n_elem == 0)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: features must be non-empty");
  for (size_t i = 0; i < features.n_elem; ++i) {
    if (static_cast<size_t>(features(i)) >= X.n_rows)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: feature index >= "
          "X.n_rows");
  }
  for (size_t j = 0; j < y.n_elem; ++j) {
    if (y(j) >= numClasses_)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: label >= numClasses");
  }

  const size_t n = X.n_cols;

  nodes_.clear();
  childIndices_.clear();
  fitted_ = false;

  ShapeFunctionNode root;
  root.height = 0;
  root.sampleIndices.resize(n);
  for (size_t i = 0; i < n; ++i)
    root.sampleIndices[i] = i;
  root.nodeIndex = 0;
  root.numPartitions = numPartitions_;
  root.score = impurityForSampleIndices(root.sampleIndices, y);
  root.isLeaf = true;
  fillLeafHistogram(root, y);
  nodes_.push_back(root);
  childIndices_.emplace_back();
  rootIndex_ = 0;

  outerTreeBuilder_.buildTree(
      nodes_[0],
      [this, &X, &y, &features](ShapeFunctionNode &node, size_t minLeaf) { // find best shape function
        const size_t ns = node.sampleIndices.size();
        node.score = impurityForSampleIndices(node.sampleIndices, y);

        if (ns < 2 * minLeaf) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          fillLeafHistogram(node, y);
          return false;
        }

        const double parentImp = node.score;
        if (parentImp <= outerTreeBuilder_.eps) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          fillLeafHistogram(node, y);
          return false;
        }

        arma::uvec subIdx(ns);
        for (size_t i = 0; i < ns; ++i)
          subIdx(i) = static_cast<arma::uword>(node.sampleIndices[i]);
        const arma::fmat Xsub = X.cols(subIdx);
        arma::Row<size_t> ysub(subIdx.n_elem);
        for (arma::uword i = 0; i < subIdx.n_elem; ++i)
          ysub(i) = y(subIdx(i));

        const size_t xSubCols = static_cast<size_t>(Xsub.n_cols);
        double bestPenalizedChild = std::numeric_limits<double>::infinity();
        ShapeBranchingResult brBest{};
        arma::uvec featOne(1);

        for (size_t fi = 0; fi < features.n_elem; ++fi) {
          const size_t f = static_cast<size_t>(features(fi));
          if (f >= Xsub.n_rows)
            throw std::invalid_argument(
                "ClassificationShapeGeneralizedTree::fit: candidate feature "
                "index >= X.n_rows");
          featOne(0) = static_cast<arma::uword>(f);

          auto disc = makeClassificationDiscretizer(criterion_);
          disc->Train(Xsub, featOne, ysub, numClasses_, innerParams_.minLeafSize,
                      innerParams_.minGainSplit, innerParams_.maxDepth,
                      innerParams_.maxLeafNodes);
          const size_t B = disc->numLeaves();
          if (B == 0)
            continue;

          auto &stats = disc->leafStats();
          auto &sizes = disc->leafNumSamples();

          const size_t mixSeed =
              cdParams_.seed ^ (0x9e3779b9u * (static_cast<unsigned>(f) + 1u));

          std::vector<size_t> assignments(B);
          if (!cdParams_.smartInit || numPartitions_ < 2 || B < numPartitions_) {
            for (size_t b = 0; b < B; ++b)
              assignments[b] = b % numPartitions_;
          } else {
            arma::mat Xk(B, numClasses_);
            arma::vec wk(B);
            for (size_t b = 0; b < B; ++b) {
              wk(b) = std::max(1.0, static_cast<double>(sizes[b]));
              double sum = 0.0;
              for (size_t c = 0; c < numClasses_; ++c)
                sum += static_cast<double>(stats[b][c]);
              if (sum <= 0.0) {
                Xk.row(b).fill(1.0 / static_cast<double>(numClasses_));
              } else {
                for (size_t c = 0; c < numClasses_; ++c)
                  Xk(b, c) = static_cast<double>(stats[b][c]) / sum;
              }
            }
            algorithms::initAssignmentsWeightedKMeans(Xk, wk, numPartitions_,
                                                      mixSeed, assignments);
          }

          auto branchObj = makeClassificationBranchAssignment(
              criterion_, assignments, numPartitions_, stats, sizes, numClasses_);
          coordinateDescent(numPartitions_, *branchObj, cdParams_.maxIters,
                            cdParams_.patience, mixSeed);

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
                             static_cast<double>(numPartitions_ > 0
                                                     ? numPartitions_ - 1
                                                     : 0);
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

            const auto &perBinCols = disc->inSampleDiscretizations();
            brBest.sampleBins.assign(xSubCols, 0);
            for (size_t b = 0; b < perBinCols.size(); ++b) {
              for (size_t colIdx : perBinCols[b]) {
                if (colIdx >= xSubCols)
                  throw std::runtime_error(
                      "ClassificationShapeGeneralizedTree: discretizer sample "
                      "index >= Xsub columns");
                brBest.sampleBins[colIdx] = b;
              }
            }

            brBest.leafStats = stats;
          }
        }

        if (!std::isfinite(bestPenalizedChild) ||
            bestPenalizedChild >= std::numeric_limits<double>::infinity() ||
            brBest.impurityDecrease <= outerTreeBuilder_.eps) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          fillLeafHistogram(node, y);
          return false;
        }

        node.isLeaf = false;
        node.routingFeature = brBest.featureIndex;
        node.innerThresholds = std::move(brBest.innerThresholds);
        node.binToPartition = std::move(brBest.binToPartition);
        node.sampleBins = std::move(brBest.sampleBins);
        node.splitLeafStats = std::move(brBest.leafStats);
        node.numPartitions = numPartitions_;
        node.informationGain = brBest.impurityDecrease;
        node.leafClassCounts.clear();
        return true;
      },
      [this](ShapeFunctionNode &parent) { // make children
        if (parent.sampleBins.size() != parent.sampleIndices.size())
          throw std::runtime_error(
              "ClassificationShapeGeneralizedTree::fit: sampleBins length "
              "mismatch");

        // map sample indices to partitions
        std::vector<std::vector<size_t>> buckets(numPartitions_);
        for (size_t i = 0; i < parent.sampleIndices.size(); ++i) {
          const size_t si = parent.sampleIndices[i];
          const size_t bin = parent.sampleBins[i];
          if (bin >= parent.binToPartition.size())
            throw std::runtime_error(
                "ClassificationShapeGeneralizedTree::fit: bin id out of range");
          size_t p = parent.binToPartition[bin];
          if (p >= numPartitions_)
            p = numPartitions_ - 1;
          buckets[p].push_back(si);
        }

        const auto &binStats = parent.splitLeafStats;
        if (binStats.size() != parent.binToPartition.size())
          throw std::runtime_error(
              "ClassificationShapeGeneralizedTree::fit: splitLeafStats / "
              "binToPartition size mismatch");

        std::vector<ShapeFunctionNode> children;
        children.reserve(numPartitions_);
        for (size_t p = 0; p < numPartitions_; ++p) {
          ShapeFunctionNode ch;
          ch.height = parent.height + 1;
          ch.sampleIndices = std::move(buckets[p]);
          ch.numPartitions = numPartitions_;
          ch.leafClassCounts.assign(numClasses_, 0);
          // aggregate per-bin class counts to get leaf class counts
          for (size_t b = 0; b < binStats.size(); ++b) {
            if (parent.binToPartition[b] != p)
              continue;
            const auto &sb = binStats[b];
            for (size_t c = 0; c < numClasses_; ++c) {
              const size_t add = (c < sb.size()) ? sb[c] : 0;
              ch.leafClassCounts[c] += add;
            }
          }
          ch.score = impurityForClassCounts(ch.leafClassCounts);
          ch.isLeaf = true;
          children.push_back(std::move(ch));
        }
        return children;
      },
      [this](ShapeFunctionNode &parent,
             std::vector<ShapeFunctionNode> &children) { // commit split
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
    node.sampleIndices.clear();
    node.sampleBins.clear();
    node.splitLeafStats.clear();
    node.nodeIndex = 0;
  }

  fitted_ = true;
}

arma::Row<size_t>
ClassificationShapeGeneralizedTree::predict(const arma::fmat &X) const {
  if (!fitted_)
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predict: model is not fitted");
  if (nodes_.empty())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predict: empty tree");
  if (rootIndex_ >= nodes_.size())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predict: invalid root index");
  if (childIndices_.size() != nodes_.size())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predict: childIndices/nodes size "
        "mismatch");

  const size_t n_samples = X.n_cols;
  arma::Row<size_t> yhat(n_samples);

  for (size_t s = 0; s < n_samples; ++s) {
    size_t idx = rootIndex_;
    while (true) {
      const auto &node = nodes_[idx];
      if (node.isLeaf) {
        yhat(s) = leafArgmaxClass(node.leafClassCounts);
        break;
      }
      if (node.routingFeature >= X.n_rows)
        throw std::invalid_argument(
            "ClassificationShapeGeneralizedTree::predict: routing feature "
            "index out of range for X");
      const float v = X(node.routingFeature, s);
      const size_t part = node.routeFeatureValueToPartition(v);
      if (part >= childIndices_[idx].size())
        throw std::runtime_error(
            "ClassificationShapeGeneralizedTree::predict: child partition "
            "out of range");
      idx = childIndices_[idx][part];
      if (idx >= nodes_.size())
        throw std::runtime_error(
            "ClassificationShapeGeneralizedTree::predict: child node index "
            "out of range");
    }
  }
  return yhat;
}

arma::fmat
ClassificationShapeGeneralizedTree::predictProba(const arma::fmat &X) const {
  if (!fitted_)
    throw std::logic_error("ClassificationShapeGeneralizedTree::predictProba: "
                           "model is not fitted");
  if (nodes_.empty())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predictProba: empty tree");
  if (rootIndex_ >= nodes_.size())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predictProba: invalid root index");
  if (childIndices_.size() != nodes_.size())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predictProba: childIndices/nodes "
        "size mismatch");

  const size_t n_samples = X.n_cols;
  const size_t K = numClasses_;
  arma::fmat P(K, n_samples);
  const float uniform = K > 0 ? 1.f / static_cast<float>(K) : 0.f;

  for (size_t s = 0; s < n_samples; ++s) {
    size_t idx = rootIndex_;
    while (true) {
      const auto &node = nodes_[idx];
      if (node.isLeaf) {
        const auto &h = node.leafClassCounts;
        double sum = 0.0;
        for (size_t c = 0; c < K; ++c)
          sum += (c < h.size()) ? static_cast<double>(h[c]) : 0.0;
        if (sum <= 0.0) {
          P.col(s).fill(uniform);
        } else {
          for (size_t c = 0; c < K; ++c) {
            const double cnt =
                (c < h.size()) ? static_cast<double>(h[c]) : 0.0;
            P(c, s) = static_cast<float>(cnt / sum);
          }
        }
        break;
      }
      if (node.routingFeature >= X.n_rows)
        throw std::invalid_argument(
            "ClassificationShapeGeneralizedTree::predictProba: routing "
            "feature index out of range for X");
      const float v = X(node.routingFeature, s);
      const size_t part = node.routeFeatureValueToPartition(v);
      if (part >= childIndices_[idx].size())
        throw std::runtime_error(
            "ClassificationShapeGeneralizedTree::predictProba: child "
            "partition out of range");
      idx = childIndices_[idx][part];
      if (idx >= nodes_.size())
        throw std::runtime_error(
            "ClassificationShapeGeneralizedTree::predictProba: child node "
            "index out of range");
    }
  }
  return P;
}

size_t ClassificationShapeGeneralizedTree::numLeaves() const {
  size_t c = 0;
  for (const auto &node : nodes_) {
    if (node.isLeaf)
      ++c;
  }
  return c;
}

size_t ClassificationShapeGeneralizedTree::numNodes() const {
  return nodes_.size();
}

bool ClassificationShapeGeneralizedTree::isFitted() const { return fitted_; }
