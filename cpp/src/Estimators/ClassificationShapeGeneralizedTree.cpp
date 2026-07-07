/**
 * @file Estimators/ClassificationShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the classification
 *        shape-generalized tree. Per-node inner fit: discretize -> search
 *        partition counts k in [2, numPartitions] (construct_mapping) ->
 *        coordinate descent -> record best branch.
 */

#include "Estimators/ClassificationShapeGeneralizedTree.h"

#include "Criterion.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "Discretizers/DiscretizerFactories.h"
#include "Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h"

#include <algorithm>
#include <armadillo>
#include <iterator>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace classificationInference {

size_t argMaxClass(const std::vector<double> &counts) {
  if (counts.empty())
    throw std::runtime_error("classification counts cannot be empty");
  auto it = std::max_element(counts.begin(), counts.end());
  return std::distance(counts.begin(), it);
}

} // namespace classificationInference

ClassificationShapeGeneralizedTree::ClassificationShapeGeneralizedTree(
    LearningCriterion criterion, size_t numClasses, size_t numPartitions,
    TreeBuildingParams outerParams, TreeBuildingParams innerParams,
    CoordinateDescentParams cdParams, uint64_t random_state,
    FeatureBaggingPickFn featureBagging)
    : ShapeGeneralizedTree(criterion, numPartitions, outerParams, innerParams),
      numClasses_(numClasses), cdParams_(cdParams),
      random_state_(random_state), rng_(),
      featureBagging_(featureBagging
                          ? std::move(featureBagging)
                          : FeatureBaggingPickFn(pickAllFeatureIndices)),
      outerTreeBuilder_(outerParams_.minLeafSize, outerParams_.minGainSplit,
                        outerParams_.maxDepth, outerParams_.maxLeafNodes) {
  if (criterion != LearningCriterion::Entropy &&
      criterion != LearningCriterion::Gini)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: criterion must be Entropy or "
        "Gini");
  if (numClasses_ < 2)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: numClasses must be >= 2");
  if (numPartitions < 2)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: numPartitions must be >= 2");
}

double ClassificationShapeGeneralizedTree::impurityForClassCounts(
    const std::vector<double> &classCounts) const {
  double totalWeight = 0.0;
  for (double c : classCounts)
    totalWeight += c;
  if (totalWeight <= 0.0)
    return 0.0;
  if (criterion_ == LearningCriterion::Gini)
    return Criterion::gini(classCounts, totalWeight);
  if (criterion_ == LearningCriterion::Entropy)
    return Criterion::entropy(classCounts, totalWeight);
  throw std::runtime_error("ClassificationShapeGeneralizedTree::"
                           "impurityForClassCounts: invalid criterion");
}

std::vector<double> ClassificationShapeGeneralizedTree::fillLeafHistogram(
    ShapeFunctionNode &node, const arma::Row<size_t> &y) const {
  std::vector<double> counts(numClasses_, 0.0);
  for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(node.sampleIndices(i));
    const size_t lab = y(si);
    if (lab >= numClasses_)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: class label out of range");
    counts[lab] += static_cast<double>(fitSampleWeights_(si));
  }
  return counts;
}

void ClassificationShapeGeneralizedTree::fit(
    const arma::fmat &X, const arma::Row<size_t> &y,
    const arma::Row<float> &sampleWeights) {

  if (X.n_cols != y.n_elem)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: X.n_cols must match "
        "y.n_elem");
  if (X.n_rows == 0)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: X must have at least one "
        "feature (row)");
  for (size_t j = 0; j < y.n_elem; ++j) {
    if (y(j) >= numClasses_)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: label >= numClasses");
  }

  const size_t n = X.n_cols;
  if (sampleWeights.n_elem != n)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: sample_weights length must "
        "match number of samples");
  fitSampleWeights_ = sampleWeights;

  rng_.seed(static_cast<std::mt19937_64::result_type>(random_state_));

  nodes_.clear();
  childIndices_.clear();
  classCounts.clear();

  fitted_ = false;

  ShapeFunctionNode root;
  root.height = 0;
  root.sampleIndices =
      arma::regspace<arma::uvec>(0, static_cast<arma::uword>(n - 1));
  root.nodeIndex = 0;
  root.numPartitions = numPartitions_;
  root.isLeaf = true;
  classCounts.push_back(fillLeafHistogram(root, y));
  root.score = impurityForClassCounts(classCounts[root.nodeIndex]);
  nodes_.push_back(root);

  childIndices_.emplace_back();
  rootIndex_ = 0;

  const arma::uvec featureCandidates =
      arma::regspace<arma::uvec>(0, X.n_rows - 1);

  const auto findBestSplit =
      [this, &X, &y, &featureCandidates](ShapeFunctionNode &node,
                                         size_t minLeaf) -> bool {
        const size_t ns = node.sampleIndices.n_elem;
        node.score = impurityForClassCounts(classCounts[node.nodeIndex]);

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
        const arma::Row<size_t> ysub = y.cols(subIdx);

        std::vector<size_t> featurePool(
            static_cast<size_t>(featureCandidates.n_elem));
        for (arma::uword i = 0; i < featureCandidates.n_elem; ++i)
          featurePool[static_cast<size_t>(i)] =
              static_cast<size_t>(featureCandidates(i));
        const std::vector<size_t> featureSubset = featureBagging_(
            std::span<const size_t>(featurePool.data(), featurePool.size()),
            rng_);

        const size_t xSubCols = static_cast<size_t>(Xsub.n_cols);
        ShapeBestBranchingState best{};
        arma::uvec featOne(1);

        const auto applyTaskFields =
            [](ShapeBestBranchingState &state,
               const ShapeBranchAssignmentSearchResult &search,
               const std::vector<std::vector<double>> &leafStats) {
              state.branching.leafStats = leafStats;
              state.partitionClassCounts = search.partitionClassCounts;
              state.partitionWeights = search.partitionWeights;
            };

        for (size_t fi = 0; fi < featureSubset.size(); ++fi) {
          const size_t f = featureSubset[fi];
          if (f >= Xsub.n_rows)
            throw std::invalid_argument(
                "ClassificationShapeGeneralizedTree::fit: feature index "
                ">= X.n_rows");
          featOne(0) = static_cast<arma::uword>(f);

          const arma::Row<float> wsub =
              subSampleWeights(fitSampleWeights_, subIdx);

          auto disc = makeClassificationDiscretizer(criterion_,
                                                    DiscretizerInputKind::Numeric);
          disc->Train(Xsub, featOne, ysub, numClasses_,
                      innerParams_.minLeafSize, innerParams_.minGainSplit,
                      innerParams_.maxDepth, innerParams_.maxLeafNodes, wsub);
          if (disc->numLeaves() < 2)
            continue;

          const ShapeBranchAssignmentSearchResult featureBest =
              searchShapeBranchAssignmentFromDiscretizer(
                  *disc, criterion_, parentImp, numPartitions_, outerParams_,
                  cdParams_, outerTreeBuilder_.eps, rng_,
                  /*useKMeansSeed=*/true, numClasses_);
          if (!featureBest.found)
            continue;

          featureHasBetterShapeBranching(
              featureBest, best, f, xSubCols,
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
        node.routingFeatures = {best.branching.featureIndex};
        node.innerDiscretizer = best.winningDiscretizer;
        node.binToPartition = std::move(best.branching.binToPartition);
        node.sampleBins = std::move(best.branching.sampleBins);
        node.splitLeafStats = std::move(best.branching.leafStats);
        node.splitBinWeights = std::move(best.binWeights);
        node.binSampleCounts = std::move(best.branching.leafNumSamples);
        node.numPartitions = best.branching.numPartitionsUsed;
        node.informationGain = best.branching.impurityDecrease;

        return true;
      };

  const auto makeChildren =
      [this, &X](const ShapeFunctionNode &parent)
          -> std::vector<ShapeFunctionNode> {
        const auto buckets = routeSamplesToPartitions(parent, X);
        const auto &binStats = parent.splitLeafStats;
        if (binStats.size() != parent.binToPartition.size())
          throw std::runtime_error(
              "ClassificationShapeGeneralizedTree::fit: splitLeafStats / "
              "binToPartition size mismatch");

        auto children =
            makeRoutedChildNodes(parent, buckets, numPartitions_);
        for (size_t p = 0; p < children.size(); ++p) {
          std::vector<double> childClassCounts(numClasses_, 0.0);
          for (size_t b = 0; b < binStats.size(); ++b) {
            if (parent.binToPartition[b] != p)
              continue;
            const auto &sb = binStats[b];
            for (size_t c = 0; c < numClasses_; ++c)
              childClassCounts[c] +=
                  (c < sb.size()) ? static_cast<double>(sb[c]) : 0.0;
          }
          children[p].score = impurityForClassCounts(childClassCounts);
          children[p].isLeaf = true;
          classCounts.push_back(childClassCounts);
        }
        return children;
      };

  outerTreeBuilder_.buildTree(
      nodes_[0], findBestSplit, makeChildren,
      [this](ShapeFunctionNode &parent,
             std::vector<ShapeFunctionNode> &children) {
        const size_t pid = parent.nodeIndex;
        nodes_[pid] = std::move(parent);
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
        yhat(s) = classificationInference::argMaxClass(classCounts[node.nodeIndex]);
        break;
      }
      const size_t part = node.routeSampleToPartition(X, static_cast<arma::uword>(s));
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
        const auto &h = classCounts[node.nodeIndex];
        double sum = 0.0;
        for (size_t c = 0; c < K; ++c)
          sum += (c < h.size()) ? static_cast<double>(h[c]) : 0.0;
        if (sum <= 0.0) {
          P.col(s).fill(uniform);
        } else {
          for (size_t c = 0; c < K; ++c) {
            const double cnt = (c < h.size()) ? static_cast<double>(h[c]) : 0.0;
            P(c, s) = static_cast<float>(cnt / sum);
          }
        }
        break;
      }
      const size_t part = node.routeSampleToPartition(X, static_cast<arma::uword>(s));
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
