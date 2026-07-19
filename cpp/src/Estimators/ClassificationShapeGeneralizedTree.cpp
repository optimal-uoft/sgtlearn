/**
 * @file Estimators/ClassificationShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the classification
 *        shape-generalized tree. Per-node inner fit: discretize -> search
 *        partition counts k in [2, numPartitions] (construct_mapping) ->
 *        coordinate descent -> record best branch.
 */

#include <memory>
#include <utility>
#include <cstdint>
#include <cstddef>
#include "Estimators/ClassificationShapeGeneralizedTree.h"

#include "Criterion.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "Discretizers/factories/DiscretizerFactories.h"
#include "Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h"

#include <algorithm>
#include <armadillo>
#include <iterator>
#include <limits>
#include <numeric>
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

// Argmax within the block [offset, offset + nClasses) of a concatenated
// histogram, returned as a class index relative to the block.
size_t argMaxClassInBlock(const std::vector<double> &counts, size_t offset,
                          size_t nClasses) {
  size_t best = 0;
  double bestVal = -1.0;
  for (size_t c = 0; c < nClasses; ++c) {
    const double v = (offset + c < counts.size()) ? counts[offset + c] : 0.0;
    if (v > bestVal) {
      bestVal = v;
      best = c;
    }
  }
  return best;
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

ClassificationShapeGeneralizedTree::ClassificationShapeGeneralizedTree(
    LearningCriterion criterion, std::vector<size_t> nClassesPerOutput,
    size_t numPartitions, TreeBuildingParams outerParams,
    TreeBuildingParams innerParams, CoordinateDescentParams cdParams,
    uint64_t random_state, FeatureBaggingPickFn featureBagging)
    : ShapeGeneralizedTree(criterion, numPartitions, outerParams, innerParams),
      numClasses_(nClassesPerOutput.empty() ? 0 : nClassesPerOutput[0]),
      configuredClassesPerOutput_(std::move(nClassesPerOutput)),
      cdParams_(cdParams), random_state_(random_state), rng_(),
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
  if (configuredClassesPerOutput_.empty())
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: nClassesPerOutput must be "
        "non-empty");
  for (size_t nc : configuredClassesPerOutput_)
    if (nc < 2)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree: each output must have >= 2 "
          "classes");
  if (numPartitions < 2)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: numPartitions must be >= 2");
}

void ClassificationShapeGeneralizedTree::resolveOutputLayout(size_t nOutputs) {
  nOutputs_ = nOutputs;
  classesPerOutput_.clear();
  if (!configuredClassesPerOutput_.empty()) {
    if (configuredClassesPerOutput_.size() == 1)
      classesPerOutput_.assign(nOutputs_, configuredClassesPerOutput_[0]);
    else if (configuredClassesPerOutput_.size() == nOutputs_)
      classesPerOutput_ = configuredClassesPerOutput_;
    else
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: nClassesPerOutput length "
          "must be 1 or match y.n_rows");
  } else {
    classesPerOutput_.assign(nOutputs_, numClasses_);
  }
  classOffsets_.assign(nOutputs_, 0);
  totalClasses_ = 0;
  for (size_t o = 0; o < nOutputs_; ++o) {
    classOffsets_[o] = totalClasses_;
    totalClasses_ += classesPerOutput_[o];
  }
}

double ClassificationShapeGeneralizedTree::impurityForClassCounts(
    const std::vector<double> &classCounts) const {
  double totalWeight = 0.0;
  for (double c : classCounts)
    totalWeight += c;
  if (totalWeight <= 0.0)
    return 0.0;
  if (criterion_ == LearningCriterion::Gini)
    return Criterion::giniMulti(classCounts, classesPerOutput_);
  if (criterion_ == LearningCriterion::Entropy)
    return Criterion::entropyMulti(classCounts, classesPerOutput_);
  throw std::runtime_error("ClassificationShapeGeneralizedTree::"
                           "impurityForClassCounts: invalid criterion");
}

std::vector<double> ClassificationShapeGeneralizedTree::fillLeafHistogram(
    ShapeFunctionNode &node, const arma::Mat<size_t> &y) const {
  std::vector<double> counts(totalClasses_, 0.0);
  for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(node.sampleIndices(i));
    const double w = static_cast<double>(fitSampleWeights_(si));
    for (size_t o = 0; o < nOutputs_; ++o) {
      const size_t lab =
          y(static_cast<arma::uword>(o), static_cast<arma::uword>(si));
      if (lab >= classesPerOutput_[o])
        throw std::invalid_argument(
            "ClassificationShapeGeneralizedTree::fit: class label out of "
            "range");
      counts[classOffsets_[o] + lab] += w;
    }
  }
  return counts;
}

void ClassificationShapeGeneralizedTree::fit(
    const arma::fmat &X, const arma::Mat<size_t> &y,
    const arma::Row<float> &sampleWeights,
    const std::vector<FeatureInfo> &features) {

  if (X.n_cols != y.n_cols)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: X.n_cols must match "
        "y.n_cols");
  if (y.n_rows == 0)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: y must have at least one "
        "output (row)");
  if (X.n_rows == 0)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: X must have at least one "
        "feature (row)");

  resolveOutputLayout(static_cast<size_t>(y.n_rows));

  for (arma::uword o = 0; o < y.n_rows; ++o)
    for (arma::uword j = 0; j < y.n_cols; ++j)
      if (y(o, j) >= classesPerOutput_[o])
        throw std::invalid_argument(
            "ClassificationShapeGeneralizedTree::fit: label >= numClasses");

  const size_t n = X.n_cols;
  if (sampleWeights.n_elem != n)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: sample_weights length must "
        "match number of samples");
  fitSampleWeights_ = sampleWeights;
  features_ = features;

  rng_.seed(static_cast<std::mt19937_64::result_type>(random_state_));

  nodes_.clear();
  childIndices_.clear();
  classCounts.clear();

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
  classCounts.push_back(fillLeafHistogram(root, y));
  root.score = impurityForClassCounts(classCounts[root.nodeIndex]);
  nodes_.push_back(root);

  childIndices_.emplace_back();
  rootIndex_ = 0;

  const size_t numLogicalFeatures = features_.size();

  const auto findBestSplit =
      [this, &X, &y, numLogicalFeatures](ShapeFunctionNode &node,
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
        const arma::Mat<size_t> ysub = y.cols(subIdx);

        std::vector<size_t> featurePool(numLogicalFeatures);
        std::iota(featurePool.begin(), featurePool.end(), 0);
        const std::vector<size_t> featureSubset = featureBagging_(
            std::span<const size_t>(featurePool.data(), featurePool.size()),
            rng_);

        const size_t xSubCols = static_cast<size_t>(Xsub.n_cols);
        ShapeBestBranchingState best{};

        const auto applyTaskFields =
            [](ShapeBestBranchingState &state,
               const ShapeBranchAssignmentSearchResult &search,
               const std::vector<std::vector<double>> &leafStats) {
              state.branching.leafStats = leafStats;
              state.partitionClassCounts = search.partitionClassCounts;
              state.partitionWeights = search.partitionWeights;
            };

        for (size_t fi = 0; fi < featureSubset.size(); ++fi) {
          const size_t logicalIdx = featureSubset[fi];
          const FeatureInfo &feature = features_[logicalIdx];

          const arma::Row<float> wsub =
              subSampleWeights(fitSampleWeights_, subIdx);

          auto disc = makeClassificationDiscretizer(criterion_, feature);
          trainClassificationDiscretizer(
              *disc, feature, Xsub, ysub, classesPerOutput_,
              innerParams_.minLeafSize, innerParams_.minGainSplit,
              innerParams_.maxDepth, innerParams_.maxLeafNodes, wsub);
          if (disc->numLeaves() < 2)
            continue;

          const ShapeBranchAssignmentSearchResult featureBest =
              searchShapeBranchAssignmentFromDiscretizer(
                  *disc, criterion_, parentImp, numPartitions_, outerParams_,
                  cdParams_, outerTreeBuilder_.eps, rng_,
                  /*useKMeansSeed=*/true, classesPerOutput_, nOutputs_);
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
          std::vector<double> childClassCounts(totalClasses_, 0.0);
          for (size_t b = 0; b < binStats.size(); ++b) {
            if (parent.binToPartition[b] != p)
              continue;
            const auto &sb = binStats[b];
            for (size_t c = 0; c < totalClasses_; ++c)
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
        sumOfNodeImportancesByFeature_(parent.splitFeatureIndex) +=
            parent.informationGain;
        totalNodeImportanceSum_ += parent.informationGain;
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

  featureImportance_.zeros(sumOfNodeImportancesByFeature_.n_elem);
  if (totalNodeImportanceSum_ > 0.0)
    featureImportance_ =
        sumOfNodeImportancesByFeature_ / totalNodeImportanceSum_;
  fitted_ = true;
}

arma::Mat<size_t>
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
  arma::Mat<size_t> yhat(nOutputs_, n_samples);

  for (size_t s = 0; s < n_samples; ++s) {
    size_t idx = rootIndex_;
    while (true) {
      const auto &node = nodes_[idx];
      if (node.isLeaf) {
        const auto &counts = classCounts[node.nodeIndex];
        for (size_t o = 0; o < nOutputs_; ++o)
          yhat(static_cast<arma::uword>(o), static_cast<arma::uword>(s)) =
              classificationInference::argMaxClassInBlock(
                  counts, classOffsets_[o], classesPerOutput_[o]);
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

std::vector<arma::fmat>
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
  std::vector<arma::fmat> probas;
  probas.reserve(nOutputs_);
  for (size_t o = 0; o < nOutputs_; ++o)
    probas.emplace_back(classesPerOutput_[o], n_samples);

  for (size_t s = 0; s < n_samples; ++s) {
    size_t idx = rootIndex_;
    while (true) {
      const auto &node = nodes_[idx];
      if (node.isLeaf) {
        const auto &h = classCounts[node.nodeIndex];
        for (size_t o = 0; o < nOutputs_; ++o) {
          const size_t off = classOffsets_[o];
          const size_t K = classesPerOutput_[o];
          const float uniform = K > 0 ? 1.f / static_cast<float>(K) : 0.f;
          double sum = 0.0;
          for (size_t c = 0; c < K; ++c)
            sum += (off + c < h.size()) ? static_cast<double>(h[off + c]) : 0.0;
          if (sum <= 0.0) {
            probas[o].col(s).fill(uniform);
          } else {
            for (size_t c = 0; c < K; ++c) {
              const double cnt =
                  (off + c < h.size()) ? static_cast<double>(h[off + c]) : 0.0;
              probas[o](static_cast<arma::uword>(c),
                        static_cast<arma::uword>(s)) =
                  static_cast<float>(cnt / sum);
            }
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
  return probas;
}
