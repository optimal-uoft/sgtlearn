/**
 * @file Estimators/ShapeGeneralizedTree.cpp
 * @brief Shared tree construction, commit, and inference routing for
 *        shape-generalized estimators.
 */

#include "Estimators/ShapeGeneralizedTree.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>

ShapeGeneralizedTree::ShapeGeneralizedTree(
    LearningCriterion criterion, size_t numPartitions,
    TreeBuildingParams outerParams, TreeBuildingParams innerParams,
    CoordinateDescentParams cdParams, uint64_t random_state,
    FeatureBaggingPickFn featureBagging, size_t pairwiseCandidates,
    double pairwisePenalty)
    : criterion_(criterion), numPartitions_(numPartitions),
      outerParams_(std::move(outerParams)),
      innerParams_(std::move(innerParams)), cdParams_(cdParams),
      random_state_(random_state), rng_(),
      featureBagging_(featureBagging
                          ? std::move(featureBagging)
                          : FeatureBaggingPickFn(pickAllFeatureIndices)),
      pairwiseCandidates_(pairwiseCandidates),
      pairwisePenalty_(pairwisePenalty), outerTreeBuilder_(outerParams_) {
  if (numPartitions_ < 2)
    throw std::invalid_argument(
        "ShapeGeneralizedTree: numPartitions must be >= 2");
  if (!std::isfinite(outerParams_.minGainSplit) ||
      outerParams_.minGainSplit < 0.0)
    throw std::invalid_argument(
        "min_impurity_decrease must be finite and non-negative");
  if (!std::isfinite(outerParams_.branchingPenalty) ||
      outerParams_.branchingPenalty < 0.0)
    throw std::invalid_argument(
        "branching_penalty must be finite and non-negative");
  if (!std::isfinite(pairwisePenalty_) || pairwisePenalty_ < 0.0)
    throw std::invalid_argument(
        "pairwise_penalty must be finite and non-negative");
}

size_t ShapeGeneralizedTree::numLeaves() const {
  size_t c = 0;
  for (const auto &node : nodes_) {
    if (node.isLeaf)
      ++c;
  }
  return c;
}

size_t ShapeGeneralizedTree::numNodes() const { return nodes_.size(); }

bool ShapeGeneralizedTree::hasPairNodes() const {
  return std::any_of(nodes_.begin(), nodes_.end(),
                     [](const ShapeFunctionNode &node) {
                       return !node.isLeaf &&
                              node.logicalFeatureIndices.size() == 2;
                     });
}

const arma::vec &ShapeGeneralizedTree::featureImportance() const {
  if (!fitted_)
    throw std::logic_error(
        "ShapeGeneralizedTree::featureImportance: model is not fitted");
  return featureImportance_;
}

void ShapeGeneralizedTree::commitSplitAndPushChildren(
    ShapeFunctionNode &parent, std::vector<ShapeFunctionNode> &children) {
  const auto &logical = parent.logicalFeatureIndices;
  if (logical.size() == 2) {
    sumOfNodeImportancesByFeature_(logical[0]) += parent.informationGain / 2.0;
    sumOfNodeImportancesByFeature_(logical[1]) += parent.informationGain / 2.0;
  } else {
    sumOfNodeImportancesByFeature_(parent.splitFeatureIndex) +=
        parent.informationGain;
  }
  totalNodeImportanceSum_ += parent.informationGain;
  const size_t pid = parent.nodeIndex;
  const size_t numChildPartitions = parent.numPartitions;
  nodes_[pid] = std::move(parent);
  nodes_[pid].isLeaf = false;
  childIndices_[pid].assign(numChildPartitions, 0);
  for (size_t p = 0; p < numChildPartitions; ++p) {
    const size_t cid = nodes_.size();
    children[p].nodeIndex = cid;
    nodes_.push_back(children[p]);
    childIndices_.push_back({});
    childIndices_[pid][p] = cid;
  }
}

void ShapeGeneralizedTree::finalizeFit() {
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

void ShapeGeneralizedTree::validateFittedForPredict(const char *caller) const {
  if (!fitted_)
    throw std::logic_error(std::string(caller) + ": model is not fitted");
  if (nodes_.empty())
    throw std::logic_error(std::string(caller) + ": empty tree");
  if (rootIndex_ >= nodes_.size())
    throw std::logic_error(std::string(caller) + ": invalid root index");
  if (childIndices_.size() != nodes_.size())
    throw std::logic_error(std::string(caller) +
                           ": childIndices/nodes size mismatch");
}

const ShapeFunctionNode &
ShapeGeneralizedTree::routeSampleToLeaf(const arma::fmat &X,
                                        arma::uword sampleCol,
                                        const char *caller) const {
  size_t idx = rootIndex_;
  while (true) {
    const auto &node = nodes_[idx];
    if (node.isLeaf)
      return node;
    const size_t part = node.routeSampleToPartition(X, sampleCol);
    if (part >= childIndices_[idx].size())
      throw std::runtime_error(std::string(caller) +
                               ": child partition out of range");
    idx = childIndices_[idx][part];
    if (idx >= nodes_.size())
      throw std::runtime_error(std::string(caller) +
                               ": child node index out of range");
  }
}
