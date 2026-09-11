/**
 * @file Estimators/ShapeGeneralizedTree.cpp
 * @brief Shared tree introspection for shape-generalized estimators.
 */

#include <cstddef>
#include "Estimators/ShapeGeneralizedTree.h"

#include <stdexcept>
#include <utility>
#include <cmath>

ShapeGeneralizedTree::ShapeGeneralizedTree(LearningCriterion criterion,
                                           size_t numPartitions,
                                           TreeBuildingParams outerParams,
                                           TreeBuildingParams innerParams)
    : criterion_(criterion), numPartitions_(numPartitions),
      outerParams_(std::move(outerParams)),
      innerParams_(std::move(innerParams)) {
  if (!std::isfinite(outerParams_.minGainSplit) || outerParams_.minGainSplit < 0.0)
    throw std::invalid_argument("min_impurity_decrease must be finite and non-negative");
  if (!std::isfinite(outerParams_.branchingPenalty) || outerParams_.branchingPenalty < 0.0)
    throw std::invalid_argument("branching_penalty must be finite and non-negative");
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

const arma::vec &ShapeGeneralizedTree::featureImportance() const {
  if (!fitted_)
    throw std::logic_error(
        "ShapeGeneralizedTree::featureImportance: model is not fitted");
  return featureImportance_;
}
