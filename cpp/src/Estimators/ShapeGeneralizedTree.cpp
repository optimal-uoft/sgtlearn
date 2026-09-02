/**
 * @file Estimators/ShapeGeneralizedTree.cpp
 * @brief Shared tree introspection for shape-generalized estimators.
 */

#include <cstddef>
#include "Estimators/ShapeGeneralizedTree.h"

#include <stdexcept>
#include <utility>

ShapeGeneralizedTree::ShapeGeneralizedTree(LearningCriterion criterion,
                                           size_t numPartitions,
                                           TreeBuildingParams outerParams,
                                           TreeBuildingParams innerParams)
    : criterion_(criterion), numPartitions_(numPartitions),
      outerParams_(std::move(outerParams)),
      innerParams_(std::move(innerParams)) {}

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

void ShapeGeneralizedTree::refreshFeatureImportances() {
  sumOfNodeImportancesByFeature_.zeros();
  totalNodeImportanceSum_ = 0.0;
  for (const ShapeFunctionNode &node : nodes_) {
    if (node.isLeaf || node.logicalFeatureIndices.empty())
      continue;
    const double gain = node.informationGain;
    if (node.logicalFeatureIndices.size() == 2) {
      sumOfNodeImportancesByFeature_(node.logicalFeatureIndices[0]) += gain / 2.0;
      sumOfNodeImportancesByFeature_(node.logicalFeatureIndices[1]) += gain / 2.0;
    } else {
      sumOfNodeImportancesByFeature_(node.logicalFeatureIndices[0]) += gain;
    }
    totalNodeImportanceSum_ += gain;
  }
  featureImportance_.zeros(sumOfNodeImportancesByFeature_.n_elem);
  if (totalNodeImportanceSum_ > 0.0)
    featureImportance_ = sumOfNodeImportancesByFeature_ / totalNodeImportanceSum_;
}
