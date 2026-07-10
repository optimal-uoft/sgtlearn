/**
 * @file Estimators/ShapeFunctions/ShapeFunctionNode.cpp
 * @brief Out-of-line members of ``ShapeFunctionNode`` that require the complete
 *        ``ShapeGeneralizedTree`` definition.
 */

#include <cstddef>
#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"

#include "Estimators/ShapeGeneralizedTree.h"

#include <stdexcept>

size_t ShapeFunctionNode::routeFeatureValuesToPartition(
    const std::vector<float> &featureValues) const {
  if (!innerDiscretizer)
    throw std::runtime_error("innerDiscretizer is not set");
  if (binToPartition.empty())
    throw std::runtime_error("binToPartition is empty");
  const size_t bin = innerDiscretizer->routeToBin(featureValues);
  if (bin >= binToPartition.size())
    throw std::runtime_error("bin out of range");
  return binToPartition[bin];
}

std::vector<float> ShapeFunctionNode::gatherRoutingFeatureValues(
    const arma::fmat &X, arma::uword sampleCol) const {
  if (routingFeatures.empty())
    throw std::runtime_error("routingFeatures is empty");
  if (sampleCol >= X.n_cols)
    throw std::invalid_argument(
        "gatherRoutingFeatureValues: sample column out of range for X");
  std::vector<float> values;
  values.reserve(routingFeatures.size());
  for (size_t f : routingFeatures) {
    if (f >= X.n_rows)
      throw std::invalid_argument(
          "gatherRoutingFeatureValues: routing feature index out of range for X");
    values.push_back(X(f, sampleCol));
  }
  return values;
}

std::vector<ShapeFunctionNode *> ShapeFunctionNode::getChildren() const {
  if (tree == nullptr)
    throw std::runtime_error("tree is null");

  std::vector<ShapeFunctionNode *> children;
  // const_cast: ShapeGeneralizedTree exposes only const node access, but the
  // alternating-optimization pass mutates the children it walks.
  auto &mutableNodes =
      const_cast<std::vector<ShapeFunctionNode> &>(tree->nodes());
  for (size_t childIndex : tree->childIndices()[nodeIndex])
    children.push_back(&mutableNodes[childIndex]);
  return children;
}
