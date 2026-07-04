/**
 * @file algorithms/TAO/ShapeGeneralizedTaoAdapter.cpp
 */

#include "algorithms/TAO/ShapeGeneralizedTaoAdapter.h"

#include <algorithm>
#include <stdexcept>

namespace tao {

ShapeGeneralizedTaoAdapter::ShapeGeneralizedTaoAdapter(
    ShapeGeneralizedTree &tree, const arma::fmat &X,
    const arma::Row<float> &sampleWeights)
    : tree_(tree), X_(X), w_(sampleWeights) {}

std::vector<ShapeFunctionNode> &ShapeGeneralizedTaoAdapter::nodes() {
  return tree_.mutableNodes();
}

std::vector<std::vector<size_t>> &
ShapeGeneralizedTaoAdapter::childIndices() {
  return tree_.mutableChildIndices();
}

size_t ShapeGeneralizedTaoAdapter::rootIndex() const {
  return tree_.rootIndex();
}

const arma::fmat &ShapeGeneralizedTaoAdapter::X() const { return X_; }

size_t ShapeGeneralizedTaoAdapter::numFeatures() const {
  return static_cast<size_t>(X_.n_rows);
}

const arma::Row<float> &ShapeGeneralizedTaoAdapter::sampleWeights() const {
  return w_;
}

LearningCriterion ShapeGeneralizedTaoAdapter::criterion() const {
  return tree_.criterion();
}

const TreeBuildingParams &ShapeGeneralizedTaoAdapter::innerParams() const {
  return tree_.innerParams();
}

size_t ShapeGeneralizedTaoAdapter::argMax(const std::vector<double> &counts) {
  return static_cast<size_t>(std::distance(
      counts.begin(), std::max_element(counts.begin(), counts.end())));
}

size_t ShapeGeneralizedTaoAdapter::walkToLeaf(size_t startNode,
                                              arma::uword col) const {
  const auto &nodes = tree_.nodes();
  const auto &childIndices = tree_.childIndices();
  size_t idx = startNode;
  while (!nodes[idx].isLeaf) {
    const ShapeFunctionNode &node = nodes[idx];
    const size_t part = node.routeSampleToPartition(X_, col);
    const auto &children = childIndices[idx];
    if (part >= children.size())
      throw std::runtime_error("tao::walkToLeaf: routed partition out of range");
    idx = children[part];
  }
  return idx;
}

} // namespace tao
