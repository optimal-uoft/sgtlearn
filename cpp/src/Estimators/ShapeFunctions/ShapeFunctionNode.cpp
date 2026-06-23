/**
 * @file Estimators/ShapeFunctions/ShapeFunctionNode.cpp
 * @brief Out-of-line members of ``ShapeFunctionNode`` that require the complete
 *        ``ShapeGeneralizedTree`` definition.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"

#include "Estimators/ShapeGeneralizedTree.h"

#include <stdexcept>

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
