/**
 * @file Estimators/ShapeGeneralizedTree.cpp
 * @brief Shared tree introspection for shape-generalized estimators.
 */

#include "Estimators/ShapeGeneralizedTree.h"

size_t ShapeGeneralizedTree::numLeaves() const {
  size_t c = 0;
  for (const auto &node : nodes_) {
    if (node.isLeaf)
      ++c;
  }
  return c;
}

size_t ShapeGeneralizedTree::numNodes() const { return nodes_.size(); }
