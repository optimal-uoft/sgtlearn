/**
 * @file Estimators/ShapeGeneralizedTree.cpp
 * @brief Shared tree introspection for shape-generalized estimators.
 */

#include "Estimators/ShapeGeneralizedTree.h"

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
