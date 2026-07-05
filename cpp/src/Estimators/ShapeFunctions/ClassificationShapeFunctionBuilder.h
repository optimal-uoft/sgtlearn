#pragma once

/**
 * @file Estimators/ShapeFunctions/ClassificationShapeFunctionBuilder.h
 * @brief Classification split search for shape-generalized trees.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h"

#include <armadillo>
#include <cstddef>
#include <vector>

class ClassificationShapeGeneralizedTree;

class ClassificationShapeFunctionBuilder {
public:
  ClassificationShapeFunctionBuilder(ClassificationShapeGeneralizedTree &tree,
                                     const arma::fmat &X,
                                     const arma::Row<size_t> &y,
                                     const arma::uvec &featureCandidates);

  bool findBestSplit(ShapeFunctionNode &node, size_t minLeafSize);

  std::vector<ShapeFunctionNode> makeChildren(const ShapeFunctionNode &parent);

private:
  ShapeBranchAssignmentSearchResult searchBestBranchAssignment(
      size_t numRoutingBins, double parentImp,
      std::vector<std::vector<double>> &stats, const std::vector<size_t> &sizes,
      std::vector<double> &weights);

  ClassificationShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<size_t> &y_;
  const arma::uvec &featureCandidates_;
};
