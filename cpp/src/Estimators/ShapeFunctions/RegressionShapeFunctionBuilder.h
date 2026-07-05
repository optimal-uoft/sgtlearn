#pragma once

/**
 * @file Estimators/ShapeFunctions/RegressionShapeFunctionBuilder.h
 * @brief Regression split search for shape-generalized trees.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h"

#include <armadillo>
#include <cstddef>
#include <vector>

class RegressionShapeGeneralizedTree;

class RegressionShapeFunctionBuilder {
public:
  RegressionShapeFunctionBuilder(RegressionShapeGeneralizedTree &tree,
                                 const arma::fmat &X, const arma::Row<float> &y,
                                 const arma::uvec &featureCandidates);

  bool findBestSplit(ShapeFunctionNode &node, size_t minLeafSize);

  std::vector<ShapeFunctionNode> makeChildren(const ShapeFunctionNode &parent);

private:
  ShapeBranchAssignmentSearchResult searchBestBranchAssignment(
      size_t numRoutingBins, double parentImp,
      std::vector<std::vector<double>> &stats, const std::vector<size_t> &sizes,
      std::vector<double> &binWeights,
      std::vector<std::vector<float>> *maeLeafYs,
      std::vector<std::vector<float>> *maeLeafWs);

  RegressionShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<float> &y_;
  const arma::uvec &featureCandidates_;
};
