#pragma once

/**
 * @file Estimators/ShapeFunctions/RegressionShapeFunctionBuilder.h
 * @brief Regression split search for shape-generalized trees.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionBuilder.h"

#include <armadillo>
#include <cstddef>
#include <vector>

class RegressionDiscretizer;
class RegressionShapeGeneralizedTree;

/**
 * Per-node split search for ``RegressionShapeGeneralizedTree``.
 *
 * For each bagged feature: train an inner regression discretizer, search
 * partition counts ``k``, refine with coordinate descent for squared error
 * (round-robin only for absolute error), and keep the best penalized branch.
 */
class RegressionShapeFunctionBuilder : public ShapeFunctionBuilder {
public:
  /**
   * @param tree               fitted estimator state (params, RNG, criterion).
   * @param X                  full training design matrix for this ``fit`` call.
   * @param y                  full training targets for this ``fit`` call.
   * @param featureCandidates  row indices into ``X`` eligible for routing.
   */
  RegressionShapeFunctionBuilder(RegressionShapeGeneralizedTree &tree,
                                 const arma::fmat &X, const arma::Row<float> &y,
                                 const arma::uvec &featureCandidates);

  bool findBestSplit(ShapeFunctionNode &node, size_t minLeafSize) override;

  std::vector<ShapeFunctionNode>
  makeChildren(const ShapeFunctionNode &parent) override;

private:
  BranchAssignmentSearchResult searchBestBranchAssignment(
      size_t numBins, double parentImp,
      std::vector<std::vector<double>> &stats,
      const std::vector<size_t> &sizes, std::vector<double> &binWeights,
      std::vector<std::vector<float>> *maeLeafYs,
      std::vector<std::vector<float>> *maeLeafWs);

  void applyTaskBranchingFields(
      BestBranchingState &best, const BranchAssignmentSearchResult &search,
      const std::vector<std::vector<double>> &leafStats) override;

  void assignNanPredictionPartition(
      ShapeFunctionNode &node,
      const std::vector<size_t> &partitionSampleCounts, const arma::fmat &Xsub,
      const arma::Row<float> &ysub, const arma::uvec &subIdx) const;

  RegressionShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<float> &y_;
  const arma::uvec &featureCandidates_;
};
