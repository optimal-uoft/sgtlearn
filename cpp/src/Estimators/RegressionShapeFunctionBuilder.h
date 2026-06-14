#pragma once

/**
 * @file Estimators/RegressionShapeFunctionBuilder.h
 * @brief Regression split search for shape-generalized trees.
 */

#include "Estimators/ShapeFunctionBuilder.h"
#include "algorithms/ShapeBranchingTypes.h"

#include <armadillo>
#include <cstddef>
#include <limits>
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

private:
  /** Best branch assignment found for one candidate feature/discretizer. */
  struct BranchAssignmentSearchResult {
    /** Lowest penalized child impurity seen for this feature. */
    double bestFeatureScore = std::numeric_limits<double>::infinity();
    /** Fan-out ``k`` that achieved ``bestFeatureScore``. */
    size_t chosenK = 0;
    /** Bin-to-partition map for ``chosenK``. */
    std::vector<size_t> assignments;
    /** Unweighted sample count per child partition after the chosen map. */
    std::vector<size_t> partitionSampleCounts;
    /** ``parentImpurity - childImpurity`` for the chosen map. */
    double impurityDecrease = 0.0;
    /** True if at least one valid ``k`` passed min-leaf and min-gain checks. */
    bool found = false;
  };

  /**
   * Search ``k = 2 .. min(numBins, numPartitions)`` for one discretized feature.
   *
   * Squared error runs coordinate descent when enabled; absolute error uses
   * round-robin seeding with per-bin ``(y, w)`` lists passed via ``maeLeaf*``.
   */
  BranchAssignmentSearchResult searchBestBranchAssignment(
      size_t numBins, double parentImp,
      std::vector<std::vector<double>> &stats,
      const std::vector<size_t> &sizes, std::vector<double> &binWeights,
      std::vector<std::vector<float>> *maeLeafYs,
      std::vector<std::vector<float>> *maeLeafWs);

  /**
   * Replace the global best split if this feature beats ``bestPenalizedChild``.
   *
   * Copies thresholds, bin map, per-bin stats, and in-sample bin ids from
   * ``disc`` into ``brBest`` when ``bestFeatureScore`` is sufficiently lower.
   *
   * @return true if ``brBest`` was updated.
   */
  bool adoptFeatureBranchIfBetter(
      double bestFeatureScore, double &bestPenalizedChild,
      ShapeBranchingResult<double> &brBest, std::vector<size_t> &binSizesForBest,
      std::vector<double> &binWeightsForBest, size_t featureIndex,
      const BranchAssignmentSearchResult &featureBest, size_t numBins,
      size_t xSubCols, RegressionDiscretizer &disc,
      const std::vector<std::vector<double>> &stats,
      const std::vector<size_t> &sizesCopy,
      const std::vector<double> &binWeights);

  /**
   * Choose the child partition used at predict time for missing routing values.
   *
   * Uses majority finite-sample partition when no NaNs are present; otherwise
   * delegates to squared-error or absolute-error NaN routing helpers.
   */
  size_t chooseNanPredictionPartition(
      const ShapeFunctionNode &node,
      const std::vector<size_t> &partitionSampleCounts,
      const arma::fmat &Xsub, const arma::Row<float> &ysub,
      const arma::uvec &subIdx) const;

  RegressionShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<float> &y_;
  const arma::uvec &featureCandidates_;
};
