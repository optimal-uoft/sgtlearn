#pragma once

/**
 * @file Estimators/ClassificationShapeFunctionBuilder.h
 * @brief Classification split search for shape-generalized trees.
 */

#include "Estimators/ShapeFunctionBuilder.h"
#include "algorithms/ShapeBranchingTypes.h"

#include <armadillo>
#include <cstddef>
#include <limits>
#include <vector>

class ClassificationDiscretizer;
class ClassificationShapeGeneralizedTree;

/**
 * Per-node split search for ``ClassificationShapeGeneralizedTree``.
 *
 * For each bagged feature: train an inner classification discretizer, search
 * partition counts ``k``, optionally refine with coordinate descent, and keep
 * the feature with the lowest penalized child impurity.
 */
class ClassificationShapeFunctionBuilder : public ShapeFunctionBuilder {
public:
  /**
   * @param tree               fitted estimator state (params, RNG, class counts).
   * @param X                  full training design matrix for this ``fit`` call.
   * @param y                  full training labels for this ``fit`` call.
   * @param featureCandidates  row indices into ``X`` eligible for routing.
   */
  ClassificationShapeFunctionBuilder(ClassificationShapeGeneralizedTree &tree,
                                     const arma::fmat &X,
                                     const arma::Row<size_t> &y,
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
    /** Weighted class counts per partition after coordinate descent. */
    std::vector<std::vector<double>> partitionClassCounts;
    /** Sum of bin weights per partition after coordinate descent. */
    std::vector<double> partitionWeights;
    /** ``parentImpurity - childImpurity`` for the chosen map. */
    double impurityDecrease = 0.0;
    /** True if at least one valid ``k`` passed min-leaf and min-gain checks. */
    bool found = false;
  };

  /**
   * Search ``k = 2 .. min(numBins, numPartitions)`` for one discretized feature.
   *
   * Seeds assignments (identity, round-robin, or k-means), runs coordinate
   * descent when ``k < numBins``, and scores with penalized child impurity.
   */
  BranchAssignmentSearchResult searchBestBranchAssignment(
      size_t numBins, double parentImp,
      std::vector<std::vector<double>> &stats,
      const std::vector<size_t> &sizes, std::vector<double> &weights);

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
      ShapeBranchingResult<double> &brBest,
      std::vector<double> &binWeightsForBest,
      std::vector<std::vector<double>> &partitionClassCountsForBest,
      std::vector<double> &partitionWeightsForBest, size_t featureIndex,
      const BranchAssignmentSearchResult &featureBest, size_t xSubCols,
      ClassificationDiscretizer &disc,
      const std::vector<std::vector<double>> &stats,
      const std::vector<size_t> &sizes,
      const std::vector<double> &weights);

  /**
   * Choose the child partition used at predict time for missing routing values.
   *
   * ``partitionClassCounts`` / ``partitionWeights`` are the finite-sample
   * aggregates from the winning branch assignment; missing columns are scored
   * by trial assignment to each partition.
   */
  size_t chooseNanPredictionPartition(
      size_t numPartitions, size_t routingFeature,
      const std::vector<size_t> &partitionSampleCounts,
      const std::vector<std::vector<double>> &partitionClassCounts,
      const std::vector<double> &partitionWeights, const arma::fmat &Xsub,
      const arma::Row<size_t> &ysub, const arma::uvec &subIdx) const;

  ClassificationShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<size_t> &y_;
  const arma::uvec &featureCandidates_;
};
