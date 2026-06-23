#pragma once

/**
 * @file Estimators/ShapeFunctions/ClassificationShapeFunctionBuilder.h
 * @brief Classification split search for shape-generalized trees.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionBuilder.h"

#include "Domain/LearningCriterion.h"

#include <armadillo>
#include <cstddef>
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

  std::vector<ShapeFunctionNode>
  makeChildren(const ShapeFunctionNode &parent) override;

private:
  BranchAssignmentSearchResult searchBestBranchAssignment(
      size_t numBins, double parentImp,
      std::vector<std::vector<double>> &stats,
      const std::vector<size_t> &sizes, std::vector<double> &weights);

  void applyTaskBranchingFields(
      BestBranchingState &best, const BranchAssignmentSearchResult &search,
      const std::vector<std::vector<double>> &leafStats) override;

  /**
   * Assign the NaN branch after coordinate descent has run on the numeric bins.
   * If the discretizer saw no NaN (``nanSeen == false``) route NaN to the
   * largest child partition; otherwise pick the partition whose impurity grows
   * least when the NaN bucket (``nanStats`` / ``nanWeight``) is merged in.
   */
  void assignNanPredictionPartition(
      ShapeFunctionNode &node,
      const std::vector<size_t> &partitionSampleCounts,
      const std::vector<std::vector<double>> &partitionClassCounts,
      const std::vector<double> &partitionWeights, bool nanSeen,
      const std::vector<double> &nanStats, double nanWeight) const;

  using PartitionImpurityFn = double (*)(const std::vector<double> &classCounts,
                                         double totalWeight);

  static PartitionImpurityFn partitionImpurityFnFor(LearningCriterion criterion);

  /** Weighted impurity for one partition histogram (Gini or entropy per tree). */
  double partitionImpurity(const std::vector<double> &classCounts,
                           double totalWeight) const;

  PartitionImpurityFn partitionImpurity_ = nullptr;

  ClassificationShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<size_t> &y_;
  const arma::uvec &featureCandidates_;
};
