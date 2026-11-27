#pragma once

/**
 * @file Estimators/ShapeGeneralizedTree.h
 * @brief Common fitted-tree storage, commit, and inference routing for
 *        shape-generalized estimators.
 */

#include "Domain/FeatureInfo.h"
#include "Domain/LearningCriterion.h"
#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "algorithms/FeatureBagging.h"
#include "algorithms/OuterTreeBuilder.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"

#include <armadillo>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

/**
 * Shared base for classification and regression shape-generalized trees.
 *
 * Holds the outer node array, child index map, shared hyperparameters,
 * feature importance tracking, split commit helper, and leaf-routing walk.
 */
class ShapeGeneralizedTree {
public:
  virtual ~ShapeGeneralizedTree() = default;

  const std::vector<ShapeFunctionNode> &nodes() const { return nodes_; }
  const std::vector<std::vector<size_t>> &childIndices() const {
    return childIndices_;
  }
  std::vector<ShapeFunctionNode> &mutableNodes() { return nodes_; }
  std::vector<std::vector<size_t>> &mutableChildIndices() {
    return childIndices_;
  }

  size_t rootIndex() const { return rootIndex_; }
  size_t numLeaves() const;
  size_t numNodes() const;
  bool isFitted() const { return fitted_; }
  size_t nOutputs() const { return nOutputs_; }
  bool hasPairNodes() const;

  const arma::vec &featureImportance() const;
  virtual LearningCriterion criterion() const { return criterion_; }
  size_t numPartitions() const { return numPartitions_; }
  const TreeBuildingParams &outerParams() const { return outerParams_; }
  const TreeBuildingParams &innerParams() const { return innerParams_; }

protected:
  ShapeGeneralizedTree(LearningCriterion criterion, size_t numPartitions,
                       TreeBuildingParams outerParams,
                       TreeBuildingParams innerParams,
                       CoordinateDescentParams cdParams, uint64_t random_state,
                       FeatureBaggingPickFn featureBagging,
                       size_t pairwiseCandidates, double pairwisePenalty);

  /** Commit an accepted outer split and push its child nodes onto ``nodes_``. */
  void commitSplitAndPushChildren(ShapeFunctionNode &parent,
                                  std::vector<ShapeFunctionNode> &children);

  /** Clear transient fit indices, normalize importances, and mark fitted. */
  void finalizeFit();

  /** Validate fitted tree invariants before prediction. */
  void validateFittedForPredict(const char *caller) const;

  /** Walk from ``rootIndex_`` to a leaf node for sample column ``s``. */
  const ShapeFunctionNode &routeSampleToLeaf(const arma::fmat &X,
                                             arma::uword sampleCol,
                                             const char *caller) const;

  bool fitted_ = false;

  LearningCriterion criterion_;
  size_t numPartitions_;
  TreeBuildingParams outerParams_;
  TreeBuildingParams innerParams_;
  CoordinateDescentParams cdParams_;
  uint64_t random_state_;
  std::mt19937_64 rng_;
  FeatureBaggingPickFn featureBagging_;
  size_t pairwiseCandidates_ = 0;
  double pairwisePenalty_ = 0.0;
  std::vector<FeatureInfo> features_;
  OuterTreeBuilder outerTreeBuilder_;
  size_t nOutputs_ = 1;
  arma::Row<float> fitSampleWeights_;

  std::vector<ShapeFunctionNode> nodes_;
  std::vector<std::vector<size_t>> childIndices_;
  size_t rootIndex_ = 0;

  arma::vec featureImportance_;
  arma::vec sumOfNodeImportancesByFeature_;
  double totalNodeImportanceSum_ = 0.0;
};
