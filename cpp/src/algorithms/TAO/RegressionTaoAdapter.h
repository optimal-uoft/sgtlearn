#pragma once

/**
 * @file algorithms/TAO/RegressionTaoAdapter.h
 * @brief TAO adapter for regression shape-generalized trees.
 *
 * Care-set logic (regression variant):
 *
 * - For each sample at an internal node, walk each child subtree to its leaf
 *   and score child ``c`` with reward ``-(y - pred_c)^2`` (MSE) or
 *   ``-|y - pred_c|`` (MAE).
 * - Samples where all children tie are excluded from the care set.
 * - Each remaining care sample contributes one expanded router-training row per
 *   non-worst child (every branch strictly better than the minimum reward).
 *
 * Leaf refresh recomputes per-leaf means (squared error) or weighted medians
 * (absolute error) and updates ``leafRegressionStats`` / ``leafNumSamples``.
 * The routing discretizer always uses Gini over child-partition pseudolabels,
 * independent of the tree's regression criterion.
 */

#include <cstddef>
#include "Estimators/RegressionShapeGeneralizedTree.h"
#include "algorithms/TAO/ShapeGeneralizedTaoAdapter.h"

namespace tao {

/**
 * ``TaoAdapter`` for ``RegressionShapeGeneralizedTree``.
 *
 * Constructed by the pybind layer (or tests) with the fitted tree and the same
 * ``(X, y, sample_weights)`` used for ``fit``; passed to ``tao::optimize``.
 */
class RegressionTaoAdapter final : public ShapeGeneralizedTaoAdapter {
public:
  /**
   * @param tree           Fitted regression tree to refine in place.
   * @param X              (numFeatures, numSamples) float32 design matrix.
   * @param y              (numSamples,) float32 targets.
   * @param sampleWeights  (numSamples,) per-sample weights.
   */
  RegressionTaoAdapter(RegressionShapeGeneralizedTree &tree, const arma::fmat &X,
                     const arma::Mat<float> &y,
                     const arma::Row<float> &sampleWeights);

  /** Always ``LearningCriterion::Gini`` for the routing discretizer. */
  LearningCriterion routerCriterion() const override;

  NodeCareSet buildCareSet(const std::vector<arma::uword> &samples,
                           const std::vector<size_t> &children) const override;

  void recomputeLeafStats(
      const std::vector<std::vector<arma::uword>> &nodeSamples) override;

  void refreshNodeBinMetadata(
      ShapeFunctionNode &node,
      const std::vector<arma::uword> &samples) override;

private:
  /**
   * Per-child negative loss rewards for one sample under the current leaf
   * predictions.
   */
  void childRewards(const std::vector<size_t> &childLeaves, arma::uword col,
                    std::vector<double> &reward) const;

  RegressionShapeGeneralizedTree &regressionTree_;
  const arma::Mat<float> &y_;
  /** ``true`` when the tree criterion is squared error (mean leaf refresh). */
  bool squared_;
};

} // namespace tao
