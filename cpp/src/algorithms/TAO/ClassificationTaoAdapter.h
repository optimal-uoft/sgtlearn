#pragma once

/**
 * @file algorithms/TAO/ClassificationTaoAdapter.h
 * @brief TAO adapter for classification shape-generalized trees.
 *
 * Care-set logic (classification variant):
 *
 * - For each sample at an internal node, walk each child subtree to its leaf
 *   and score child ``c`` with reward ``1`` if the leaf's majority class equals
 *   the sample label, else ``0``.
 * - Samples where all children tie are excluded from the care set.
 * - Samples with multiple equally good children contribute one expanded router-
 *   training row per good child (multi-to-single pseudolabel expansion).
 *
 * Leaf refresh recomputes weighted class histograms at every node from
 * ``nodeSamples``. The routing discretizer uses the tree's own criterion
 * (Gini or entropy).
 */

#include "Estimators/ClassificationShapeGeneralizedTree.h"
#include "algorithms/TAO/ShapeGeneralizedTaoAdapter.h"

namespace tao {

/**
 * ``TaoAdapter`` for ``ClassificationShapeGeneralizedTree``.
 *
 * Constructed by the pybind layer (or tests) with the fitted tree and the same
 * ``(X, y, sample_weights)`` used for ``fit``; passed to ``tao::optimize``.
 */
class ClassificationTaoAdapter final : public ShapeGeneralizedTaoAdapter {
public:
  /**
   * @param tree           Fitted classification tree to refine in place.
   * @param X              (numFeatures, numSamples) float32 design matrix.
   * @param y              (numSamples,) integer class labels in [0, numClasses).
   * @param sampleWeights  (numSamples,) per-sample weights.
   */
  ClassificationTaoAdapter(ClassificationShapeGeneralizedTree &tree,
                           const arma::fmat &X, const arma::Row<size_t> &y,
                           const arma::Row<float> &sampleWeights);

  /** Same as the tree's impurity criterion (Gini or entropy). */
  LearningCriterion routerCriterion() const override;

  NodeCareSet buildCareSet(const std::vector<arma::uword> &samples,
                           const std::vector<size_t> &children) const override;

  void recomputeLeafStats(
      const std::vector<std::vector<arma::uword>> &nodeSamples) override;

private:
  /**
   * Per-child correctness rewards for one sample: ``1`` if child leaf predicts
   * the sample label, else ``0``.
   */
  void childRewards(const std::vector<size_t> &childLeaves, arma::uword col,
                    std::vector<double> &reward) const;

  ClassificationShapeGeneralizedTree &classificationTree_;
  const arma::Row<size_t> &y_;
};

} // namespace tao
