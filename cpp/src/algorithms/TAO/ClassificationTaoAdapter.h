#pragma once

/**
 * @file algorithms/TAO/ClassificationTaoAdapter.h
 * @brief TAO adapter for classification shape-generalized trees.
 *
 * Care-set logic (classification variant):
 *
 * - For each sample at an internal node, walk each child subtree to its leaf
 *   and score child ``c`` with reward ``1/x`` if the leaf's majority class equals
 *   the sample label, where ``x`` is the number of correct children; else ``0``.
 * - Samples where all children tie are excluded from the care set.
 * - Samples with multiple equally good children contribute one expanded router-
 *   training row per good child (multi-to-single pseudolabel expansion).
 *
 * Leaf refresh recomputes weighted class histograms at every node from
 * ``nodeSamples``. The routing discretizer uses the tree's own criterion
 * (Gini or entropy).
 */

#include <cstddef>
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
                           const arma::fmat &X, const arma::Mat<size_t> &y,
                           const arma::Row<float> &sampleWeights);

  /** Same as the tree's impurity criterion (Gini or entropy). */
  LearningCriterion routerCriterion() const override;

  NodeCareSet buildCareSet(const std::vector<arma::uword> &samples,
                           const std::vector<size_t> &children) const override;

  void recomputeLeafStats(
      const std::vector<std::vector<arma::uword>> &nodeSamples) override;

private:
  /**
   * Per-child correctness rewards for one sample.
   *
   * Single-output: ``1/x`` split uniformly across correct children (``x`` =
   * count of correct), else ``0``. Multi-output: fraction of outputs correctly
   * classified by each child's leaf.
   */
  void childRewards(const std::vector<size_t> &childLeaves, arma::uword col,
                    std::vector<double> &reward) const;

  ClassificationShapeGeneralizedTree &classificationTree_;
  const arma::Mat<size_t> &y_;
  /** Per-output class counts (copied from the fitted tree). */
  std::vector<size_t> classesPerOutput_;
  /** Number of outputs (``y_.n_rows``). */
  size_t nOutputs_;
};

} // namespace tao
