#pragma once

/**
 * @file algorithms/TAO/ShapeGeneralizedTaoAdapter.h
 * @brief Shared ``TaoAdapter`` implementation for ``ShapeGeneralizedTree`` estimators.
 *
 * Both classification and regression shape-generalized trees share the same outer
 * routing structure and hyperparameter accessors. This intermediate base implements
 * all ``TaoAdapter`` methods that forward directly to ``ShapeGeneralizedTree`` or
 * to the refinement-time training data passed into the constructor.
 *
 * Concrete subclasses supply only ``routerCriterion``, ``buildCareSet``, and
 * ``recomputeLeafStats``.
 */

#include <cstddef>
#include "Estimators/ShapeGeneralizedTree.h"
#include "algorithms/TAO/TaoAdapter.h"

namespace tao {

/**
 * Partial ``TaoAdapter`` for ``ShapeGeneralizedTree`` subclasses.
 *
 * Holds non-owning references to the tree being refined and to the training
 * matrices supplied at TAO call time. Mutable tree access goes through
 * ``ShapeGeneralizedTree::mutableNodes()`` / ``mutableChildIndices()`` so
 * routing updates do not require ``const_cast``.
 */
class ShapeGeneralizedTaoAdapter : public TaoAdapter {
public:
  /**
   * @param tree           Fitted estimator whose routing rules will be refined.
   * @param X              Training design matrix, (numFeatures, numSamples).
   * @param sampleWeights  Per-sample weights, shape (numSamples,).
   */
  ShapeGeneralizedTaoAdapter(ShapeGeneralizedTree &tree, const arma::fmat &X,
                           const arma::Row<float> &sampleWeights);

  std::vector<ShapeFunctionNode> &nodes() override;
  std::vector<std::vector<size_t>> &childIndices() override;
  size_t rootIndex() const override;

  const arma::fmat &X() const override;
  size_t numFeatures() const override;
  const arma::Row<float> &sampleWeights() const override;

  LearningCriterion criterion() const override;
  const TreeBuildingParams &innerParams() const override;

protected:
  /** Index of the largest entry in ``counts`` (ties: first max wins). */
  static size_t argMax(const std::vector<double> &counts);

  /**
   * Follow routing from ``startNode`` until a leaf is reached for sample ``col``.
   *
   * Uses the fitted tree and design matrix held by this adapter.
   */
  size_t walkToLeaf(size_t startNode, arma::uword col) const;

  ShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<float> &w_;
};

} // namespace tao
