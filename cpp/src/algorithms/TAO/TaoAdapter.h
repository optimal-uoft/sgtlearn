#pragma once

/**
 * @file algorithms/TAO/TaoAdapter.h
 * @brief Base adapter between TAO and fitted ``ShapeGeneralizedTree`` estimators.
 *
 * TAO (Tree-Alternating Optimization) refines routing rules on an already-fitted
 * outer tree without changing its topology. Shared tree accessors, care-set
 * construction, and bin-metadata refresh live here; concrete subclasses
 * (``ClassificationTaoAdapter``, ``RegressionTaoAdapter``) supply task-specific
 * child rewards, leaf recomputation, and per-bin stats.
 */

#include "Domain/LearningCriterion.h"
#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "Estimators/ShapeGeneralizedTree.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"

#include <armadillo>
#include <cstddef>
#include <vector>

namespace tao {

/**
 * Per-node care set: samples whose routing choice matters for scoring and for
 * training the candidate routing discretizer.
 */
struct NodeCareSet {
  std::vector<arma::uword> careCols;
  std::vector<std::vector<double>> careRewards;
  /** Per care sample; empty means uniform weight 1. */
  std::vector<double> careWeights;

  arma::fmat Xexp;
  arma::Row<size_t> yexp;
  arma::Row<float> wexp;
  size_t dummyChild = 0;

  bool empty() const { return careCols.empty(); }
  size_t size() const { return careCols.size(); }
};

/**
 * Base adapter for one TAO refinement pass over a ``ShapeGeneralizedTree``.
 */
class TaoAdapter {
public:
  TaoAdapter(ShapeGeneralizedTree &tree, const arma::fmat &X,
             const arma::Row<float> &sampleWeights);
  virtual ~TaoAdapter() = default;

  std::vector<ShapeFunctionNode> &nodes() { return tree_.mutableNodes(); }
  std::vector<std::vector<size_t>> &childIndices() {
    return tree_.mutableChildIndices();
  }
  size_t rootIndex() const { return tree_.rootIndex(); }

  const arma::fmat &X() const { return X_; }
  size_t numFeatures() const { return static_cast<size_t>(X_.n_rows); }
  const arma::Row<float> &sampleWeights() const { return w_; }

  LearningCriterion criterion() const { return tree_.criterion(); }
  const TreeBuildingParams &innerParams() const { return tree_.innerParams(); }

  /** Criterion for the per-node routing discretizer (always over partition labels). */
  virtual LearningCriterion routerCriterion() const = 0;

  /**
   * Build the care set and router-training data for one internal node.
   * Shared across classification and regression; delegates reward evaluation
   * to ``childRewards`` and soft-label vs non-worst policy to ``careSetKeepsOnlyBest``.
   */
  NodeCareSet buildCareSet(const std::vector<arma::uword> &samples,
                           const std::vector<size_t> &children) const;

  /** Refresh task-specific leaf state after routing changes. */
  virtual void recomputeLeafStats(
      const std::vector<std::vector<arma::uword>> &nodeSamples) = 0;

  /** Refresh exported per-bin metadata for a router accepted by TAO. */
  virtual void refreshNodeBinMetadata(
      ShapeFunctionNode &node,
      const std::vector<arma::uword> &samples) = 0;

protected:
  /** Per-child rewards for one sample at an internal node. */
  virtual void childRewards(const std::vector<size_t> &childLeaves,
                            arma::uword col,
                            std::vector<double> &reward) const = 0;

  /**
   * If true (single-output classification), keep only children tied with the
   * best reward and split sample weight across them. Otherwise keep every
   * child strictly better than the worst reward at full sample weight.
   */
  virtual bool careSetKeepsOnlyBest() const { return false; }

  /** Index of the largest entry in ``counts`` (ties: first max wins). */
  static size_t argMax(const std::vector<double> &counts);

  /** Follow routing from ``startNode`` until a leaf is reached for ``col``. */
  size_t walkToLeaf(size_t startNode, arma::uword col) const;

  /** Shared transform + count/weight refresh for ``refreshNodeBinMetadata``. */
  void initBinMetadata(ShapeFunctionNode &node,
                       const std::vector<arma::uword> &samples,
                       arma::Row<size_t> &binsOut,
                       const char *caller) const;

  ShapeGeneralizedTree &tree_;
  const arma::fmat &X_;
  const arma::Row<float> &w_;
};

} // namespace tao
