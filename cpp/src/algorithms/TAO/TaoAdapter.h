#pragma once

/**
 * @file algorithms/TAO/TaoAdapter.h
 * @brief Abstract adapter between TAO and fitted shape-generalized trees.
 *
 * TAO (Tree-Alternating Optimization) refines routing rules on an already-fitted
 * outer tree without changing its topology. The shared driver in
 * ``TreeAlternatingOptimization`` operates only through this interface; task-specific
 * care-set construction, pseudolabeling, and leaf refresh live in concrete
 * adapters (see ``ClassificationTaoAdapter`` and ``RegressionTaoAdapter``).
 *
 * Training data (``X``, targets, sample weights) are supplied again at refinement
 * time because sample partitions are not retained after ``fit``.
 */

#include "Domain/LearningCriterion.h"
#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"

#include <armadillo>
#include <cstddef>
#include <vector>

namespace tao {

/**
 * Per-node care set: samples whose routing choice matters for scoring and for
 * training the candidate routing discretizer.
 *
 * For each care sample ``i``, ``careRewards[i][c]`` is the reward of sending
 * that sample down child partition ``c``. ``Xexp`` / ``yexp`` / ``wexp`` are the
 * expanded router-training matrices (one or more rows per care sample, depending
 * on task). ``dummyChild`` is the fallback partition when a feature value is NaN
 * or when the best rule routes every care sample to a single child.
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
 * Adapter interface for one TAO refinement pass.
 *
 * Concrete adapters wrap a fitted tree plus refinement-time training data and
 * implement the task-specific hooks:
 *
 * - ``buildCareSet`` — identify care samples at a node, compute per-child
 *   rewards, and build router-training matrices for the discretizer search.
 * - ``recomputeLeafStats`` — refresh leaf predictions / histograms after a
 *   routing change.
 *
 * All other methods expose tree structure, training design matrix, and
 * hyperparameters needed by the shared per-node optimizer.
 */
class TaoAdapter {
public:
  virtual ~TaoAdapter() = default;

  /** Mutable fitted node array; routing fields are updated in place by TAO. */
  virtual std::vector<ShapeFunctionNode> &nodes() = 0;

  /** Mutable per-node child index lists (empty at leaves). */
  virtual std::vector<std::vector<size_t>> &childIndices() = 0;

  /** Root node index (currently always 0 after fit). */
  virtual size_t rootIndex() const = 0;

  /**
   * Training design matrix, shape (numFeatures, numSamples); column-major.
   * Same convention as ``fit``.
   */
  virtual const arma::fmat &X() const = 0;

  /** Number of routing features (rows of ``X``). */
  virtual size_t numFeatures() const = 0;

  /** Per-sample weights from the refinement call, shape (numSamples,). */
  virtual const arma::Row<float> &sampleWeights() const = 0;

  /** Loss / impurity criterion for this tree (classification or regression). */
  virtual LearningCriterion criterion() const = 0;

  /**
   * Criterion for the per-node routing discretizer.
   *
   * The discretizer always treats child partitions as class labels; regression
   * typically uses Gini here even when the tree criterion is MAE.
   */
  virtual LearningCriterion routerCriterion() const = 0;

  /** Inner discretizer hyperparameters (``minLeafSize``, ``maxDepth``, etc.). */
  virtual const TreeBuildingParams &innerParams() const = 0;

  /**
   * Build the care set and router-training data for one internal node.
   *
   * @param samples  Column indices of training samples reaching this node.
   * @param children Node indices of this node's child partitions.
   * @returns Care samples, reward matrix, expanded training matrices, and
   *          ``dummyChild``. Returns an empty care set when no sample's routing
   *          choice affects reward (all children tie).
   */
  virtual NodeCareSet
  buildCareSet(const std::vector<arma::uword> &samples,
               const std::vector<size_t> &children) const = 0;

  /**
   * Refresh task-specific leaf state after routing changes.
   *
   * @param nodeSamples  For each node index, column indices of samples routed
   *                     there under the current routing rules.
   */
  virtual void recomputeLeafStats(
      const std::vector<std::vector<arma::uword>> &nodeSamples) = 0;

  /** Refresh exported per-bin metadata for a router accepted by TAO. */
  virtual void refreshNodeBinMetadata(
      ShapeFunctionNode &node,
      const std::vector<arma::uword> &samples) = 0;

  /** Rebuild feature importances after one or more routing changes. */
  virtual void refreshFeatureImportances() = 0;
};

} // namespace tao
