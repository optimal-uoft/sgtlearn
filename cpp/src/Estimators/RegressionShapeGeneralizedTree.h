#pragma once

/**
 * @file Estimators/RegressionShapeGeneralizedTree.h
 * @brief Multivariate shape-generalized regression tree (outer ``TreeBuilder`` +
 *        inner branching fits).
 */

#include <stdexcept>
#include <functional>
#include "Domain/LearningCriterion.h"
#include "Domain/FeatureInfo.h"
#include "Estimators/ShapeGeneralizedTree.h"
#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "algorithms/FeatureBagging.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"
#include "algorithms/TreeBuilder.h"

#include <armadillo>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

/**
 * Shape-Generalized Tree, regression variant.
 *
 * Responsibilities (by phase):
 * - **Outer growth** (`TreeBuilder`): best-first or depth-first expansion;
 *   per-node split search and child creation via lambdas in ``fit``; commit
 *   step remains a local lambda in ``fit``.
 * - **Per-node split search** (``fit`` lambdas): for each
 *   discretize -> round-robin bin-to-partition seed -> ``coordinateDescent`` on
 *   ``SquaredError`` only; ``AbsoluteError`` keeps the round-robin map (no CD).
 * - **Leaf state**: per-leaf mean (squared error) or median (absolute error)
 *   plus optional ``[sum y, sum y^2]`` stats for squared error.
 * - **Inference**: `predict` walks `childIndices_` using
 *   `routeFeatureValueToPartition`.
 *
 * At every outer-tree node, for each candidate feature:
 *   1. **Discretize**: train univariate regression discretizer on the node's
 *      samples (per-bin stats and training column indices).
 *   2. **Initial assignment**: round-robin by discretizer bin index
 *      (``b % numPartitions``); regression does not use k-means seeding.
 *   3. **Refinement**: ``SquaredError`` runs coordinate descent on the
 *      bin-to-partition map; if the post-CD objective clearly worsens vs the
 *      seed, restore the assignment snapshot and rebuild. ``AbsoluteError``
 *      keeps the round-robin seed (no CD).
 *
 * The best-scoring feature wins; its inner discretizer + bin->partition
 * mapping become the routing rule for that node, producing `numPartitions`
 * children. Inner-node fitting matches the Python `BranchingTree` pattern;
 * the outer loop uses `TreeBuilder` like Python's heap over
 * `best_impurity_decrease`.
 *
 * Inputs use Armadillo's column-major convention: X has shape
 * (numFeatures, numSamples); each column is one sample. Outer routing
 * considers every row index ``0 .. numFeatures-1`` (no separate candidate list).
 *
 * @note Only `LearningCriterion::SquaredError` and
 *       `LearningCriterion::AbsoluteError` are accepted; other criteria throw
 *       from the constructor.
 */
class RegressionShapeGeneralizedTree : public ShapeGeneralizedTree {
public:
  /**
   * @param criterion       impurity for inner splits, partition scoring, and
   *                        coordinate descent (SquaredError or AbsoluteError).
   * @param numPartitions   fan-out of every internal routing node (>= 2).
   * @param outerParams     tree-building params for the outer routing tree.
   * @param innerParams     tree-building params for the per-feature inner
   *                        univariate discretizer.
   * @param cdParams        coordinate-descent loop parameters.
   * @param random_state    seed for the tree-owned RNG (reseeds at each ``fit``).
   * @param featureBagging  per-node subset of candidate feature indices; empty
   *                        ``std::function`` defaults to all candidates.
   */
  RegressionShapeGeneralizedTree(
      LearningCriterion criterion, size_t numPartitions,
      TreeBuildingParams outerParams = {},
      TreeBuildingParams innerParams = {},
      CoordinateDescentParams cdParams = {}, uint64_t random_state = 42,
      FeatureBaggingPickFn featureBagging = {});

  ~RegressionShapeGeneralizedTree() = default;

  RegressionShapeGeneralizedTree(const RegressionShapeGeneralizedTree &) = delete;
  RegressionShapeGeneralizedTree &
  operator=(const RegressionShapeGeneralizedTree &) = delete;
  RegressionShapeGeneralizedTree(RegressionShapeGeneralizedTree &&) noexcept =
      default;
  RegressionShapeGeneralizedTree &
  operator=(RegressionShapeGeneralizedTree &&) noexcept = default;

  /**
   * Fit the routing tree.
   *
   * @param X  (numFeatures, numSamples) column-major; one sample per column.
   *           Routing candidates are row indices ``0 .. numFeatures-1``.
   * @param y  (numOutputs, numSamples) real-valued targets; single-output is
   *           ``n_rows == 1``.
   *
   * @param features  logical feature groups resolved in Python.
   *
   * @throws std::invalid_argument on shape mismatch.
   */
  void fit(const arma::fmat &X, const arma::Mat<float> &y,
           const arma::Row<float> &sampleWeights,
           const std::vector<FeatureInfo> &features);

  /** Predicted responses, shape (numOutputs, numSamples). */
  arma::Mat<double> predict(const arma::fmat &X) const;

  /** Number of outputs the tree was fitted on (>= 1). */
  size_t nOutputs() const { return nOutputs_; }

  /**
   * Per outer-tree node index: for squared error, concatenated
   * ``[Σw·y0, Σw·y0², Σw·y1, Σw·y1², ...]`` (length ``2 * nOutputs``) at leaves
   * (empty at internal nodes after fit). For absolute error, empty at every
   * leaf (no extra MAE statistic storage).
   */
  std::vector<std::vector<float>> leafRegressionStats;
  /** Sample count per leaf (same indexing as ``leafRegressionStats``). */
  std::vector<size_t> leafNumSamples;

  /**
   * Per-leaf prediction, keyed by node id; inner vector has length
   * ``nOutputs`` (mean per output for squared error, median for absolute).
   */
  const std::vector<std::vector<double>> &leafPredictions() const {
    return leafPredictions_;
  }

  /** Mutable leaf predictions for in-place refinement (e.g. TAO). */
  std::vector<std::vector<double>> &mutableLeafPredictions() {
    return leafPredictions_;
  }

private:
  CoordinateDescentParams cdParams_;
  uint64_t random_state_;
  std::mt19937_64 rng_;
  FeatureBaggingPickFn featureBagging_;
  std::vector<FeatureInfo> features_;

  /** Outer routing expansion; `fit` passes split logic via buildTree callbacks. */
  TreeBuilder<ShapeFunctionNode> outerTreeBuilder_;

  /** Number of outputs (``y.n_rows``), set at ``fit``. */
  size_t nOutputs_ = 1;

  /**
   * Leaf constant prediction keyed by ``nodeIndex``; inner vector length
   * ``nOutputs_`` (mean per output for squared error, median for absolute).
   */
  std::vector<std::vector<double>> leafPredictions_;

  double impurityAtNode(const arma::Mat<float> &y,
                        const ShapeFunctionNode &node) const;

  /** Nested ``[output][Σw·y, Σw·y²]`` per node sample set. */
  std::vector<std::vector<double>> aggregateYSquaredStats(
      const ShapeFunctionNode &node, const arma::Mat<float> &y) const;

  arma::Row<float> fitSampleWeights_;
};
