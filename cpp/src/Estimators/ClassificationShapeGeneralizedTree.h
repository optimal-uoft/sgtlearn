#pragma once

/**
 * @file Estimators/ClassificationShapeGeneralizedTree.h
 * @brief Multivariate shape-generalized classification tree (outer ``TreeBuilder`` + inner branching fits).
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
 * Shape-Generalized Tree, classification variant.
 *
 * Responsibilities (by phase):
 * - **Outer growth** (`TreeBuilder`): best-first or depth-first expansion;
 *   per-node split search and child creation via lambdas in ``fit``; commit
 *   step remains a local lambda in ``fit``.
 * - **Per-node split search** (``fit`` lambdas): for each
 *   discretize -> k-means-style bin init -> `coordinateDescent` on bin-to-
 *   partition map; if the post-CD objective is **clearly worse** than the seed
 *   (absolute margin ``kShapeFunctionCdImprovementEps``),
 *   restore the
 *   assignment snapshot and rebuild the ``BranchAssignment``; keep the best branch
 *   by penalized child impurity.
 * - **Leaf state**: `fillLeafHistogram` or aggregated discretizer stats after a
 *   committed split.
 * - **Inference**: `predict` / `predictProba` walk childIndices_ using
 *   `routeFeatureValueToPartition`.
 *
 * At every outer-tree node, for each candidate feature:
 *   1. **Discretize**: train `UnivariateClassificationDiscretizer` on the
 *      node's samples (per-bin class counts and training column indices).
 *   2. **Partition search** : for each
 *      ``k`` in ``[2, min(numBins, numPartitions)]``, seed assignments
 *      (identity when ``k == numBins``, else k-means or round-robin), run
 *      coordinate descent when ``k < numBins``, score
 *      ``impurity + branchingPenalty * (k - 1)``, keep the best ``k``.
 *   3. **Coordinate descent**: if the objective clearly worsens vs the seed,
 *      restore the snapshot. Enforce ``minLeafSize`` per child partition.
 *
 * The best-scoring feature wins; its inner discretizer + bin->partition
 * mapping become the routing rule for that node, with ``k`` children (``k``
 * may be less than ``numPartitions``). The outer loop uses `TreeBuilder`
 * like Python's heap over impurity decrease.
 *
 * Inputs use Armadillo's column-major convention: X has shape
 * (numFeatures, numSamples); each column is one sample. Outer routing
 * considers every row index ``0 .. numFeatures-1`` (no separate candidate list).
 *
 * @note Only `LearningCriterion::Entropy` and `LearningCriterion::Gini` are
 *       accepted; other criteria throw from the constructor.
 */
class ClassificationShapeGeneralizedTree : public ShapeGeneralizedTree {
public:
  /**
   * @param criterion       impurity for inner splits, partition scoring, and
   *                        coordinate descent (Entropy or Gini).
   * @param numClasses      per-output class counts. A single entry is expanded
   *                        to match ``y.n_rows`` at ``fit``; length may also
   *                        equal the number of outputs. Scalar single-output
   *                        is just ``{K}``.
   * @param numPartitions   fan-out of every internal routing node (>= 2).
   * @param outerParams     tree-building params for the outer routing tree.
   * @param innerParams     tree-building params for the per-feature inner
   *                        univariate discretizer.
   * @param cdParams         coordinate-descent loop parameters.
   * @param random_state     seed for the tree-owned RNG (reseeds at each ``fit``).
   * @param featureBagging   per-node subset of candidate feature indices; empty
   *                         ``std::function`` defaults to all candidates.
   */
  ClassificationShapeGeneralizedTree(
      LearningCriterion criterion, std::vector<size_t> numClasses,
      size_t numPartitions, TreeBuildingParams outerParams = {},
      TreeBuildingParams innerParams = {},
      CoordinateDescentParams cdParams = {}, uint64_t random_state = 42,
      FeatureBaggingPickFn featureBagging = {});

  /** Convenience overload: single-output / shared class count ``{numClasses}``. */
  ClassificationShapeGeneralizedTree(
      LearningCriterion criterion, size_t numClasses, size_t numPartitions,
      TreeBuildingParams outerParams = {},
      TreeBuildingParams innerParams = {},
      CoordinateDescentParams cdParams = {}, uint64_t random_state = 42,
      FeatureBaggingPickFn featureBagging = {})
      : ClassificationShapeGeneralizedTree(
            criterion, std::vector<size_t>{numClasses}, numPartitions,
            outerParams, innerParams, cdParams, random_state,
            std::move(featureBagging)) {}

  ~ClassificationShapeGeneralizedTree() = default;

  ClassificationShapeGeneralizedTree(
      const ClassificationShapeGeneralizedTree &) = delete;
  ClassificationShapeGeneralizedTree &
  operator=(const ClassificationShapeGeneralizedTree &) = delete;
  ClassificationShapeGeneralizedTree(ClassificationShapeGeneralizedTree &&) noexcept =
      default;
  ClassificationShapeGeneralizedTree &
  operator=(ClassificationShapeGeneralizedTree &&) noexcept = default;

  /**
   * Fit the routing tree.
   *
   * @param X  (numFeatures, numSamples) column-major; one sample per column.
   *           Routing candidates are row indices ``0 .. numFeatures-1``.
   * @param y  (numOutputs, numSamples) integer class labels; output ``o`` has
   *           labels in ``[0, classesPerOutput[o])``. Single-output is
   *           ``n_rows == 1``.
   *
   * @param features  logical feature groups resolved in Python.
   *
   * @throws std::invalid_argument on shape / label-range mismatch.
   */
  void fit(const arma::fmat &X, const arma::Mat<size_t> &y,
           const arma::Row<float> &sampleWeights,
           const std::vector<FeatureInfo> &features);

  /** Hard class predictions, shape (numOutputs, numSamples). */
  arma::Mat<size_t> predict(const arma::fmat &X) const;

  /**
   * Class probabilities: one ``(classesPerOutput[o], numSamples)`` matrix per
   * output. Single-output returns a length-1 vector.
   */
  std::vector<arma::fmat> predictProba(const arma::fmat &X) const;

  /**
   * Per-node class histograms (also populated at internal nodes).
   * Shape: ``classCounts[node][output][class]``.
   */
  std::vector<std::vector<std::vector<double>>> classCounts;

  /** Per-output class counts (configured, or fit-resolved when available). */
  const std::vector<size_t> &numClasses() const {
    return classesPerOutput_.empty() ? numClasses_ : classesPerOutput_;
  }

  /** Number of outputs the tree was fitted on (>= 1). */
  size_t nOutputs() const { return nOutputs_; }

  /** Fit-resolved per-output class counts (empty before ``fit``). */
  const std::vector<size_t> &classesPerOutput() const {
    return classesPerOutput_;
  }

private:
  /** Configured class counts from the ctor (length 1 or nOutputs). */
  std::vector<size_t> numClasses_;
  /** Resolved per-output class counts, set at ``fit`` (length ``nOutputs_``). */
  std::vector<size_t> classesPerOutput_;
  /** Number of outputs, set at ``fit``. */
  size_t nOutputs_ = 1;
  CoordinateDescentParams cdParams_;
  uint64_t random_state_;
  std::mt19937_64 rng_;
  FeatureBaggingPickFn featureBagging_;
  std::vector<FeatureInfo> features_;

  /** Outer routing expansion; `fit` passes split logic via buildTree callbacks. */
  TreeBuilder<ShapeFunctionNode> outerTreeBuilder_;

  /** Resolve ``classesPerOutput_`` from config and ``y``. */
  void resolveOutputLayout(size_t nOutputs);

  /** Summed Gini/entropy over per-output histograms. */
  double impurityForClassCounts(
      const std::vector<std::vector<double>> &classCounts) const;

  /** Empty nested histogram with the correct per-output lengths. */
  std::vector<std::vector<double>> makeEmptyHistogram() const;

  std::vector<std::vector<double>>
  fillLeafHistogram(ShapeFunctionNode &node,
                    const arma::Mat<size_t> &y) const;

  arma::Row<float> fitSampleWeights_;
};
