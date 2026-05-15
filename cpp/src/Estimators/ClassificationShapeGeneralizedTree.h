#pragma once

/**
 * @file Estimators/ClassificationShapeGeneralizedTree.h
 * @brief Multivariate shape-generalized classification tree (outer ``TreeBuilder`` + inner branching fits).
 */

#include "Domain/LearningCriterion.h"
#include "Estimators/ShapeFunctionNode.h"
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
 *   split / child / commit steps are local lambdas in `fit`.
 * - **Per-node split search** (inside `fit`): for each candidate feature,
 *   discretize -> k-means-style bin init -> `coordinateDescent` on bin-to-
 *   partition map; if the post-CD objective is **clearly worse** than the seed
 *   (absolute margin ``kCdObjectiveImprovementEps`` in the ``.cpp``), restore the
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
 *   2. **K-means init** (when enabled): cluster bin histograms to seed a
 *      bin-to-partition assignment; else round-robin by bin index.
 *   3. **Coordinate descent**: refine that assignment to reduce impurity; if
 *      the objective **clearly worsens** vs the pre-CD value, restore the snapshot
 *      and rebuild the objective. If feasible sizes and gain pass thresholds,
 *      compete to update the node's best branch.
 *
 * The best-scoring feature wins; its inner discretizer + bin->partition
 * mapping become the routing rule for that node, producing `numPartitions`
 * children. Inner-node fitting matches the Python `BranchingTree` pattern
 * (discretizer + coordinate descent over bins); the outer loop uses
 * `TreeBuilder` like Python's heap over `best_impurity_decrease`.
 *
 * Inputs use Armadillo's column-major convention: X has shape
 * (numFeatures, numSamples); each column is one sample. Outer routing
 * considers every row index ``0 .. numFeatures-1`` (no separate candidate list).
 *
 * @note Only `LearningCriterion::Entropy` and `LearningCriterion::Gini` are
 *       accepted; other criteria throw from the constructor.
 */
class ClassificationShapeGeneralizedTree {
public:
  /**
   * @param criterion       impurity for inner splits, partition scoring, and
   *                        coordinate descent (Entropy or Gini).
   * @param numClasses      number of distinct class labels in [0, numClasses).
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
      LearningCriterion criterion, size_t numClasses, size_t numPartitions,
      TreeBuildingParams outerParams = {},
      TreeBuildingParams innerParams = {},
      CoordinateDescentParams cdParams = {}, uint64_t random_state = 42,
      FeatureBaggingPickFn featureBagging = {});

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
   * @param y  (numSamples,) integer class labels in [0, numClasses).
   *
   * @throws std::invalid_argument on shape / label-range mismatch.
   */
  void fit(const arma::fmat &X, const arma::Row<size_t> &y);

  /** Hard class predictions, shape (numSamples,). */
  arma::Row<size_t> predict(const arma::fmat &X) const;

  /** Class probabilities, shape (numClasses, numSamples). */
  arma::fmat predictProba(const arma::fmat &X) const;

  /** Number of leaf nodes in the fitted outer tree. */
  size_t numLeaves() const;

  /** Total number of nodes (internal + leaf) in the fitted outer tree. */
  size_t numNodes() const;

  /** True if `fit` has completed successfully. */
  bool isFitted() const;

  // TODO: flat introspection accessors for graphviz / text export, e.g.
  //   featureOf(node), thresholdsOf(node), binToPartitionOf(node),
  //   childrenOf(node), classCountsOf(leaf), depthOf(node).
    std::vector<std::vector<size_t>> classCounts;
private:
  LearningCriterion criterion_;
  size_t numClasses_;
  size_t numPartitions_;
  TreeBuildingParams outerParams_;
  TreeBuildingParams innerParams_;
  CoordinateDescentParams cdParams_;
  uint64_t random_state_;
  std::mt19937_64 rng_;
  FeatureBaggingPickFn featureBagging_;
  bool fitted_ = false;

  std::vector<ShapeFunctionNode> nodes_;
  /**
   * For node i, childIndices_[i][p] is the node index of child partition p.
   * Empty for leaves.
   */
  std::vector<std::vector<size_t>> childIndices_;
  size_t rootIndex_ = 0;

  /** Outer routing expansion; `fit` passes split logic via buildTree callbacks. */
  TreeBuilder<ShapeFunctionNode> outerTreeBuilder_;

  /** Gini or entropy from an aggregated class histogram (``N`` = sum of counts). */
  double impurityForClassCounts(const std::vector<size_t> &classCounts) const;

  std::vector<size_t> fillLeafHistogram(ShapeFunctionNode &node,
                         const arma::Row<size_t> &y) const;
};
