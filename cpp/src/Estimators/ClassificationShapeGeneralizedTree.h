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
 *   split search via ``ClassificationShapeFunctionBuilder``; child / commit
 *   steps remain local lambdas in ``fit``.
 * - **Per-node split search** (``ClassificationShapeFunctionBuilder``): for each
 *   discretize -> k-means-style bin init -> `coordinateDescent` on bin-to-
 *   partition map; if the post-CD objective is **clearly worse** than the seed
 *   (absolute margin ``ShapeFunctionBuilder::kCdObjectiveImprovementEps``),
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
class ClassificationShapeFunctionBuilder;

class ClassificationShapeGeneralizedTree {
  friend class ClassificationShapeFunctionBuilder;

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
  void fit(const arma::fmat &X, const arma::Row<size_t> &y,
           const arma::Row<float> &sampleWeights);

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

  /** Per-node class histograms (also populated at internal nodes). */
  std::vector<std::vector<double>> classCounts;

  /** Read-only access to the fitted node array for introspection / export. */
  const std::vector<ShapeFunctionNode> &nodes() const { return nodes_; }

  /** Per-node child indices; empty inner vector at leaves. */
  const std::vector<std::vector<size_t>> &childIndices() const { return childIndices_; }

  /** Index of the root node (currently always 0 after fit). */
  size_t rootIndex() const { return rootIndex_; }

  /** Fan-out used by this tree (constructor arg). */
  size_t numPartitions() const { return numPartitions_; }

  /** Number of class labels expected by this estimator. */
  size_t numClasses() const { return numClasses_; }

  /** Impurity criterion this tree was constructed with. */
  LearningCriterion criterion() const { return criterion_; }
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
  double impurityForClassCounts(const std::vector<double> &classCounts) const;

  std::vector<double> fillLeafHistogram(ShapeFunctionNode &node,
                                        const arma::Row<size_t> &y) const;

  arma::Row<float> fitSampleWeights_;
};
