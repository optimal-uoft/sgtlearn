#pragma once

#include "Domain/LearningCriterion.h"
#include "algorithms/TreeBuilder.h"

#include <algorithm>
#include <armadillo>
#include <compare>
#include <cstddef>
#include <vector>

/**
 * Standard tree-building hyperparameters. Shared by the outer routing tree
 * and the inner per-feature univariate discretizer.
 *
 * - minLeafSize:  minimum samples in a node for it to remain split-eligible.
 * - minGainSplit: minimum impurity reduction required to commit a split.
 * - maxDepth:     0 = unlimited; otherwise expansion stops at this depth.
 * - maxLeafNodes: 0 = depth-first / unlimited; otherwise best-first growth up
 *                 to this many leaves (Heap frontier in TreeBuilder).
 */
struct TreeBuildingParams {
  size_t minLeafSize = 1;
  double minGainSplit = 1e-7;
  size_t maxDepth = 0;
  size_t maxLeafNodes = 0;
  /** Added to effective child impurity when ranking splits (Python branching_penalty * (k-1)). */
  double branchingPenalty = 0.0;
};

/**
 * Coordinate-descent hyperparameters for the bin->partition assignment loop
 * (see algorithms/CoordinateDescent.h).
 */
struct CoordinateDescentParams {
  size_t maxIters = 10;
  size_t patience = 5;
  size_t seed = 42;
  /**
   * If true (default), initialize bin->partition assignments with weighted
   * k-means (k = numPartitions) on per-bin normalized class counts before
   * coordinate descent; otherwise use round-robin.
   */
  bool smartInit = true;
};

/**
 * Persistent state for one outer-tree node after a single build pass (and the
 * object TreeBuilder / makeChildren mutate while growing the tree).
 *
 * Training / child construction:
 *   - The outer node can route on different features at different depths, so
 *     sample membership is not modeled here as a single univariate slice.
 *   - After a committed split, childSampleBounds is an exclusive prefix (length
 *     numPartitions + 1) over whatever index buffer the tree builder uses to
 *     group training rows into children—partition p owns
 *     [childSampleBounds[p], childSampleBounds[p + 1]).
 *
 * Out-of-sample routing:
 *   - routingFeature selects the row of X; innerThresholds are sorted
 *     ascending (same convention as UnivariateDiscretizer::transform).
 *   - Bin id is lower_bound(innerThresholds, x); bin count is
 *     binToPartition.size() (typically innerThresholds.size() + 1 for axis
 *     splits).
 *   - binToPartition maps that bin to a child partition in [0, numPartitions).
 *
 * Routing and histogram sizes come from the vectors (innerThresholds,
 * binToPartition, childSampleBounds, leafClassCounts). Class cardinality is a
 * model-level property, not duplicated on each node.
 *
 * @see TreeBuilder — requires score, informationGain, height, and a weak
 *      ordering consistent with SplitCandidate (higher informationGain first
 *      in the best-first heap).
 */
struct ClassificationShapeGeneralizedNode {
  size_t height = 0;
  double score = 0.0;
  double informationGain = 0.0;

  /**
   * During `fit`, half-open range into `sampleOrder_` listing column indices of
   * training samples at this node. Cleared after training.
   */
  size_t sampleBegin = 0;
  size_t sampleEnd = 0;
  /** Node id in `nodes_` while building; cleared after training. */
  size_t nodeIndex = 0;

  bool isLeaf = true;
  /** Fan-out when split; childSampleBounds.size() should be numPartitions + 1. */
  size_t numPartitions = 0;

  /** Row index into X for routing; undefined if isLeaf. */
  size_t routingFeature = 0;
  /** Sorted ascending; same convention as UnivariateDiscretizer::transform. */
  std::vector<float> innerThresholds;
  /** Length equals inner discretizer bin count; maps bin -> child partition. */
  std::vector<size_t> binToPartition;

  /**
   * Exclusive prefix into the builder's sample-index buffer after partitioning
   * into numPartitions children. Valid when !isLeaf.
   */
  std::vector<size_t> childSampleBounds;

  /** Leaf-only: counts per class label (encoding matches training y). */
  std::vector<size_t> leafClassCounts;

  std::weak_ordering
  operator<=>(const ClassificationShapeGeneralizedNode &o) const {
    return std::compare_weak_order_fallback(informationGain,
                                            o.informationGain);
  }

  bool operator==(const ClassificationShapeGeneralizedNode &o) const = default;

  /**
   * Map a scalar feature value to a child partition index, using the same
   * binning rule as UnivariateDiscretizer::transform.
   *
   * @pre Typically !isLeaf with a populated binToPartition; if empty, returns 0.
   */
  size_t routeFeatureValueToPartition(float featureValue) const {
    if (binToPartition.empty())
      return 0;
    const auto it = std::lower_bound(innerThresholds.begin(),
                                     innerThresholds.end(), featureValue);
    size_t bin = static_cast<size_t>(it - innerThresholds.begin());
    if (bin >= binToPartition.size())
      bin = binToPartition.size() - 1;
    return binToPartition[bin];
  }
};

/**
 * Shape-Generalized Tree, classification variant.
 *
 * At every outer-tree node, for each candidate feature:
 *   1. Train a UnivariateClassificationDiscretizer (inner tree) on the
 *      node's samples to obtain a bin index per sample and per-bin class
 *      counts.
 *   2. Build a classification BranchAssignment from those bin stats and run
 *      coordinateDescent to find a bin -> partition mapping that minimises
 *      the criterion impurity over `numPartitions` partitions.
 *   3. Score the resulting partition impurity.
 *
 * The best-scoring feature wins; its inner discretizer + bin->partition
 * mapping become the routing rule for that node, producing `numPartitions`
 * children. Inner-node fitting matches the Python `BranchingTree` pattern
 * (discretizer + coordinate descent over bins); the outer loop uses
 * `TreeBuilder` like Python's heap over `best_impurity_decrease`.
 *
 * Inputs use Armadillo's column-major convention: X has shape
 * (numFeatures, numSamples); each column is one sample.
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
   * @param cdParams        coordinate-descent loop parameters.
   */
  ClassificationShapeGeneralizedTree(
      LearningCriterion criterion, size_t numClasses, size_t numPartitions,
      TreeBuildingParams outerParams = {},
      TreeBuildingParams innerParams = {},
      CoordinateDescentParams cdParams = {});

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
   * @param X         (numFeatures, numSamples) column-major; one sample per
   *                  column.
   * @param features  feature indices considered as routing candidates at
   *                  every node. Non-const because internal sort scratch
   *                  may reuse it.
   * @param y         (numSamples,) integer class labels in [0, numClasses).
   *
   * @throws std::invalid_argument on shape / label-range mismatch.
   */
  void fit(const arma::fmat &X, arma::uvec &features,
           const arma::Row<size_t> &y);

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

private:
  LearningCriterion criterion_;
  size_t numClasses_;
  size_t numPartitions_;
  TreeBuildingParams outerParams_;
  TreeBuildingParams innerParams_;
  CoordinateDescentParams cdParams_;
  bool fitted_ = false;

  std::vector<ClassificationShapeGeneralizedNode> nodes_;
  /**
   * For node i, childIndices_[i][p] is the node index of child partition p.
   * Empty for leaves.
   */
  std::vector<std::vector<size_t>> childIndices_;
  size_t rootIndex_ = 0;

  /** Outer routing expansion; `fit` passes split logic via buildTree callbacks. */
  TreeBuilder<ClassificationShapeGeneralizedNode> outerTreeBuilder_;

  /** Column permutation of training data; used only inside `fit`. */
  arma::uvec sampleOrder_;

  double impurityAtRange(size_t begin, size_t end,
                         const arma::Row<size_t> &y) const;

  void fillLeafHistogram(ClassificationShapeGeneralizedNode &node,
                         const arma::Row<size_t> &y) const;

  bool findBestSplitNode(ClassificationShapeGeneralizedNode &node,
                         size_t minLeafSize, const arma::fmat &X,
                         const arma::Row<size_t> &y, const arma::uvec &features);

  std::vector<ClassificationShapeGeneralizedNode>
  makeChildrenNode(ClassificationShapeGeneralizedNode &parent,
                     const arma::fmat &X, const arma::Row<size_t> &y);

  void commitSplitNode(ClassificationShapeGeneralizedNode &parent,
                       std::vector<ClassificationShapeGeneralizedNode> &children);
};
