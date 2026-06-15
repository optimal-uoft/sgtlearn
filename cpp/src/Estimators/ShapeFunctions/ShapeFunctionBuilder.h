#pragma once

/**
 * @file Estimators/ShapeFunctions/ShapeFunctionBuilder.h
 * @brief Interface for per-node shape-function split search in shape-generalized trees.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "Discretizers/ShapeDiscretizer.h"
#include "algorithms/ShapeBranchingTypes.h"

#include <armadillo>
#include <cstddef>
#include <limits>
#include <vector>

/**
 * Searches for the best inner discretizer + bin-to-partition mapping at one
 * outer-tree node. Used as the ``findBestSplit`` callback for ``TreeBuilder``.
 */
class ShapeFunctionBuilder {
public:
  virtual ~ShapeFunctionBuilder() = default;

  /**
   * Minimum objective improvement required to keep a coordinate-descent
   * branch assignment; smaller changes revert to the seed map.
   */
  static constexpr double kCdObjectiveImprovementEps = 1e-10;

  /**
   * Search for and apply the best shape-function split at ``node``.
   *
   * Mutates ``node`` in place when a split is found; marks it a leaf otherwise.
   *
   * @return true if ``node`` is a splittable internal node after this call.
   */
  virtual bool findBestSplit(ShapeFunctionNode &node, size_t minLeafSize) = 0;

  /**
   * Route samples through ``parent``'s split and build child nodes with
   * objective-specific leaf statistics and scores.
   */
  virtual std::vector<ShapeFunctionNode>
  makeChildren(const ShapeFunctionNode &parent) = 0;

  /**
   * Route each sample at ``parent`` to a child partition bucket using the
   * parent's chosen split (finite values via ``sampleBins`` /
   * ``binToPartition``; non-finite values via ``nanPredictionPartition``).
   */
  static std::vector<std::vector<size_t>>
  routeSamplesToPartitions(const ShapeFunctionNode &parent, const arma::fmat &X);

  /**
   * Build one child node per partition bucket with ``height``, ``sampleIndices``,
   * and ``numPartitions`` set; objective-specific fields are left to the caller.
   */
  static std::vector<ShapeFunctionNode>
  makeRoutedChildNodes(const ShapeFunctionNode &parent,
                       const std::vector<std::vector<size_t>> &buckets,
                       size_t treeNumPartitions);

protected:
  /** Best branch assignment found for one candidate feature/discretizer. */
  struct BranchAssignmentSearchResult {
    double bestFeatureScore = std::numeric_limits<double>::infinity();
    size_t chosenK = 0;
    std::vector<size_t> assignments;
    std::vector<size_t> partitionSampleCounts;
    /** Classification only: weighted class counts per partition after CD. */
    std::vector<std::vector<double>> partitionClassCounts;
    /** Classification only: sum of bin weights per partition after CD. */
    std::vector<double> partitionWeights;
    double impurityDecrease = 0.0;
    bool found = false;
  };

  /** Best split accumulated across feature candidates at one node. */
  struct BestBranchingState {
    double penalizedChildScore = std::numeric_limits<double>::infinity();
    ShapeBranchingResult<double> branching;
    std::vector<double> binWeights;
    /** Classification only: partition aggregates used for NaN routing. */
    std::vector<std::vector<double>> partitionClassCounts;
    std::vector<double> partitionWeights;
  };

  /**
   * Mark ``node`` as a terminal leaf with no routing state from a failed split
   * search (clears ``sampleBins``, split-side stats, and ``informationGain``).
   */
  static void markLeafNoSplit(ShapeFunctionNode &node);

  /**
   * Fill ``sampleBins[col]`` with the inner discretizer bin index for each
   * column of the node's subsampled ``X`` matrix.
   *
   * ``perBinCols[b]`` lists column indices assigned to bin ``b`` by the
   * discretizer (same layout as ``UnivariateDiscretizer::inSampleDiscretizations``).
   */
  static void fillSampleBinsFromDiscretizer(
      size_t xSubCols, const std::vector<std::vector<size_t>> &perBinCols,
      std::vector<size_t> &sampleBins);

  /**
   * Align sample weights with a node's subsampled training columns.
   *
   * During outer-tree growth each node holds ``sampleIndices``: original column
   * ids into the full ``X`` / ``y`` passed to ``fit``. Split search then uses
   * ``Xsub = X.cols(sampleIndices)`` and ``ysub = y.cols(sampleIndices)``, so
   * column ``j`` of the subsample is global sample ``sampleIndices(j)``.
   *
   * ``weights`` is the full-length weight row (one entry per column of ``X``).
   * This returns ``wsub`` with ``wsub.n_elem == subIdx.n_elem`` and
   * ``wsub(j) = weights(subIdx(j))``, i.e. weights in the same order as
   * ``Xsub`` / ``ysub`` for discretizer training and NaN partition scoring.
   *
   * @param weights  (numSamples,) weights for the entire dataset.
   * @param subIdx   ``node.sampleIndices`` (or any subset of column ids).
   */
  static arma::Row<float> subSampleWeights(const arma::Row<float> &weights,
                                           const arma::uvec &subIdx);

  /** Copy routing metadata shared by classification and regression builders. */
  static void applySharedBranchingFields(
      BestBranchingState &best, const BranchAssignmentSearchResult &search,
      size_t featureIndex, size_t xSubCols,
      const std::vector<double> &thresholds,
      const std::vector<std::vector<size_t>> &perBinCols);

  /**
   * If ``search`` improves ``best``, copy discretizer routing state into
   * ``best`` and apply task-specific leaf statistics.
   */
  bool featureHasBetterBranching(const BranchAssignmentSearchResult &search,
                                 BestBranchingState &best, size_t featureIndex,
                                 size_t xSubCols, ShapeDiscretizer &disc,
                                 double scoreEpsilon);

  /** Fill objective-specific fields after shared branching metadata is copied. */
  virtual void applyTaskBranchingFields(
      BestBranchingState &best, const BranchAssignmentSearchResult &search,
      const std::vector<std::vector<double>> &leafStats) = 0;
};
