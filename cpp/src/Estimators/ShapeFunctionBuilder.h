#pragma once

/**
 * @file Estimators/ShapeFunctionBuilder.h
 * @brief Interface for per-node shape-function split search in shape-generalized trees.
 */

#include "Estimators/ShapeFunctionNode.h"

#include <armadillo>
#include <cstddef>
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

protected:
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
};
