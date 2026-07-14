#pragma once

/**
 * @file Estimators/ShapeFunctions/ShapeFunctionNode.h
 * @brief One node in a shape-generalized / shape-function tree: routing, fit
 *        sample set, and per-bin stats for the winning inner split (classification
 *        or regression).
 */

#include "Discretizers/InnerDiscretizerBase.h"

#include <armadillo>
#include <compare>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <vector>

class ShapeGeneralizedTree;

/**
 * Outer-tree node: routing rule when internal, plus training sample indices
 * during fit.
 *
 * After `fit`, `sampleIndices` is cleared on stored nodes; routing uses
 * ``innerDiscretizer`` + ``binToPartition`` (including a trailing NaN bin for
 * univariate numeric discretizers).
 *
 * `TreeBuilder` orders nodes by `informationGain` for the best-first heap.
 */
class ShapeFunctionNode {
public:
  ShapeGeneralizedTree* tree = nullptr;

  size_t height = 0;
  double score = 0.0;
  double informationGain = 0.0;

  /** Node id in `nodes_` while building; cleared after training. */
  size_t nodeIndex = 0;

  bool isLeaf = true;
  size_t numPartitions = 0;

  /**
   * Index into the fit-time logical features sequence for the winning split.
   * Meaningful on internal nodes after ``findBestSplit`` succeeds.
   */
  size_t splitFeatureIndex = 0;

  /** Row indices into X used for routing; undefined if isLeaf. */
  std::vector<size_t> routingFeatures;
  /** Maps each inner discretizer bin (including NaN) to a child partition. */
  std::vector<size_t> binToPartition;
  /** Winning inner discretizer for this split; required on internal nodes. */
  std::shared_ptr<const InnerDiscretizerBase<double>> innerDiscretizer;

  /**
   * During fit: original column indices into X at this node (Python
   * `point_idxs`), as a column index vector for `X.cols(...)` / `y.cols(...)`.
   * Cleared after training.
   */
  arma::uvec sampleIndices;
  /**
   * During fit, after a split is chosen: inner discretizer bin per entry of
   * `sampleIndices` (same length). Cleared after training.
   */
  std::vector<size_t> sampleBins;
  /**
   * Per-bin sufficient statistics from the winning inner discretizer (same
   * length as ``binToPartition``). Classification: weighted class counts.
   * Regression (squared error): ``[sum w·y, sum w·y²]``. Empty for MAE leaves.
   */
  std::vector<std::vector<double>> splitLeafStats;
  /**
   * Per-bin sum of sample weights (``sum w``) from the inner discretizer.
   * Same length as ``binToPartition``.
   */
  std::vector<double> splitBinWeights;
  /**
   * After a split is chosen and kept: number of training samples that fell
   * into each bin of the winning inner discretizer (same length as
   * ``binToPartition``). Retained after training so plotting/introspection
   * can recover per-bin histograms without re-routing data. Empty at leaves.
   */
  std::vector<size_t> binSampleCounts;

  std::weak_ordering operator<=>(const ShapeFunctionNode &o) const {
    return std::compare_weak_order_fallback(informationGain, o.informationGain);
  }

  bool operator==(const ShapeFunctionNode &o) const {
    return informationGain == o.informationGain;
  }

  /** Map routing feature value(s) to an outer child partition index. */
  size_t routeFeatureValuesToPartition(
      const std::vector<float> &featureValues) const;

  /** Values at ``sampleCol`` for each index in ``routingFeatures``. */
  std::vector<float> gatherRoutingFeatureValues(const arma::fmat &X,
                                                arma::uword sampleCol) const;

  /** Route one sample column of ``X`` to an outer child partition index. */
  size_t routeSampleToPartition(const arma::fmat &X, arma::uword sampleCol) const {
    return routeFeatureValuesToPartition(gatherRoutingFeatureValues(X, sampleCol));
  }

  /**
   * Child nodes of this internal node via the owning tree's child index map.
   * Defined out-of-line in ShapeFunctionNode.cpp because it needs the complete
   * ``ShapeGeneralizedTree`` type (only forward-declared here to break the
   * ShapeGeneralizedTree <-> ShapeFunctionNode include cycle).
   */
  std::vector<ShapeFunctionNode *> getChildren() const;
};
