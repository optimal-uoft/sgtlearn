#pragma once

#include <cmath>

/**
 * @file Estimators/ShapeFunctions/ShapeFunctionNode.h
 * @brief One node in a shape-generalized / shape-function tree: routing, fit
 *        sample set, and per-bin stats for the winning inner split (classification
 *        or regression).
 */

#include <algorithm>
#include <armadillo>
#include <compare>
#include <cstddef>
#include <vector>

class ShapeGeneralizedTree;

/**
 * Outer-tree node: routing rule when internal, plus training sample indices
 * during fit.
 *
 * After `fit`, `sampleIndices` is cleared on stored nodes; routing.
 *
 * `TreeBuilder` orders nodes by `informationGain` for the best-first heap.
 */
struct ShapeFunctionNode {
  ShapeGeneralizedTree* tree = nullptr;

  size_t height = 0;
  double score = 0.0;
  double informationGain = 0.0;

  /** Node id in `nodes_` while building; cleared after training. */
  size_t nodeIndex = 0;

  bool isLeaf = true;
  size_t numPartitions = 0;

  /** Row index into X for routing; undefined if isLeaf. */
  size_t routingFeature = 0;
  /** Sorted ascending; same convention as UnivariateDiscretizer::transform. */
  std::vector<float> innerThresholds;
  /** Length equals inner discretizer bin count; maps bin -> child partition. */
  std::vector<size_t> binToPartition;
  /**
   * Outer child partition for non-finite values of ``routingFeature``.
   * Set during training: best-scoring partition when NaN was seen, else the
   * finite-sample majority partition (smallest index on ties).
   */
  size_t nanPredictionPartition = 0;

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
  /**
   * Weighted sufficient statistics for samples with non-finite
   * ``routingFeature``. Classification: weighted class counts. Regression
   * (squared error): ``[sum w·y, sum w·y²]``. Empty when no missing values,
   * at leaves, or for MAE splits.
   */
  std::vector<double> splitMissingStats;
  /** Sum of sample weights in ``splitMissingStats`` (regression squared error). */
  double splitMissingWeight = 0.0;

  std::weak_ordering operator<=>(const ShapeFunctionNode &o) const {
    return std::compare_weak_order_fallback(informationGain, o.informationGain);
  }

  bool operator==(const ShapeFunctionNode &o) const {
    return informationGain == o.informationGain;
  }

  size_t routeFeatureValueToPartition(float featureValue) const {
    if (binToPartition.empty())
      throw std::runtime_error("binToPartition is empty");
    if (!std::isfinite(static_cast<double>(featureValue)))
      return nanPredictionPartition;
    const auto it = std::lower_bound(innerThresholds.begin(),
                                     innerThresholds.end(), featureValue);
    size_t bin = static_cast<size_t>(it - innerThresholds.begin());
    if (bin >= binToPartition.size())
      throw std::runtime_error("bin out of range");
    return binToPartition[bin];
  }

  /**
   * Child nodes of this internal node via the owning tree's child index map.
   * Defined out-of-line in ShapeFunctionNode.cpp because it needs the complete
   * ``ShapeGeneralizedTree`` type (only forward-declared here to break the
   * ShapeGeneralizedTree <-> ShapeFunctionNode include cycle).
   */
  std::vector<ShapeFunctionNode *> getChildren() const;
};
