#pragma once

/**
 * @file ShapeFunctionNode.h
 * @brief One node in a shape-generalized / shape-function tree: routing, fit
 *        sample set, and classification leaf histogram.
 */

#include <algorithm>
#include <compare>
#include <cstddef>
#include <vector>

/**
 * Outer-tree node: routing rule when internal, plus training sample indices
 * during fit and per-class counts at leaves (classification).
 *
 * After `fit`, `sampleIndices` is cleared on stored nodes; routing and
 * `leafClassCounts` remain for prediction.
 *
 * `TreeBuilder` orders nodes by `informationGain` for the best-first heap.
 */
struct ShapeFunctionNode {
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
   * During fit: original column indices into X at this node (Python
   * `point_idxs`). Cleared after training.
   */
  std::vector<size_t> sampleIndices;
  /**
   * During fit, after a split is chosen: inner discretizer bin per entry of
   * `sampleIndices` (same length). Cleared after training.
   */
  std::vector<size_t> sampleBins;
  /**
   * During fit, after a split: per-bin class counts from the winning inner
   * discretizer (same row count as ``binToPartition``). Cleared after training.
   */
  std::vector<std::vector<size_t>> splitLeafStats;

  /** Leaf-only: counts per class label (encoding matches training y). */
  std::vector<size_t> leafClassCounts;

  std::weak_ordering operator<=>(const ShapeFunctionNode &o) const {
    return std::compare_weak_order_fallback(informationGain,
                                            o.informationGain);
  }

  bool operator==(const ShapeFunctionNode &o) const {
    return informationGain == o.informationGain;
  }

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
