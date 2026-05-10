#pragma once

/**
 * @file ShapeGeneralizedTreeNodeBase.h
 * @brief Routing fields shared by shape-generalized tree nodes (classification and future regression).
 */

#include <algorithm>
#include <compare>
#include <cstddef>
#include <vector>

/**
 * Routing topology and transient builder state for one outer-tree node.
 *
 * Shared contract:
 * - During `fit`, sampleBegin/sampleEnd index into the builder's sample-order
 *   buffer; nodeIndex indexes `nodes_`. These are cleared after training.
 * - After a split, childSampleBounds is an exclusive prefix (length
 *   numPartitions + 1) over that buffer; partition p owns
 *   [childSampleBounds[p], childSampleBounds[p + 1]).
 * - For prediction, routingFeature selects a row of X; innerThresholds are
 *   sorted ascending (UnivariateDiscretizer::transform convention).
 *   binToPartition maps bin id to child partition in [0, numPartitions).
 *
 * Task-specific payloads (class histograms, regression sufficient stats, …)
 * live on derived node types so the same routing machinery can serve
 * classification and regression.
 */
struct ShapeGeneralizedTreeNodeBase {
  size_t height = 0;
  double score = 0.0;
  double informationGain = 0.0;

  /**
   * During `fit`, half-open range into the tree's sample-order vector listing
   * column indices of training samples at this node. Cleared after training.
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
