#pragma once

/**
 * @file algorithms/TAO/TaoObjective.h
 * @brief Per-node TAO reward objective over a care set.
 *
 * Encapsulates scoring of candidate routing rules at one internal node. Rewards
 * come from the task-specific ``NodeCareSet``; this class only sums per-sample
 * rewards under a routing map (current node rule, dummy constant rule, or a
 * single-feature rule induced by a trained classification discretizer).
 */

#include "Discretizers/ClassificationDiscretizer.h"
#include "Estimators/ShapeFunctions/ShapeFunctionNode.h"
#include "algorithms/TAO/TaoAdapter.h"

#include <cstddef>
#include <vector>

namespace tao {

/**
 * Mean care-set reward under a routing rule, with optional split penalty.
 *
 * ``dummyScore`` omits ``lambda``; ``scoreCurrent`` and discretizer scores
 * subtract ``lambda`` to penalize non-constant splits.
 */
class TaoObjective {
public:
  TaoObjective(const NodeCareSet &care, const arma::fmat &X, double lambda);

  /** Number of care samples at this node. */
  size_t nCare() const { return nCare_; }

  /** Fallback child partition for NaN routing and dummy rules. */
  size_t dummyChild() const { return care_.dummyChild; }

  const NodeCareSet &careSet() const { return care_; }

  /**
   * Mean care reward under the node's current routing rule, minus ``lambda_``.
   */
  double scoreCurrent(const ShapeFunctionNode &node) const;

  /** Mean care reward when every care sample routes to ``dummyChild``. */
  double scoreDummy() const;

  /**
   * Extract routing from a trained discretizer and score it on the care set.
   *
   * Bins are mapped to child partitions by argmax over discretizer leaf stats.
   * Writes the induced thresholds and bin-to-partition map to the out-params.
   *
   * @returns Penalized mean reward, or ``-infinity`` when ``disc`` has no bins.
   */
  double scoreDiscretizer(size_t feature, ClassificationDiscretizer &disc,
                          std::vector<float> &thresholdsOut,
                          std::vector<size_t> &binToPartitionOut) const;

  /**
   * Mean care reward for an explicit single-feature routing rule, minus
   * ``lambda_``.
   */
  double scoreRouting(size_t feature, const std::vector<float> &thresholds,
                      const std::vector<size_t> &binToPartition) const;

private:
  static size_t argMax(const std::vector<double> &counts);

  static size_t routeValue(float value, const std::vector<float> &thresholds,
                           const std::vector<size_t> &binToPartition,
                           size_t nanPartition);

  double rewardSumForPartition(size_t feature,
                               const std::vector<float> &thresholds,
                               const std::vector<size_t> &binToPartition) const;

  double meanReward(double rewardSum) const;
  double penalizedScore(double rewardSum) const;

  const NodeCareSet &care_;
  const arma::fmat &X_;
  double lambda_;
  size_t nCare_;
};

} // namespace tao
