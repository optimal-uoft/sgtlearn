#pragma once

/**
 * @file algorithms/TAO/TaoObjective.h
 * @brief Per-node TAO reward objective over a care set.
 *
 * Encapsulates scoring of candidate routing rules at one internal node. Rewards
 * come from the task-specific ``NodeCareSet``; this class only sums per-sample
 * rewards under a routing map (current node rule, dummy constant rule, or a
 * rule induced by a trained classification discretizer).
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
 * Candidates subtract ``complexityScale * lambda * nodeSampleCount`` from the
 * weighted reward sum. ``scoreDummy`` uses scale zero.
 */
class TaoObjective {
public:
  TaoObjective(const NodeCareSet &care, const arma::fmat &X, double lambda,
               double nodeSampleCount);

  /** Number of care samples at this node. */
  size_t nCare() const { return nCare_; }

  /** Fallback child partition for NaN routing and dummy rules. */
  size_t dummyChild() const { return care_.dummyChild; }

  const NodeCareSet &careSet() const { return care_; }

  /** Mean care reward under the node's current routing rule, minus split penalty. */
  double scoreCurrent(const ShapeFunctionNode &node,
                      double complexityScale) const;

  /** Mean care reward when every care sample routes to ``dummyChild``. */
  double scoreDummy() const;

  /**
   * Extract routing from a trained discretizer and score it on the care set.
   *
   * Bins are mapped to child partitions by argmax over discretizer leaf stats.
   * Writes the induced bin-to-partition map to the out-param.
   *
   * @returns Penalized mean reward, or ``-infinity`` when ``disc`` has no bins.
   */
  double scoreDiscretizer(ClassificationDiscretizer &disc,
                          std::vector<size_t> &binToPartitionOut,
                          double complexityScale) const;

private:
  static size_t argMax(const std::vector<double> &counts);

  double rewardSumForDiscretizer(
      const ClassificationDiscretizer &disc,
      const std::vector<size_t> &binToPartition) const;

  double meanReward(double rewardSum) const;
  double penalizedScore(double rewardSum, double complexityScale) const;
  double careWeight(size_t i) const;

  const NodeCareSet &care_;
  const arma::fmat &X_;
  double lambda_;
  double nodeSampleCount_;
  size_t nCare_;
  double totalCareWeight_;
};

} // namespace tao
