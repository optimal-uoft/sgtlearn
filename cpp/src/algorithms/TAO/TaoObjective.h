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
#include <limits>
#include <optional>
#include <vector>

namespace tao {

/**
 * One univariate threshold routing for a two-child node: finite
 * ``x <= leftMax`` goes to ``leftChild``, finite ``x > leftMax`` to
 * ``1 - leftChild``, non-finite to ``nanChild``.
 */
struct ThresholdCut {
  /** Largest finite care value on the left side; the cut lies above it. */
  float leftMax = 0.0f;
  size_t leftChild = 0;
  size_t nanChild = 0;
  /** Weighted care reward of this routing (no complexity penalty). */
  double rewardSum = -std::numeric_limits<double>::infinity();
};

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
   * Each bin goes to the child with the largest summed care reward over the
   * care samples in it (ties: lowest child index), not to the argmax of the
   * discretizer's fitted histogram. Bins with zero care weight, including an
   * unused NaN bin, go to ``dummyChild``. Writes the bin-to-partition map to
   * the out-param.
   *
   * @returns Penalized mean reward, or ``-infinity`` when ``disc`` has no bins.
   */
  double scoreDiscretizer(ClassificationDiscretizer &disc,
                          std::vector<size_t> &binToPartitionOut,
                          double complexityScale) const;

  /**
   * Penalized mean reward when care sample ``i`` routes to
   * ``binToPartition[bins(careCols[i])]``.
   *
   * @param bins Bin of every column of ``X`` (as from ``transform(X, bins)``).
   */
  double scoreBinAssignment(const arma::Row<size_t> &bins,
                            const std::vector<size_t> &binToPartition,
                            double complexityScale) const;

  /**
   * Exact best threshold routing on raw ``feature`` for a two-child node.
   *
   * Cuts lie between consecutive distinct finite care values (using the inner
   * splitter's ``1e-7`` tie rule) with at least ``minLeafSize`` finite care
   * samples per side; both orientations are tried. Non-finite care samples go
   * to the child with the larger summed reward over them (``dummyChild`` when
   * they carry no weight).
   *
   * @returns The best cut, or ``std::nullopt`` when no cut is feasible.
   */
  std::optional<ThresholdCut> bestThresholdCut(size_t feature,
                                               size_t minLeafSize) const;

private:
  static size_t argMax(const std::vector<double> &counts);

  double rewardSumForBins(const arma::Row<size_t> &bins,
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
