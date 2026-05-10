#pragma once

/**
 * @file ShapeBranchingFit_internal.h
 * @brief Implementation details for ``fitShapeBranch``: k-means-style CD initialization,
 * partition feasibility checks, and per-feature discriminator evaluation.
 *
 * Not included from public headers; keep call sites limited to
 * ShapeBranchingFit.cpp (and any future regression branching TU).
 */

#include "algorithms/CoordinateDescent.h"
#include "algorithms/ShapeBranchingTypes.h"
#include "algorithms/ShapeGeneralizedTreeParams.h"
#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "Domain/LearningCriterion.h"

#include <armadillo>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

namespace shape_branching_fit::detail {

/**
 * Weighted Lloyd on rows of X (B x C), weights w; k clusters. Writes labels
 * in [0, k). Uses spaced bin indices as initial centroids (deterministic
 * given B,k). Falls back to round-robin if B < k.
 */
inline void initAssignmentsWeightedKMeans(const arma::mat &X, const arma::vec &w,
                                            size_t k, uint64_t seed,
                                            std::vector<size_t> &assign) {
  const size_t B = X.n_rows;
  const size_t C = X.n_cols;
  assign.resize(B);
  if (B == 0 || k == 0)
    return;
  if (B < k) {
    for (size_t b = 0; b < B; ++b)
      assign[b] = b % k;
    return;
  }

  arma::mat centroids(k, C);
  for (size_t j = 0; j < k; ++j) {
    const size_t idx = (j * B) / k;
    centroids.row(j) = X.row(idx);
  }

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<size_t> pickBin(0, B - 1);

  for (size_t iter = 0; iter < 50; ++iter) {
    bool changed = false;
    for (size_t b = 0; b < B; ++b) {
      double bestD = std::numeric_limits<double>::infinity();
      size_t bestJ = 0;
      for (size_t j = 0; j < k; ++j) {
        double d = 0.0;
        for (size_t c = 0; c < C; ++c) {
          const double diff = X(b, c) - centroids(j, c);
          d += diff * diff;
        }
        if (d < bestD) {
          bestD = d;
          bestJ = j;
        }
      }
      if (assign[b] != bestJ) {
        assign[b] = bestJ;
        changed = true;
      }
    }

    centroids.zeros();
    arma::vec sumW(k, arma::fill::zeros);
    for (size_t b = 0; b < B; ++b) {
      const size_t j = assign[b];
      for (size_t c = 0; c < C; ++c)
        centroids(j, c) += w(b) * X(b, c);
      sumW(j) += w(b);
    }
    for (size_t j = 0; j < k; ++j) {
      if (sumW(j) > 1e-12)
        centroids.row(j) /= sumW(j);
      else
        centroids.row(j) = X.row(pickBin(rng));
    }

    if (!changed && iter > 0)
      break;
  }
}

inline void initBinAssignmentsForCd(bool smartInit,
                                    const std::vector<std::vector<size_t>> &stats,
                                    const std::vector<size_t> &sizes,
                                    size_t numPartitions, size_t numClasses,
                                    uint64_t seed, std::vector<size_t> &assignments) {
  const size_t B = stats.size();
  assignments.resize(B);
  if (B == 0)
    return;
  if (!smartInit || numPartitions < 2 || B < numPartitions) {
    for (size_t b = 0; b < B; ++b)
      assignments[b] = b % numPartitions;
    return;
  }

  arma::mat X(B, numClasses);
  arma::vec w(B);
  for (size_t b = 0; b < B; ++b) {
    w(b) = std::max(1.0, static_cast<double>(sizes[b]));
    double sum = 0.0;
    for (size_t c = 0; c < numClasses; ++c)
      sum += static_cast<double>(stats[b][c]);
    if (sum <= 0.0) {
      X.row(b).fill(1.0 / static_cast<double>(numClasses));
    } else {
      for (size_t c = 0; c < numClasses; ++c)
        X(b, c) = static_cast<double>(stats[b][c]) / sum;
    }
  }

  initAssignmentsWeightedKMeans(X, w, numPartitions, seed, assignments);
}

inline bool partitionSizesOk(const std::vector<size_t> &assignments,
                             const std::vector<size_t> &binSizes, size_t numPartitions,
                             size_t outerMinLeaf) {
  std::vector<size_t> wt(numPartitions, 0);
  for (size_t b = 0; b < assignments.size(); ++b)
    wt[assignments[b]] += binSizes[b];
  for (size_t p = 0; p < numPartitions; ++p) {
    if (wt[p] < outerMinLeaf)
      return false;
  }
  return true;
}

template <typename Disc>
void evaluateDiscriminator(
    Disc &disc, size_t fIdx, LearningCriterion criterion, size_t numClasses,
    size_t numPartitions, const CoordinateDescentParams &cd, double parentImp,
    double branchingPenalty, double outerMinImpurityDecrease, size_t outerMinLeaf,
    double eps, double &bestObjective, ShapeBranchingResult &best) {
  const size_t B = disc.numLeaves;
  if (B == 0)
    return;

  auto &stats = disc.getLeafStats();
  auto &sizes = disc.getLeafNumSamples();

  const size_t mixSeed =
      cd.seed ^ (0x9e3779b9u * (static_cast<unsigned>(fIdx) + 1u));
  std::vector<size_t> assignments(B);
  initBinAssignmentsForCd(cd.smartInit, stats, sizes, numPartitions, numClasses,
                          mixSeed, assignments);

  auto branchObj = makeClassificationBranchAssignment(
      criterion, assignments, numPartitions, stats, sizes, numClasses);
  coordinateDescent(numPartitions, *branchObj, cd.maxIters, cd.patience,
                    mixSeed);

  if (!partitionSizesOk(assignments, sizes, numPartitions, outerMinLeaf))
    return;

  const double childImp = branchObj->objective();
  const double penalizedChild =
      childImp + branchingPenalty * static_cast<double>(numPartitions > 0 ? numPartitions - 1 : 0);
  const double impurityDecrease = parentImp - childImp;
  if (impurityDecrease < outerMinImpurityDecrease - eps)
    return;

  if (penalizedChild < bestObjective - eps) {
    bestObjective = penalizedChild;
    best.featureIndex = fIdx;
    const auto &dth = disc.getThresholds();
    best.innerThresholds.resize(dth.size());
    for (size_t t = 0; t < dth.size(); ++t)
      best.innerThresholds[t] = static_cast<float>(dth[t]);
    best.binToPartition = assignments;
    best.impurityDecrease = impurityDecrease;
  }
}

} // namespace shape_branching_fit::detail
