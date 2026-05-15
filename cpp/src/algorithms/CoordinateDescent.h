#pragma once

/**
 * @file CoordinateDescent.h
 * @brief Greedy coordinate updates that reassign inner-tree bins to partitions to improve ``BranchAssignment::objective``.
 */

#include "BranchAssignmentObjectives/BranchAssignmentVariants.h"
#include <algorithm>
#include <armadillo>
#include <numeric>
#include <random>
#include <vector>

/**
 * Shuffle bins each outer iteration and, for each bin, try all partition moves that improve the objective.
 *
 * Each bin update compares candidate objectives to the value **before** moving
 * that bin (holding all other assignments fixed). Reusing a single global
 * ``best`` across bins was incorrect and could accept worsening moves or reject
 * valid improvements (observed on MAE regression vs sklearn fidelity).
 *
 * @param numPartitions number of child partitions (fan-out).
 * @param assignmentObjective live objective; ``assignments`` updated in place.
 * @param rng          non-const generator for shuffling bin order each outer round.
 * @param maxIters     outer shuffle rounds.
 * @param patience     stop after this many rounds without improvement.
 * @return final ``assignmentObjective.objective()`` after the last accepted move.
 */
inline double coordinateDescent(size_t numPartitions,
                                BranchAssignment &assignmentObjective,
                                std::mt19937_64 &rng, size_t maxIters = 10,
                                size_t patience = 5) {
  size_t numBins = assignmentObjective.assignments.size();
  size_t consecutiveTrialsWithoutImprovement = 0;

  for (size_t i = 0; i < maxIters; ++i) {
    bool improved = false;

    std::vector<size_t> permutation(numBins);
    std::iota(permutation.begin(), permutation.end(), size_t{0});
    std::shuffle(permutation.begin(), permutation.end(), rng);

    for (size_t j : permutation) {
      const double objectiveBeforeMovingBinJ = assignmentObjective.objective();
      size_t currentAssignedPartition = assignmentObjective.assignments[j];
      size_t bestPartition = currentAssignedPartition;
      double bestImpurityForBinJ = objectiveBeforeMovingBinJ;

      assignmentObjective.removeLeaf(j);

      for (size_t partition = 0; partition < numPartitions; ++partition) {
        if (partition == currentAssignedPartition)
          continue;

        assignmentObjective.addLeaf(j, partition);
        double impurity = assignmentObjective.objective();

        if (impurity < bestImpurityForBinJ) {
          bestImpurityForBinJ = impurity;
          bestPartition = partition;
          improved = true;
        }
        assignmentObjective.removeLeaf(j);
      }
      assignmentObjective.addLeaf(j, bestPartition);
    }

    if (improved)
      consecutiveTrialsWithoutImprovement = 0;
    else
      consecutiveTrialsWithoutImprovement++;

    if (consecutiveTrialsWithoutImprovement >= patience)
      break;
  }
  return assignmentObjective.objective();
}
