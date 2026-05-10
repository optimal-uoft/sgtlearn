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
 * @param numPartitions number of child partitions (fan-out).
 * @param assignmentObjective live objective; ``assignments`` updated in place.
 * @param maxIters outer shuffle rounds.
 * @param patience stop after this many rounds without improvement.
 * @param seed RNG for shuffling bin order.
 * @return final ``assignmentObjective.objective()`` after the last accepted move.
 */
inline double coordinateDescent(size_t numPartitions,
                                BranchAssignment &assignmentObjective,
                                size_t maxIters = 10, size_t patience = 5,
                                size_t seed = 42) {
  size_t numBins = assignmentObjective.assignments.size();
  size_t consecutiveTrialsWithoutImprovement = 0;
  std::mt19937 g(seed);
  double bestImpurity = assignmentObjective.objective();

  for (size_t i = 0; i < maxIters; ++i) {
    bool improved = false;

    std::vector<size_t> permutation(numBins);
    std::iota(permutation.begin(), permutation.end(), size_t{0});
    std::shuffle(permutation.begin(), permutation.end(), g);

    for (size_t j : permutation) {
      size_t currentAssignedPartition = assignmentObjective.assignments[j];
      size_t bestPartition = currentAssignedPartition;
      assignmentObjective.removeLeaf(j);

      for (size_t partition = 0; partition < numPartitions; ++partition) {
        if (partition == currentAssignedPartition)
          continue;

        assignmentObjective.addLeaf(j, partition);
        double impurity = assignmentObjective.objective();

        if (impurity < bestImpurity) {
          bestImpurity = impurity;
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
