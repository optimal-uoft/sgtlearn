
#pragma once
#include "BranchAssignmentObjectives/BranchAssignmentVariants.h"
#include <algorithm>
#include <armadillo>
#include <random>
#include <ranges>

/**
 *
 * @param numPartitions number of partitions to try to assign bins to
 * @param assignmentObjective
 * @param maxIters number of permutations to attempt
 * @param patience number of permutations to try without seeing improvements
 * before breaking
 * @param seed
 * @return the final impurity of the assignment
 */
inline float coordinateDescent(size_t numPartitions,
                               BranchAssignment assignmentObjective,
                               size_t maxIters = 10, size_t patience = 5,
                               size_t seed = 42) {
  size_t numBins = assignmentObjective.assignments.size();
  size_t consecutiveTrialsWithoutImprovement = 0;
  std::mt19937 g(seed);
  double bestImpurity = assignmentObjective.objective();

  for (int i = 0; i < maxIters; ++i) {
    bool improved = false;

    auto permutation =
        std::views::iota(0, numBins) | std::ranges::to<std::vector>();
    std::ranges::shuffle(permutation, g);

    for (int j : permutation) {
      size_t currentAssignedPartition = assignmentObjective.assignments[j];
      size_t bestPartition = currentAssignedPartition;
      assignmentObjective.removeLeaf(j);

      for (int partition = 0; partition < numPartitions; ++partition) {
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
  return 0.0;
}