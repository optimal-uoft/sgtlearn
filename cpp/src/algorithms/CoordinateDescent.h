#pragma once

/**
 * @file CoordinateDescent.h
 * @brief Greedy coordinate updates that reassign inner-tree bins to partitions to improve ``BranchAssignment::objective``.
 */

#include <cstddef>
#include "BranchAssignmentObjectives/BranchAssignmentVariants.h"
#include "algorithms/missing_values.h"
#include <algorithm>
#include <armadillo>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

/**
 * Shuffle bins each outer iteration and, for each bin, try all partition moves that improve the objective.
 * The NaN routing bin (always the last bin) is omitted from the main optimization loop to avoid 
 * skewing the numeric boundaries, and is greedily assigned to the optimal partition afterward.
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
  if (numBins <= 1) return assignmentObjective.objective();

  const size_t nanBinIndex = numBins - 1;

  // 1. Omit the NaN bin from the objective state during standard coordinate descent
  assignmentObjective.removeLeaf(nanBinIndex);

  size_t consecutiveTrialsWithoutImprovement = 0;

  for (size_t i = 0; i < maxIters; ++i) {
    bool improved = false;

    // 2. Shuffle and optimize ONLY the finite numeric bins (0 to numBins - 2)
    std::vector<size_t> permutation(nanBinIndex);
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


  // 3. Factor the NaN bin back in by greedily finding its optimal partition
  size_t bestNanPartition = missing_values::partition_with_max_count_min_index_tie(assignmentObjective.partitionSampleCounts()); // Fallback
  assignmentObjective.addLeaf(
    nanBinIndex, 
    bestNanPartition
  );
  double bestFinalObjective =assignmentObjective.objective();
  
  assignmentObjective.removeLeaf(nanBinIndex);
  for (size_t partition = 0; partition < numPartitions; ++partition) {
  
    assignmentObjective.addLeaf(nanBinIndex, partition);
    double currentObjective = assignmentObjective.objective();
    
    if (currentObjective < bestFinalObjective) {
      bestFinalObjective = currentObjective;
      bestNanPartition = partition;
    }
    
    assignmentObjective.removeLeaf(nanBinIndex);
  }

  // Commit the best placement for the NaN bin
  assignmentObjective.addLeaf(nanBinIndex, bestNanPartition);

  return assignmentObjective.objective();
}