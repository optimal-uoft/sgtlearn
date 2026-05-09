
#pragma once
#include "BranchAssignmentObjectives/BranchAssignmentObjective.h"
#include <armadillo>
#include <stdexcept>

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
                               BranchAssignmentObjective assignmentObjective,
                               size_t maxIters = 10, size_t patience = 5,
                               size_t seed = 42) {
  throw std::runtime_error("not implemented yet");
  return 0.0;
}