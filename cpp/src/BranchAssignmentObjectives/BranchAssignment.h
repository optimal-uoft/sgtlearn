#pragma once
#include <vector>

class BranchAssignment {
public:
  BranchAssignment(std::vector<size_t> &assignments, size_t numPartitions)
      : assignments(assignments), numPartitions(numPartitions) {}

  virtual ~BranchAssignment() = default;

  std::vector<size_t> &assignments;
  size_t numPartitions;

  virtual double objective() = 0;

  virtual void addLeaf(size_t leaf, size_t partition) = 0;

  virtual void removeLeaf(size_t leaf) = 0;
};
