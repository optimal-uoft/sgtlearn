#pragma once
#include <vector>

class BranchAssignmentObjective {
protected:
  ~BranchAssignmentObjective() = default;

public:
  BranchAssignmentObjective(std::vector<size_t> &assignments)
      : assignments(assignments) {}

  std::vector<size_t> &assignments;

  virtual double objective();

  virtual void add_leaf(int leaf, int partition);

  virtual void remove_leaf(int leaf);
};