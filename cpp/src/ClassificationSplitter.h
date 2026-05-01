#pragma once

#include "Splitter.h"
#include <limits>

class ClassificationSplitter : public Splitter<size_t> {

public:
  ClassificationSplitter(size_t numClasses) : Splitter(numClasses) {}
  SplitCandidate makeRoot(const arma::Row<size_t> y) override {
    auto stats = makeEmptyStats();
    for (size_t idx = 0; idx < y.n_cols; idx++)
      stats[y(idx)]++;

    splitStats[0][y.n_cols - 1] = stats;

    return {.height = 0,
            .start = 0,
            .end = y.n_cols - 1,
            .score = score(stats, y.n_cols),
            .routingThreshold = std::numeric_limits<double>::infinity()};
  }
  ~ClassificationSplitter() override = default;

  double predict(const SplitCandidate &split) override {
    auto v = getStats(split);

    auto it = std::max_element(v.begin(), v.end());

    return std::distance(v.begin(), it);
  };

protected:
  void moveSample(std::vector<size_t> &rightStats, std::vector<size_t> &leftStats,
                  size_t v) override {
    leftStats[v]++;
    rightStats[v]--;
  }
};