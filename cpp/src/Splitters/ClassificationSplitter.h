#pragma once

#include "Splitter.h"
#include <armadillo>
#include <limits>

class ClassificationSplitter : public Splitter<size_t> {

public:
  ClassificationSplitter(arma::frowvec &X, arma::Mat<size_t> &y,
                         size_t numClasses)
      : Splitter(X, y, numClasses) {}
  SplitCandidate makeRoot() override {
    auto stats = makeEmptyStats();
    for (size_t idx = 0; idx < y.n_cols; idx++)
      stats[y(0, idx)]++;

    splitStats[0][y.n_cols - 1] = stats;

    return {.height = 0,
            .start = 0,
            .end = y.n_cols - 1,
            .score = score(stats, 0, y.n_cols - 1),
            .routingThreshold = std::numeric_limits<double>::infinity()};
  }
  ~ClassificationSplitter() override = default;

  size_t predict(const SplitCandidate &split) override {
    auto v = getStats(split);

    auto it = std::max_element(v.begin(), v.end());

    return std::distance(v.begin(), it);
  }


protected:
  void moveSample(std::vector<size_t> &rightStats,
                  std::vector<size_t> &leftStats, size_t idx) override {
    const size_t cls = y(0, idx);
    leftStats[cls]++;
    rightStats[cls]--;
  }
};