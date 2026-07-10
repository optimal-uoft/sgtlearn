#pragma once

/**
 * @file ClassificationSplitter.h
 * @brief Classification impurity splitter: per-interval weighted class count vectors and majority-vote prediction.
 */

#include <algorithm>
#include <cstddef>
#include "Splitter.h"
#include <armadillo>
#include <limits>

class ClassificationSplitter : public Splitter<double, size_t> {

public:
  ClassificationSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                         arma::Mat<size_t> &y, size_t numClasses)
      : Splitter(X, sampleWeights, numClasses), labels(y) {}
  UnivariateSplitCandidate makeRoot() override {
    auto stats = makeEmptyStats();
    for (size_t idx = 0; idx < labels.n_cols; idx++)
      stats[labels(0, idx)] += static_cast<double>(sampleWeights(idx));

    splitStats[0][labels.n_cols - 1] = stats;

    UnivariateSplitCandidate root{.height = 0,
                        .start = 0,
                        .end = labels.n_cols - 1,
                        .score = score(stats, 0, labels.n_cols - 1),
                        .routingThreshold =
                            std::numeric_limits<double>::infinity()};
    fillIntervalMeta(root);
    return root;
  }
  ~ClassificationSplitter() override = default;

  size_t predict(const UnivariateSplitCandidate &split) override {
    auto v = getStats(split);

    auto it = std::max_element(v.begin(), v.end());

    return static_cast<size_t>(std::distance(v.begin(), it));
  }

  double score(const std::vector<double> &stats, size_t l, size_t r) override = 0;

protected:
  const arma::Mat<size_t> &labels;

  void moveSample(std::vector<double> &rightStats, std::vector<double> &leftStats,
                  size_t idx) override {
    const size_t cls = labels(0, idx);
    const double w = static_cast<double>(sampleWeights(idx));
    leftStats[cls] += w;
    rightStats[cls] -= w;
  }
};
