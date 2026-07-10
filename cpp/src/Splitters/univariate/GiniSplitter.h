#pragma once

/**
 * @file GiniSplitter.h
 * @brief ``ClassificationSplitter`` using Gini impurity.
 */

#include "ClassificationSplitter.h"
#include "Criterion.h"

class GiniSplitter : public ClassificationSplitter {
public:
  GiniSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
               arma::Mat<size_t> &y, size_t numClasses)
      : ClassificationSplitter(X, sampleWeights, y, numClasses) {};

  double score(const std::vector<double> &stats, size_t l, size_t r) override {
    (void)l;
    (void)r;
    double W = 0.0;
    for (double c : stats)
      W += c;
    return Criterion::gini(stats, W);
  }

  double score(const UnivariateSplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  ~GiniSplitter() override = default;
};
