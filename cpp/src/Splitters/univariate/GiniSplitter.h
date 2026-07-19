#pragma once

/**
 * @file GiniSplitter.h
 * @brief ``ClassificationSplitter`` using Gini impurity.
 */

#include <cstddef>
#include "ClassificationSplitter.h"
#include "Criterion.h"

class GiniSplitter : public ClassificationSplitter {
public:
  GiniSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
               arma::Mat<size_t> &y,
               const std::vector<size_t> &nClassesPerOutput)
      : ClassificationSplitter(X, sampleWeights, y, nClassesPerOutput) {};

  double score(const std::vector<double> &stats, size_t l, size_t r) override {
    (void)l;
    (void)r;
    return Criterion::giniMulti(stats, classesPerOutput);
  }

  double score(const UnivariateSplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  ~GiniSplitter() override = default;
};
