#pragma once

/**
 * @file EntropySplitter.h
 * @brief ``ClassificationSplitter`` using Shannon entropy impurity.
 */

#include "ClassificationSplitter.h"
#include "Criterion.h"

#include <cmath>

class EntropySplitter : public ClassificationSplitter {
public:
  EntropySplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                  arma::Mat<size_t> &y, size_t numClasses)
      : ClassificationSplitter(X, sampleWeights, y, numClasses) {}

  double score(const std::vector<double> &stats, size_t l, size_t r) override {
    (void)l;
    (void)r;
    double W = 0.0;
    for (double c : stats)
      W += c;
    return Criterion::entropy(stats, W);
  }

  double score(const SplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  ~EntropySplitter() override = default;
};
