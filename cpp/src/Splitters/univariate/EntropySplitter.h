#pragma once

/**
 * @file EntropySplitter.h
 * @brief ``ClassificationSplitter`` using Shannon entropy impurity.
 */

#include <cstddef>
#include "ClassificationSplitter.h"
#include "Criterion.h"

#include <cmath>

class EntropySplitter : public ClassificationSplitter {
public:
  EntropySplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                  arma::Mat<size_t> &y,
                  const std::vector<size_t> &nClassesPerOutput)
      : ClassificationSplitter(X, sampleWeights, y, nClassesPerOutput) {}

  double score(const std::vector<double> &stats, size_t l, size_t r) override {
    (void)l;
    (void)r;
    return Criterion::entropyMulti(stats, classesPerOutput);
  }

  double score(const UnivariateSplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  ~EntropySplitter() override = default;
};
