#pragma once
#include "ClassificationSplitter.h"
#include "Criterion.h"

class GiniSplitter : public ClassificationSplitter {
public:
  GiniSplitter(arma::frowvec &X, arma::Mat<size_t> &y, size_t numClasses)
      : ClassificationSplitter(X, y, numClasses) {};

  double score(const std::vector<size_t> &stats, size_t l, size_t r) override {
    return Criterion::gini(stats, r - l + 1);
  }

  double score(const SplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  ~GiniSplitter() override = default;
};