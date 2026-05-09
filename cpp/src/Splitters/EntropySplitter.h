#pragma once
#include "ClassificationSplitter.h"
#include <cmath>

class EntropySplitter : public ClassificationSplitter {
public:
  EntropySplitter(arma::frowvec &X, arma::Row<size_t> &y, size_t numClasses)
      : ClassificationSplitter(X, y, numClasses) {}

  double score(const std::vector<size_t> &stats, size_t l, size_t r) override {
    size_t N = r - l + 1;
    if (N == 0)
      return 0;
    double ent = 0;
    for (size_t count : stats) {
      if (count == 0)
        continue;
      const double p = static_cast<double>(count) / static_cast<double>(N);
      ent -= p * std::log2(p);
    }
    return ent;
  }

  double score(const SplitCandidate &split) override {
    return score(getStats(split), split.start, split.end);
  }

  ~EntropySplitter() override = default;
};
