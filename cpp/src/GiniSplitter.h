#pragma once
#include "ClassificationSplitter.h"
#include <numeric>

class GiniSplitter : public ClassificationSplitter {
public:
  GiniSplitter(size_t numClasses) : ClassificationSplitter(numClasses) {};

  double score(const std::vector<size_t> &stats, size_t N) override {
    if (N == 0)
      return 0.0;
    const double sumP2 =
        std::accumulate(stats.begin(), stats.end(), 0.0,
                        [N](const double acc, const size_t count) {
                          const double p = static_cast<double>(count) / N;
                          return acc + p * p;
                        });
    return 1.0 - sumP2;
  }

  double score(const SplitCandidate &split) override {
    return score(getStats(split), split.end - split.start + 1);
  }

  ~GiniSplitter() override = default;
};