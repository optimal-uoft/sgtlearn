#pragma once
#include "Splitter.h"

class SquaredErrorSplitter : public Splitter<float> {
public:
  SquaredErrorSplitter(arma::frowvec &X, arma::Row<float> &y)
      : Splitter(X, y, 2) {}
  SplitCandidate makeRoot() override;
  float predict(const SplitCandidate &split) override;
  double score(const std::vector<float> &stats, size_t N) override;
  double score(const SplitCandidate &split) override {
    return score(getStats(split), split.end - split.start + 1);
  };

protected:
  void moveSample(std::vector<float> &rightStats, std::vector<float> &leftStats,
                  float v) override;
};