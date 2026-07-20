#pragma once

/**
 * @file ClassificationSplitter.h
 * @brief Classification impurity splitter: per-interval weighted class count vectors and majority-vote prediction.
 */

#include <algorithm>
#include <cstddef>
#include <iterator>
#include "Splitter.h"
#include <armadillo>
#include <limits>
#include <vector>

/**
 * Multi-output classification splitter. ``labels`` has shape
 * ``(nOutputs, nSamples)``. Per-interval stats are nested weighted class
 * histograms ``stats[output][class]``. Concrete subclasses sum the per-output
 * impurity across outputs. ``predict`` returns the per-output argmax.
 * Single-output matches the old scalar path.
 */
class ClassificationSplitter
    : public Splitter<std::vector<double>, std::vector<size_t>> {

public:
  ClassificationSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                         arma::Mat<size_t> &y,
                         const std::vector<size_t> &nClassesPerOutput)
      : Splitter(X, sampleWeights, nClassesPerOutput.size()), labels(y),
        classesPerOutput(nClassesPerOutput),
        nOutputs_(nClassesPerOutput.size()) {}

  UnivariateSplitCandidate makeRoot() override {
    auto stats = makeEmptyStats();
    for (size_t idx = 0; idx < labels.n_cols; idx++) {
      const double w = static_cast<double>(sampleWeights(idx));
      for (size_t o = 0; o < nOutputs_; ++o)
        stats[o][labels(o, idx)] += w;
    }

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

  std::vector<size_t> predict(const UnivariateSplitCandidate &split) override {
    const auto &v = getStats(split);
    std::vector<size_t> preds(nOutputs_, 0);
    for (size_t o = 0; o < nOutputs_; ++o) {
      auto it = std::max_element(v[o].begin(), v[o].end());
      preds[o] = static_cast<size_t>(std::distance(v[o].begin(), it));
    }
    return preds;
  }

  double score(const std::vector<std::vector<double>> &stats, size_t l,
               size_t r) override = 0;

protected:
  const arma::Mat<size_t> &labels;
  std::vector<size_t> classesPerOutput;
  size_t nOutputs_;

  std::vector<std::vector<double>> makeEmptyStats() override {
    std::vector<std::vector<double>> stats(nOutputs_);
    for (size_t o = 0; o < nOutputs_; ++o)
      stats[o].assign(classesPerOutput[o], 0.0);
    return stats;
  }

  void moveSample(std::vector<std::vector<double>> &rightStats,
                  std::vector<std::vector<double>> &leftStats,
                  size_t idx) override {
    const double w = static_cast<double>(sampleWeights(idx));
    for (size_t o = 0; o < nOutputs_; ++o) {
      const size_t lab = labels(o, idx);
      leftStats[o][lab] += w;
      rightStats[o][lab] -= w;
    }
  }
};
