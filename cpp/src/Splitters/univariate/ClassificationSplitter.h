#pragma once

/**
 * @file ClassificationSplitter.h
 * @brief Classification impurity splitter: per-interval weighted class count vectors and majority-vote prediction.
 */

#include <algorithm>
#include <cstddef>
#include <numeric>
#include "Splitter.h"
#include <armadillo>
#include <limits>
#include <vector>

/**
 * Multi-output classification splitter. ``labels`` has shape
 * ``(nOutputs, nSamples)``. Per-interval stats are the concatenation of one
 * weighted class histogram per output (block ``o`` has ``classesPerOutput[o]``
 * entries). Concrete subclasses sum the per-output impurity across blocks.
 * ``predict`` returns the per-output argmax. Single-output matches the old
 * scalar path.
 */
class ClassificationSplitter : public Splitter<double, std::vector<size_t>> {

public:
  ClassificationSplitter(arma::frowvec &X, arma::frowvec &sampleWeights,
                         arma::Mat<size_t> &y,
                         const std::vector<size_t> &nClassesPerOutput)
      : Splitter(X, sampleWeights, totalClasses(nClassesPerOutput)), labels(y),
        classesPerOutput(nClassesPerOutput),
        nOutputs_(nClassesPerOutput.size()) {
    classOffsets_.assign(nOutputs_, 0);
    size_t off = 0;
    for (size_t o = 0; o < nOutputs_; ++o) {
      classOffsets_[o] = off;
      off += classesPerOutput[o];
    }
  }
  UnivariateSplitCandidate makeRoot() override {
    auto stats = makeEmptyStats();
    for (size_t idx = 0; idx < labels.n_cols; idx++) {
      const double w = static_cast<double>(sampleWeights(idx));
      for (size_t o = 0; o < nOutputs_; ++o)
        stats[classOffsets_[o] + labels(o, idx)] += w;
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
      const size_t off = classOffsets_[o];
      const size_t nc = classesPerOutput[o];
      auto it = std::max_element(v.begin() + static_cast<std::ptrdiff_t>(off),
                                 v.begin() + static_cast<std::ptrdiff_t>(off + nc));
      preds[o] = static_cast<size_t>(
          std::distance(v.begin() + static_cast<std::ptrdiff_t>(off), it));
    }
    return preds;
  }

  double score(const std::vector<double> &stats, size_t l, size_t r) override = 0;

protected:
  const arma::Mat<size_t> &labels;
  std::vector<size_t> classesPerOutput;
  std::vector<size_t> classOffsets_;
  size_t nOutputs_;

  static size_t totalClasses(const std::vector<size_t> &nClassesPerOutput) {
    return std::accumulate(nClassesPerOutput.begin(), nClassesPerOutput.end(),
                           size_t{0});
  }

  void moveSample(std::vector<double> &rightStats, std::vector<double> &leftStats,
                  size_t idx) override {
    const double w = static_cast<double>(sampleWeights(idx));
    for (size_t o = 0; o < nOutputs_; ++o) {
      const size_t pos = classOffsets_[o] + labels(o, idx);
      leftStats[pos] += w;
      rightStats[pos] -= w;
    }
  }
};
