#pragma once

/**
 * @file CategoricalSplitter.h
 * @brief One-hot categorical splitter: partition by active category without sorting.
 */

#include "Domain/CategoricalSplitCandidate.h"

#include <armadillo>
#include <cstddef>
#include <vector>

/**
 * @tparam StatsT leaf sufficient-statistic element type.
 * @tparam PredictT leaf prediction type routed to downstream code.
 */
template <typename StatsT, typename PredictT = StatsT>
class CategoricalSplitter {
public:
  virtual ~CategoricalSplitter() = default;

  CategoricalSplitCandidate makeRoot();

  virtual PredictT predict(const std::vector<size_t> &samples) = 0;

  virtual double score(const std::vector<size_t> &samples) = 0;

  virtual std::vector<StatsT> statsForSamples(
      const std::vector<size_t> &samples) = 0;

  bool findBestSplit(CategoricalSplitCandidate &node, size_t minLeafSize);

  std::vector<CategoricalSplitCandidate>
  makeChildren(const CategoricalSplitCandidate &parent);

  double totalWeight(const std::vector<size_t> &samples) const;

protected:
  const arma::fmat &X;
  const arma::Row<float> &sampleWeights;
  const std::vector<size_t> &featureIndices;

  CategoricalSplitter(const arma::fmat &X, const arma::Row<float> &sampleWeights,
                      const std::vector<size_t> &featureIndices)
      : X(X), sampleWeights(sampleWeights), featureIndices(featureIndices) {}

  static bool isActive(float v) { return v >= 0.5f; }

  void fillNodeMeta(CategoricalSplitCandidate &node) const;
};

#include "CategoricalSplitter.tpp"
