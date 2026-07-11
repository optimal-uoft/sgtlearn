#pragma once

/**
 * @file Splitter.h
 * @brief Abstract univariate tree splitter: maintain sufficient statistics on a sorted sample interval and search for threshold cuts.
 */

#include <cstddef>
#include "Domain/UnivariateSplitCandidate.h"

#include <armadillo>
#include <array>
#include <unordered_map>
#include <vector>

/**
 * @tparam StatsT aggregated statistic element type (e.g. weighted class count, float sum).
 * @tparam PredictT leaf prediction type (e.g. ``size_t`` class id, ``float`` target).
 */
template <typename StatsT, typename PredictT = StatsT> class Splitter {
public:
  virtual ~Splitter() = default;

  virtual UnivariateSplitCandidate makeRoot() = 0;

  virtual PredictT predict(const UnivariateSplitCandidate &split) = 0;

  virtual double score(const std::vector<StatsT> &stats, size_t l, size_t r) = 0;

  virtual double score(const UnivariateSplitCandidate &split) = 0;

  virtual bool findBestSplit(UnivariateSplitCandidate &split, size_t minLeafSize);

  std::vector<UnivariateSplitCandidate> makeChildren(const UnivariateSplitCandidate &parent);

  virtual const std::vector<StatsT> &getStats(const UnivariateSplitCandidate &split);

  double totalSampleWeight() const { return weightPrefix.empty() ? 0.0 : weightPrefix.back(); }

protected:
  const arma::frowvec &X;
  const arma::frowvec &sampleWeights;
  size_t statsSize;

  Splitter(arma::frowvec &X, arma::frowvec &sampleWeights, size_t statsSize)
      : X(X), sampleWeights(sampleWeights), statsSize(statsSize) {
    buildWeightPrefix();
  }

  double intervalWeight(size_t l, size_t r) const;
  static size_t intervalNumSamples(size_t l, size_t r);

  void fillIntervalMeta(UnivariateSplitCandidate &split) const;

  std::vector<StatsT> makeEmptyStats();

  std::unordered_map<size_t,
                     std::unordered_map<size_t, std::vector<std::vector<StatsT>>>>
      childrenSplitStats;
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<StatsT>>>
      splitStats;

  virtual void moveSample(std::vector<StatsT> &rightStats,
                          std::vector<StatsT> &leftStats, size_t idx) = 0;

private:
  std::vector<double> weightPrefix;

  void buildWeightPrefix();
};

#include "algorithms/missing_values.h"

#include "Splitter.tpp"
