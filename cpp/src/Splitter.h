#pragma once
#include "SplitCandidate.h"
#include <armadillo>
#include <array>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

template <typename T> class Splitter {
public:
  virtual ~Splitter() = default;

  virtual SplitCandidate makeRoot(const arma::Row<T> y) = 0;

  virtual double predict(const SplitCandidate &split) = 0;

  virtual double score(const std::vector<T> &stats, size_t N) = 0;

  virtual double score(const SplitCandidate &split) = 0;

  bool findBestSplit(SplitCandidate &split, const arma::frowvec &X,
                     const arma::Row<T> &y, size_t minLeafSize);

  std::array<SplitCandidate, 2> makeChildren(const SplitCandidate &parent);

protected:
  size_t statsSize;

  Splitter(size_t statsSize) : statsSize(statsSize) {}


  const std::vector<T> &getStats(const SplitCandidate &split);
  std::vector<T> makeEmptyStats();

  std::unordered_map<size_t,
                     std::unordered_map<size_t, std::vector<std::vector<T>>>>
      childrenSplitStats;
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<T>>>
      splitStats;

  virtual void moveSample(std::vector<T> &rightStats, std::vector<T> &leftStats,
                          T v) = 0;
};

#include "Splitter.tpp"