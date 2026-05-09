#pragma once
#include "../Domain/SplitCandidate.h"

#include <armadillo>
#include <array>
#include <unordered_map>
#include <vector>

template <typename T> class Splitter {
public:
  virtual ~Splitter() = default;

  virtual SplitCandidate makeRoot() = 0;

  virtual T predict(const SplitCandidate &split) = 0;

  virtual double score(const std::vector<T> &stats, size_t l, size_t r) = 0;

  virtual double score(const SplitCandidate &split) = 0;

  virtual bool findBestSplit(SplitCandidate &split, size_t minLeafSize);

  std::vector<SplitCandidate> makeChildren(const SplitCandidate &parent);

protected:
  const arma::Mat<T> &y;
  const arma::frowvec &X;
  size_t statsSize;

  Splitter(arma::frowvec &X, arma::Mat<T> &y, size_t statsSize)
      : X(X), y(y), statsSize(statsSize) {}

  virtual const std::vector<T> &getStats(const SplitCandidate &split);
  std::vector<T> makeEmptyStats();

  std::unordered_map<size_t,
                     std::unordered_map<size_t, std::vector<std::vector<T>>>>
      childrenSplitStats;
  std::unordered_map<size_t, std::unordered_map<size_t, std::vector<T>>>
      splitStats;

  virtual void moveSample(std::vector<T> &rightStats, std::vector<T> &leftStats,
                          size_t idx) = 0;
};

#include "Splitter.tpp"