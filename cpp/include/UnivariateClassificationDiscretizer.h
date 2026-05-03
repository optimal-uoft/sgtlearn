#pragma once
#include "Discretizers/UnivariateDiscretizer.h"
#include "Splitters/ClassificationSplitter.h"
#include "Splitters/EntropySplitter.h"
#include "Splitters/GiniSplitter.h"

#include <armadillo>

template <typename T>
concept TClassificationSplitter =
    std::derived_from<T, ClassificationSplitter> &&
    std::constructible_from<T, arma::frowvec &, arma::Row<size_t> &, size_t>;

template <TClassificationSplitter Tsplitter = GiniSplitter>
class UnivariateClassificationDiscretizer
    : public UnivariateDiscretizer<size_t> {
public:
  ~UnivariateClassificationDiscretizer() = default;

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<size_t> &y, size_t numClasses,
             size_t minLeafSize = 1, double minGainSplit = 1e-7,
             size_t maxDepth = 0, size_t maxLeafNodes = 0);
};
