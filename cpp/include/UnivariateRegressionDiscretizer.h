#pragma once

#include "Discretizers/UnivariateDiscretizer.h"
#include "Splitters/Splitter.h"
#include "Splitters/SquaredErrorSplitter.h"

#include <armadillo>

template <typename T>
concept TRegressionSplitter =
    std::derived_from<T, Splitter<float>> &&
    std::constructible_from<T, arma::frowvec &, arma::Mat<float> &>;

template <TRegressionSplitter = SquaredErrorSplitter>
class UnivariateRegressionDiscretizer : public UnivariateDiscretizer<float> {
public:
  ~UnivariateRegressionDiscretizer() = default;

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<float> &y, size_t minLeafSize = 1,
             double minGainSplit = 1e-7, size_t maxDepth = 0,
             size_t maxLeafNodes = 0);
};

