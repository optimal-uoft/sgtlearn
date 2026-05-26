#pragma once

#include "Domain/LearningCriterion.h"
#include "Splitters/AbsoluteErrorSplitter.h"
#include "Discretizers/UnivariateRegressionDiscretizer.h"

#include <armadillo>
#include <memory>
#include <stdexcept>
#include <vector>

class RegressionDiscretizer {
public:
  virtual ~RegressionDiscretizer() = default;

  virtual void Train(const arma::fmat &X, arma::uvec &features,
                     const arma::Row<float> &y, size_t minLeafSize,
                     double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
                     const arma::Row<float> &sampleWeights) = 0;

  virtual size_t numLeaves() const = 0;
  virtual std::vector<std::vector<double>> &leafStats() = 0;
  virtual std::vector<size_t> &leafNumSamples() = 0;
  virtual std::vector<double> &leafNodeWeights() = 0;
  virtual const std::vector<double> &thresholds() const = 0;
  virtual std::vector<std::vector<size_t>> &inSampleDiscretizations() = 0;
};

template <typename TSplitter>
class RegressionDiscretizerImpl final : public RegressionDiscretizer {
public:
  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
             size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights) override {
    impl_.Train(X, features, y, minLeafSize, minGainSplit, maxDepth,
                maxLeafNodes, sampleWeights);
  }

  size_t numLeaves() const override { return impl_.numLeaves; }
  std::vector<std::vector<double>> &leafStats() override {
    return impl_.getLeafStats();
  }

  std::vector<size_t> &leafNumSamples() override {
    return impl_.getLeafNumSamples();
  }
  std::vector<double> &leafNodeWeights() override {
    return impl_.getLeafNodeWeights();
  }
  const std::vector<double> &thresholds() const override {
    return impl_.getThresholds();
  }
  std::vector<std::vector<size_t>> &inSampleDiscretizations() override {
    return impl_.getInSampleDiscretizations();
  }

private:
  UnivariateRegressionDiscretizer<TSplitter> impl_;
};

inline std::unique_ptr<RegressionDiscretizer>
makeRegressionDiscretizer(LearningCriterion criterion) {
  if (criterion == LearningCriterion::SquaredError) {
    return std::make_unique<RegressionDiscretizerImpl<SquaredErrorSplitter>>();
  }
  if (criterion == LearningCriterion::AbsoluteError) {
    return std::make_unique<RegressionDiscretizerImpl<AbsoluteErrorSplitter>>();
  }
  throw std::invalid_argument(
      "makeRegressionDiscretizer: criterion must be SquaredError or "
      "AbsoluteError");
}
