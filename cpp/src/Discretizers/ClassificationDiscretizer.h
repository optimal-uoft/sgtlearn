#pragma once

#include "Domain/LearningCriterion.h"
#include "UnivariateClassificationDiscretizer.h"

#include <armadillo>
#include <memory>
#include <stdexcept>
#include <vector>

class ClassificationDiscretizer {
public:
  virtual ~ClassificationDiscretizer() = default;

  virtual void Train(const arma::fmat &X, arma::uvec &features,
                     const arma::Row<size_t> &y, size_t numClasses,
                     size_t minLeafSize, double minGainSplit, size_t maxDepth,
                     size_t maxLeafNodes) = 0;

  virtual size_t numLeaves() const = 0;
  virtual std::vector<std::vector<size_t>> &leafStats() = 0;
  virtual std::vector<size_t> &leafNumSamples() = 0;
  virtual const std::vector<double> &thresholds() const = 0;
  virtual std::vector<std::vector<size_t>> &inSampleDiscretizations() = 0;
};

template <typename TSplitter>
class ClassificationDiscretizerImpl final : public ClassificationDiscretizer {
public:
  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<size_t> &y, size_t numClasses,
             size_t minLeafSize, double minGainSplit, size_t maxDepth,
             size_t maxLeafNodes) override {
    impl_.Train(X, features, y, numClasses, minLeafSize, minGainSplit, maxDepth,
                maxLeafNodes);
  }

  size_t numLeaves() const override { return impl_.numLeaves; }
  std::vector<std::vector<size_t>> &leafStats() override {
    return impl_.getLeafStats();
  }
  std::vector<size_t> &leafNumSamples() override {
    return impl_.getLeafNumSamples();
  }
  const std::vector<double> &thresholds() const override {
    return impl_.getThresholds();
  }
  std::vector<std::vector<size_t>> &inSampleDiscretizations() override {
    return impl_.getInSampleDiscretizations();
  }

private:
  UnivariateClassificationDiscretizer<TSplitter> impl_;
};

inline std::unique_ptr<ClassificationDiscretizer>
makeClassificationDiscretizer(LearningCriterion criterion) {
  if (criterion == LearningCriterion::Gini) {
    return std::make_unique<ClassificationDiscretizerImpl<GiniSplitter>>();
  }
  if (criterion == LearningCriterion::Entropy) {
    return std::make_unique<ClassificationDiscretizerImpl<EntropySplitter>>();
  }
  throw std::invalid_argument(
      "makeClassificationDiscretizer: criterion must be Entropy or Gini");
}
