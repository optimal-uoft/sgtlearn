#pragma once

/**
 * @file RegressionDiscretizer.h
 * @brief Runtime-polymorphic (type-erased) wrapper over the splitter-templated
 *        regression discretizers, selectable by ``LearningCriterion``.
 *
 * Discretizers are organized in three tiers:
 *
 *  1. ``UnivariateDiscretizer<StatsT, PredictT>`` (template base) owns the
 *     discretization algorithm (buildTree / processLeaves / transform / NaN
 *     bucket), parameterized on the stat/prediction types.
 *  2. ``UnivariateRegressionDiscretizer<TSplitter>`` derives from tier 1 and is
 *     templated on the *splitter* (``SquaredErrorSplitter`` /
 *     ``AbsoluteErrorSplitter``). The splitter is chosen at **compile time**, so
 *     each instantiation is a distinct, unrelated type.
 *  3. ``RegressionDiscretizer`` (this file) is the abstract base that erases
 *     that splitter template behind virtual methods, and
 *     ``RegressionDiscretizerImpl<TSplitter>`` is the concrete bridge that owns
 *     a tier-2 object and forwards each virtual call to it.
 *
 * Callers (the shape-function builders) hold a
 * ``std::unique_ptr<RegressionDiscretizer>`` produced by
 * ``makeRegressionDiscretizer`` and never name the splitter type: the criterion
 * picks the concrete instantiation at **runtime**. This keeps the fast,
 * monomorphized splitter code (templates) while letting one builder code path
 * work for any criterion (one virtual call per discretizer method).
 */

#include "Domain/LearningCriterion.h"
#include "Discretizers/ShapeDiscretizer.h"
#include "Splitters/AbsoluteErrorSplitter.h"
#include "Discretizers/UnivariateRegressionDiscretizer.h"

#include <armadillo>
#include <memory>
#include <stdexcept>
#include <vector>

class RegressionDiscretizer : public ShapeDiscretizer {
public:
  ~RegressionDiscretizer() override = default;

  virtual void Train(const arma::fmat &X, arma::uvec &features,
                     const arma::Row<float> &y, size_t minLeafSize,
                     double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
                     const arma::Row<float> &sampleWeights) = 0;
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
  bool nanSeen() const override { return impl_.nanSeen(); }
  const std::vector<double> &nanStats() const override {
    return impl_.getNanStats();
  }
  size_t nanNumSamples() const override { return impl_.getNanNumSamples(); }
  double nanNodeWeight() const override { return impl_.getNanNodeWeight(); }
  const std::vector<size_t> &nanInSampleIndices() const override {
    return impl_.getNanInSampleIndices();
  }
  size_t routeToBin(const std::vector<float> &featureValues) const override {
    return impl_.routeToBin(featureValues);
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
