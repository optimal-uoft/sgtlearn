#pragma once

/**
 * @file ClassificationDiscretizer.h
 * @brief Runtime-polymorphic (type-erased) wrapper over the splitter-templated
 *        classification discretizers, selectable by ``LearningCriterion``.
 *
 * Discretizers are organized in three tiers:
 *
 *  1. ``UnivariateDiscretizer<StatsT, PredictT>`` (template base) owns the
 *     discretization algorithm (buildTree / processLeaves / transform / NaN
 *     bucket), parameterized on the stat/prediction types.
 *  2. ``UnivariateClassificationDiscretizer<TSplitter>`` derives from tier 1 and
 *     is templated on the *splitter* (``GiniSplitter`` / ``EntropySplitter``).
 *     The splitter is chosen at **compile time**, so ``...<GiniSplitter>`` and
 *     ``...<EntropySplitter>`` are two distinct, unrelated types.
 *  3. ``ClassificationDiscretizer`` (this file) is the abstract base that erases
 *     that splitter template behind virtual methods, and
 *     ``ClassificationDiscretizerImpl<TSplitter>`` is the concrete bridge that
 *     owns a tier-2 object and forwards each virtual call to it.
 *
 * Callers (the shape-function builders) hold a
 * ``std::unique_ptr<ClassificationDiscretizer>`` produced by
 * ``makeClassificationDiscretizer`` and never name the splitter type: the
 * criterion picks the concrete instantiation at **runtime**. This keeps the
 * fast, monomorphized splitter code (templates) while letting one builder code
 * path work for any criterion (one virtual call per discretizer method).
 */

#include "Domain/LearningCriterion.h"
#include "Discretizers/ShapeDiscretizer.h"
#include "Discretizers/UnivariateClassificationDiscretizer.h"

#include <armadillo>
#include <memory>
#include <stdexcept>
#include <vector>

class ClassificationDiscretizer : public ShapeDiscretizer {
public:
  ~ClassificationDiscretizer() override = default;

  virtual void Train(const arma::fmat &X, arma::uvec &features,
                     const arma::Row<size_t> &y, size_t numClasses,
                     size_t minLeafSize, double minGainSplit, size_t maxDepth,
                     size_t maxLeafNodes,
                     const arma::Row<float> &sampleWeights = arma::Row<float>()) = 0;
};

template <typename TSplitter>
class ClassificationDiscretizerImpl final : public ClassificationDiscretizer {
public:
  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<size_t> &y, size_t numClasses,
             size_t minLeafSize, double minGainSplit, size_t maxDepth,
             size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights) override {
    impl_.Train(X, features, y, numClasses, minLeafSize, minGainSplit, maxDepth,
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
