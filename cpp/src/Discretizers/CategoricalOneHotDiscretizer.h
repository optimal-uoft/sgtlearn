#pragma once

/**
 * @file CategoricalOneHotDiscretizer.h
 * @brief One-hot categorical inner discretizers for classification and regression.
 */

#include "Discretizers/CategoricalDiscretizer.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "Discretizers/RegressionDiscretizer.h"

#include <armadillo>
#include <cstddef>
#include <string>
#include <variant>

enum class OneHotDiscretizerCriterion {
  Gini,
  Entropy,
  SquaredError,
  AbsoluteError
};

class CategoricalClassificationDiscretizer
    : public CategoricalDiscretizer<double, size_t>,
      public ClassificationDiscretizer {
public:
  explicit CategoricalClassificationDiscretizer(
      OneHotDiscretizerCriterion criterion = OneHotDiscretizerCriterion::Gini)
      : criterion_(criterion) {}

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override {
    CategoricalDiscretizer<double, size_t>::transform(X, binLoc);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const override {
    return CategoricalDiscretizer<double, size_t>::routeToBin(featureValues);
  }

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<size_t> &y, size_t numClasses, size_t minLeafSize,
             double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights = arma::Row<float>()) override;

private:
  OneHotDiscretizerCriterion criterion_;
};

class CategoricalRegressionDiscretizer
    : public CategoricalDiscretizer<double, float>,
      public RegressionDiscretizer {
public:
  explicit CategoricalRegressionDiscretizer(
      OneHotDiscretizerCriterion criterion =
          OneHotDiscretizerCriterion::SquaredError)
      : criterion_(criterion) {}

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override {
    CategoricalDiscretizer<double, float>::transform(X, binLoc);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const override {
    return CategoricalDiscretizer<double, float>::routeToBin(featureValues);
  }

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
             size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights) override;

private:
  OneHotDiscretizerCriterion criterion_;
};

/** Backward-compatible alias for bindings and legacy call sites. */
class CategoricalOneHotDiscretizer {
public:
  void setCriterion(OneHotDiscretizerCriterion c) { criterion_ = c; }

  bool isClassification() const { return isClassification_; }

  bool isTrained() const {
    return std::visit([](const auto &d) { return d.isTrained(); }, impl_);
  }

  size_t numLeaves() const {
    return std::visit([](const auto &d) { return d.numLeaves(); }, impl_);
  }

  void setNumLeaves(size_t count) {
    std::visit([count](auto &d) { d.setNumLeaves(count); }, impl_);
  }

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const {
    std::visit([&](const auto &d) { d.transform(X, binLoc); }, impl_);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const {
    return std::visit(
        [&](const auto &d) { return d.routeToBin(featureValues); }, impl_);
  }

  const std::vector<std::vector<size_t>> &inSampleDiscretizations() const {
    return std::visit(
        [](const auto &d) -> const std::vector<std::vector<size_t>> & {
          return d.inSampleDiscretizations();
        },
        impl_);
  }

  const std::vector<size_t> &leafNumSamples() const {
    return std::visit(
        [](const auto &d) -> const std::vector<size_t> & {
          return d.leafNumSamples();
        },
        impl_);
  }

  const std::vector<double> &leafNodeWeights() const {
    return std::visit(
        [](const auto &d) -> const std::vector<double> & {
          return d.leafNodeWeights();
        },
        impl_);
  }

  const std::vector<size_t> &binPredictionsClass() const {
    return std::get<CategoricalClassificationDiscretizer>(impl_)
        .getBinPredictions();
  }

  const std::vector<float> &binPredictionsReg() const {
    return std::get<CategoricalRegressionDiscretizer>(impl_).getBinPredictions();
  }

  void TrainClassification(
      const arma::fmat &X, const arma::uvec &featureIndices,
      const arma::Row<size_t> &y, size_t numClasses, size_t minLeafSize,
      double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
      const arma::Row<float> &sampleWeights = arma::Row<float>());

  void TrainRegression(
      const arma::fmat &X, const arma::uvec &featureIndices,
      const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
      size_t maxDepth, size_t maxLeafNodes,
      const arma::Row<float> &sampleWeights = arma::Row<float>());

private:
  OneHotDiscretizerCriterion criterion_ = OneHotDiscretizerCriterion::Gini;
  bool isClassification_ = true;
  std::variant<CategoricalClassificationDiscretizer, CategoricalRegressionDiscretizer>
      impl_{CategoricalClassificationDiscretizer{criterion_}};
};

OneHotDiscretizerCriterion
parseOneHotDiscretizerCriterion(const std::string &criterion, bool classification);
