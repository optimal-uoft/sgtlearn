#pragma once

/**
 * @file InnerDiscretizerBase.h
 * @brief Shared trained-state and leaf-output contract for inner discretizers.
 */

#include <armadillo>
#include <cstddef>
#include <stdexcept>
#include <vector>

enum class DiscretizerTrainingState { Untrained, Trained };

/**
 * Non-templated routing / leaf-metadata contract shared by classification and
 * regression inner discretizers (different ``StatsT``).
 */
class InnerDiscretizerBase {
public:
  virtual ~InnerDiscretizerBase() = default;

  virtual size_t numLeaves() const = 0;
  virtual bool isTrained() const = 0;
  virtual void transform(const arma::fmat &X,
                         arma::Row<size_t> &binLoc) const = 0;
  virtual size_t routeToBin(const std::vector<float> &featureValues) const = 0;

  virtual const std::vector<std::vector<size_t>> &
  inSampleDiscretizations() const = 0;
  virtual std::vector<std::vector<size_t>> &inSampleDiscretizations() = 0;

  virtual const std::vector<size_t> &leafNumSamples() const = 0;
  virtual std::vector<size_t> &leafNumSamples() = 0;

  virtual const std::vector<double> &leafNodeWeights() const = 0;
  virtual std::vector<double> &leafNodeWeights() = 0;
};

/**
 * Common leaf routing outputs and accessors for inner discretizers.
 *
 * @tparam StatsT per-leaf sufficient-statistic element type.
 */
template <typename StatsT = double>
class InnerDiscretizer : public InnerDiscretizerBase {
protected:
  DiscretizerTrainingState trainingState_ = DiscretizerTrainingState::Untrained;

  std::vector<std::vector<size_t>> inSampleDiscretizations_;
  std::vector<size_t> leafNumSamples_;
  std::vector<double> leafNodeWeights_;
  std::vector<std::vector<StatsT>> leafStats_;

  void markTrained() { trainingState_ = DiscretizerTrainingState::Trained; }

  void resetTrainedOutputs() {
    trainingState_ = DiscretizerTrainingState::Untrained;
    inSampleDiscretizations_.clear();
    leafNumSamples_.clear();
    leafNodeWeights_.clear();
    leafStats_.clear();
    numLeaves_ = 0;
  }

  void ensureTrained() const {
    if (trainingState_ != DiscretizerTrainingState::Trained)
      throw std::runtime_error(
          "Cannot use discretizer outputs without first training it");
  }

  size_t numLeaves_ = 0;

public:
  size_t numLeaves() const override { return numLeaves_; }

  void setNumLeaves(size_t count) { numLeaves_ = count; }

  bool isTrained() const override {
    return trainingState_ == DiscretizerTrainingState::Trained;
  }

  void transform(const arma::fmat &X,
                 arma::Row<size_t> &binLoc) const override = 0;

  size_t routeToBin(const std::vector<float> &featureValues) const override = 0;

  const std::vector<std::vector<size_t>> &inSampleDiscretizations() const override {
    ensureTrained();
    return inSampleDiscretizations_;
  }

  std::vector<std::vector<size_t>> &inSampleDiscretizations() override {
    ensureTrained();
    return inSampleDiscretizations_;
  }

  const std::vector<size_t> &leafNumSamples() const override {
    ensureTrained();
    return leafNumSamples_;
  }

  std::vector<size_t> &leafNumSamples() override {
    ensureTrained();
    return leafNumSamples_;
  }

  const std::vector<double> &leafNodeWeights() const override {
    ensureTrained();
    return leafNodeWeights_;
  }

  std::vector<double> &leafNodeWeights() override {
    ensureTrained();
    return leafNodeWeights_;
  }

  const std::vector<std::vector<StatsT>> &leafStats() const {
    ensureTrained();
    return leafStats_;
  }

  std::vector<std::vector<StatsT>> &leafStats() {
    ensureTrained();
    return leafStats_;
  }

  ~InnerDiscretizer() override = default;
};
