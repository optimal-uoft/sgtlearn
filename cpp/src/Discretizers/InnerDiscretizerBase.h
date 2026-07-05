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
 * Common leaf routing outputs and accessors for inner discretizers.
 *
 * @tparam StatsT per-leaf sufficient-statistic element type.
 */
template <typename StatsT = double>
class InnerDiscretizerBase {
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
  virtual size_t numLeaves() const { return numLeaves_; }

  void setNumLeaves(size_t count) { numLeaves_ = count; }

  bool isTrained() const {
    return trainingState_ == DiscretizerTrainingState::Trained;
  }

  virtual void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const = 0;

  virtual size_t
  routeToBin(const std::vector<float> &featureValues) const = 0;

  const std::vector<std::vector<size_t>> &inSampleDiscretizations() const {
    ensureTrained();
    return inSampleDiscretizations_;
  }

  std::vector<std::vector<size_t>> &inSampleDiscretizations() {
    ensureTrained();
    return inSampleDiscretizations_;
  }

  const std::vector<size_t> &leafNumSamples() const {
    ensureTrained();
    return leafNumSamples_;
  }

  std::vector<size_t> &leafNumSamples() {
    ensureTrained();
    return leafNumSamples_;
  }

  const std::vector<double> &leafNodeWeights() const {
    ensureTrained();
    return leafNodeWeights_;
  }

  std::vector<double> &leafNodeWeights() {
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

  virtual ~InnerDiscretizerBase() = default;
};
