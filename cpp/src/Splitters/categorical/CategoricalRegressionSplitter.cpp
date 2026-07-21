/**
 * @file CategoricalRegressionSplitter.cpp
 */

#include <cstddef>
#include "Splitters/categorical/CategoricalRegressionSplitter.h"

#include <stdexcept>

std::vector<float> CategoricalRegressionSplitter::ysForSamples(
    size_t output, const std::vector<size_t> &samples) const {
  std::vector<float> out;
  out.reserve(samples.size());
  for (size_t i : samples)
    out.push_back(y_(output, i));
  return out;
}

std::vector<float> CategoricalRegressionSplitter::wsForSamples(
    const std::vector<size_t> &samples) const {
  std::vector<float> out;
  out.reserve(samples.size());
  for (size_t i : samples)
    out.push_back(sampleWeights(i));
  return out;
}

double CategoricalRegressionSplitter::score(const std::vector<size_t> &samples) {
  const double wTot = totalWeight(samples);
  if (wTot <= 0.0)
    return 0.0;
  switch (criterion_) {
  case LearningCriterion::SquaredError: {
    const auto stats = statsForSamples(samples);
    return Criterion::squaredError(stats, wTot);
  }
  case LearningCriterion::AbsoluteError: {
    const auto ws = wsForSamples(samples);
    double total = 0.0;
    for (size_t o = 0; o < nOutputs_; ++o)
      total += Criterion::absoluteError(ysForSamples(o, samples), ws).mae;
    return total;
  }
  default:
    throw std::invalid_argument(
        "CategoricalRegressionSplitter requires SquaredError or AbsoluteError");
  }
}

std::vector<std::vector<double>>
CategoricalRegressionSplitter::statsForSamples(
    const std::vector<size_t> &samples) {
  switch (criterion_) {
  case LearningCriterion::SquaredError: {
    std::vector<std::vector<double>> stats(nOutputs_,
                                           std::vector<double>(2, 0.0));
    for (size_t i : samples) {
      const double wi = static_cast<double>(sampleWeights(i));
      for (size_t o = 0; o < nOutputs_; ++o) {
        const double v = static_cast<double>(y_(o, i));
        stats[o][0] += wi * v;
        stats[o][1] += wi * v * v;
      }
    }
    return stats;
  }
  case LearningCriterion::AbsoluteError:
    return std::vector<std::vector<double>>(nOutputs_);
  default:
    throw std::invalid_argument(
        "CategoricalRegressionSplitter requires SquaredError or AbsoluteError");
  }
}

std::vector<float>
CategoricalRegressionSplitter::predict(const std::vector<size_t> &samples) {
  const double wTot = totalWeight(samples);
  std::vector<float> preds(nOutputs_, 0.f);
  switch (criterion_) {
  case LearningCriterion::SquaredError: {
    const auto stats = statsForSamples(samples);
    for (size_t o = 0; o < nOutputs_; ++o)
      preds[o] = wTot > 0.0 ? static_cast<float>(stats[o][0] / wTot) : 0.f;
    return preds;
  }
  case LearningCriterion::AbsoluteError: {
    const auto ws = wsForSamples(samples);
    for (size_t o = 0; o < nOutputs_; ++o)
      preds[o] = static_cast<float>(
          Criterion::absoluteError(ysForSamples(o, samples), ws).median);
    return preds;
  }
  default:
    throw std::invalid_argument(
        "CategoricalRegressionSplitter requires SquaredError or AbsoluteError");
  }
}
