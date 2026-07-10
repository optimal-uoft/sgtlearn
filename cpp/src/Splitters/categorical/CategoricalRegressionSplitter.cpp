/**
 * @file CategoricalRegressionSplitter.cpp
 */

#include "Splitters/categorical/CategoricalRegressionSplitter.h"

#include <stdexcept>

std::vector<float> CategoricalRegressionSplitter::ysForSamples(
    const std::vector<size_t> &samples) const {
  std::vector<float> out;
  out.reserve(samples.size());
  for (size_t i : samples)
    out.push_back(y_(i));
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
    std::vector<double> stats(2, 0.0);
    for (size_t i : samples) {
      const double v = static_cast<double>(y_(i));
      const double wi = static_cast<double>(sampleWeights(i));
      stats[0] += wi * v;
      stats[1] += wi * v * v;
    }
    return Criterion::squaredError(stats, wTot);
  }
  case LearningCriterion::AbsoluteError: {
    const auto ys = ysForSamples(samples);
    const auto ws = wsForSamples(samples);
    return Criterion::absoluteError(ys, ws).mae;
  }
  default:
    throw std::invalid_argument(
        "CategoricalRegressionSplitter requires SquaredError or AbsoluteError");
  }
}

std::vector<double>
CategoricalRegressionSplitter::statsForSamples(
    const std::vector<size_t> &samples) {
  switch (criterion_) {
  case LearningCriterion::SquaredError: {
    std::vector<double> stats(2, 0.0);
    for (size_t i : samples) {
      const double v = static_cast<double>(y_(i));
      const double wi = static_cast<double>(sampleWeights(i));
      stats[0] += wi * v;
      stats[1] += wi * v * v;
    }
    return stats;
  }
  case LearningCriterion::AbsoluteError:
    return {};
  default:
    throw std::invalid_argument(
        "CategoricalRegressionSplitter requires SquaredError or AbsoluteError");
  }
}

float CategoricalRegressionSplitter::predict(const std::vector<size_t> &samples) {
  const double wTot = totalWeight(samples);
  switch (criterion_) {
  case LearningCriterion::SquaredError: {
    const auto stats = statsForSamples(samples);
    return wTot > 0.0 ? static_cast<float>(stats[0] / wTot) : 0.f;
  }
  case LearningCriterion::AbsoluteError: {
    const auto ys = ysForSamples(samples);
    const auto ws = wsForSamples(samples);
    return static_cast<float>(Criterion::absoluteError(ys, ws).median);
  }
  default:
    throw std::invalid_argument(
        "CategoricalRegressionSplitter requires SquaredError or AbsoluteError");
  }
}
