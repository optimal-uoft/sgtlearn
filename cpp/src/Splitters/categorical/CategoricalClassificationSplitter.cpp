/**
 * @file CategoricalClassificationSplitter.cpp
 */

#include <cstddef>
#include "Splitters/categorical/CategoricalClassificationSplitter.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

namespace {

size_t majorityClass(const std::vector<double> &counts) {
  return static_cast<size_t>(std::distance(
      counts.begin(), std::max_element(counts.begin(), counts.end())));
}

} // namespace

std::vector<double> CategoricalClassificationSplitter::classCounts(
    const std::vector<size_t> &samples) const {
  std::vector<double> counts(numClasses_, 0.0);
  for (size_t i : samples)
    counts[y_(i)] += static_cast<double>(sampleWeights(i));
  return counts;
}

double CategoricalClassificationSplitter::score(
    const std::vector<size_t> &samples) {
  const double wTot = totalWeight(samples);
  if (wTot <= 0.0)
    return 0.0;
  const auto counts = classCounts(samples);
  switch (criterion_) {
  case LearningCriterion::Gini:
    return Criterion::gini(counts, wTot);
  case LearningCriterion::Entropy:
    return Criterion::entropy(counts, wTot);
  default:
    throw std::invalid_argument(
        "CategoricalClassificationSplitter requires Gini or Entropy");
  }
}

std::vector<double> CategoricalClassificationSplitter::statsForSamples(
    const std::vector<size_t> &samples) {
  return classCounts(samples);
}

size_t CategoricalClassificationSplitter::predict(
    const std::vector<size_t> &samples) {
  return majorityClass(classCounts(samples));
}
