/**
 * @file CategoricalClassificationSplitter.cpp
 */

#include "Splitters/CategoricalClassificationSplitter.h"

#include <algorithm>
#include <iterator>

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
  if (criterion_ == CategoricalClassificationCriterion::Gini)
    return Criterion::gini(counts, wTot);
  return Criterion::entropy(counts, wTot);
}

std::vector<double> CategoricalClassificationSplitter::statsForSamples(
    const std::vector<size_t> &samples) {
  return classCounts(samples);
}

size_t CategoricalClassificationSplitter::predict(
    const std::vector<size_t> &samples) {
  return majorityClass(classCounts(samples));
}
