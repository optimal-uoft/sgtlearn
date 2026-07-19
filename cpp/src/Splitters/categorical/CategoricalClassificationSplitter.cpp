/**
 * @file CategoricalClassificationSplitter.cpp
 */

#include <cstddef>
#include "Splitters/categorical/CategoricalClassificationSplitter.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>

std::vector<double> CategoricalClassificationSplitter::classCounts(
    const std::vector<size_t> &samples) const {
  std::vector<double> counts(totalClasses_, 0.0);
  for (size_t i : samples) {
    const double w = static_cast<double>(sampleWeights(i));
    for (size_t o = 0; o < nOutputs_; ++o)
      counts[classOffsets_[o] + y_(o, i)] += w;
  }
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
    return Criterion::giniMulti(counts, classesPerOutput_);
  case LearningCriterion::Entropy:
    return Criterion::entropyMulti(counts, classesPerOutput_);
  default:
    throw std::invalid_argument(
        "CategoricalClassificationSplitter requires Gini or Entropy");
  }
}

std::vector<double> CategoricalClassificationSplitter::statsForSamples(
    const std::vector<size_t> &samples) {
  return classCounts(samples);
}

std::vector<size_t> CategoricalClassificationSplitter::predict(
    const std::vector<size_t> &samples) {
  const auto counts = classCounts(samples);
  std::vector<size_t> preds(nOutputs_, 0);
  for (size_t o = 0; o < nOutputs_; ++o) {
    const size_t off = classOffsets_[o];
    const size_t nc = classesPerOutput_[o];
    auto it = std::max_element(
        counts.begin() + static_cast<std::ptrdiff_t>(off),
        counts.begin() + static_cast<std::ptrdiff_t>(off + nc));
    preds[o] = static_cast<size_t>(
        std::distance(counts.begin() + static_cast<std::ptrdiff_t>(off), it));
  }
  return preds;
}
