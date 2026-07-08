#pragma once

/**
 * @file CategoricalSplitCandidate.h
 * @brief Node state for one-hot categorical tree expansion via ``TreeBuilder``.
 */

#include <compare>
#include <cstddef>
#include <vector>

/**
 * One inner-tree node: a sample subset, remaining category features, and optional
 * committed split bookkeeping for the active/inactive partition.
 */
struct CategoricalSplitCandidate {
  size_t height{0};
  /** Index into the discretizer routing table. */
  size_t nodeId{0};
  std::vector<size_t> samples;
  /** Row indices into ``X`` still eligible as split candidates. */
  std::vector<size_t> availableCategoryFeatures;

  size_t numSamples{0};
  double nodeWeight{0.0};
  double score{0.0};
  double informationGain{0.0};

  size_t splitFeature{0};
  std::vector<size_t> activeSamples;
  std::vector<size_t> inactiveSamples;

  /** When true, ``TreeBuilder`` must not expand this branch (active category leaf). */
  bool isActiveLeafBranch{false};

  bool operator==(const CategoricalSplitCandidate &o) const {
    return informationGain == o.informationGain;
  }

  std::weak_ordering operator<=>(const CategoricalSplitCandidate &o) const {
    return std::compare_weak_order_fallback(informationGain,
                                            o.informationGain);
  }
};
