#pragma once

/**
 * @file SplitCandidate.h
 * @brief Interval and scores for one node in a univariate ``Splitter`` search tree.
 */

#include <compare>
#include <cstddef>
#include <limits>
#include <tuple>

/** Half-open ``[start,end]`` sample indices on the sorted axis plus impurity bookkeeping for left/right children. */
struct SplitCandidate {
  size_t height{0}, start{0}, end{0};
  double score{0.0}, informationGain{0.0};
  double threshold{0.0},
      routingThreshold{std::numeric_limits<double>::infinity()};

  size_t leftStart{0}, leftEnd{0}, rightStart{0}, rightEnd{0};
  double leftScore{0.0}, rightScore{0.0};

  bool operator==(const SplitCandidate &o) const {
    return informationGain == o.informationGain;
  }

  std::weak_ordering operator<=>(const SplitCandidate &o) const {
    return std::compare_weak_order_fallback(informationGain,
                                            o.informationGain);
  }
};
