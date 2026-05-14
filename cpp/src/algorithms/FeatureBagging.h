#pragma once

/**
 * @file FeatureBagging.h
 * @brief Runtime and static policies for random subspaces: pick feature column indices.
 */

#include <algorithm>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <functional>
#include <random>
#include <span>
#include <stdexcept>
#include <vector>

/** Use every candidate column index (no bagging). */
[[nodiscard]] inline std::vector<size_t>
pickAllFeatureIndices(std::span<const size_t> candidateFeatureIndices,
                      std::mt19937_64 &) {
  return {candidateFeatureIndices.begin(), candidateFeatureIndices.end()};
}

/**
 * Callable bagging strategy (typically built in bindings / trainers).
 * Receives the candidate pool and the tree-owned RNG (non-const: may draw).
 */
using FeatureBaggingPickFn =
    std::function<std::vector<size_t>(std::span<const size_t>, std::mt19937_64 &)>;

/**
 * Stateless policies with a single static entry point (no CRTP, no instance).
 */
template <typename Policy>
concept FeatureBaggingPolicy =
    requires(std::span<const size_t> candidateFeatureIndices, std::mt19937_64 &rng) {
      { Policy::pickFeatureSubset(candidateFeatureIndices, rng) }
      -> std::same_as<std::vector<size_t>>;
    };

template <FeatureBaggingPolicy Policy>
[[nodiscard]] inline FeatureBaggingPickFn wrapFeatureBaggingPolicy() {
  return [](std::span<const size_t> s, std::mt19937_64 &rng) {
    return Policy::pickFeatureSubset(s, rng);
  };
}

/** ``max(1, int(sqrt(n)))`` with sklearn-style truncation. */
[[nodiscard]] inline size_t featureSubsetCountSqrt(size_t n) {
  if (n == 0)
    return 0;
  return std::max<size_t>(
      1, static_cast<size_t>(std::floor(std::sqrt(static_cast<double>(n)))));
}

/** ``max(1, int(log2(n)))`` with sklearn-style truncation. */
[[nodiscard]] inline size_t featureSubsetCountLog2(size_t n) {
  if (n == 0)
    return 0;
  return std::max<size_t>(
      1, static_cast<size_t>(std::floor(std::log2(static_cast<double>(n)))));
}

/**
 * ``max(1, int(c * n))`` capped at ``n``; ``c`` must lie in ``(0, 1]``.
 * Matches ``max(1, int(max_features * n_features))`` for a float ``max_features``.
 */
[[nodiscard]] inline size_t featureSubsetCountFraction(double c, size_t n) {
  if (n == 0)
    return 0;
  if (!(c > 0.0) || c > 1.0)
    throw std::invalid_argument(
        "feature bagging fraction must satisfy 0 < c <= 1");
  const size_t k = static_cast<size_t>(std::floor(c * static_cast<double>(n)));
  return std::max<size_t>(1, std::min(n, k));
}

/** Shuffle and keep at most ``k`` distinct indices (``k`` already clamped to ``[1, n]``). */
[[nodiscard]] inline std::vector<size_t>
pickRandomSubsetOfSize(std::span<const size_t> candidates, std::mt19937_64 &rng,
                       size_t k) {
  if (candidates.empty())
    return {};
  k = std::min(k, candidates.size());
  k = std::max<size_t>(1, k);
  if (k >= candidates.size())
    return {candidates.begin(), candidates.end()};
  std::vector<size_t> pool(candidates.begin(), candidates.end());
  std::shuffle(pool.begin(), pool.end(), rng);
  pool.resize(k);
  return pool;
}

[[nodiscard]] inline FeatureBaggingPickFn makeFeatureBaggingAll() {
  return FeatureBaggingPickFn(pickAllFeatureIndices);
}

/** At most ``cap`` features chosen uniformly without replacement (``cap >= 1``). */
[[nodiscard]] inline FeatureBaggingPickFn makeFeatureBaggingCap(size_t cap) {
  if (cap == 0)
    return makeFeatureBaggingAll();
  return [cap](std::span<const size_t> candidates,
               std::mt19937_64 &rng) -> std::vector<size_t> {
    const size_t n = candidates.size();
    if (n == 0)
      return {};
    const size_t k = std::min(cap, n);
    return pickRandomSubsetOfSize(candidates, rng, std::max<size_t>(1, k));
  };
}

[[nodiscard]] inline FeatureBaggingPickFn makeFeatureBaggingSqrt() {
  return [](std::span<const size_t> candidates,
            std::mt19937_64 &rng) -> std::vector<size_t> {
    const size_t n = candidates.size();
    if (n == 0)
      return {};
    return pickRandomSubsetOfSize(candidates, rng, featureSubsetCountSqrt(n));
  };
}

[[nodiscard]] inline FeatureBaggingPickFn makeFeatureBaggingLog2() {
  return [](std::span<const size_t> candidates,
            std::mt19937_64 &rng) -> std::vector<size_t> {
    const size_t n = candidates.size();
    if (n == 0)
      return {};
    return pickRandomSubsetOfSize(candidates, rng, featureSubsetCountLog2(n));
  };
}

[[nodiscard]] inline FeatureBaggingPickFn makeFeatureBaggingFraction(double c) {
  return [c](std::span<const size_t> candidates,
             std::mt19937_64 &rng) -> std::vector<size_t> {
    const size_t n = candidates.size();
    if (n == 0)
      return {};
    return pickRandomSubsetOfSize(candidates, rng,
                                  featureSubsetCountFraction(c, n));
  };
}

// ---- Static policy types (for ``FeatureBaggingPolicy`` + ``wrapFeatureBaggingPolicy``) ----

struct AllFeaturesPolicy {
  static std::vector<size_t> pickFeatureSubset(std::span<const size_t> c,
                                              std::mt19937_64 &rng) {
    return pickAllFeatureIndices(c, rng);
  }
};

struct SqrtFeaturesPolicy {
  static std::vector<size_t> pickFeatureSubset(std::span<const size_t> c,
                                              std::mt19937_64 &rng) {
    const size_t n = c.size();
    if (n == 0)
      return {};
    return pickRandomSubsetOfSize(c, rng, featureSubsetCountSqrt(n));
  }
};

struct Log2FeaturesPolicy {
  static std::vector<size_t> pickFeatureSubset(std::span<const size_t> c,
                                             std::mt19937_64 &rng) {
    const size_t n = c.size();
    if (n == 0)
      return {};
    return pickRandomSubsetOfSize(c, rng, featureSubsetCountLog2(n));
  }
};
