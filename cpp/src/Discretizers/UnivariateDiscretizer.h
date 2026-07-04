#pragma once

/**
 * @file UnivariateDiscretizer.h
 * @brief Train a one-dimensional axis-aligned partition of the feature space and expose bins, thresholds, and leaf aggregates.
 */

#include "Domain/SplitCandidate.h"
#include "Splitters/Splitter.h"

#include <armadillo>
#include <map>
#include <tuple>
#include <vector>

/**
 * @tparam StatsT leaf sufficient-statistic element type (weighted class counts, float sums, …).
 * @tparam PredictT leaf prediction type routed to downstream code.
 */
template <typename StatsT, typename PredictT = StatsT>
class UnivariateDiscretizer {
  enum class Step { Untrained, FitTree, LeavesProcessed };
  Step step = Step::Untrained;

public:
  size_t feature;

protected:
  bool leavesProcessed = false;

  std::vector<std::vector<size_t>> inSampleDiscretizations;
  std::vector<PredictT> binPredictions;
  std::vector<double> thresholds;
  std::vector<std::vector<StatsT>> leafStats;
  std::vector<size_t> leafNumSamples;
  std::vector<double> leafNodeWeights;
  std::map<std::tuple<size_t, size_t>, SplitCandidate> leaves;

  /**
   * NaN bucket: conceptually the trailing bin (index ``numLeaves``) past the
   * ``+inf`` threshold, used to route non-finite feature values. The inner
   * tree / splitter is fit on finite values only; these aggregate the
   * non-finite training samples so callers can decide the NaN branch after
   * coordinate descent has run on the numeric bins.
   */
  bool nanSeen_ = false;
  std::vector<StatsT> nanStats_;
  size_t nanNumSamples_ = 0;
  double nanNodeWeight_ = 0.0;
  /** Training column indices (into the columns passed to ``Train``) of NaN rows. */
  std::vector<size_t> nanInSampleIndices_;

  void processLeaves(arma::uvec sortedOrder,
                     Splitter<StatsT, PredictT> &splitter);

  void buildTree(Splitter<StatsT, PredictT> &splitter, size_t minLeafSize,
                 double minGainSplit, size_t maxDepth, size_t maxLeafNodes);

public:
  UnivariateDiscretizer() = default;

  size_t numLeaves;
  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc);

  /**
   * Route one sample's feature value(s) to an inner bin (univariate: uses
   * ``featureValues[0]`` against ``thresholds``).
   */
  size_t routeToBin(const std::vector<float> &featureValues) const;

  std::vector<std::vector<size_t>> &getInSampleDiscretizations();

  std::vector<PredictT> &getBinPredictions();

  std::vector<std::vector<StatsT>> &getLeafStats();

  std::vector<size_t> &getLeafNumSamples();

  std::vector<double> &getLeafNodeWeights();

  /** Sorted-axis cut points; same ordering as `transform` / inner bins. */
  const std::vector<double> &getThresholds() const { return thresholds; }

  /** True if any non-finite feature value was observed during ``Train``. */
  bool nanSeen() const { return nanSeen_; }

  /** Aggregated NaN-bucket stats (class counts, or ``[sum w*y, sum w*y^2]``). */
  const std::vector<StatsT> &getNanStats() const { return nanStats_; }

  /** Unweighted count of non-finite training samples. */
  size_t getNanNumSamples() const { return nanNumSamples_; }

  /** Summed sample weight of the NaN bucket. */
  double getNanNodeWeight() const { return nanNodeWeight_; }

  /** Training column indices of the NaN-bucket samples. */
  const std::vector<size_t> &getNanInSampleIndices() const {
    return nanInSampleIndices_;
  }

  ~UnivariateDiscretizer() = default;
};

#include "Discretizers/UnivariateDiscretizer.tpp"
