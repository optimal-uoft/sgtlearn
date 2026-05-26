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

  void processLeaves(arma::uvec sortedOrder,
                     Splitter<StatsT, PredictT> &splitter);

  void buildTree(Splitter<StatsT, PredictT> &splitter, size_t minLeafSize,
                 double minGainSplit, size_t maxDepth, size_t maxLeafNodes);

public:
  UnivariateDiscretizer() = default;

  size_t numLeaves;
  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc);

  std::vector<std::vector<size_t>> &getInSampleDiscretizations();

  std::vector<PredictT> &getBinPredictions();

  std::vector<std::vector<StatsT>> &getLeafStats();

  std::vector<size_t> &getLeafNumSamples();

  std::vector<double> &getLeafNodeWeights();

  /** Sorted-axis cut points; same ordering as `transform` / inner bins. */
  const std::vector<double> &getThresholds() const { return thresholds; }

  ~UnivariateDiscretizer() = default;
};

#include "Discretizers/UnivariateDiscretizer.tpp"
