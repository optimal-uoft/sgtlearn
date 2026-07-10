#pragma once

/**
 * @file UnivariateDiscretizer.h
 * @brief Train a one-dimensional axis-aligned partition of the feature space and expose bins, thresholds, and leaf aggregates.
 */

#include "Discretizers/InnerDiscretizerBase.h"
#include "Domain/UnivariateSplitCandidate.h"
#include "Splitters/univariate/Splitter.h"

#include <armadillo>
#include <map>
#include <tuple>
#include <vector>

/** Numeric univariate cut points; not part of the generic inner discretizer contract. */
class UnivariateThresholds {
public:
  virtual ~UnivariateThresholds() = default;
  virtual const std::vector<double> &thresholds() const = 0;
};

/**
 * @tparam StatsT leaf sufficient-statistic element type (weighted class counts, float sums, …).
 * @tparam PredictT leaf prediction type routed to downstream code.
 */
template <typename StatsT, typename PredictT = StatsT>
class UnivariateDiscretizer : public virtual InnerDiscretizerBase<StatsT>,
                              public UnivariateThresholds {
  enum class Step { Untrained, FitTree, LeavesProcessed };
  Step step = Step::Untrained;

public:
  size_t feature;

protected:
  bool leavesProcessed = false;

  std::vector<PredictT> binPredictions;
  std::vector<double> thresholds_;
  std::map<std::tuple<size_t, size_t>, UnivariateSplitCandidate> leaves;

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

  void appendNanRoutingBin();

public:
  UnivariateDiscretizer() = default;

  /** Finite inner-tree leaf count (excludes the trailing NaN routing bin). */
  size_t numLeaves() const override { return this->numLeaves_; }

  /** Index of the NaN routing bin (always the trailing leaf-stats entry). */
  size_t nanBinIndex() const {
    this->ensureTrained();
    return this->leafStats_.size() - 1;
  }

  /** Total routing bins including the trailing NaN bin. */
  size_t numRoutingBins() const { return this->numLeaves_ + 1; }

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override;

  /**
   * Route one sample's feature value(s) to an inner bin (univariate: uses
   * ``featureValues[0]`` against ``thresholds``; non-finite values map to the
   * trailing NaN bin).
   */
  size_t routeToBin(const std::vector<float> &featureValues) const override;

  const std::vector<double> &thresholds() const override { return thresholds_; }

  std::vector<std::vector<size_t>> &getInSampleDiscretizations() {
    return this->inSampleDiscretizations();
  }

  std::vector<PredictT> &getBinPredictions();

  std::vector<std::vector<StatsT>> &getLeafStats() { return this->leafStats(); }

  std::vector<size_t> &getLeafNumSamples() { return this->leafNumSamples(); }

  std::vector<double> &getLeafNodeWeights() { return this->leafNodeWeights(); }

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

  ~UnivariateDiscretizer() override = default;
};

/** Empty when ``disc`` is not a numeric univariate discretizer. */
inline const std::vector<double> &
numericInnerThresholds(const InnerDiscretizerBase<double> &disc) {
  static const std::vector<double> kEmpty;
  const auto *uni = dynamic_cast<const UnivariateThresholds *>(&disc);
  return uni ? uni->thresholds() : kEmpty;
}

#include "Discretizers/univariate/UnivariateDiscretizer.tpp"
