#pragma once

/**
 * @file CategoricalDiscretizer.h
 * @brief Train a one-hot categorical partition and expose bins, routing, and leaf aggregates.
 */

#include "Discretizers/InnerDiscretizerBase.h"

#include <armadillo>
#include <cstddef>
#include <vector>

template <typename StatsT, typename PredictT>
class CategoricalSplitter;

/**
 * @tparam StatsT per-leaf sufficient-statistic element type.
 * @tparam PredictT leaf prediction type routed to downstream code.
 */
template <typename StatsT, typename PredictT = StatsT>
class CategoricalDiscretizer : public virtual InnerDiscretizerBase<StatsT> {
  enum class Step { Untrained, FitTree, LeavesProcessed };
  Step step = Step::Untrained;

public:
  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override;

  size_t routeToBin(const std::vector<float> &featureValues) const override;

  const std::vector<PredictT> &getBinPredictions() const {
    this->ensureTrained();
    return binPredictions_;
  }

  /** For each routing bin, column indices whose one-hot level routes there. */
  std::vector<std::vector<size_t>> categoriesPerBin() const;

  /** Index of the trailing NaN / catch-all routing bin. */
  size_t nanBinIndex() const {
    this->ensureTrained();
    return this->leafStats_.size() - 1;
  }

protected:
  std::vector<size_t> featureIndices_;
  std::vector<PredictT> binPredictions_;

  struct RoutingNode {
    bool isLeaf = true;
    size_t leafBin = 0;
    size_t splitFeature = 0;
    size_t inactiveChild = 0;
    size_t activeLeafBin = 0;
  };

  struct LeafRecord {
    std::vector<size_t> samples;
    size_t categoryFeature = SIZE_MAX;
  };

  std::vector<RoutingNode> routing_;
  std::vector<LeafRecord> leaves_;

  void buildTree(const arma::fmat &X,
                 CategoricalSplitter<StatsT, PredictT> &splitter,
                 size_t minLeafSize, double minGainSplit, size_t maxDepth,
                 size_t maxLeafNodes);

  void processLeaves(CategoricalSplitter<StatsT, PredictT> &splitter);

  void appendNanRoutingBin();

  size_t routeOne(const arma::fmat &X, arma::uword col) const;

  static bool isActive(float v) { return v >= 0.5f; }

private:
  size_t appendLeaf(const std::vector<size_t> &samples, size_t categoryFeature);

  void finalizeNodeAsLeaf(const arma::fmat &X, size_t nodeId,
                          const std::vector<size_t> &samples,
                          const std::vector<size_t> &availableCategories);

  size_t dominantActiveCategory(const arma::fmat &X,
                                const std::vector<size_t> &samples,
                                const std::vector<size_t> &availableCategories) const;
};

#include "Discretizers/CategoricalDiscretizer.tpp"
