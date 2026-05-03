#pragma once

#include "Domain/SplitCandidate.h"
#include "Splitters/Splitter.h"

#include <armadillo>
#include <map>
#include <tuple>
#include <vector>

template <typename T> class UnivariateDiscretizer {
public:
  struct TrainingContext {
    arma::frowvec sortedX;
    arma::Row<T> sortedY;
    arma::uvec sortedOrder;
    size_t feature = 0;

    TrainingContext() = default;
    TrainingContext(arma::frowvec sorted_x, arma::Row<T> sorted_y,
                    arma::uvec order, size_t feature_index)
        : sortedX(std::move(sorted_x)), sortedY(std::move(sorted_y)),
          sortedOrder(std::move(order)), feature(feature_index) {}
  };

protected:
  bool leavesProcessed = false;
  TrainingContext training_;
  std::vector<std::vector<size_t>> inSampleDiscretizations;
  std::vector<T> binPredictions;
  std::vector<double> thresholds;

  void processLeaves(
      const std::map<std::tuple<size_t, size_t>, SplitCandidate> &leaves,
      Splitter<T> &splitter);

  void Train(Splitter<T> &splitter, size_t minLeafSize, double minGainSplit,
             size_t maxDepth, size_t maxLeafNodes);

  void setTrainingContext(TrainingContext ctx) { training_ = std::move(ctx); }

public:
  UnivariateDiscretizer() = default;

  size_t numLeaves;
  void transform(const arma::fmat &X, arma::Row<T> &binLoc);

  std::vector<std::vector<T>> &getInSampleDiscretizations();

  std::vector<size_t> &getBinPredictions();

  ~UnivariateDiscretizer() = default;
};

#include "Discretizers/UnivariateDiscretizer.tpp"