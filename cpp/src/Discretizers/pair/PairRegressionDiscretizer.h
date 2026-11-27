#pragma once

#include "Discretizers/RegressionDiscretizer.h"
#include "Discretizers/pair/PairDiscretizerBase.h"

/** Ordinary axis-aligned CART over exactly two logical features (regression). */
class PairRegressionDiscretizer final : public RegressionDiscretizer,
                                        public PairDiscretizerMixin {
public:
  PairRegressionDiscretizer(LearningCriterion criterion, FeatureInfo first,
                            FeatureInfo second);

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Mat<float> &y, size_t minLeafSize, double minGainSplit,
             size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights = arma::Row<float>()) override;

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override {
    this->ensureTrained();
    transformRows(X, binLoc);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const override {
    this->ensureTrained();
    return routeValues(featureValues);
  }

  std::vector<size_t> rootBinAssignments(size_t missingBranch = 0) const override {
    return pairRootBinAssignments(tree_, numLeaves_, missingBranch);
  }

private:
  double impurity(const std::vector<size_t> &samples, const arma::Mat<float> &y,
                  const arma::Row<float> &weights) const;
  PairSplit bestSplit(const PairBuildNode &node, const arma::fmat &X,
                      const arma::Mat<float> &y,
                      const arma::Row<float> &weights, size_t minLeafSize,
                      double totalWeight) const;
};
