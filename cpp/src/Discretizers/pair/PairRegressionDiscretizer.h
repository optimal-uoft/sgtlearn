#pragma once

#include "Discretizers/RegressionDiscretizer.h"
#include "Discretizers/pair/PairClassificationDiscretizer.h"
#include "Domain/LearningCriterion.h"

#include <armadillo>
#include <array>
#include <cstddef>
#include <vector>

/** Ordinary axis-aligned CART over exactly two logical features. */
class PairRegressionDiscretizer final : public RegressionDiscretizer {
public:
  PairRegressionDiscretizer(LearningCriterion criterion, FeatureInfo first,
                            FeatureInfo second);

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Mat<float> &y, size_t minLeafSize,
             double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights = arma::Row<float>()) override;

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override;
  size_t routeToBin(const std::vector<float> &featureValues) const override;

  const std::vector<PairRoutingTreeNode> &routingTree() const { return tree_; }
  const std::array<FeatureInfo, 2> &axes() const { return axes_; }

private:
  struct BuildNode {
    PairRoutingTreeNode routing;
    std::vector<size_t> samples;
    std::vector<std::vector<double>> stats;
    double weight = 0.0;
    double impurity = 0.0;
    size_t depth = 0;
  };

  struct Split {
    bool found = false;
    size_t featurePosition = 0;
    size_t rawFeature = 0;
    FeatureType featureType = FeatureType::Continuous;
    double threshold = 0.0;
    double gain = 0.0;
    std::vector<size_t> left;
    std::vector<size_t> right;
    std::vector<size_t> missing;
  };

  double impurity(const std::vector<size_t> &samples,
                  const arma::Mat<float> &y,
                  const arma::Row<float> &weights) const;
  Split bestSplit(const BuildNode &node, const arma::fmat &X,
                  const arma::Mat<float> &y,
                  const arma::Row<float> &weights, size_t minLeafSize,
                  double totalWeight) const;
  size_t routeValues(const std::vector<float> &values) const;
  bool axisMissing(size_t axis, const arma::fmat &X, size_t sample) const;
  bool axisMissing(size_t axis, const std::vector<float> &values) const;
  size_t routingPosition(size_t rawFeature) const;

  LearningCriterion criterion_;
  std::array<FeatureInfo, 2> axes_;
  arma::uvec routingFeatures_;
  std::array<size_t, 3> axisOffsets_{};
  std::vector<PairRoutingTreeNode> tree_;
};
