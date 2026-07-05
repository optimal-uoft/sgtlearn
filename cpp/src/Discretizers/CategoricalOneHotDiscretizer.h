#pragma once

/**
 * @file CategoricalOneHotDiscretizer.h
 * @brief One-hot categorical inner discretizers for classification and regression.
 */

#include "Discretizers/ClassificationDiscretizer.h"
#include "Discretizers/InnerDiscretizerBase.h"
#include "Discretizers/RegressionDiscretizer.h"

#include <armadillo>
#include <cstddef>
#include <string>
#include <vector>

enum class OneHotDiscretizerCriterion {
  Gini,
  Entropy,
  SquaredError,
  AbsoluteError
};

/**
 * Shared one-hot routing tree: binary feature block, active branch is terminal.
 *
 * ``featureIndices`` lists row indices into ``X`` (one binary column per category).
 */
class CategoricalOneHotDiscretizerBase : public virtual InnerDiscretizerBase<double> {
public:
  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override;

  /** Route values aligned with ``featureIndices`` order. */
  size_t routeToBin(const std::vector<float> &featureValues) const override;

  const std::vector<size_t> &binPredictionsClass() const {
    return binPredictionsClass_;
  }
  const std::vector<float> &binPredictionsReg() const {
    return binPredictionsReg_;
  }
  bool isClassification() const { return isClassification_; }

  void setCriterion(OneHotDiscretizerCriterion c);

protected:
  void trainClassification(const arma::fmat &X, const arma::uvec &featureIndices,
                           const arma::Row<size_t> &y, size_t numClasses,
                           size_t minLeafSize, double minGainSplit, size_t maxDepth,
                           size_t maxLeafNodes,
                           const arma::Row<float> &sampleWeights);

  void trainRegression(const arma::fmat &X, const arma::uvec &featureIndices,
                       const arma::Row<float> &y, size_t minLeafSize,
                       double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
                       const arma::Row<float> &sampleWeights);

private:
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

  bool isClassification_ = true;
  OneHotDiscretizerCriterion criterion_ = OneHotDiscretizerCriterion::Gini;
  size_t numClasses_ = 0;
  std::vector<size_t> featureIndices_;
  arma::Row<size_t> yClass_;
  arma::Row<float> yReg_;
  arma::Row<float> sampleWeights_;

  std::vector<RoutingNode> routing_;
  std::vector<LeafRecord> leaves_;
  std::vector<size_t> binPredictionsClass_;
  std::vector<float> binPredictionsReg_;

  void buildTree(const arma::fmat &X, size_t minLeafSize, double minGainSplit,
                 size_t maxDepth, size_t maxLeafNodes);
  double impurity(const std::vector<size_t> &samples) const;
  bool bestSplit(const arma::fmat &X, const std::vector<size_t> &samples,
                 double minGainSplit, size_t minLeafSize, double &gainOut,
                 size_t &featureOut, std::vector<size_t> &activeOut,
                 std::vector<size_t> &inactiveOut) const;
  size_t appendLeaf(const std::vector<size_t> &samples, size_t categoryFeature);
  void finalizeLeafOutputs();
  size_t routeOne(const arma::fmat &X, arma::uword col) const;
  static bool isActive(float v);
};

class CategoricalClassificationDiscretizer : public ClassificationDiscretizer,
                                             public CategoricalOneHotDiscretizerBase {
public:
  explicit CategoricalClassificationDiscretizer(
      OneHotDiscretizerCriterion criterion = OneHotDiscretizerCriterion::Gini) {
    setCriterion(criterion);
  }

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override {
    CategoricalOneHotDiscretizerBase::transform(X, binLoc);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const override {
    return CategoricalOneHotDiscretizerBase::routeToBin(featureValues);
  }

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<size_t> &y, size_t numClasses, size_t minLeafSize,
             double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights = arma::Row<float>()) override;
};

class CategoricalRegressionDiscretizer : public RegressionDiscretizer,
                                         public CategoricalOneHotDiscretizerBase {
public:
  explicit CategoricalRegressionDiscretizer(
      OneHotDiscretizerCriterion criterion =
          OneHotDiscretizerCriterion::SquaredError) {
    setCriterion(criterion);
  }

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const override {
    CategoricalOneHotDiscretizerBase::transform(X, binLoc);
  }

  size_t routeToBin(const std::vector<float> &featureValues) const override {
    return CategoricalOneHotDiscretizerBase::routeToBin(featureValues);
  }

  void Train(const arma::fmat &X, arma::uvec &features,
             const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
             size_t maxDepth, size_t maxLeafNodes,
             const arma::Row<float> &sampleWeights) override;
};

/** Backward-compatible alias for bindings and legacy call sites. */
class CategoricalOneHotDiscretizer : public CategoricalOneHotDiscretizerBase {
public:
  void TrainClassification(
      const arma::fmat &X, const arma::uvec &featureIndices,
      const arma::Row<size_t> &y, size_t numClasses, size_t minLeafSize,
      double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
      const arma::Row<float> &sampleWeights = arma::Row<float>()) {
    trainClassification(X, featureIndices, y, numClasses, minLeafSize,
                        minGainSplit, maxDepth, maxLeafNodes, sampleWeights);
  }

  void TrainRegression(
      const arma::fmat &X, const arma::uvec &featureIndices,
      const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
      size_t maxDepth, size_t maxLeafNodes,
      const arma::Row<float> &sampleWeights = arma::Row<float>()) {
    trainRegression(X, featureIndices, y, minLeafSize, minGainSplit, maxDepth,
                    maxLeafNodes, sampleWeights);
  }
};

OneHotDiscretizerCriterion
parseOneHotDiscretizerCriterion(const std::string &criterion, bool classification);
