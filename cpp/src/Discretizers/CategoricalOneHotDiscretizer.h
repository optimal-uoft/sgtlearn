#pragma once

/**
 * @file Discretizers/CategoricalOneHotDiscretizer.h
 * @brief One-hot categorical inner discretizer: binary features, active branch is
 *        always terminal.
 */

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
 * Trains a specialized tree on a one-hot feature block.
 *
 * ``featureIndices`` lists row indices into ``X`` (one binary column per category).
 * Splitting on feature ``f`` sends ``X(f,:)==1`` samples to a terminal category
 * leaf; ``X(f,:)==0`` samples continue on the inactive branch.
 */
class CategoricalOneHotDiscretizer {
public:
  size_t numLeaves = 0;

  void TrainClassification(
      const arma::fmat &X, const arma::uvec &featureIndices,
      const arma::Row<size_t> &y, size_t numClasses, size_t minLeafSize,
      double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
      const arma::Row<float> &sampleWeights = arma::Row<float>());

  void TrainRegression(
      const arma::fmat &X, const arma::uvec &featureIndices,
      const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
      size_t maxDepth, size_t maxLeafNodes,
      const arma::Row<float> &sampleWeights = arma::Row<float>());

  void transform(const arma::fmat &X, arma::Row<size_t> &binLoc) const;

  /** Route values aligned with ``featureIndices`` order. */
  size_t routeToBin(const std::vector<float> &featureValues) const;

  const std::vector<std::vector<size_t>> &inSampleDiscretizations() const {
    return inSampleDiscretizations_;
  }
  const std::vector<size_t> &leafNumSamples() const { return leafNumSamples_; }
  const std::vector<double> &leafNodeWeights() const { return leafNodeWeights_; }
  const std::vector<std::vector<double>> &leafStats() const { return leafStats_; }
  const std::vector<size_t> &binPredictionsClass() const {
    return binPredictionsClass_;
  }
  const std::vector<float> &binPredictionsReg() const {
    return binPredictionsReg_;
  }
  bool isClassification() const { return isClassification_; }

  void setCriterion(OneHotDiscretizerCriterion c);

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
  std::vector<std::vector<size_t>> inSampleDiscretizations_;
  std::vector<size_t> leafNumSamples_;
  std::vector<double> leafNodeWeights_;
  std::vector<std::vector<double>> leafStats_;
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

OneHotDiscretizerCriterion
parseOneHotDiscretizerCriterion(const std::string &criterion, bool classification);
