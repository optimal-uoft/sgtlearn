/**
 * @file Discretizers/CategoricalOneHotDiscretizer.cpp
 */

#include "Discretizers/CategoricalOneHotDiscretizer.h"

#include "Splitters/CategoricalClassificationSplitter.h"
#include "Splitters/CategoricalRegressionSplitter.h"

#include <cctype>
#include <stdexcept>
#include <string>

namespace {

CategoricalClassificationCriterion
classificationCriterion(OneHotDiscretizerCriterion c) {
  if (c == OneHotDiscretizerCriterion::Gini)
    return CategoricalClassificationCriterion::Gini;
  if (c == OneHotDiscretizerCriterion::Entropy)
    return CategoricalClassificationCriterion::Entropy;
  throw std::invalid_argument("invalid classification one-hot criterion");
}

CategoricalRegressionCriterion regressionCriterion(OneHotDiscretizerCriterion c) {
  if (c == OneHotDiscretizerCriterion::SquaredError)
    return CategoricalRegressionCriterion::SquaredError;
  if (c == OneHotDiscretizerCriterion::AbsoluteError)
    return CategoricalRegressionCriterion::AbsoluteError;
  throw std::invalid_argument("invalid regression one-hot criterion");
}

arma::Row<float> normalizedSampleWeights(const arma::fmat &X,
                                         const arma::Row<float> &sampleWeights) {
  if (sampleWeights.n_elem == 0) {
    arma::Row<float> w(X.n_cols);
    w.ones();
    return w;
  }
  if (sampleWeights.n_elem != X.n_cols)
    throw std::invalid_argument("sample_weights length must match X.n_cols");
  return sampleWeights;
}

void validateFeatureIndices(const arma::fmat &X, const arma::uvec &featureIndices) {
  for (size_t f : featureIndices) {
    if (f >= X.n_rows)
      throw std::invalid_argument("feature index out of range for X");
  }
}

} // namespace

OneHotDiscretizerCriterion
parseOneHotDiscretizerCriterion(const std::string &raw, bool classification) {
  std::string s = raw;
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (classification) {
    if (s == "gini")
      return OneHotDiscretizerCriterion::Gini;
    if (s == "entropy" || s == "log_loss")
      return OneHotDiscretizerCriterion::Entropy;
    throw std::invalid_argument(
        "classification criterion must be 'gini' or 'entropy'");
  }
  if (s == "squared_error" || s == "mse")
    return OneHotDiscretizerCriterion::SquaredError;
  if (s == "absolute_error" || s == "mae")
    return OneHotDiscretizerCriterion::AbsoluteError;
  throw std::invalid_argument(
      "regression criterion must be 'squared_error'/'mse' or "
      "'absolute_error'/'mae'");
}

void CategoricalClassificationDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<size_t> &y,
    size_t numClasses, size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes, const arma::Row<float> &sampleWeights) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  if (numClasses < 2)
    throw std::invalid_argument("numClasses must be >= 2");
  validateFeatureIndices(X, features);
  featureIndices_.assign(features.begin(), features.end());
  const arma::Row<float> w = normalizedSampleWeights(X, sampleWeights);

  CategoricalClassificationSplitter splitter(
      X, w, y, numClasses, featureIndices_,
      classificationCriterion(criterion_));
  buildTree(X, splitter, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
  processLeaves(splitter);
}

void CategoricalRegressionDiscretizer::Train(
    const arma::fmat &X, arma::uvec &features, const arma::Row<float> &y,
    size_t minLeafSize, double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  if (y.n_elem != X.n_cols)
    throw std::invalid_argument("y length must equal X.n_cols");
  validateFeatureIndices(X, features);
  featureIndices_.assign(features.begin(), features.end());
  const arma::Row<float> w = normalizedSampleWeights(X, sampleWeights);

  CategoricalRegressionSplitter splitter(X, w, y, featureIndices_,
                                         regressionCriterion(criterion_));
  buildTree(X, splitter, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
  processLeaves(splitter);
}

void CategoricalOneHotDiscretizer::TrainClassification(
    const arma::fmat &X, const arma::uvec &featureIndices,
    const arma::Row<size_t> &y, size_t numClasses, size_t minLeafSize,
    double minGainSplit, size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  isClassification_ = true;
  CategoricalClassificationDiscretizer disc(criterion_);
  arma::uvec features = featureIndices;
  disc.Train(X, features, y, numClasses, minLeafSize, minGainSplit, maxDepth,
             maxLeafNodes, sampleWeights);
  impl_ = std::move(disc);
}

void CategoricalOneHotDiscretizer::TrainRegression(
    const arma::fmat &X, const arma::uvec &featureIndices,
    const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
    size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  isClassification_ = false;
  CategoricalRegressionDiscretizer disc(criterion_);
  arma::uvec features = featureIndices;
  disc.Train(X, features, y, minLeafSize, minGainSplit, maxDepth, maxLeafNodes,
             sampleWeights);
  impl_ = std::move(disc);
}
