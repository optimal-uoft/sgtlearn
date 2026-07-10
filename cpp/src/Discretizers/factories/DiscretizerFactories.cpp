/**
 * @file DiscretizerFactories.cpp
 * @brief Runtime factories for ``ClassificationDiscretizer`` and ``RegressionDiscretizer``.
 */

#include <memory>
#include <cstddef>
#include "Discretizers/factories/DiscretizerFactories.h"

#include "Discretizers/categorical/CategoricalClassificationDiscretizer.h"
#include "Discretizers/categorical/CategoricalRegressionDiscretizer.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "Discretizers/RegressionDiscretizer.h"
#include "Discretizers/univariate/UnivariateClassificationDiscretizer.h"
#include "Discretizers/univariate/UnivariateRegressionDiscretizer.h"
#include "Splitters/univariate/AbsoluteErrorSplitter.h"
#include "Splitters/univariate/EntropySplitter.h"
#include "Splitters/univariate/GiniSplitter.h"
#include "Splitters/univariate/SquaredErrorSplitter.h"

#include <stdexcept>

std::unique_ptr<ClassificationDiscretizer>
makeClassificationDiscretizer(LearningCriterion criterion,
                              DiscretizerInputKind inputKind) {
  if (inputKind == DiscretizerInputKind::CategoricalOneHot) {
    return std::make_unique<CategoricalClassificationDiscretizer>(criterion);
  }
  if (inputKind != DiscretizerInputKind::Numeric)
    throw std::invalid_argument(
        "makeClassificationDiscretizer: unsupported DiscretizerInputKind");

  if (criterion == LearningCriterion::Gini)
    return std::make_unique<UnivariateClassificationDiscretizer<GiniSplitter>>();
  if (criterion == LearningCriterion::Entropy)
    return std::make_unique<
        UnivariateClassificationDiscretizer<EntropySplitter>>();
  throw std::invalid_argument(
      "makeClassificationDiscretizer: numeric input requires Entropy or Gini");
}

std::unique_ptr<ClassificationDiscretizer>
makeClassificationDiscretizer(LearningCriterion criterion,
                              const FeatureInfo &feature) {
  return makeClassificationDiscretizer(criterion,
                                       discretizerInputKindFor(feature));
}

void trainClassificationDiscretizer(
    ClassificationDiscretizer &disc, const FeatureInfo &feature,
    const arma::fmat &X, const arma::Row<size_t> &y, size_t numClasses,
    size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes, const arma::Row<float> &sampleWeights) {
  arma::uvec feats = feature.indices;
  disc.Train(X, feats, y, numClasses, minLeafSize, minGainSplit, maxDepth,
             maxLeafNodes, sampleWeights);
}

std::unique_ptr<RegressionDiscretizer>
makeRegressionDiscretizer(LearningCriterion criterion,
                          DiscretizerInputKind inputKind) {
  if (inputKind == DiscretizerInputKind::CategoricalOneHot) {
    return std::make_unique<CategoricalRegressionDiscretizer>(criterion);
  }
  if (inputKind != DiscretizerInputKind::Numeric)
    throw std::invalid_argument(
        "makeRegressionDiscretizer: unsupported DiscretizerInputKind");

  if (criterion == LearningCriterion::SquaredError)
    return std::make_unique<
        UnivariateRegressionDiscretizer<SquaredErrorSplitter>>();
  if (criterion == LearningCriterion::AbsoluteError)
    return std::make_unique<
        UnivariateRegressionDiscretizer<AbsoluteErrorSplitter>>();
  throw std::invalid_argument(
      "makeRegressionDiscretizer: numeric input requires SquaredError or "
      "AbsoluteError");
}

std::unique_ptr<RegressionDiscretizer>
makeRegressionDiscretizer(LearningCriterion criterion,
                          const FeatureInfo &feature) {
  return makeRegressionDiscretizer(criterion, discretizerInputKindFor(feature));
}

void trainRegressionDiscretizer(
    RegressionDiscretizer &disc, const FeatureInfo &feature, const arma::fmat &X,
    const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
    size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights) {
  arma::uvec feats = feature.indices;
  disc.Train(X, feats, y, minLeafSize, minGainSplit, maxDepth, maxLeafNodes,
             sampleWeights);
}
