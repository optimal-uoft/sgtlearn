/**
 * @file DiscretizerFactories.cpp
 * @brief Runtime factories for ``ClassificationDiscretizer`` and ``RegressionDiscretizer``.
 */

#include "Discretizers/DiscretizerFactories.h"

#include "Discretizers/CategoricalOneHotDiscretizer.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "Discretizers/RegressionDiscretizer.h"
#include "Discretizers/UnivariateClassificationDiscretizer.h"
#include "Discretizers/UnivariateRegressionDiscretizer.h"
#include "Splitters/AbsoluteErrorSplitter.h"
#include "Splitters/EntropySplitter.h"
#include "Splitters/GiniSplitter.h"
#include "Splitters/SquaredErrorSplitter.h"

#include <stdexcept>

namespace {

OneHotDiscretizerCriterion
oneHotCriterionForClassification(LearningCriterion criterion) {
  if (criterion == LearningCriterion::Gini)
    return OneHotDiscretizerCriterion::Gini;
  if (criterion == LearningCriterion::Entropy)
    return OneHotDiscretizerCriterion::Entropy;
  throw std::invalid_argument(
      "makeClassificationDiscretizer: categorical one-hot requires Entropy or "
      "Gini");
}

OneHotDiscretizerCriterion
oneHotCriterionForRegression(LearningCriterion criterion) {
  if (criterion == LearningCriterion::SquaredError)
    return OneHotDiscretizerCriterion::SquaredError;
  if (criterion == LearningCriterion::AbsoluteError)
    return OneHotDiscretizerCriterion::AbsoluteError;
  throw std::invalid_argument(
      "makeRegressionDiscretizer: categorical one-hot requires SquaredError or "
      "AbsoluteError");
}

} // namespace

std::unique_ptr<ClassificationDiscretizer>
makeClassificationDiscretizer(LearningCriterion criterion,
                              DiscretizerInputKind inputKind) {
  if (inputKind == DiscretizerInputKind::CategoricalOneHot) {
    return std::make_unique<CategoricalClassificationDiscretizer>(
        oneHotCriterionForClassification(criterion));
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

std::unique_ptr<RegressionDiscretizer>
makeRegressionDiscretizer(LearningCriterion criterion,
                          DiscretizerInputKind inputKind) {
  if (inputKind == DiscretizerInputKind::CategoricalOneHot) {
    return std::make_unique<CategoricalRegressionDiscretizer>(
        oneHotCriterionForRegression(criterion));
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
