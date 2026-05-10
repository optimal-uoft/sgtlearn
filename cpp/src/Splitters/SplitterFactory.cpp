/**
 * @file SplitterFactory.cpp
 * @brief ``makeClassificationSplitter`` / ``makeFloatSplitter`` implementations.
 */

#include "SplitterFactory.h"

#include "AbsoluteErrorSplitter.h"
#include "EntropySplitter.h"
#include "GainHessianSplitter.h"
#include "GiniSplitter.h"
#include "SquaredErrorSplitter.h"

#include <stdexcept>

std::unique_ptr<ClassificationSplitter>
makeClassificationSplitter(LearningCriterion criterion, arma::frowvec &X,
                           arma::Mat<size_t> &labels, size_t numClasses) {
  switch (criterion) {
  case LearningCriterion::Entropy:
    return std::make_unique<EntropySplitter>(X, labels, numClasses);
  case LearningCriterion::Gini:
    return std::make_unique<GiniSplitter>(X, labels, numClasses);
  case LearningCriterion::SquaredError:
  case LearningCriterion::GainHessian:
  case LearningCriterion::AbsoluteError:
    throw std::invalid_argument(
        "makeClassificationSplitter: criterion is not a classification objective");
  default:
    throw std::invalid_argument("makeClassificationSplitter: unknown criterion");
  }
}

std::unique_ptr<Splitter<float>>
makeFloatSplitter(LearningCriterion criterion, arma::frowvec &X,
                  arma::Mat<float> &y, double gainHessianLambda) {
  switch (criterion) {
  case LearningCriterion::SquaredError:
    return std::make_unique<SquaredErrorSplitter>(X, y);
  case LearningCriterion::GainHessian:
    return std::make_unique<GainHessianSplitter>(X, y, gainHessianLambda);
  case LearningCriterion::AbsoluteError:
    return std::make_unique<AbsoluteErrorSplitter>(X, y);
  case LearningCriterion::Entropy:
  case LearningCriterion::Gini:
    throw std::invalid_argument(
        "makeFloatSplitter: criterion is not a float-target objective");
  default:
    throw std::invalid_argument("makeFloatSplitter: unknown criterion");
  }
}
