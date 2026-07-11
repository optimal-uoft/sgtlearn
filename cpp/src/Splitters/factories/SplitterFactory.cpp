/**
 * @file SplitterFactory.cpp
 * @brief ``makeClassificationSplitter`` / ``makeFloatSplitter`` implementations.
 */

#include <memory>
#include <cstddef>
#include "Splitters/factories/SplitterFactory.h"

#include "Splitters/univariate/AbsoluteErrorSplitter.h"
#include "Splitters/univariate/EntropySplitter.h"
#include "Splitters/univariate/GainHessianSplitter.h"
#include "Splitters/univariate/GiniSplitter.h"
#include "Splitters/univariate/SquaredErrorSplitter.h"

#include <stdexcept>

std::unique_ptr<ClassificationSplitter>
makeClassificationSplitter(LearningCriterion criterion, arma::frowvec &X,
                           arma::frowvec &sampleWeights,
                           arma::Mat<size_t> &labels, size_t numClasses) {
  switch (criterion) {
  case LearningCriterion::Entropy:
    return std::make_unique<EntropySplitter>(X, sampleWeights, labels,
                                             numClasses);
  case LearningCriterion::Gini:
    return std::make_unique<GiniSplitter>(X, sampleWeights, labels, numClasses);
  case LearningCriterion::SquaredError:
  case LearningCriterion::GainHessian:
  case LearningCriterion::AbsoluteError:
    throw std::invalid_argument(
        "makeClassificationSplitter: criterion is not a classification objective");
  default:
    throw std::invalid_argument("makeClassificationSplitter: unknown criterion");
  }
}

std::unique_ptr<Splitter<float, float>>
makeFloatSplitter(LearningCriterion criterion, arma::frowvec &X,
                  arma::frowvec &sampleWeights, arma::Mat<float> &y,
                  double gainHessianLambda) {
  switch (criterion) {
  case LearningCriterion::SquaredError:
    throw std::invalid_argument(
        "makeFloatSplitter: SquaredError path uses regression discretizer directly");
  case LearningCriterion::GainHessian:
    return std::make_unique<GainHessianSplitter>(X, sampleWeights, y,
                                                 gainHessianLambda);
  case LearningCriterion::AbsoluteError:
    throw std::invalid_argument(
        "makeFloatSplitter: AbsoluteError path uses regression discretizer directly");
  case LearningCriterion::Entropy:
  case LearningCriterion::Gini:
    throw std::invalid_argument(
        "makeFloatSplitter: criterion is not a float-target objective");
  default:
    throw std::invalid_argument("makeFloatSplitter: unknown criterion");
  }
}
