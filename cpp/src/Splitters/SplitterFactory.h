#pragma once

/**
 * @file SplitterFactory.h
 * @brief Construct a ``Splitter`` implementation from ``LearningCriterion`` and data views.
 */

#include "ClassificationSplitter.h"
#include "Splitter.h"
#include "Domain/LearningCriterion.h"

#include <armadillo>
#include <memory>

/** Classification labels (integer class ids). Throws if criterion is not Entropy or Gini. */
std::unique_ptr<ClassificationSplitter>
makeClassificationSplitter(LearningCriterion criterion, arma::frowvec &X,
                           arma::Mat<size_t> &labels, size_t numClasses);

/**
 * Regression / gradient targets (float matrix). Throws if criterion is not
 * SquaredError, GainHessian, or AbsoluteError.
 * @param gainHessianLambda used only for GainHessian (L2 leaf regularization).
 */
std::unique_ptr<Splitter<float>>
makeFloatSplitter(LearningCriterion criterion, arma::frowvec &X,
                  arma::Mat<float> &y, double gainHessianLambda = 1.0);
