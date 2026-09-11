#pragma once

#include "Discretizers/InnerDiscretizerBase.h"
#include "Domain/LearningCriterion.h"

#include <memory>

/** Outer-only CART stump; feasibility and loss include both missing routes. */
std::shared_ptr<InnerDiscretizer<std::vector<double>>>
makeNumericFallbackDiscretizer(
    LearningCriterion criterion, const arma::fmat &X, size_t feature,
    const arma::Mat<size_t> &y, const arma::Row<float> &weights,
    size_t minLeafSize, const std::vector<size_t> &classesPerOutput);

std::shared_ptr<InnerDiscretizer<std::vector<double>>>
makeNumericFallbackDiscretizer(
    LearningCriterion criterion, const arma::fmat &X, size_t feature,
    const arma::Mat<float> &y, const arma::Row<float> &weights,
    size_t minLeafSize);
