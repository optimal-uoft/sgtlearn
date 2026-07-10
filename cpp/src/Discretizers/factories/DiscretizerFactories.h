#pragma once

/**
 * @file DiscretizerFactories.h
 * @brief Runtime factories for task-scoped inner discretizers.
 */

#include <cstddef>
#include "Discretizers/DiscretizerInputKind.h"
#include "Domain/FeatureInfo.h"
#include "Domain/LearningCriterion.h"

#include <armadillo>
#include <memory>

class ClassificationDiscretizer;
class RegressionDiscretizer;

/** ``criterion`` selects the impurity/loss splitter; ``inputKind`` selects the algorithm family. */
std::unique_ptr<ClassificationDiscretizer>
makeClassificationDiscretizer(LearningCriterion criterion,
                              DiscretizerInputKind inputKind);

/** ``criterion`` and ``feature.type`` select the discretizer implementation. */
std::unique_ptr<ClassificationDiscretizer>
makeClassificationDiscretizer(LearningCriterion criterion,
                              const FeatureInfo &feature);

void trainClassificationDiscretizer(
    ClassificationDiscretizer &disc, const FeatureInfo &feature,
    const arma::fmat &X, const arma::Row<size_t> &y, size_t numClasses,
    size_t minLeafSize, double minGainSplit, size_t maxDepth,
    size_t maxLeafNodes, const arma::Row<float> &sampleWeights = arma::Row<float>());

/** ``criterion`` selects the regression splitter; ``inputKind`` selects the algorithm family. */
std::unique_ptr<RegressionDiscretizer>
makeRegressionDiscretizer(LearningCriterion criterion,
                          DiscretizerInputKind inputKind);

/** ``criterion`` and ``feature.type`` select the discretizer implementation. */
std::unique_ptr<RegressionDiscretizer>
makeRegressionDiscretizer(LearningCriterion criterion, const FeatureInfo &feature);

void trainRegressionDiscretizer(
    RegressionDiscretizer &disc, const FeatureInfo &feature, const arma::fmat &X,
    const arma::Row<float> &y, size_t minLeafSize, double minGainSplit,
    size_t maxDepth, size_t maxLeafNodes,
    const arma::Row<float> &sampleWeights = arma::Row<float>());
