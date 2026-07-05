#pragma once

/**
 * @file DiscretizerFactories.h
 * @brief Runtime factories for task-scoped inner discretizers.
 */

#include "Discretizers/DiscretizerInputKind.h"
#include "Domain/LearningCriterion.h"

#include <memory>

class ClassificationDiscretizer;
class RegressionDiscretizer;

/** ``criterion`` selects the impurity/loss splitter; ``inputKind`` selects the algorithm family. */
std::unique_ptr<ClassificationDiscretizer>
makeClassificationDiscretizer(LearningCriterion criterion,
                              DiscretizerInputKind inputKind);

/** ``criterion`` selects the regression splitter; ``inputKind`` selects the algorithm family. */
std::unique_ptr<RegressionDiscretizer>
makeRegressionDiscretizer(LearningCriterion criterion,
                          DiscretizerInputKind inputKind);
