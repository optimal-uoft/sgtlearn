#pragma once

/**
 * @file LearningCriterion.h
 * @brief Impurity / loss kind used by splitter and branch-assignment factories (classification vs regression families).
 */
enum class LearningCriterion {
  Entropy,
  Gini,
  SquaredError,
  GainHessian,
  AbsoluteError,
};
