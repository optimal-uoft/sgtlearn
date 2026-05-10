#pragma once

/** Shared routing key for splitter and branch-assignment factories. */
enum class LearningCriterion {
  Entropy,
  Gini,
  SquaredError,
  GainHessian,
  AbsoluteError,
};
