#pragma once

/**
 * @file DiscretizerInputKind.h
 * @brief How routing features are encoded for inner discretizer training.
 */
enum class DiscretizerInputKind {
  /** Single continuous column; threshold splits via ``UnivariateDiscretizer``. */
  Numeric,
  /** Block of binary one-hot columns; categorical classification/regression discretizers. */
  CategoricalOneHot,
};
