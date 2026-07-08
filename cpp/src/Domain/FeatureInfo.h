#pragma once

/**
 * @file FeatureInfo.h
 * @brief Logical feature groups passed from Python into shape-generalized trees.
 */

#include "Discretizers/DiscretizerInputKind.h"

#include <armadillo>
#include <cstddef>

enum class FeatureType {
  Continuous,
  Categorical,
};

/** One logical feature: a single numeric column or a one-hot column block. */
struct FeatureInfo {
  FeatureType type = FeatureType::Continuous;
  /** Column indices into ``X`` (rows / Armadillo feature dimension). */
  arma::uvec indices;
};

inline DiscretizerInputKind discretizerInputKindFor(const FeatureInfo &feature) {
  return feature.type == FeatureType::Categorical
             ? DiscretizerInputKind::CategoricalOneHot
             : DiscretizerInputKind::Numeric;
}
