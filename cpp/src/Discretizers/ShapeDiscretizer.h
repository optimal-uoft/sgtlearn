#pragma once

/**
 * @file Discretizers/ShapeDiscretizer.h
 * @brief Task-agnostic contract for inner univariate discretizers.
 */

#include <cstddef>
#include <vector>

/**
 * Post-training view shared by classification and regression discretizers.
 *
 * After ``Train``, callers read thresholds, per-bin sample routing, and leaf
 * statistics through this interface regardless of the learning objective.
 */
class ShapeDiscretizer {
public:
  virtual ~ShapeDiscretizer() = default;

  virtual size_t numLeaves() const = 0;
  virtual std::vector<std::vector<double>> &leafStats() = 0;
  virtual std::vector<size_t> &leafNumSamples() = 0;
  virtual std::vector<double> &leafNodeWeights() = 0;
  virtual const std::vector<double> &thresholds() const = 0;
  virtual std::vector<std::vector<size_t>> &inSampleDiscretizations() = 0;
};
