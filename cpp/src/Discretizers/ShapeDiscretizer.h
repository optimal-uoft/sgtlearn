#pragma once

/**
 * @file Discretizers/ShapeDiscretizer.h
 * @brief Task-agnostic contract for inner discretizers.
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

  /**
   * NaN-bucket view: aggregates over non-finite training rows for the routing
   * feature. Used to assign the NaN branch after coordinate descent has run on
   * the numeric bins.
   */
  virtual bool nanSeen() const = 0;
  virtual const std::vector<double> &nanStats() const = 0;
  virtual size_t nanNumSamples() const = 0;
  virtual double nanNodeWeight() const = 0;
  virtual const std::vector<size_t> &nanInSampleIndices() const = 0;

  /** Map finite routing feature value(s) to an inner bin index. */
  virtual size_t
  routeToBin(const std::vector<float> &featureValues) const = 0;
};
