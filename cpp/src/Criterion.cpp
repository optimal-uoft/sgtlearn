/**
 * @file Criterion.cpp
 * @brief Definitions of entropy, Gini, MSE, and gain/hessian criterion helpers.
 */

#include "Criterion.h"

#include <cmath>
#include <numeric>

double Criterion::entropy(const std::vector<double> &classCounts,
                        double totalWeight) {
  if (totalWeight <= 0.0)
    return 0;
  double ent = 0;
  for (double count : classCounts) {
    if (count <= 0.0)
      continue;
    const double p = count / totalWeight;
    ent -= p * std::log2(p);
  }
  return ent;
}
double Criterion::gini(const std::vector<double> &classCounts,
                       double totalWeight) {
  if (totalWeight <= 0.0)
    return 0.0;
  const double sumP2 =
      std::accumulate(classCounts.begin(), classCounts.end(), 0.0,
                      [totalWeight](const double acc, const double count) {
                        const double p = count / totalWeight;
                        return acc + p * p;
                      });
  return 1.0 - sumP2;
}
double Criterion::squaredError(const std::vector<float> &yPowerSum,
                               double totalWeight) {
  if (totalWeight <= 0.0)
    return 0;
  const double ySum = yPowerSum[0];
  const double ySqrdSum = yPowerSum[1];

  const double mean = ySum / totalWeight;

  return ySqrdSum - 2.0 * mean * ySum + totalWeight * mean * mean;
}
double Criterion::gainAndHessian(const std::vector<float> &derivatives,
                                 double lambda) {
  const double g = static_cast<double>(derivatives[0]);
  const double h = static_cast<double>(derivatives[1]);
  return g * g / (h + lambda);
}
