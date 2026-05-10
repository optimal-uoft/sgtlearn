#include "Criterion.h"

#include <cmath>
#include <numeric>

double Criterion::entropy(const std::vector<size_t> &classCounts, size_t N) {
  if (N == 0)
    return 0;
  double ent = 0;
  for (size_t count : classCounts) {
    if (count == 0)
      continue;
    const double p = static_cast<double>(count) / static_cast<double>(N);
    ent -= p * std::log2(p);
  }
  return ent;
}
double Criterion::gini(const std::vector<size_t> &classCounts, size_t N) {
  if (N == 0)
    return 0.0;
  const double sumP2 =
      std::accumulate(classCounts.begin(), classCounts.end(), 0.0,
                      [N](const double acc, const size_t count) {
                        const double p = static_cast<double>(count) / N;
                        return acc + p * p;
                      });
  return 1.0 - sumP2;
}
double Criterion::squaredError(const std::vector<float> &yPowerSum, size_t N) {
  if (N == 0)
    return 0;
  double ySum = yPowerSum[0], ySqrdSum = yPowerSum[1];

  double mean = ySum / N;

  return ySqrdSum - 2 * mean * ySum + N * mean * mean;
}
double Criterion::gainAndHessian(const std::vector<float> &derivatives,
                                 double lambda) {
  const double g = static_cast<double>(derivatives[0]);
  const double h = static_cast<double>(derivatives[1]);
  return g * g / (h + lambda);
}
