/**
 * @file Criterion.cpp
 * @brief Definitions of entropy, Gini, MSE, and gain/hessian criterion helpers.
 */

#include <cstddef>
#include "Criterion.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>
#include <vector>

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
double Criterion::squaredError(const std::vector<double> &yPowerSum,
                               double totalWeight) {
  if (totalWeight <= 0.0)
    return 0;
  const double ySum = yPowerSum[0];
  const double ySqrdSum = yPowerSum[1];

  const double mean = ySum / totalWeight;

  // Weighted mean squared error: (1 / sum_w) * sum_i w_i (y_i - mean_w)^2
  const double sse =
      ySqrdSum - 2.0 * mean * ySum + totalWeight * mean * mean;
  return sse / totalWeight;
}

Criterion::AbsoluteErrorStats
Criterion::absoluteError(const std::vector<float> &ys,
                         const std::vector<float> &weights) {
  AbsoluteErrorStats out;
  const size_t n = ys.size();
  if (n == 0 || n != weights.size())
    return out;

  std::vector<std::pair<double, double>> pairs;
  pairs.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    const double w = static_cast<double>(weights[i]);
    if (w < 0.0)
      return out;
    pairs.emplace_back(static_cast<double>(ys[i]), w);
    out.totalWeight += w;
  }
  if (out.totalWeight <= 0.0)
    return out;

  std::sort(pairs.begin(), pairs.end(),
            [](const auto &a, const auto &b) { return a.first < b.first; });

  const double half = 0.5 * out.totalWeight;
  double wLeft = 0.0;
  double wyLeft = 0.0;
  double totalWy = 0.0;
  for (const auto &[y, w] : pairs)
    totalWy += w * y;

  int medianRank = static_cast<int>(pairs.size()) - 1;
  int medianPrevRank = medianRank > 0 ? medianRank - 1 : -1;
  bool found = false;
  for (size_t rank = 0; rank < pairs.size(); ++rank) {
    const double w = pairs[rank].second;
    if (wLeft + w > half) {
      medianRank = static_cast<int>(rank);
      medianPrevRank = rank > 0 ? static_cast<int>(rank - 1) : -1;
      found = true;
      break;
    }
    wLeft += w;
    wyLeft += w * pairs[rank].first;
  }
  if (!found) {
    wLeft = out.totalWeight - pairs.back().second;
    wyLeft = totalWy - pairs.back().second * pairs.back().first;
  }

  if (medianPrevRank >= 0 && std::fabs(wLeft - half) <= 1e-12) {
    out.median = 0.5 * (pairs[static_cast<size_t>(medianPrevRank)].first +
                        pairs[static_cast<size_t>(medianRank)].first);
  } else {
    out.median = pairs[static_cast<size_t>(medianRank)].first;
  }

  const double wRight = out.totalWeight - wLeft;
  const double wyRight = totalWy - wyLeft;
  const double pinball = (wyRight - out.median * wRight) +
                         (out.median * wLeft - wyLeft);
  out.mae = pinball / out.totalWeight;
  return out;
}

double Criterion::gainAndHessian(const std::vector<float> &derivatives,
                                 double lambda) {
  const double g = static_cast<double>(derivatives[0]);
  const double h = static_cast<double>(derivatives[1]);
  return g * g / (h + lambda);
}
