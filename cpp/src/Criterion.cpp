/**
 * @file Criterion.cpp
 * @brief Definitions of entropy, Gini, MSE, and gain/hessian criterion helpers.
 */

#include <cstddef>
#include "Criterion.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

double Criterion::entropy(
    const std::vector<std::vector<double>> &countsByOutput) {
  double total = 0.0;
  for (const auto &block : countsByOutput) {
    double blockWeight = 0.0;
    for (double c : block)
      blockWeight += c;
    if (blockWeight <= 0.0)
      continue;
    for (double count : block) {
      if (count <= 0.0)
        continue;
      const double p = count / blockWeight;
      total -= p * std::log2(p);
    }
  }
  return total;
}

double Criterion::gini(const std::vector<std::vector<double>> &countsByOutput) {
  double total = 0.0;
  for (const auto &block : countsByOutput) {
    double blockWeight = 0.0;
    for (double c : block)
      blockWeight += c;
    if (blockWeight <= 0.0)
      continue;
    const double sumP2 =
        std::accumulate(block.begin(), block.end(), 0.0,
                        [blockWeight](const double acc, const double count) {
                          const double p = count / blockWeight;
                          return acc + p * p;
                        });
    total += 1.0 - sumP2;
  }
  return total;
}

double Criterion::squaredError(
    const std::vector<std::vector<double>> &momentsByOutput,
    double totalWeight) {
  if (totalWeight <= 0.0)
    return 0.0;
  double total = 0.0;
  for (const auto &moments : momentsByOutput) {
    if (moments.size() < 2)
      continue;
    const double ySum = moments[0];
    const double ySqrdSum = moments[1];
    const double mean = ySum / totalWeight;
    const double sse =
        ySqrdSum - 2.0 * mean * ySum + totalWeight * mean * mean;
    total += sse / totalWeight;
  }
  return total;
}

Criterion::AbsoluteErrorStats
Criterion::absoluteErrorPresorted(const std::vector<float> &ys,
                                  const std::vector<float> &weights) {
  AbsoluteErrorStats out;
  const size_t n = ys.size();
  if (n == 0 || n != weights.size())
    return out;

  for (size_t i = 0; i < n; ++i) {
    const double w = static_cast<double>(weights[i]);
    if (w < 0.0)
      return AbsoluteErrorStats{};
    out.totalWeight += w;
  }
  if (out.totalWeight <= 0.0)
    return out;

  const double half = 0.5 * out.totalWeight;
  double wLeft = 0.0;
  double wyLeft = 0.0;
  double totalWy = 0.0;
  for (size_t i = 0; i < n; ++i)
    totalWy += static_cast<double>(weights[i]) * static_cast<double>(ys[i]);

  int medianRank = static_cast<int>(n) - 1;
  int medianPrevRank = medianRank > 0 ? medianRank - 1 : -1;
  bool found = false;
  for (size_t rank = 0; rank < n; ++rank) {
    const double w = static_cast<double>(weights[rank]);
    if (wLeft + w > half) {
      medianRank = static_cast<int>(rank);
      medianPrevRank = rank > 0 ? static_cast<int>(rank - 1) : -1;
      found = true;
      break;
    }
    wLeft += w;
    wyLeft += w * static_cast<double>(ys[rank]);
  }
  if (!found) {
    const double wLast = static_cast<double>(weights.back());
    wLeft = out.totalWeight - wLast;
    wyLeft = totalWy - wLast * static_cast<double>(ys.back());
  }

  if (medianPrevRank >= 0 && std::fabs(wLeft - half) <= 1e-12) {
    out.median = 0.5 * (static_cast<double>(ys[static_cast<size_t>(medianPrevRank)]) +
                        static_cast<double>(ys[static_cast<size_t>(medianRank)]));
  } else {
    out.median = static_cast<double>(ys[static_cast<size_t>(medianRank)]);
  }

  const double wRight = out.totalWeight - wLeft;
  const double wyRight = totalWy - wyLeft;
  const double pinball = (wyRight - out.median * wRight) +
                         (out.median * wLeft - wyLeft);
  out.mae = pinball / out.totalWeight;
  return out;
}

Criterion::AbsoluteErrorStats
Criterion::absoluteError(const std::vector<float> &ys,
                         const std::vector<float> &weights) {
  AbsoluteErrorStats out;
  const size_t n = ys.size();
  if (n == 0 || n != weights.size())
    return out;

  std::vector<float> ysSorted;
  std::vector<float> wsSorted;
  ysSorted.reserve(n);
  wsSorted.reserve(n);
  std::vector<size_t> order(n);
  for (size_t i = 0; i < n; ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(),
            [&ys](size_t a, size_t b) { return ys[a] < ys[b]; });
  for (size_t idx : order) {
    ysSorted.push_back(ys[idx]);
    wsSorted.push_back(weights[idx]);
  }
  return absoluteErrorPresorted(ysSorted, wsSorted);
}

double Criterion::gainAndHessian(const std::vector<float> &derivatives,
                                 double lambda) {
  const double g = static_cast<double>(derivatives[0]);
  const double h = static_cast<double>(derivatives[1]);
  return g * g / (h + lambda);
}
