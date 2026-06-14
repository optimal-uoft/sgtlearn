#pragma once

/**
 * @file NanPartitionRouting.h
 * @brief Choose the outer partition for non-finite routing-feature values.
 */

#include "Criterion.h"
#include "algorithms/missing_values.h"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

namespace nan_partition_routing {

inline std::vector<size_t>
missing_column_indices(const arma::frowvec &featureRow) {
  std::vector<size_t> cols;
  for (arma::uword col = 0; col < featureRow.n_elem; ++col) {
    if (!missing_values::is_finite(featureRow(col)))
      cols.push_back(static_cast<size_t>(col));
  }
  return cols;
}

inline size_t choose_nan_partition_squared_error(
    size_t numPartitions, const std::vector<size_t> &binToPartition,
    const std::vector<std::vector<double>> &binStats,
    const std::vector<double> &binWeights,
    const std::vector<size_t> &missingCols, const arma::Row<float> &ysub,
    const arma::Row<float> &wsub) {
  std::vector<double> partSumWY(numPartitions, 0.0);
  std::vector<double> partSumWY2(numPartitions, 0.0);
  std::vector<double> partWeights(numPartitions, 0.0);
  for (size_t b = 0; b < binStats.size(); ++b) {
    if (b >= binToPartition.size())
      continue;
    const size_t p = binToPartition[b];
    if (p >= numPartitions)
      continue;
    if (b < binWeights.size())
      partWeights[p] += binWeights[b];
    const auto &sb = binStats[b];
    if (sb.size() >= 2) {
      partSumWY[p] += sb[0];
      partSumWY2[p] += sb[1];
    }
  }

  double missingSumWY = 0.0;
  double missingSumWY2 = 0.0;
  double missingWeight = 0.0;
  for (size_t col : missingCols) {
    if (col >= static_cast<size_t>(ysub.n_elem))
      continue;
    const double v = static_cast<double>(ysub(col));
    const double w = static_cast<double>(wsub(col));
    missingSumWY += w * v;
    missingSumWY2 += w * v * v;
    missingWeight += w;
  }

  double baseWeightedLoss = 0.0;
  double totalWeight = missingWeight;
  for (size_t p = 0; p < numPartitions; ++p) {
    totalWeight += partWeights[p];
    if (partWeights[p] > 0.0) {
      const std::vector<double> st{partSumWY[p], partSumWY2[p]};
      baseWeightedLoss +=
          partWeights[p] * Criterion::squaredError(st, partWeights[p]);
    }
  }
  if (totalWeight <= 0.0)
    return 0;

  std::vector<double> trialScores(numPartitions,
                                  std::numeric_limits<double>::infinity());
  for (size_t p = 0; p < numPartitions; ++p) {
    const double trialSumWY = partSumWY[p] + missingSumWY;
    const double trialSumWY2 = partSumWY2[p] + missingSumWY2;
    const double trialWeight = partWeights[p] + missingWeight;
    const std::vector<double> trialSt{trialSumWY, trialSumWY2};
    double trialLoss = 0.0;
    if (trialWeight > 0.0)
      trialLoss = Criterion::squaredError(trialSt, trialWeight);
    double basePartLoss = 0.0;
    if (partWeights[p] > 0.0) {
      const std::vector<double> st{partSumWY[p], partSumWY2[p]};
      basePartLoss = Criterion::squaredError(st, partWeights[p]);
    }
    const double weightedLoss =
        baseWeightedLoss - partWeights[p] * basePartLoss + trialWeight * trialLoss;
    trialScores[p] = weightedLoss / totalWeight;
  }
  return missing_values::pick_lowest_score_min_index_tie(trialScores);
}

inline double partition_absolute_error(const std::vector<float> &ys,
                                       const std::vector<float> &ws) {
  if (ys.empty())
    return 0.0;
  if (ys.size() <= 1)
    return Criterion::absoluteError(ys, ws).mae;
  std::vector<size_t> order(ys.size());
  for (size_t i = 0; i < order.size(); ++i)
    order[i] = i;
  std::sort(order.begin(), order.end(),
            [&ys](size_t a, size_t b) { return ys[a] < ys[b]; });
  std::vector<float> ysSorted;
  std::vector<float> wsSorted;
  ysSorted.reserve(ys.size());
  wsSorted.reserve(ws.size());
  for (size_t idx : order) {
    ysSorted.push_back(ys[idx]);
    wsSorted.push_back(ws[idx]);
  }
  return Criterion::absoluteError(ysSorted, wsSorted).mae;
}

inline size_t choose_nan_partition_absolute_error(
    size_t numPartitions, const arma::frowvec &featureRow,
    const std::vector<size_t> &sampleBins,
    const std::vector<size_t> &binToPartition,
    const std::vector<size_t> &missingCols, const arma::Row<float> &ysub,
    const arma::Row<float> &wsub) {
  std::vector<std::vector<float>> partYs(numPartitions);
  std::vector<std::vector<float>> partWs(numPartitions);
  std::vector<double> partWeights(numPartitions, 0.0);
  for (size_t col = 0; col < sampleBins.size(); ++col) {
    if (col >= static_cast<size_t>(featureRow.n_elem) ||
        !missing_values::is_finite(featureRow(col)))
      continue;
    const size_t bin = sampleBins[col];
    if (bin >= binToPartition.size())
      continue;
    const size_t p = binToPartition[bin];
    if (p >= numPartitions)
      continue;
    partYs[p].push_back(ysub(col));
    partWs[p].push_back(wsub(col));
    partWeights[p] += static_cast<double>(wsub(col));
  }

  std::vector<float> missingYs;
  std::vector<float> missingWs;
  double missingWeight = 0.0;
  for (size_t col : missingCols) {
    if (col >= static_cast<size_t>(ysub.n_elem))
      continue;
    missingYs.push_back(ysub(col));
    missingWs.push_back(wsub(col));
    missingWeight += static_cast<double>(wsub(col));
  }

  double baseWeightedLoss = 0.0;
  double totalWeight = missingWeight;
  for (size_t p = 0; p < numPartitions; ++p) {
    totalWeight += partWeights[p];
    if (!partYs[p].empty())
      baseWeightedLoss +=
          partWeights[p] * partition_absolute_error(partYs[p], partWs[p]);
  }
  if (totalWeight <= 0.0)
    return 0;

  std::vector<double> trialScores(numPartitions,
                                  std::numeric_limits<double>::infinity());
  for (size_t p = 0; p < numPartitions; ++p) {
    std::vector<float> trialYs = partYs[p];
    std::vector<float> trialWs = partWs[p];
    trialYs.insert(trialYs.end(), missingYs.begin(), missingYs.end());
    trialWs.insert(trialWs.end(), missingWs.begin(), missingWs.end());
    const double trialWeight = partWeights[p] + missingWeight;
    const double trialLoss = partition_absolute_error(trialYs, trialWs);
    const double basePartLoss = partition_absolute_error(partYs[p], partWs[p]);
    const double weightedLoss = baseWeightedLoss - partWeights[p] * basePartLoss +
                                trialWeight * trialLoss;
    trialScores[p] = weightedLoss / totalWeight;
  }
  return missing_values::pick_lowest_score_min_index_tie(trialScores);
}

} // namespace nan_partition_routing
