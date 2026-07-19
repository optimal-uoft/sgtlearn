#pragma once

/**
 * @file Estimators/ShapeFunctions/NanPartitionRouting.h
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

/**
 * Squared-error NaN routing from precomputed NaN-bucket moments. Picks the
 * partition whose weighted MSE grows least when the NaN bucket is merged into
 * it. Multi-output: ``binStats[b]`` holds the concatenated
 * ``[Σw·y0, Σw·y0², Σw·y1, Σw·y1², ...]`` (length ``2 * nOutputs``), and
 * ``missingMoments`` holds the same layout for the NaN bucket. Loss is summed
 * across outputs via ``Criterion::squaredErrorMulti``.
 */
inline size_t choose_nan_partition_squared_error_from_moments(
    size_t numPartitions, const std::vector<size_t> &binToPartition,
    const std::vector<std::vector<double>> &binStats,
    const std::vector<double> &binWeights,
    const std::vector<double> &missingMoments, double missingWeight,
    size_t nOutputs = 1) {
  const size_t dim = 2 * nOutputs;
  std::vector<std::vector<double>> partMoments(numPartitions,
                                               std::vector<double>(dim, 0.0));
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
    for (size_t d = 0; d < dim && d < sb.size(); ++d)
      partMoments[p][d] += sb[d];
  }

  double baseWeightedLoss = 0.0;
  double totalWeight = missingWeight;
  for (size_t p = 0; p < numPartitions; ++p) {
    totalWeight += partWeights[p];
    if (partWeights[p] > 0.0)
      baseWeightedLoss +=
          partWeights[p] *
          Criterion::squaredErrorMulti(partMoments[p], partWeights[p], nOutputs);
  }
  if (totalWeight <= 0.0)
    return 0;

  std::vector<double> trialScores(numPartitions,
                                  std::numeric_limits<double>::infinity());
  for (size_t p = 0; p < numPartitions; ++p) {
    std::vector<double> trialSt(dim, 0.0);
    for (size_t d = 0; d < dim; ++d)
      trialSt[d] =
          partMoments[p][d] + (d < missingMoments.size() ? missingMoments[d] : 0.0);
    const double trialWeight = partWeights[p] + missingWeight;
    double trialLoss = 0.0;
    if (trialWeight > 0.0)
      trialLoss = Criterion::squaredErrorMulti(trialSt, trialWeight, nOutputs);
    double basePartLoss = 0.0;
    if (partWeights[p] > 0.0)
      basePartLoss =
          Criterion::squaredErrorMulti(partMoments[p], partWeights[p], nOutputs);
    const double weightedLoss =
        baseWeightedLoss - partWeights[p] * basePartLoss + trialWeight * trialLoss;
    trialScores[p] = weightedLoss / totalWeight;
  }
  return missing_values::pick_lowest_score_min_index_tie(trialScores);
}

inline size_t choose_nan_partition_squared_error(
    size_t numPartitions, const std::vector<size_t> &binToPartition,
    const std::vector<std::vector<double>> &binStats,
    const std::vector<double> &binWeights,
    const std::vector<size_t> &missingCols, const arma::Mat<float> &ysub,
    const arma::Row<float> &wsub,
    std::vector<double> *missingMomentsOut = nullptr,
    double *missingWeightOut = nullptr) {
  const size_t nOutputs = std::max<size_t>(ysub.n_rows, 1);
  const size_t dim = 2 * nOutputs;

  std::vector<double> missingMoments(dim, 0.0);
  double missingWeight = 0.0;
  for (size_t col : missingCols) {
    if (col >= static_cast<size_t>(ysub.n_cols))
      continue;
    const double w = static_cast<double>(wsub(col));
    for (size_t o = 0; o < nOutputs; ++o) {
      const double v = static_cast<double>(
          ysub(static_cast<arma::uword>(o), static_cast<arma::uword>(col)));
      missingMoments[2 * o] += w * v;
      missingMoments[2 * o + 1] += w * v * v;
    }
    missingWeight += w;
  }
  if (missingMomentsOut)
    *missingMomentsOut = missingMoments;
  if (missingWeightOut)
    *missingWeightOut = missingWeight;

  return choose_nan_partition_squared_error_from_moments(
      numPartitions, binToPartition, binStats, binWeights, missingMoments,
      missingWeight, nOutputs);
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
    const std::vector<size_t> &missingCols, const arma::Mat<float> &ysub,
    const arma::Row<float> &wsub) {
  const size_t nOutputs = std::max<size_t>(ysub.n_rows, 1);
  // Per partition, per output: raw y samples; weights are shared per sample.
  std::vector<std::vector<std::vector<float>>> partYs(
      numPartitions, std::vector<std::vector<float>>(nOutputs));
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
    for (size_t o = 0; o < nOutputs; ++o)
      partYs[p][o].push_back(
          ysub(static_cast<arma::uword>(o), static_cast<arma::uword>(col)));
    partWs[p].push_back(wsub(col));
    partWeights[p] += static_cast<double>(wsub(col));
  }

  std::vector<std::vector<float>> missingYs(nOutputs);
  std::vector<float> missingWs;
  double missingWeight = 0.0;
  for (size_t col : missingCols) {
    if (col >= static_cast<size_t>(ysub.n_cols))
      continue;
    for (size_t o = 0; o < nOutputs; ++o)
      missingYs[o].push_back(
          ysub(static_cast<arma::uword>(o), static_cast<arma::uword>(col)));
    missingWs.push_back(wsub(col));
    missingWeight += static_cast<double>(wsub(col));
  }

  // Summed-over-outputs MAE for one partition given its per-output y lists.
  const auto partitionLoss =
      [nOutputs](const std::vector<std::vector<float>> &ys,
                 const std::vector<float> &ws) -> double {
    double loss = 0.0;
    for (size_t o = 0; o < nOutputs; ++o)
      loss += partition_absolute_error(ys[o], ws);
    return loss;
  };

  double baseWeightedLoss = 0.0;
  double totalWeight = missingWeight;
  for (size_t p = 0; p < numPartitions; ++p) {
    totalWeight += partWeights[p];
    if (!partWs[p].empty())
      baseWeightedLoss += partWeights[p] * partitionLoss(partYs[p], partWs[p]);
  }
  if (totalWeight <= 0.0)
    return 0;

  std::vector<double> trialScores(numPartitions,
                                  std::numeric_limits<double>::infinity());
  for (size_t p = 0; p < numPartitions; ++p) {
    std::vector<std::vector<float>> trialYs = partYs[p];
    for (size_t o = 0; o < nOutputs; ++o)
      trialYs[o].insert(trialYs[o].end(), missingYs[o].begin(),
                        missingYs[o].end());
    std::vector<float> trialWs = partWs[p];
    trialWs.insert(trialWs.end(), missingWs.begin(), missingWs.end());
    const double trialWeight = partWeights[p] + missingWeight;
    const double trialLoss = partitionLoss(trialYs, trialWs);
    const double basePartLoss = partitionLoss(partYs[p], partWs[p]);
    const double weightedLoss = baseWeightedLoss - partWeights[p] * basePartLoss +
                                trialWeight * trialLoss;
    trialScores[p] = weightedLoss / totalWeight;
  }
  return missing_values::pick_lowest_score_min_index_tie(trialScores);
}

} // namespace nan_partition_routing
