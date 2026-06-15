/**
 * @file Estimators/ShapeFunctions/ShapeFunctionBuilder.cpp
 * @brief Shared helpers for shape-function split builders.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionBuilder.h"

#include "Discretizers/ShapeDiscretizer.h"

#include "algorithms/missing_values.h"

#include <armadillo>
#include <stdexcept>

void ShapeFunctionBuilder::markLeafNoSplit(ShapeFunctionNode &node) {
  node.isLeaf = true;
  node.informationGain = 0.0;
  node.sampleBins.clear();
  node.splitLeafStats.clear();
  node.splitBinWeights.clear();
  node.binSampleCounts.clear();
  node.splitMissingStats.clear();
  node.splitMissingWeight = 0.0;
}

void ShapeFunctionBuilder::fillSampleBinsFromDiscretizer(
    size_t xSubCols, const std::vector<std::vector<size_t>> &perBinCols,
    std::vector<size_t> &sampleBins) {
  sampleBins.assign(xSubCols, 0);
  for (size_t b = 0; b < perBinCols.size(); ++b) {
    for (size_t colIdx : perBinCols[b]) {
      if (colIdx >= xSubCols)
        throw std::runtime_error(
            "ShapeFunctionBuilder: discretizer sample index >= Xsub columns");
      sampleBins[colIdx] = b;
    }
  }
}

arma::Row<float> ShapeFunctionBuilder::subSampleWeights(
    const arma::Row<float> &weights, const arma::uvec &subIdx) {
  arma::Row<float> wsub(subIdx.n_elem);
  for (arma::uword j = 0; j < subIdx.n_elem; ++j)
    wsub(j) = weights(subIdx(j));
  return wsub;
}

void ShapeFunctionBuilder::applySharedBranchingFields(
    BestBranchingState &best, const BranchAssignmentSearchResult &search,
    size_t featureIndex, size_t xSubCols,
    const std::vector<double> &thresholds,
    const std::vector<std::vector<size_t>> &perBinCols) {
  best.penalizedChildScore = search.bestFeatureScore;
  best.branching.featureIndex = featureIndex;
  best.branching.innerThresholds.resize(thresholds.size());
  for (size_t t = 0; t < thresholds.size(); ++t)
    best.branching.innerThresholds[t] = static_cast<float>(thresholds[t]);
  best.branching.binToPartition = search.assignments;
  best.branching.impurityDecrease = search.impurityDecrease;
  best.branching.numPartitionsUsed = search.chosenK;
  best.branching.partitionSampleCounts = search.partitionSampleCounts;
  fillSampleBinsFromDiscretizer(xSubCols, perBinCols, best.branching.sampleBins);
}

bool ShapeFunctionBuilder::featureHasBetterBranching(
    const BranchAssignmentSearchResult &search, BestBranchingState &best,
    size_t featureIndex, size_t xSubCols, ShapeDiscretizer &disc,
    double scoreEpsilon) {
  if (search.bestFeatureScore >= best.penalizedChildScore - scoreEpsilon)
    return false;

  applySharedBranchingFields(best, search, featureIndex, xSubCols,
                             disc.thresholds(), disc.inSampleDiscretizations());
  best.branching.leafNumSamples = disc.leafNumSamples();
  best.binWeights.assign(disc.leafNodeWeights().begin(),
                         disc.leafNodeWeights().end());
  applyTaskBranchingFields(best, search, disc.leafStats());
  return true;
}

std::vector<std::vector<size_t>> ShapeFunctionBuilder::routeSamplesToPartitions(
    const ShapeFunctionNode &parent, const arma::fmat &X) {
  if (parent.sampleBins.size() != parent.sampleIndices.n_elem)
    throw std::runtime_error(
        "ShapeFunctionBuilder::routeSamplesToPartitions: sampleBins length "
        "mismatch");

  const size_t numChildPartitions = parent.numPartitions;
  std::vector<std::vector<size_t>> buckets(numChildPartitions);
  for (arma::uword i = 0; i < parent.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(parent.sampleIndices(i));
    size_t p = parent.nanPredictionPartition;
    if (missing_values::is_finite(
            X(parent.routingFeature, static_cast<arma::uword>(si)))) {
      const size_t bin = parent.sampleBins[static_cast<size_t>(i)];
      if (bin >= parent.binToPartition.size())
        throw std::runtime_error(
            "ShapeFunctionBuilder::routeSamplesToPartitions: bin id out of "
            "range");
      p = parent.binToPartition[bin];
    }
    if (p >= numChildPartitions)
      throw std::runtime_error(
          "ShapeFunctionBuilder::routeSamplesToPartitions: partition out of "
          "range");
    buckets[p].push_back(si);
  }
  return buckets;
}

std::vector<ShapeFunctionNode> ShapeFunctionBuilder::makeRoutedChildNodes(
    const ShapeFunctionNode &parent,
    const std::vector<std::vector<size_t>> &buckets, size_t treeNumPartitions) {
  std::vector<ShapeFunctionNode> children;
  children.reserve(buckets.size());
  for (const auto &bucket : buckets) {
    ShapeFunctionNode ch;
    ch.height = parent.height + 1;
    ch.sampleIndices = arma::conv_to<arma::uvec>::from(bucket);
    ch.numPartitions = treeNumPartitions;
    children.push_back(std::move(ch));
  }
  return children;
}
