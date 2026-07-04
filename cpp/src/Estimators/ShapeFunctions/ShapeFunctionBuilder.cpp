/**
 * @file Estimators/ShapeFunctions/ShapeFunctionBuilder.cpp
 * @brief Shared helpers for shape-function split builders.
 */

#include "Estimators/ShapeFunctions/ShapeFunctionBuilder.h"

#include "Discretizers/ShapeDiscretizer.h"

#include <armadillo>
#include <stdexcept>

void ShapeFunctionBuilder::markLeafNoSplit(ShapeFunctionNode &node) {
  node.isLeaf = true;
  node.informationGain = 0.0;
  node.innerDiscretizer.reset();
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
    const std::vector<std::vector<size_t>> &perBinCols) {
  best.penalizedChildScore = search.bestFeatureScore;
  best.branching.featureIndex = featureIndex;
  best.branching.binToPartition = search.assignments;
  best.branching.impurityDecrease = search.impurityDecrease;
  best.branching.numPartitionsUsed = search.chosenK;
  best.branching.partitionSampleCounts = search.partitionSampleCounts;
  fillSampleBinsFromDiscretizer(xSubCols, perBinCols, best.branching.sampleBins);
}

bool ShapeFunctionBuilder::featureHasBetterBranching(
    const BranchAssignmentSearchResult &search, BestBranchingState &best,
    size_t featureIndex, size_t xSubCols,
    std::unique_ptr<ShapeDiscretizer> disc, double scoreEpsilon) {
  if (search.bestFeatureScore >= best.penalizedChildScore - scoreEpsilon)
    return false;

  applySharedBranchingFields(best, search, featureIndex, xSubCols,
                             disc->inSampleDiscretizations());
  best.branching.leafNumSamples = disc->leafNumSamples();
  best.binWeights.assign(disc->leafNodeWeights().begin(),
                         disc->leafNodeWeights().end());
  best.nanSeen = disc->nanSeen();
  best.nanStats = disc->nanStats();
  best.nanNumSamples = disc->nanNumSamples();
  best.nanNodeWeight = disc->nanNodeWeight();
  best.nanInSampleIndices = disc->nanInSampleIndices();
  applyTaskBranchingFields(best, search, disc->leafStats());
  best.winningDiscretizer =
      std::shared_ptr<const ShapeDiscretizer>(std::move(disc));
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
    if (parent.routingFeatureValuesAreFinite(
            X, static_cast<arma::uword>(si))) {
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
