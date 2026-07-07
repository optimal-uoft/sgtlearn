/**
 * @file Estimators/ShapeFunctions/ShapeFunctionSplitSearch.cpp
 */

#include "Estimators/ShapeFunctions/ShapeFunctionSplitSearch.h"

#include <stdexcept>

void markShapeFunctionNodeAsLeaf(ShapeFunctionNode &node) {
  node.isLeaf = true;
  node.informationGain = 0.0;
  node.innerDiscretizer.reset();
  node.sampleBins.clear();
  node.splitLeafStats.clear();
  node.splitBinWeights.clear();
  node.binSampleCounts.clear();
}

void fillSampleBinsFromDiscretizer(
    size_t xSubCols, const std::vector<std::vector<size_t>> &perBinCols,
    std::vector<size_t> &sampleBins) {
  sampleBins.assign(xSubCols, 0);
  for (size_t b = 0; b < perBinCols.size(); ++b) {
    for (size_t colIdx : perBinCols[b]) {
      if (colIdx >= xSubCols)
        throw std::runtime_error(
            "fillSampleBinsFromDiscretizer: sample index >= Xsub columns");
      sampleBins[colIdx] = b;
    }
  }
}

arma::Row<float> subSampleWeights(const arma::Row<float> &weights,
                                  const arma::uvec &subIdx) {
  arma::Row<float> wsub(subIdx.n_elem);
  for (arma::uword j = 0; j < subIdx.n_elem; ++j)
    wsub(j) = weights(subIdx(j));
  return wsub;
}

void applySharedShapeBranchingFields(
    ShapeBestBranchingState &best, const ShapeBranchAssignmentSearchResult &search,
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

bool featureHasBetterShapeBranching(
    const ShapeBranchAssignmentSearchResult &search,
    ShapeBestBranchingState &best, size_t featureIndex, size_t xSubCols,
    std::unique_ptr<InnerDiscretizerBase<double>> disc, double scoreEpsilon,
    const std::function<void(ShapeBestBranchingState &,
                             const ShapeBranchAssignmentSearchResult &,
                             const std::vector<std::vector<double>> &)> &
        applyTaskFields) {
  if (!search.found || search.assignments.empty())
    return false;
  if (search.bestFeatureScore >= best.penalizedChildScore - scoreEpsilon)
    return false;

  applySharedShapeBranchingFields(best, search, featureIndex, xSubCols,
                                  disc->inSampleDiscretizations());
  best.branching.leafNumSamples = disc->leafNumSamples();
  best.binWeights.assign(disc->leafNodeWeights().begin(),
                         disc->leafNodeWeights().end());
  applyTaskFields(best, search, disc->leafStats());
  best.winningDiscretizer =
      std::shared_ptr<const InnerDiscretizerBase<double>>(std::move(disc));
  return true;
}

std::vector<std::vector<size_t>>
routeSamplesToPartitions(const ShapeFunctionNode &parent, const arma::fmat &X) {
  (void)X;
  if (parent.sampleBins.size() != parent.sampleIndices.n_elem)
    throw std::runtime_error(
        "routeSamplesToPartitions: sampleBins length mismatch");

  const size_t numChildPartitions = parent.numPartitions;
  std::vector<std::vector<size_t>> buckets(numChildPartitions);
  for (arma::uword i = 0; i < parent.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(parent.sampleIndices(i));
    const size_t bin = parent.sampleBins[static_cast<size_t>(i)];
    if (bin >= parent.binToPartition.size())
      throw std::runtime_error("routeSamplesToPartitions: bin id out of range");
    const size_t p = parent.binToPartition[bin];
    if (p >= numChildPartitions)
      throw std::runtime_error(
          "routeSamplesToPartitions: partition out of range");
    buckets[p].push_back(si);
  }
  return buckets;
}

std::vector<ShapeFunctionNode>
makeRoutedChildNodes(const ShapeFunctionNode &parent,
                     const std::vector<std::vector<size_t>> &buckets,
                     size_t treeNumPartitions) {
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
