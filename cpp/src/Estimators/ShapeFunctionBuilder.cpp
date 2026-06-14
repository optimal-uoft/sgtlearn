/**
 * @file Estimators/ShapeFunctionBuilder.cpp
 * @brief Shared helpers for shape-function split builders.
 */

#include "Estimators/ShapeFunctionBuilder.h"

#include <armadillo>
#include <stdexcept>

void ShapeFunctionBuilder::markLeafNoSplit(ShapeFunctionNode &node) {
  node.isLeaf = true;
  node.informationGain = 0.0;
  node.sampleBins.clear();
  node.splitLeafStats.clear();
  node.splitBinWeights.clear();
  node.binSampleCounts.clear();
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
