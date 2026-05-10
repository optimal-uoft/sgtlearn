/**
 * @file ShapeBranchingFit.cpp
 * @brief Classification inner branching: try Gini/entropy discretizers per feature and pick the best coordinate-descent mapping.
 */

#include "algorithms/ShapeBranchingFit.h"
#include "algorithms/ShapeBranchingFit_internal.h"

#include "UnivariateClassificationDiscretizer.h"
#include "Splitters/EntropySplitter.h"
#include "Splitters/GiniSplitter.h"

#include <cmath>
#include <limits>
#include <stdexcept>

using shape_branching_fit::detail::evaluateDiscriminator;

std::optional<ShapeBranchingResult> fitShapeBranch(
    LearningCriterion criterion, size_t numClasses, size_t numPartitions,
    const TreeBuildingParams &innerParams,
    const CoordinateDescentParams &cdParams, double branchingPenalty,
    const arma::fmat &Xsub, const arma::Row<size_t> &ysub,
    const arma::uvec &features, double parentImpurity,
    double outerMinImpurityDecrease, size_t outerMinSamplesLeaf,
    double comparatorEps) {
  double bestPenalizedChild = std::numeric_limits<double>::infinity();
  ShapeBranchingResult best{};
  arma::uvec featOne(1);

  for (size_t fi = 0; fi < features.n_elem; ++fi) {
    const size_t f = static_cast<size_t>(features(fi));
    if (f >= Xsub.n_rows)
      throw std::invalid_argument("fitShapeBranch: feature index >= X.n_rows");
    featOne(0) = static_cast<arma::uword>(f);

    if (criterion == LearningCriterion::Gini) {
      UnivariateClassificationDiscretizer<GiniSplitter> disc;
      try {
        disc.Train(Xsub, featOne, ysub, numClasses, innerParams.minLeafSize,
                   innerParams.minGainSplit, innerParams.maxDepth,
                   innerParams.maxLeafNodes);
      } catch (...) {
        continue;
      }
      evaluateDiscriminator(disc, f, criterion, numClasses, numPartitions,
                            cdParams, parentImpurity, branchingPenalty,
                            outerMinImpurityDecrease, outerMinSamplesLeaf,
                            comparatorEps, bestPenalizedChild, best);
    } else {
      UnivariateClassificationDiscretizer<EntropySplitter> disc;
      try {
        disc.Train(Xsub, featOne, ysub, numClasses, innerParams.minLeafSize,
                   innerParams.minGainSplit, innerParams.maxDepth,
                   innerParams.maxLeafNodes);
      } catch (...) {
        continue;
      }
      evaluateDiscriminator(disc, f, criterion, numClasses, numPartitions,
                            cdParams, parentImpurity, branchingPenalty,
                            outerMinImpurityDecrease, outerMinSamplesLeaf,
                            comparatorEps, bestPenalizedChild, best);
    }
  }

  if (!std::isfinite(bestPenalizedChild) ||
      bestPenalizedChild >= std::numeric_limits<double>::infinity())
    return std::nullopt;
  if (best.impurityDecrease <= comparatorEps)
    return std::nullopt;
  return best;
}
