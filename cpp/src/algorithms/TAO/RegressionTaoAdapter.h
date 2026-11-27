#pragma once

/**
 * @file algorithms/TAO/RegressionTaoAdapter.h
 * @brief TAO adapter for regression shape-generalized trees.
 */

#include <cstddef>
#include "Estimators/RegressionShapeGeneralizedTree.h"
#include "algorithms/TAO/TaoAdapter.h"

namespace tao {

class RegressionTaoAdapter final : public TaoAdapter {
public:
  RegressionTaoAdapter(RegressionShapeGeneralizedTree &tree,
                       const arma::fmat &X, const arma::Mat<float> &y,
                       const arma::Row<float> &sampleWeights);

  /** Always ``LearningCriterion::Gini`` for the routing discretizer. */
  LearningCriterion routerCriterion() const override;

  void recomputeLeafStats(
      const std::vector<std::vector<arma::uword>> &nodeSamples) override;

  void refreshNodeBinMetadata(
      ShapeFunctionNode &node,
      const std::vector<arma::uword> &samples) override;

protected:
  void childRewards(const std::vector<size_t> &childLeaves, arma::uword col,
                    std::vector<double> &reward) const override;

private:
  RegressionShapeGeneralizedTree &regressionTree_;
  const arma::Mat<float> &y_;
  bool squared_;
};

} // namespace tao
