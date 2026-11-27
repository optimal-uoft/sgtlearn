#pragma once

/**
 * @file algorithms/TAO/ClassificationTaoAdapter.h
 * @brief TAO adapter for classification shape-generalized trees.
 */

#include <cstddef>
#include "Estimators/ClassificationShapeGeneralizedTree.h"
#include "algorithms/TAO/TaoAdapter.h"

namespace tao {

class ClassificationTaoAdapter final : public TaoAdapter {
public:
  ClassificationTaoAdapter(ClassificationShapeGeneralizedTree &tree,
                           const arma::fmat &X, const arma::Mat<size_t> &y,
                           const arma::Row<float> &sampleWeights);

  /** Same as the tree's impurity criterion (Gini or entropy). */
  LearningCriterion routerCriterion() const override;

  void recomputeLeafStats(
      const std::vector<std::vector<arma::uword>> &nodeSamples) override;

  void refreshNodeBinMetadata(
      ShapeFunctionNode &node,
      const std::vector<arma::uword> &samples) override;

protected:
  void childRewards(const std::vector<size_t> &childLeaves, arma::uword col,
                    std::vector<double> &reward) const override;

  bool careSetKeepsOnlyBest() const override { return nOutputs_ == 1; }

private:
  ClassificationShapeGeneralizedTree &classificationTree_;
  const arma::Mat<size_t> &y_;
  std::vector<size_t> classesPerOutput_;
  size_t nOutputs_;
};

} // namespace tao
