/**
 * @file CategoricalSplitter.tpp
 * @brief Template implementation for one-hot categorical split search.
 */

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

template <typename StatsT, typename PredictT>
double
CategoricalSplitter<StatsT, PredictT>::totalWeight(
    const std::vector<size_t> &samples) const {
  double sum = 0.0;
  for (size_t i : samples)
    sum += static_cast<double>(sampleWeights(i));
  return sum;
}

template <typename StatsT, typename PredictT>
void CategoricalSplitter<StatsT, PredictT>::fillNodeMeta(
    CategoricalSplitCandidate &node) const {
  node.numSamples = node.samples.size();
  node.nodeWeight = totalWeight(node.samples);
}

template <typename StatsT, typename PredictT>
CategoricalSplitCandidate CategoricalSplitter<StatsT, PredictT>::makeRoot() {
  CategoricalSplitCandidate root;
  root.height = 0;
  root.samples.resize(X.n_cols);
  for (size_t i = 0; i < X.n_cols; ++i)
    root.samples[i] = i;
  root.availableCategoryFeatures = featureIndices;
  fillNodeMeta(root);
  root.score = score(root.samples);
  return root;
}

template <typename StatsT, typename PredictT>
bool CategoricalSplitter<StatsT, PredictT>::findBestSplit(
    CategoricalSplitCandidate &node, size_t minLeafSize) {
  if (node.isActiveLeafBranch)
    return false;

  const double parentImp = score(node.samples);
  const double wTot = totalWeight(node.samples);
  if (wTot <= 0.0 || node.availableCategoryFeatures.empty()) {
    node.informationGain = 0.0;
    return false;
  }

  double bestGain = -std::numeric_limits<double>::infinity();
  bool found = false;
  size_t bestFeature = 0;
  std::vector<size_t> bestActive;
  std::vector<size_t> bestInactive;
  std::vector<size_t> bestNextAvailable;

  for (size_t feat : node.availableCategoryFeatures) {
    std::vector<size_t> active;
    std::vector<size_t> inactive;
    active.reserve(node.samples.size());
    inactive.reserve(node.samples.size());
    for (size_t i : node.samples) {
      if (isActive(X(feat, i)))
        active.push_back(i);
      else
        inactive.push_back(i);
    }
    if (active.size() < minLeafSize || inactive.size() < minLeafSize)
      continue;

    const double wActive = totalWeight(active);
    const double wInactive = totalWeight(inactive);
    const double childImp =
        (wInactive * score(inactive) + wActive * score(active)) / wTot;
    const double gain = parentImp - childImp;
    if (gain > bestGain) {
      bestGain = gain;
      bestFeature = feat;
      bestActive = std::move(active);
      bestInactive = std::move(inactive);
      bestNextAvailable.clear();
      bestNextAvailable.reserve(node.availableCategoryFeatures.size() - 1);
      for (size_t f : node.availableCategoryFeatures) {
        if (f != feat)
          bestNextAvailable.push_back(f);
      }
      found = true;
    }
  }

  if (!found) {
    node.informationGain = 0.0;
    return false;
  }

  node.splitFeature = bestFeature;
  node.activeSamples = std::move(bestActive);
  node.inactiveSamples = std::move(bestInactive);
  node.informationGain = bestGain;
  return true;
}

template <typename StatsT, typename PredictT>
std::vector<CategoricalSplitCandidate>
CategoricalSplitter<StatsT, PredictT>::makeChildren(
    const CategoricalSplitCandidate &parent) {
  CategoricalSplitCandidate inactive;
  inactive.height = parent.height + 1;
  inactive.samples = parent.inactiveSamples;
  inactive.availableCategoryFeatures.reserve(parent.availableCategoryFeatures.size());
  for (size_t f : parent.availableCategoryFeatures) {
    if (f != parent.splitFeature)
      inactive.availableCategoryFeatures.push_back(f);
  }
  fillNodeMeta(inactive);
  inactive.score = score(inactive.samples);

  CategoricalSplitCandidate active;
  active.height = parent.height + 1;
  active.samples = parent.activeSamples;
  active.isActiveLeafBranch = true;
  fillNodeMeta(active);
  active.score = score(active.samples);
  active.informationGain = 0.0;

  return {inactive, active};
}
