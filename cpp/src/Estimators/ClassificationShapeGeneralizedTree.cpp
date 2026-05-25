/**
 * @file Estimators/ClassificationShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the classification
 *        shape-generalized tree. Per-node inner fit: discretize -> search
 *        partition counts k in [2, numPartitions] (construct_mapping) ->
 *        coordinate descent -> record best branch.
 */

#include "Estimators/ClassificationShapeGeneralizedTree.h"

#include "BranchAssignmentObjectives/BranchAssignmentFactory.h"
#include "Criterion.h"
#include "Discretizers/ClassificationDiscretizer.h"
#include "algorithms/BinPartitionAssignments.h"
#include "algorithms/CoordinateDescent.h"
#include "algorithms/KMeansUtils.h"
#include "algorithms/ShapeBranchingTypes.h"

#include <algorithm>
#include <armadillo>
#include <cmath>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

size_t leafArgmaxClass(const std::vector<double> &counts) {
  if (counts.empty())
    return 0;
  size_t best = 0;
  for (size_t c = 1; c < counts.size(); ++c) {
    if (counts[c] > counts[best])
      best = c;
  }
  return best;
}

arma::Row<float> subSampleWeights(const arma::Row<float> &weights,
                                  const arma::uvec &subIdx) {
  arma::Row<float> wsub(subIdx.n_elem);
  for (arma::uword j = 0; j < subIdx.n_elem; ++j)
    wsub(j) = weights(subIdx(j));
  return wsub;
}

/** Revert CD if post-CD objective is worse than the seed by more than this margin. */
constexpr double kCdObjectiveImprovementEps = 1e-10;

} // namespace

ClassificationShapeGeneralizedTree::ClassificationShapeGeneralizedTree(
    LearningCriterion criterion, size_t numClasses, size_t numPartitions,
    TreeBuildingParams outerParams, TreeBuildingParams innerParams,
    CoordinateDescentParams cdParams, uint64_t random_state,
    FeatureBaggingPickFn featureBagging)
    : criterion_(criterion), numClasses_(numClasses),
      numPartitions_(numPartitions), outerParams_(outerParams),
      innerParams_(innerParams), cdParams_(cdParams),
      random_state_(random_state), rng_(),
      featureBagging_(featureBagging ? std::move(featureBagging)
                                    : FeatureBaggingPickFn(pickAllFeatureIndices)),
      outerTreeBuilder_(outerParams_.minLeafSize, outerParams_.minGainSplit,
                        outerParams_.maxDepth, outerParams_.maxLeafNodes) {
  if (criterion_ != LearningCriterion::Entropy &&
      criterion_ != LearningCriterion::Gini)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: criterion must be Entropy or "
        "Gini");
  if (numClasses_ < 2)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: numClasses must be >= 2");
  if (numPartitions_ < 2)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree: numPartitions must be >= 2");
}



double ClassificationShapeGeneralizedTree::impurityForClassCounts(
    const std::vector<double> &classCounts) const {
  double totalWeight = 0.0;
  for (double c : classCounts)
    totalWeight += c;
  if (totalWeight <= 0.0)
    return 0.0;
  if (criterion_ == LearningCriterion::Gini)
    return Criterion::gini(classCounts, totalWeight);
  if (criterion_ == LearningCriterion::Entropy)
    return Criterion::entropy(classCounts, totalWeight);
  throw std::runtime_error("ClassificationShapeGeneralizedTree::"
                           "impurityForClassCounts: invalid criterion");
}

std::vector<double> ClassificationShapeGeneralizedTree::fillLeafHistogram(
    ShapeFunctionNode &node, const arma::Row<size_t> &y) const {
  std::vector<double> counts(numClasses_, 0.0);
  for (arma::uword i = 0; i < node.sampleIndices.n_elem; ++i) {
    const size_t si = static_cast<size_t>(node.sampleIndices(i));
    const size_t lab = y(si);
    if (lab >= numClasses_)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: class label out of range");
    counts[lab] += static_cast<double>(fitSampleWeights_(si));
  }
  return counts;
}

void ClassificationShapeGeneralizedTree::seedBinAssignmentsKMeans(
    size_t k, size_t numBins,
    const std::vector<std::vector<double>> &binClassCounts,
    const std::vector<size_t> &binSizes, const std::vector<double> &binWeights,
    std::vector<size_t> &assignments) {
  arma::mat Xk(numBins, numClasses_);
  arma::vec wk(numBins);
  for (size_t b = 0; b < numBins; ++b) {
    wk(b) = b < binWeights.size() ? std::max(1.0, binWeights[b])
                                  : std::max(1.0, static_cast<double>(binSizes[b]));
    double sum = 0.0;
    for (size_t c = 0; c < numClasses_; ++c)
      sum += binClassCounts[b][c];
    if (sum <= 0.0) {
      Xk.row(b).fill(1.0 / static_cast<double>(numClasses_));
    } else {
      for (size_t c = 0; c < numClasses_; ++c)
        Xk(b, c) = static_cast<double>(binClassCounts[b][c]) / sum;
    }
  }
  algorithms::initAssignmentsWeightedKMeans(Xk, wk, k, rng_, assignments);
}

void ClassificationShapeGeneralizedTree::fit(
    const arma::fmat &X, const arma::Row<size_t> &y,
    const arma::Row<float> &sampleWeights) {
  if (X.n_cols != y.n_elem)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: X.n_cols must match "
        "y.n_elem");
  if (X.n_rows == 0)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: X must have at least one "
        "feature (row)");
  for (size_t j = 0; j < y.n_elem; ++j) {
    if (y(j) >= numClasses_)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: label >= numClasses");
  }

  const size_t n = X.n_cols;
  if (sampleWeights.n_elem != n)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: sample_weights length must "
        "match number of samples");
  fitSampleWeights_ = sampleWeights;

  rng_.seed(static_cast<std::mt19937_64::result_type>(random_state_));

  nodes_.clear();
  childIndices_.clear();
  classCounts.clear();

  fitted_ = false;

  ShapeFunctionNode root;
  root.height = 0;
  root.sampleIndices =
      arma::regspace<arma::uvec>(0, static_cast<arma::uword>(n - 1));
  root.nodeIndex = 0;
  root.numPartitions = numPartitions_;
  root.isLeaf = true;
  classCounts.push_back(fillLeafHistogram(root, y));
  root.score = impurityForClassCounts(classCounts[root.nodeIndex]);
  nodes_.push_back(root);

  childIndices_.emplace_back();
  rootIndex_ = 0;

  const arma::uvec featureCandidates =
      arma::regspace<arma::uvec>(0, X.n_rows - 1);

  outerTreeBuilder_.buildTree(
      nodes_[0],
      [this, &X, &y, &featureCandidates](ShapeFunctionNode &node,
                                        size_t minLeaf) { // find best shape function
        const size_t ns = node.sampleIndices.n_elem;
        node.score = impurityForClassCounts(classCounts[node.nodeIndex]);

        if (ns < 2 * minLeaf) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          node.binSampleCounts.clear();

          return false;
        }

        const double parentImp = node.score;
        if (parentImp <= outerTreeBuilder_.eps) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          node.binSampleCounts.clear();
          return false;
        }

        const arma::uvec &subIdx = node.sampleIndices;
        const arma::fmat Xsub = X.cols(subIdx);
        const arma::Row<size_t> ysub = y.cols(subIdx);

        std::vector<size_t> featurePool(
            static_cast<size_t>(featureCandidates.n_elem));
        for (arma::uword i = 0; i < featureCandidates.n_elem; ++i)
          featurePool[static_cast<size_t>(i)] =
              static_cast<size_t>(featureCandidates(i));
        const std::vector<size_t> featureSubset = featureBagging_(
            std::span<const size_t>(featurePool.data(), featurePool.size()), rng_);

        const size_t xSubCols = static_cast<size_t>(Xsub.n_cols);
        double bestPenalizedChild = std::numeric_limits<double>::infinity();
        ShapeBranchingResult<double> brBest{};
        arma::uvec featOne(1);

        for (size_t fi = 0; fi < featureSubset.size(); ++fi) {
          const size_t f = featureSubset[fi];
          if (f >= Xsub.n_rows)
            throw std::invalid_argument(
                "ClassificationShapeGeneralizedTree::fit: candidate feature "
                "index >= X.n_rows");
          featOne(0) = static_cast<arma::uword>(f);

          const arma::Row<float> wsub =
              subSampleWeights(fitSampleWeights_, subIdx);

          auto disc = makeClassificationDiscretizer(criterion_);
          disc->Train(Xsub, featOne, ysub, numClasses_,
                      innerParams_.minLeafSize, innerParams_.minGainSplit,
                      innerParams_.maxDepth, innerParams_.maxLeafNodes, wsub);
          const size_t B = disc->numLeaves();
          if (B == 0)
            continue;

          auto &stats = disc->leafStats();
          auto &sizes = disc->leafNumSamples();
          auto &weights = disc->leafNodeWeights();

          if (B < 2)
            continue;

          const size_t kMax = std::min(B, numPartitions_);
          double bestFeatureScore = std::numeric_limits<double>::infinity();
          size_t chosenK = 0;
          std::vector<size_t> assignments;
          double impurityDecrease = 0.0;
          bool foundK = false;

          for (size_t k = 2; k <= kMax; ++k) {
            std::vector<size_t> trialAssignments;
            if (k == B) {
              algorithms::identityBinAssignments(B, trialAssignments);
            } else if (!cdParams_.smartInit || k < 2 || B < k) {
              algorithms::roundRobinBinAssignments(B, k, trialAssignments);
            } else {
              seedBinAssignmentsKMeans(k, B, stats, sizes, weights,
                                       trialAssignments);
            }

            auto branchObj = makeClassificationBranchAssignment(
                criterion_, trialAssignments, k, stats, weights, numClasses_);

            if (k < B) {
              const std::vector<size_t> snapshot = trialAssignments;
              const double objBeforeCd = branchObj->objective();
              coordinateDescent(k, *branchObj, rng_, cdParams_.maxIters,
                                cdParams_.patience);
              const double objAfterCd = branchObj->objective();
              if (!std::isfinite(objAfterCd) ||
                  objAfterCd > objBeforeCd + kCdObjectiveImprovementEps) {
                trialAssignments = snapshot;
                branchObj = makeClassificationBranchAssignment(
                    criterion_, trialAssignments, k, stats, weights,
                    numClasses_);
              }
            }

            if (!algorithms::partitionCountsMeetMinLeaf(trialAssignments, sizes,
                                                        k,
                                                        outerParams_.minLeafSize))
              continue;

            const double childImp = branchObj->objective();
            const double gain = parentImp - childImp;
            if (gain < outerParams_.minGainSplit - outerTreeBuilder_.eps)
              continue;

            const double score = algorithms::penalizedBranchingScore(
                childImp, k, outerParams_.branchingPenalty);
            if (score < bestFeatureScore - outerTreeBuilder_.eps) {
              bestFeatureScore = score;
              chosenK = k;
              assignments = std::move(trialAssignments);
              impurityDecrease = gain;
              foundK = true;
            }
          }

          if (!foundK)
            continue;

          if (bestFeatureScore < bestPenalizedChild - outerTreeBuilder_.eps) {
            bestPenalizedChild = bestFeatureScore;
            brBest.featureIndex = f;
            const auto &dth = disc->thresholds();
            brBest.innerThresholds.resize(dth.size());
            for (size_t t = 0; t < dth.size(); ++t)
              brBest.innerThresholds[t] = static_cast<float>(dth[t]);
            brBest.binToPartition = std::move(assignments);
            brBest.impurityDecrease = impurityDecrease;
            brBest.numPartitionsUsed = chosenK;

            const auto &perBinCols = disc->inSampleDiscretizations();
            brBest.sampleBins.assign(xSubCols, 0);
            for (size_t b = 0; b < perBinCols.size(); ++b) {
              for (size_t colIdx : perBinCols[b]) {
                if (colIdx >= xSubCols)
                  throw std::runtime_error(
                      "ClassificationShapeGeneralizedTree: discretizer sample "
                      "index >= Xsub columns");
                brBest.sampleBins[colIdx] = b;
              }
            }

            brBest.leafStats = stats;
            brBest.leafNumSamples = sizes;
          }
        }

        if (!std::isfinite(bestPenalizedChild) ||
            bestPenalizedChild >= std::numeric_limits<double>::infinity() ||
            brBest.impurityDecrease <= outerTreeBuilder_.eps) {
          node.isLeaf = true;
          node.informationGain = 0.0;
          node.sampleBins.clear();
          node.splitLeafStats.clear();
          node.binSampleCounts.clear();
          return false;
        }

        node.isLeaf = false;
        node.routingFeature = brBest.featureIndex;
        node.innerThresholds = std::move(brBest.innerThresholds);
        node.binToPartition = std::move(brBest.binToPartition);
        node.sampleBins = std::move(brBest.sampleBins);
        node.splitLeafStats = std::move(brBest.leafStats);
        node.binSampleCounts = std::move(brBest.leafNumSamples);
        node.numPartitions = brBest.numPartitionsUsed;
        node.informationGain = brBest.impurityDecrease;

        return true;
      },
      [this](ShapeFunctionNode &parent) { // make children
        const size_t numChildPartitions = parent.numPartitions;
        if (parent.sampleBins.size() != parent.sampleIndices.n_elem)
          throw std::runtime_error(
              "ClassificationShapeGeneralizedTree::fit: sampleBins length "
              "mismatch");

        // map sample indices to partitions
        std::vector<std::vector<size_t>> buckets(numChildPartitions);
        for (arma::uword i = 0; i < parent.sampleIndices.n_elem; ++i) {
          const size_t si = static_cast<size_t>(parent.sampleIndices(i));
          const size_t bin = parent.sampleBins[static_cast<size_t>(i)];
          if (bin >= parent.binToPartition.size())
            throw std::runtime_error(
                "ClassificationShapeGeneralizedTree::fit: bin id out of range");
          size_t p = parent.binToPartition[bin];
          if (p >= numChildPartitions)
            p = numChildPartitions - 1;
          buckets[p].push_back(si);
        }

        const auto &binStats = parent.splitLeafStats;
        if (binStats.size() != parent.binToPartition.size())
          throw std::runtime_error(
              "ClassificationShapeGeneralizedTree::fit: splitLeafStats / "
              "binToPartition size mismatch");

        std::vector<ShapeFunctionNode> children;
        children.reserve(numChildPartitions);
        for (size_t p = 0; p < numChildPartitions; ++p) {
          ShapeFunctionNode ch;
          ch.height = parent.height + 1;
          ch.sampleIndices = arma::conv_to<arma::uvec>::from(buckets[p]);
          ch.numPartitions = numPartitions_;
          std::vector<double> childClassCounts(numClasses_, 0.0);
          // aggregate per-bin class counts to get leaf class counts
          for (size_t b = 0; b < binStats.size(); ++b) {
            if (parent.binToPartition[b] != p)
              continue;
            const auto &sb = binStats[b];
            for (size_t c = 0; c < numClasses_; ++c) {
              const double add =
                  (c < sb.size()) ? static_cast<double>(sb[c]) : 0.0;
              childClassCounts[c] += add;
            }
          }
          ch.score = impurityForClassCounts(childClassCounts);
          ch.isLeaf = true;
          classCounts.push_back(childClassCounts);
          children.push_back(std::move(ch));
        }
        return children;
      },
      [this](ShapeFunctionNode &parent,
             std::vector<ShapeFunctionNode> &children) { // commit split
        const size_t pid = parent.nodeIndex;
        nodes_[pid] = parent;
        nodes_[pid].isLeaf = false;
        const size_t numChildPartitions = parent.numPartitions;
        childIndices_[pid].assign(numChildPartitions, 0);
        for (size_t p = 0; p < numChildPartitions; ++p) {
          const size_t cid = nodes_.size();
          children[p].nodeIndex = cid;
          nodes_.push_back(children[p]);
          childIndices_.push_back({});
          childIndices_[pid][p] = cid;
        }
      });

  for (auto &node : nodes_) {
    node.sampleIndices.set_size(0);
    node.sampleBins.clear();
  }

  fitted_ = true;
}

arma::Row<size_t>
ClassificationShapeGeneralizedTree::predict(const arma::fmat &X) const {
  if (!fitted_)
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predict: model is not fitted");
  if (nodes_.empty())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predict: empty tree");
  if (rootIndex_ >= nodes_.size())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predict: invalid root index");
  if (childIndices_.size() != nodes_.size())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predict: childIndices/nodes size "
        "mismatch");

  const size_t n_samples = X.n_cols;
  arma::Row<size_t> yhat(n_samples);

  for (size_t s = 0; s < n_samples; ++s) {
    size_t idx = rootIndex_;
    while (true) {
      const auto &node = nodes_[idx];
      if (node.isLeaf) {
        yhat(s) = leafArgmaxClass(classCounts[node.nodeIndex]);
        break;
      }
      if (node.routingFeature >= X.n_rows)
        throw std::invalid_argument(
            "ClassificationShapeGeneralizedTree::predict: routing feature "
            "index out of range for X");
      const float v = X(node.routingFeature, s);
      const size_t part = node.routeFeatureValueToPartition(v);
      if (part >= childIndices_[idx].size())
        throw std::runtime_error(
            "ClassificationShapeGeneralizedTree::predict: child partition "
            "out of range");
      idx = childIndices_[idx][part];
      if (idx >= nodes_.size())
        throw std::runtime_error(
            "ClassificationShapeGeneralizedTree::predict: child node index "
            "out of range");
    }
  }
  return yhat;
}

arma::fmat
ClassificationShapeGeneralizedTree::predictProba(const arma::fmat &X) const {
  if (!fitted_)
    throw std::logic_error("ClassificationShapeGeneralizedTree::predictProba: "
                           "model is not fitted");
  if (nodes_.empty())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predictProba: empty tree");
  if (rootIndex_ >= nodes_.size())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predictProba: invalid root index");
  if (childIndices_.size() != nodes_.size())
    throw std::logic_error(
        "ClassificationShapeGeneralizedTree::predictProba: childIndices/nodes "
        "size mismatch");

  const size_t n_samples = X.n_cols;
  const size_t K = numClasses_;
  arma::fmat P(K, n_samples);
  const float uniform = K > 0 ? 1.f / static_cast<float>(K) : 0.f;

  for (size_t s = 0; s < n_samples; ++s) {
    size_t idx = rootIndex_;
    while (true) {
      const auto &node = nodes_[idx];
      if (node.isLeaf) {
        const auto &h = classCounts[node.nodeIndex];
        double sum = 0.0;
        for (size_t c = 0; c < K; ++c)
          sum += (c < h.size()) ? static_cast<double>(h[c]) : 0.0;
        if (sum <= 0.0) {
          P.col(s).fill(uniform);
        } else {
          for (size_t c = 0; c < K; ++c) {
            const double cnt = (c < h.size()) ? static_cast<double>(h[c]) : 0.0;
            P(c, s) = static_cast<float>(cnt / sum);
          }
        }
        break;
      }
      if (node.routingFeature >= X.n_rows)
        throw std::invalid_argument(
            "ClassificationShapeGeneralizedTree::predictProba: routing "
            "feature index out of range for X");
      const float v = X(node.routingFeature, s);
      const size_t part = node.routeFeatureValueToPartition(v);
      if (part >= childIndices_[idx].size())
        throw std::runtime_error(
            "ClassificationShapeGeneralizedTree::predictProba: child "
            "partition out of range");
      idx = childIndices_[idx][part];
      if (idx >= nodes_.size())
        throw std::runtime_error(
            "ClassificationShapeGeneralizedTree::predictProba: child node "
            "index out of range");
    }
  }
  return P;
}

size_t ClassificationShapeGeneralizedTree::numLeaves() const {
  size_t c = 0;
  for (const auto &node : nodes_) {
    if (node.isLeaf)
      ++c;
  }
  return c;
}

size_t ClassificationShapeGeneralizedTree::numNodes() const {
  return nodes_.size();
}

bool ClassificationShapeGeneralizedTree::isFitted() const { return fitted_; }
