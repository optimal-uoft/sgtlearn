/**
 * @file Estimators/ClassificationShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the classification shape-generalized tree.
 */

#include "Estimators/ClassificationShapeGeneralizedTree.h"

#include "algorithms/ShapeBranchingFit.h"
#include "Criterion.h"

#include <stdexcept>
#include <vector>

namespace {

/** Majority class index from a histogram (first argmax on ties). */
size_t leafArgmaxClass(const std::vector<size_t> &counts) {
  if (counts.empty())
    return 0;
  size_t best = 0;
  for (size_t c = 1; c < counts.size(); ++c) {
    if (counts[c] > counts[best])
      best = c;
  }
  return best;
}

} // namespace

ClassificationShapeGeneralizedTree::ClassificationShapeGeneralizedTree(
    LearningCriterion criterion, size_t numClasses, size_t numPartitions,
    TreeBuildingParams outerParams, TreeBuildingParams innerParams,
    CoordinateDescentParams cdParams)
    : criterion_(criterion), numClasses_(numClasses),
      numPartitions_(numPartitions), outerParams_(outerParams),
      innerParams_(innerParams), cdParams_(cdParams),
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

double ClassificationShapeGeneralizedTree::impurityAtRange(
    size_t begin, size_t end, const arma::Row<size_t> &y) const {
  if (end <= begin)
    return 0.0;
  std::vector<size_t> counts(numClasses_, 0);
  size_t n = 0;
  for (size_t i = begin; i < end; ++i) {
    const size_t si = sampleOrder_(i);
    const size_t lab = y(si);
    counts[lab]++;
    n++;
  }
  if (criterion_ == LearningCriterion::Gini)
    return Criterion::gini(counts, n);
  return Criterion::entropy(counts, n);
}

void ClassificationShapeGeneralizedTree::fillLeafHistogram(
    ClassificationShapeGeneralizedNode &node, const arma::Row<size_t> &y) const {
  node.leafClassCounts.assign(numClasses_, 0);
  for (size_t i = node.sampleBegin; i < node.sampleEnd; ++i) {
    const size_t si = sampleOrder_(i);
    const size_t lab = y(si);
    if (lab >= numClasses_)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: class label out of range");
    node.leafClassCounts[lab]++;
  }
}

bool ClassificationShapeGeneralizedTree::findBestSplitNode(
    ClassificationShapeGeneralizedNode &node, size_t minLeafSize,
    const arma::fmat &X, const arma::Row<size_t> &y,
    const arma::uvec &features) {
  const size_t n = node.sampleEnd - node.sampleBegin;
  node.score = impurityAtRange(node.sampleBegin, node.sampleEnd, y);

  if (n < 2 * minLeafSize) {
    node.isLeaf = true;
    node.informationGain = 0.0;
    fillLeafHistogram(node, y);
    return false;
  }

  const double parentImp = node.score;
  if (parentImp <= outerTreeBuilder_.eps) {
    node.isLeaf = true;
    node.informationGain = 0.0;
    fillLeafHistogram(node, y);
    return false;
  }

  const arma::uvec subIdx =
      sampleOrder_.subvec(node.sampleBegin, node.sampleEnd - 1);
  const arma::fmat Xsub = X.cols(subIdx);
  arma::Row<size_t> ysub(subIdx.n_elem);
  for (arma::uword i = 0; i < subIdx.n_elem; ++i)
    ysub(i) = y(subIdx(i));

  const auto branch = fitShapeBranch(
      criterion_, numClasses_, numPartitions_, innerParams_, cdParams_,
      outerParams_.branchingPenalty, Xsub, ysub, features, parentImp,
      outerParams_.minGainSplit, outerParams_.minLeafSize,
      outerTreeBuilder_.eps);

  if (!branch) {
    node.isLeaf = true;
    node.informationGain = 0.0;
    fillLeafHistogram(node, y);
    return false;
  }

  ShapeBranchingResult br = std::move(*branch);
  node.isLeaf = false;
  node.routingFeature = br.featureIndex;
  node.innerThresholds = std::move(br.innerThresholds);
  node.binToPartition = std::move(br.binToPartition);
  node.numPartitions = numPartitions_;
  node.informationGain = br.impurityDecrease;
  node.leafClassCounts.clear();
  return true;
}

std::vector<ClassificationShapeGeneralizedNode>
ClassificationShapeGeneralizedTree::makeChildrenNode(
    ClassificationShapeGeneralizedNode &parent, const arma::fmat &X,
    const arma::Row<size_t> &y) {
  const size_t beg = parent.sampleBegin;
  const size_t end = parent.sampleEnd;

  std::vector<std::vector<size_t>> buckets(numPartitions_);
  for (size_t i = beg; i < end; ++i) {
    const size_t si = sampleOrder_(i);
    const float v = X(parent.routingFeature, si);
    size_t p = parent.routeFeatureValueToPartition(v);
    if (p >= numPartitions_)
      p = numPartitions_ - 1;
    buckets[p].push_back(si);
  }

  parent.childSampleBounds.resize(numPartitions_ + 1);
  parent.childSampleBounds[0] = beg;
  size_t pos = beg;
  for (size_t p = 0; p < numPartitions_; ++p) {
    for (const size_t si : buckets[p])
      sampleOrder_(pos++) = si;
    parent.childSampleBounds[p + 1] = pos;
  }

  std::vector<ClassificationShapeGeneralizedNode> children;
  children.reserve(numPartitions_);
  for (size_t p = 0; p < numPartitions_; ++p) {
    ClassificationShapeGeneralizedNode ch;
    ch.height = parent.height + 1;
    ch.sampleBegin = parent.childSampleBounds[p];
    ch.sampleEnd = parent.childSampleBounds[p + 1];
    ch.numPartitions = numPartitions_;
    ch.score = impurityAtRange(ch.sampleBegin, ch.sampleEnd, y);
    ch.isLeaf = true;
    fillLeafHistogram(ch, y);
    children.push_back(std::move(ch));
  }
  return children;
}

void ClassificationShapeGeneralizedTree::commitSplitNode(
    ClassificationShapeGeneralizedNode &parent,
    std::vector<ClassificationShapeGeneralizedNode> &children) {
  const size_t pid = parent.nodeIndex;
  nodes_[pid] = parent;
  nodes_[pid].isLeaf = false;
  childIndices_[pid].assign(numPartitions_, 0);
  for (size_t p = 0; p < numPartitions_; ++p) {
    const size_t cid = nodes_.size();
    children[p].nodeIndex = cid;
    nodes_.push_back(children[p]);
    childIndices_.push_back({});
    childIndices_[pid][p] = cid;
  }
}

void ClassificationShapeGeneralizedTree::fit(const arma::fmat &X,
                                             arma::uvec &features,
                                             const arma::Row<size_t> &y) {
  if (X.n_cols != y.n_elem)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: X.n_cols must match "
        "y.n_elem");
  if (features.n_elem == 0)
    throw std::invalid_argument(
        "ClassificationShapeGeneralizedTree::fit: features must be non-empty");
  for (size_t i = 0; i < features.n_elem; ++i) {
    if (static_cast<size_t>(features(i)) >= X.n_rows)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: feature index >= "
          "X.n_rows");
  }
  for (size_t j = 0; j < y.n_elem; ++j) {
    if (y(j) >= numClasses_)
      throw std::invalid_argument(
          "ClassificationShapeGeneralizedTree::fit: label >= numClasses");
  }

  const size_t n = X.n_cols;
  sampleOrder_.set_size(n);
  for (size_t i = 0; i < n; ++i)
    sampleOrder_(i) = static_cast<arma::uword>(i);

  nodes_.clear();
  childIndices_.clear();
  fitted_ = false;

  ClassificationShapeGeneralizedNode root;
  root.height = 0;
  root.sampleBegin = 0;
  root.sampleEnd = n;
  root.nodeIndex = 0;
  root.numPartitions = numPartitions_;
  root.score = impurityAtRange(0, n, y);
  root.isLeaf = true;
  fillLeafHistogram(root, y);
  nodes_.push_back(root);
  childIndices_.emplace_back();
  rootIndex_ = 0;

  outerTreeBuilder_.buildTree(
      nodes_[0],
      [this, &X, &y, &features](ClassificationShapeGeneralizedNode &node,
                                size_t minLeaf) {
        return findBestSplitNode(node, minLeaf, X, y, features);
      },
      [this, &X, &y](ClassificationShapeGeneralizedNode &parent) {
        return makeChildrenNode(parent, X, y);
      },
      [this](ClassificationShapeGeneralizedNode &parent,
             std::vector<ClassificationShapeGeneralizedNode> &kids) {
        commitSplitNode(parent, kids);
      });

  sampleOrder_.clear();
  for (auto &node : nodes_) {
    node.sampleBegin = 0;
    node.sampleEnd = 0;
    node.nodeIndex = 0;
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
        yhat(s) = leafArgmaxClass(node.leafClassCounts);
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
        const auto &h = node.leafClassCounts;
        double sum = 0.0;
        for (size_t c = 0; c < K; ++c)
          sum += (c < h.size()) ? static_cast<double>(h[c]) : 0.0;
        if (sum <= 0.0) {
          P.col(s).fill(uniform);
        } else {
          for (size_t c = 0; c < K; ++c) {
            const double cnt =
                (c < h.size()) ? static_cast<double>(h[c]) : 0.0;
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
