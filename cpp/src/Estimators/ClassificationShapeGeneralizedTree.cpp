/**
 * @file Estimators/ClassificationShapeGeneralizedTree.cpp
 * @brief Training, child partitioning, and prediction for the classification
 *        shape-generalized tree. Per-node inner fit: discretize -> search
 *        partition counts k in [2, numPartitions] (construct_mapping) ->
 *        coordinate descent -> record best branch.
 */

 #include "Estimators/ClassificationShapeGeneralizedTree.h"
 
 #include "Estimators/ClassificationShapeFunctionBuilder.h"
 #include "Criterion.h"
 #include "algorithms/missing_values.h"
 
 #include <algorithm>
 #include <armadillo>
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
 
 }
 
 
 
 ClassificationShapeGeneralizedTree::ClassificationShapeGeneralizedTree(
     LearningCriterion criterion, size_t numClasses, size_t numPartitions,
     TreeBuildingParams outerParams, TreeBuildingParams innerParams,
     CoordinateDescentParams cdParams, uint64_t random_state,
     FeatureBaggingPickFn featureBagging)
     : criterion_(criterion), numClasses_(numClasses),
       numPartitions_(numPartitions), outerParams_(outerParams),
       innerParams_(innerParams), cdParams_(cdParams),
       random_state_(random_state), rng_(),
       featureBagging_(featureBagging
                           ? std::move(featureBagging)
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
 
   ClassificationShapeFunctionBuilder splitBuilder(*this, X, y,
                                                   featureCandidates);
 
   outerTreeBuilder_.buildTree(
       nodes_[0],
       [&splitBuilder](ShapeFunctionNode &node, size_t minLeaf) {
         return splitBuilder.findBestSplit(node, minLeaf);
       },
       [this, &X, &y](ShapeFunctionNode &parent) {
 
         const size_t numChildPartitions = parent.numPartitions;
         if (parent.sampleBins.size() != parent.sampleIndices.n_elem)
           throw std::runtime_error(
               "ClassificationShapeGeneralizedTree::fit: sampleBins length "
               "mismatch");
 
         std::vector<std::vector<size_t>> buckets(numChildPartitions);
         for (arma::uword i = 0; i < parent.sampleIndices.n_elem; ++i) {
           const size_t si = static_cast<size_t>(parent.sampleIndices(i));
           size_t p = parent.nanPredictionPartition;
           if (missing_values::is_finite(
                   X(parent.routingFeature, static_cast<arma::uword>(si)))) {
             const size_t bin = parent.sampleBins[static_cast<size_t>(i)];
             if (bin >= parent.binToPartition.size())
               throw std::runtime_error("ClassificationShapeGeneralizedTree::"
                                        "fit: bin id out of range");
             p = parent.binToPartition[bin];
           }
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
           for (size_t si : buckets[p]) {
             if (missing_values::is_finite(
                     X(parent.routingFeature, static_cast<arma::uword>(si))))
               continue;
             const size_t lab = y(static_cast<arma::uword>(si));
             if (lab < numClasses_)
               childClassCounts[lab] +=
                   static_cast<double>(fitSampleWeights_(si));
           }
           ch.score = impurityForClassCounts(childClassCounts);
           ch.isLeaf = true;
           classCounts.push_back(childClassCounts);
           children.push_back(std::move(ch));
         }
 
 
         return children;
       },
       [this](ShapeFunctionNode &parent,
              std::vector<ShapeFunctionNode> &children) {
 
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
 
 