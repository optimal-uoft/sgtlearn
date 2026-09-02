#pragma once

/**
 * @file _sgt_estimators.h
 * @brief Shared pybind11 wrapper classes for the shape-generalized tree
 *        estimators, plus the NumPy <-> Armadillo argument helpers they need.
 *
 * The wrappers live in the named namespace ``sgt::bindings`` (not an anonymous
 * namespace) so their mangled type names are identical across translation
 * units. pybind11 keys its cross-module type registry on the mangled name, so
 * a class registered by the ``ShapeGeneralizedTrees`` module can be received as
 * a function argument by other modules (e.g. ``TreeAlternatingOptimization``)
 * that merely include this header without re-registering the type.
 */

#include "_arma_bridge.h"

#include "Discretizers/categorical/CategoricalClassificationDiscretizer.h"
#include "Discretizers/categorical/CategoricalRegressionDiscretizer.h"
#include "Discretizers/pair/PairClassificationDiscretizer.h"
#include "Discretizers/pair/PairRegressionDiscretizer.h"
#include "Domain/LearningCriterion.h"
#include "Domain/FeatureInfo.h"
#include "Discretizers/univariate/UnivariateDiscretizer.h"
#include "Estimators/ClassificationShapeGeneralizedTree.h"
#include "Estimators/RegressionShapeGeneralizedTree.h"
#include "algorithms/FeatureBagging.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <memory>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

namespace sgt::bindings {

namespace py = pybind11;

inline py::list routingFeaturesPy(const ShapeFunctionNode &n) {
  py::list feats;
  for (size_t f : n.routingFeatures)
    feats.append(f);
  return feats;
}

inline py::list logicalFeaturesPy(const ShapeFunctionNode &n) {
  py::list feats;
  for (size_t f : n.logicalFeatureIndices)
    feats.append(f);
  return feats;
}

inline py::list pairAxesPy(const ShapeFunctionNode &n,
                           const std::array<FeatureInfo, 2> &axes) {
  py::list out;
  for (size_t axis = 0; axis < axes.size(); ++axis) {
    py::dict item;
    item["logical_feature"] = n.logicalFeatureIndices.at(axis);
    item["kind"] = axes[axis].type == FeatureType::Categorical
                       ? "categorical"
                       : "continuous";
    py::list columns;
    py::list categories;
    for (size_t raw : axes[axis].indices) {
      columns.append(raw);
      if (axes[axis].type == FeatureType::Categorical)
        categories.append(raw);
    }
    item["columns"] = columns;
    item["categories"] = categories;
    item["catchall"] = axes[axis].type == FeatureType::Categorical
                            ? py::cast("missing")
                            : py::none();
    out.append(item);
  }
  return out;
}

inline py::object primaryRoutingFeaturePy(const ShapeFunctionNode &n) {
  if (n.routingFeatures.empty())
    return py::none();
  if (n.routingFeatures.size() == 1)
    return py::int_(n.routingFeatures.front());
  return py::none();
}

inline py::list innerThresholdsPy(const ShapeFunctionNode &n) {
  if (!n.innerDiscretizer)
    throw std::runtime_error("tree export: internal node missing innerDiscretizer");
  py::list th;
  for (double t : numericInnerThresholds(*n.innerDiscretizer))
    th.append(t);
  return th;
}

inline bool isCategoricalInnerDiscretizer(const InnerDiscretizerBase &disc) {
  return dynamic_cast<const CategoricalClassificationDiscretizer *>(&disc) !=
             nullptr ||
         dynamic_cast<const CategoricalRegressionDiscretizer *>(&disc) !=
             nullptr;
}

inline std::vector<std::vector<size_t>>
categoricalCategoriesPerBin(const InnerDiscretizerBase &disc) {
  if (const auto *cc =
          dynamic_cast<const CategoricalClassificationDiscretizer *>(&disc))
    return cc->categoriesPerBin();
  if (const auto *cr =
          dynamic_cast<const CategoricalRegressionDiscretizer *>(&disc))
    return cr->categoriesPerBin();
  return {};
}

inline py::list categoricalBinCategoriesPy(const ShapeFunctionNode &n) {
  if (!n.innerDiscretizer)
    throw std::runtime_error(
        "tree export: internal node missing innerDiscretizer");
  py::list bins;
  for (const auto &cats : categoricalCategoriesPerBin(*n.innerDiscretizer)) {
    py::list c;
    for (size_t col : cats)
      c.append(col);
    bins.append(c);
  }
  return bins;
}

inline std::string normalizeCriterion(std::string s) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

inline arma::Row<float> sampleWeightRowFromPy(const py::object &sample_weight,
                                              arma::uword n_samples) {
  if (sample_weight.is_none()) {
    arma::Row<float> w(n_samples);
    w.ones();
    return w;
  }
  auto wb = as1DRow<float>(sample_weight, "sample_weight");
  if (wb.view().n_elem != n_samples)
    throw std::invalid_argument("sample_weight length must equal X.shape[0]");
  arma::Row<float> w(wb.view().n_elem);
  for (arma::uword i = 0; i < wb.view().n_elem; ++i)
    w(i) = wb.view()(i);
  return w;
}

inline LearningCriterion parseClassificationCriterion(const std::string &raw) {
  const std::string s = normalizeCriterion(raw);
  if (s == "gini")
    return LearningCriterion::Gini;
  if (s == "entropy" || s == "log_loss")
    return LearningCriterion::Entropy;
  throw std::invalid_argument(
      "criterion must be 'gini', 'entropy', or 'log_loss'; got '" + raw + "'");
}

inline LearningCriterion parseRegressionCriterion(const std::string &raw) {
  const std::string s = normalizeCriterion(raw);
  if (s == "squared_error" || s == "mse" || s == "friedman_mse")
    return LearningCriterion::SquaredError;
  if (s == "absolute_error" || s == "mae")
    return LearningCriterion::AbsoluteError;
  throw std::invalid_argument(
      "criterion must be 'squared_error', 'mse', 'absolute_error', or 'mae'; "
      "got '" +
      raw + "'");
}

inline FeatureBaggingPickFn parseMaxFeaturesPy(py::handle mf_h) {
  py::object mf = py::reinterpret_borrow<py::object>(mf_h);
  if (mf.is_none())
    return makeFeatureBaggingAll();
  if (py::isinstance<py::str>(mf)) {
    std::string s = py::str(mf).cast<std::string>();
    const auto not_space = [](unsigned char c) { return !std::isspace(c); };
    s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
    s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
      return static_cast<char>(std::tolower(c));
    });
    if (s == "sqrt")
      return makeFeatureBaggingSqrt();
    if (s == "log2")
      return makeFeatureBaggingLog2();
    throw std::invalid_argument(
        "max_features string must be 'sqrt' or 'log2'; got '" + s + "'");
  }
  if (py::isinstance<py::bool_>(mf))
    throw std::invalid_argument("max_features cannot be bool");

  py::object numbers = py::module_::import("numbers");
  if (py::isinstance(mf, numbers.attr("Integral"))) {
    const long long vll = py::cast<long long>(mf);
    if (vll < 0)
      throw std::invalid_argument("max_features int must be non-negative");
    if (vll == 0)
      return makeFeatureBaggingAll();
    return makeFeatureBaggingCap(static_cast<size_t>(vll));
  }
  if (py::isinstance(mf, numbers.attr("Real"))) {
    const double c = py::cast<double>(mf);
    if (!(c > 0.0) || c > 1.0)
      throw std::invalid_argument(
          "max_features float must satisfy 0 < max_features <= 1");
    return makeFeatureBaggingFraction(c);
  }
  throw std::invalid_argument(
      "max_features must be None, a non-negative int, a float in (0, 1], "
      "or the string 'sqrt' / 'log2'");
}

inline FeatureType parseFeatureTypePy(const std::string &raw) {
  std::string s = normalizeCriterion(raw);
  if (s == "continuous" || s == "numeric")
    return FeatureType::Continuous;
  if (s == "categorical" || s == "one_hot" || s == "onehot")
    return FeatureType::Categorical;
  throw std::invalid_argument(
      "feature type must be 'continuous' or 'categorical'; got '" + raw + "'");
}

inline std::vector<FeatureInfo> parseFeaturesPy(const py::object &features) {
  if (features.is_none())
    throw std::invalid_argument("features is required");
  if (!py::isinstance<py::list>(features))
    throw std::invalid_argument("features must be a list of feature dicts");
  std::vector<FeatureInfo> out;
  for (const py::handle item : features) {
    if (!py::isinstance<py::dict>(item))
      throw std::invalid_argument(
          "each feature must be a dict with 'type' and 'indices'");
    const py::dict d = py::reinterpret_borrow<py::dict>(item);
    if (!d.contains("type") || !d.contains("indices"))
      throw std::invalid_argument(
          "each feature dict must contain 'type' and 'indices'");
    FeatureInfo feature;
    feature.type = parseFeatureTypePy(py::str(d["type"]).cast<std::string>());
    const py::list idxs = py::cast<py::list>(d["indices"]);
    feature.indices.set_size(idxs.size());
    for (arma::uword i = 0; i < feature.indices.n_elem; ++i)
      feature.indices(i) =
          static_cast<arma::uword>(py::cast<size_t>(idxs[static_cast<py::ssize_t>(i)]));
    out.push_back(std::move(feature));
  }
  return out;
}

/**
 * Thin Python adapter around `ClassificationShapeGeneralizedTree`. Owns the
 * C++ implementation and handles NumPy <-> Armadillo plumbing.
 */
/**
 * Parse the Python ``num_classes`` argument into a per-output class-count
 * vector. Accepts an integer (shared count expanded to match ``y.n_rows`` at
 * fit) or a sequence of integers (one entry per output).
 */
inline std::vector<size_t> parseNumClassesPy(const py::object &num_classes) {
  py::object numbers = py::module_::import("numbers");
  if (py::isinstance<py::bool_>(num_classes))
    throw std::invalid_argument("num_classes cannot be bool");
  if (py::isinstance(num_classes, numbers.attr("Integral")))
    return {py::cast<size_t>(num_classes)};
  if (py::isinstance<py::iterable>(num_classes)) {
    std::vector<size_t> out;
    for (const py::handle item : num_classes)
      out.push_back(py::cast<size_t>(py::reinterpret_borrow<py::object>(item)));
    if (out.empty())
      throw std::invalid_argument(
          "num_classes sequence must contain at least one output count");
    return out;
  }
  throw std::invalid_argument(
      "num_classes must be an int or a sequence of ints");
}

class ClassificationShapeGeneralizedTreePy {
public:
  ClassificationShapeGeneralizedTreePy(
      std::string criterion, py::object numClasses, size_t numPartitions,
      size_t outerMinLeafSize, double outerMinGainSplit, size_t outerMaxDepth,
      size_t outerMaxLeafNodes, size_t innerMinLeafSize,
      double innerMinGainSplit, size_t innerMaxDepth, size_t innerMaxLeafNodes,
      size_t coordinateDescentMaxIters, size_t coordinateDescentPatience,
      bool coordinateDescentSmartInit, uint64_t random_state,
      py::object max_features = py::none(), size_t pairwiseCandidates = 0,
      double pairwisePenalty = 0.0) {
    criterionStr_ = criterion;
    const LearningCriterion crit = parseClassificationCriterion(criterion);
    const TreeBuildingParams outer{outerMinLeafSize, outerMinGainSplit,
                                   outerMaxDepth, outerMaxLeafNodes};
    const TreeBuildingParams inner{innerMinLeafSize, innerMinGainSplit,
                                   innerMaxDepth, innerMaxLeafNodes};
    CoordinateDescentParams cd;
    cd.maxIters = coordinateDescentMaxIters;
    cd.patience = coordinateDescentPatience;
    cd.smartInit = coordinateDescentSmartInit;
    impl_ = std::make_unique<ClassificationShapeGeneralizedTree>(
        crit, parseNumClassesPy(numClasses), numPartitions, outer, inner, cd,
        random_state, parseMaxFeaturesPy(max_features), pairwiseCandidates,
        pairwisePenalty);
  }

  void fit(const py::array &X, const py::array &y,
           py::object sample_weight, const py::object &features) {
    auto Xb = asSamplesByFeatures<float>(X, "X");
    /** Owning copy: zero-copy views from NumPy often fail Armadillo strict
     *  checks on some dtypes / strides; C++ expects `arma::Mat<size_t>`.
     *  Accepts 1-D (n_samples,) or 2-D (n_samples, n_outputs). */
    arma::Mat<size_t> y_mat = asSamplesByOutputsOwning<size_t>(y, "y");

    if (y_mat.n_cols != Xb.view().n_cols)
      throw std::invalid_argument(
          "y.shape[0] must equal X.shape[0] (number of samples)");

    const arma::Row<float> w_row =
        sampleWeightRowFromPy(sample_weight, Xb.view().n_cols);

    const std::vector<FeatureInfo> featuresVec = parseFeaturesPy(features);

    py::gil_scoped_release release;
    impl_->fit(Xb.view(), y_mat, w_row, featuresVec);
  }

  py::array_t<size_t> predict(const py::array &X) {
    auto Xb = asSamplesByFeatures<float>(X, "X");
    arma::Mat<size_t> preds;
    {
      py::gil_scoped_release release;
      preds = impl_->predict(Xb.view());
    }
    // arma (numOutputs, numSamples) -> numpy (numSamples,) or (numSamples, numOutputs)
    return samplesByOutputsToNumpy(preds);
  }

  py::object predictProba(const py::array &X) {
    auto Xb = asSamplesByFeatures<float>(X, "X");
    std::vector<arma::fmat> proba;
    {
      py::gil_scoped_release release;
      proba = impl_->predictProba(Xb.view());
    }
    // Single output: return one (numSamples, numClasses) array. Multi-output:
    // return a list with one such array per output (sklearn convention).
    if (proba.size() == 1)
      return samplesByFeaturesToNumpy(proba.front());
    py::list out;
    for (const arma::fmat &p : proba)
      out.append(samplesByFeaturesToNumpy(p));
    return out;
  }

  /**
   * Access the underlying C++ estimator. Used by sibling modules (e.g.
   * ``TreeAlternatingOptimization``) that operate on the fitted tree without
   * being coupled to this Python wrapper's method surface.
   */
  ClassificationShapeGeneralizedTree &impl() { return *impl_; }

  size_t numLeaves() const { return impl_->numLeaves(); }
  size_t numNodes() const { return impl_->numNodes(); }
  bool isFitted() const { return impl_->isFitted(); }
  size_t nOutputs() const { return impl_->nOutputs(); }
  bool hasPairNodes() const { return impl_->hasPairNodes(); }

  py::list classesPerOutput() const {
    py::list out;
    for (size_t k : impl_->classesPerOutput())
      out.append(k);
    return out;
  }

  py::array_t<double> featureImportance() const {
    return colToNumpy(impl_->featureImportance());
  }

  py::dict tree_export() const {
    if (!impl_->isFitted())
      throw std::logic_error("tree_export: model is not fitted");
    py::dict out;
    out["num_partitions"] = impl_->numPartitions();
    out["num_nodes"] = impl_->numNodes();
    out["root_index"] = impl_->rootIndex();
    out["num_classes"] = classesPerOutput();
    out["num_outputs"] = impl_->nOutputs();
    out["classes_per_output"] = classesPerOutput();
    out["criterion"] = criterionStr_;

    const auto &nodes = impl_->nodes();
    const auto &childIdx = impl_->childIndices();
    const auto &classCounts = impl_->classCounts;

    py::list nodes_list;
    for (size_t i = 0; i < nodes.size(); ++i) {
      const auto &n = nodes[i];
      py::dict d;
      d["id"] = n.nodeIndex;
      d["depth"] = n.height;
      d["is_leaf"] = n.isLeaf;
      d["impurity"] = n.score;

      // class_counts[output][class] at every node (also populated at internals).
      py::list cc;
      size_t total = 0;
      if (i < classCounts.size()) {
        for (size_t o = 0; o < classCounts[i].size(); ++o) {
          py::list row;
          double rowSum = 0.0;
          for (double c : classCounts[i][o]) {
            row.append(c);
            rowSum += c;
          }
          cc.append(row);
          if (o == 0)
            total = static_cast<size_t>(std::llround(rowSum));
        }
      }
      d["class_counts"] = cc;
      d["n_samples"] = total;

      if (n.isLeaf) {
        d["feature"] = py::none();
        d["features"] = py::list();
        d["thresholds"] = py::list();
        d["bin_to_partition"] = py::list();
        d["bin_counts"] = py::list();
        d["bin_sample_counts"] = py::list();
        d["children"] = py::list();
      } else {
        d["feature"] = primaryRoutingFeaturePy(n);
        d["features"] = routingFeaturesPy(n);
        const auto *pairDisc = dynamic_cast<const PairClassificationDiscretizer *>(
            n.innerDiscretizer.get());
        if (pairDisc) {
          d["routing_kind"] = "pair";
          d["pair_features"] = logicalFeaturesPy(n);
          d["pair_axes"] = pairAxesPy(n, pairDisc->axes());
          py::list innerTree;
          py::list leafBins;
          const auto &routingTree = pairDisc->routingTree();
          for (size_t nodeIndex = 0; nodeIndex < routingTree.size(); ++nodeIndex) {
            const PairRoutingTreeNode &inner = routingTree[nodeIndex];
            py::dict innerNode;
            innerNode["id"] = nodeIndex;
            innerNode["is_leaf"] = inner.isLeaf;
            if (inner.isLeaf) {
              innerNode["bin"] = inner.bin;
              leafBins.append(inner.bin);
            } else {
              innerNode["feature"] = inner.rawFeature;
              innerNode["axis"] = inner.featurePosition;
              innerNode["kind"] = inner.featureType == FeatureType::Categorical
                                      ? "categorical"
                                      : "continuous";
              innerNode["threshold"] =
                  inner.featureType == FeatureType::Continuous
                      ? py::cast(inner.threshold)
                      : py::none();
              innerNode["category"] =
                  inner.featureType == FeatureType::Categorical
                      ? py::cast(inner.rawFeature)
                      : py::none();
              innerNode["left"] = inner.left;
              innerNode["right"] = inner.right;
              innerNode["missing"] = inner.missing;
            }
            innerTree.append(innerNode);
          }
          d["pair_inner_tree"] = innerTree;
          d["pair_leaf_bins"] = leafBins;
        }
        const bool isCategorical =
            isCategoricalInnerDiscretizer(*n.innerDiscretizer);
        d["is_categorical"] = isCategorical;
        d["thresholds"] = innerThresholdsPy(n);
        if (isCategorical)
          d["bin_categories"] = categoricalBinCategoriesPy(n);
        py::list b2p;
        for (size_t p : n.binToPartition) b2p.append(p);
        d["bin_to_partition"] = b2p;
        py::list bc;
        for (const auto &bin : n.splitClassCounts) {
          py::list outputs;
          for (const auto &hist : bin) {
            py::list classes;
            for (double c : hist) classes.append(c);
            outputs.append(classes);
          }
          bc.append(outputs);
        }
        d["bin_counts"] = bc;
        py::list bw;
        for (double w : n.splitBinWeights) bw.append(w);
        d["bin_weights"] = bw;
        py::list bsc;
        for (size_t v : n.binSampleCounts) bsc.append(v);
        d["bin_sample_counts"] = bsc;
        if (!pairDisc)
          d["nan_prediction_partition"] =
              n.binToPartition.empty() ? 0 : n.binToPartition.back();
        py::list ch;
        if (i < childIdx.size()) {
          for (size_t c : childIdx[i]) ch.append(c);
        }
        d["children"] = ch;
      }
      nodes_list.append(d);
    }
    out["nodes"] = nodes_list;
    return out;
  }

private:
  std::unique_ptr<ClassificationShapeGeneralizedTree> impl_;
  std::string criterionStr_;
};

class RegressionShapeGeneralizedTreePy {
public:
  RegressionShapeGeneralizedTreePy(
      std::string criterion, size_t numPartitions, size_t outerMinLeafSize,
      double outerMinGainSplit, size_t outerMaxDepth, size_t outerMaxLeafNodes,
      size_t innerMinLeafSize, double innerMinGainSplit, size_t innerMaxDepth,
      size_t innerMaxLeafNodes, size_t coordinateDescentMaxIters,
      size_t coordinateDescentPatience, bool coordinateDescentSmartInit,
      uint64_t random_state, py::object max_features = py::none(),
      size_t pairwiseCandidates = 0, double pairwisePenalty = 0.0) {
    criterionStr_ = criterion;
    const LearningCriterion crit = parseRegressionCriterion(criterion);
    const TreeBuildingParams outer{outerMinLeafSize, outerMinGainSplit,
                                   outerMaxDepth, outerMaxLeafNodes};
    const TreeBuildingParams inner{innerMinLeafSize, innerMinGainSplit,
                                   innerMaxDepth, innerMaxLeafNodes};
    CoordinateDescentParams cd;
    cd.maxIters = coordinateDescentMaxIters;
    cd.patience = coordinateDescentPatience;
    cd.smartInit = coordinateDescentSmartInit;
    impl_ = std::make_unique<RegressionShapeGeneralizedTree>(
        crit, numPartitions, outer, inner, cd, random_state,
        parseMaxFeaturesPy(max_features), pairwiseCandidates, pairwisePenalty);
  }

  void fit(const py::array &X, const py::array &y,
           py::object sample_weight, const py::object &features) {
    auto Xb = asSamplesByFeatures<float>(X, "X");
    // Accepts 1-D (n_samples,) or 2-D (n_samples, n_outputs).
    auto yb = asSamplesByOutputs<float>(y, "y");
    if (yb.view().n_cols != Xb.view().n_cols)
      throw std::invalid_argument(
          "y.shape[0] must equal X.shape[0] (number of samples)");
    const arma::Row<float> w_row =
        sampleWeightRowFromPy(sample_weight, Xb.view().n_cols);

    const std::vector<FeatureInfo> featuresVec = parseFeaturesPy(features);

    py::gil_scoped_release release;
    impl_->fit(Xb.view(), yb.view(), w_row, featuresVec);
  }

  py::array_t<double> predict(const py::array &X) {
    auto Xb = asSamplesByFeatures<float>(X, "X");
    arma::Mat<double> preds;
    {
      py::gil_scoped_release release;
      preds = impl_->predict(Xb.view());
    }
    // arma (numOutputs, numSamples) -> numpy (numSamples,) or (numSamples, numOutputs)
    return samplesByOutputsToNumpy(preds);
  }

  /**
   * Access the underlying C++ estimator (see the classification overload).
   */
  RegressionShapeGeneralizedTree &impl() { return *impl_; }

  size_t numLeaves() const { return impl_->numLeaves(); }
  size_t numNodes() const { return impl_->numNodes(); }
  bool isFitted() const { return impl_->isFitted(); }
  size_t nOutputs() const { return impl_->nOutputs(); }
  bool hasPairNodes() const { return impl_->hasPairNodes(); }

  py::array_t<double> featureImportance() const {
    return colToNumpy(impl_->featureImportance());
  }

  std::vector<std::vector<float>> leafRegressionStats() const {
    return impl_->leafRegressionStats;
  }

  std::vector<size_t> leafNumSamples() const { return impl_->leafNumSamples; }

  py::dict tree_export() const {
    if (!impl_->isFitted())
      throw std::logic_error("tree_export: model is not fitted");
    py::dict out;
    out["num_partitions"] = impl_->numPartitions();
    out["num_nodes"] = impl_->numNodes();
    out["root_index"] = impl_->rootIndex();
    out["num_outputs"] = impl_->nOutputs();
    out["criterion"] = criterionStr_;

    const auto &nodes = impl_->nodes();
    const auto &childIdx = impl_->childIndices();
    const auto &leafPred = impl_->leafPredictions();
    const auto &leafN = impl_->leafNumSamples;

    py::list nodes_list;
    for (size_t i = 0; i < nodes.size(); ++i) {
      const auto &n = nodes[i];
      py::dict d;
      d["id"] = n.nodeIndex;
      d["depth"] = n.height;
      d["is_leaf"] = n.isLeaf;
      d["impurity"] = n.score;

      if (n.isLeaf) {
        // Single output: scalar ``value`` (backward compatible). Multi-output:
        // a list of per-output leaf predictions.
        const std::vector<double> leafValue =
            i < leafPred.size() ? leafPred[i] : std::vector<double>{0.0};
        if (leafValue.size() == 1) {
          d["value"] = leafValue.front();
        } else {
          py::list vals;
          for (double v : leafValue)
            vals.append(v);
          d["value"] = vals;
        }
        d["n_samples"] = i < leafN.size() ? leafN[i] : static_cast<size_t>(0);
        d["feature"] = py::none();
        d["features"] = py::list();
        d["thresholds"] = py::list();
        d["bin_to_partition"] = py::list();
        d["bin_counts"] = py::list();
        d["bin_sample_counts"] = py::list();
        d["children"] = py::list();
      } else {
        d["value"] = py::none();
        size_t total = 0;
        for (size_t v : n.binSampleCounts) total += v;
        d["n_samples"] = total;
        d["feature"] = primaryRoutingFeaturePy(n);
        d["features"] = routingFeaturesPy(n);
        const auto *pairDisc = dynamic_cast<const PairRegressionDiscretizer *>(
            n.innerDiscretizer.get());
        const auto *taoPairDisc =
            dynamic_cast<const PairClassificationDiscretizer *>(
                n.innerDiscretizer.get());
        if (pairDisc || taoPairDisc) {
          d["routing_kind"] = "pair";
          d["pair_features"] = logicalFeaturesPy(n);
          d["pair_axes"] = pairAxesPy(
              n, pairDisc ? pairDisc->axes() : taoPairDisc->axes());
          py::list innerTree;
          py::list leafBins;
          const auto &routingTree =
              pairDisc ? pairDisc->routingTree() : taoPairDisc->routingTree();
          for (size_t nodeIndex = 0; nodeIndex < routingTree.size(); ++nodeIndex) {
            const PairRoutingTreeNode &inner = routingTree[nodeIndex];
            py::dict innerNode;
            innerNode["id"] = nodeIndex;
            innerNode["is_leaf"] = inner.isLeaf;
            if (inner.isLeaf) {
              innerNode["bin"] = inner.bin;
              leafBins.append(inner.bin);
            } else {
              innerNode["feature"] = inner.rawFeature;
              innerNode["axis"] = inner.featurePosition;
              innerNode["kind"] = inner.featureType == FeatureType::Categorical
                                      ? "categorical"
                                      : "continuous";
              innerNode["threshold"] =
                  inner.featureType == FeatureType::Continuous
                      ? py::cast(inner.threshold)
                      : py::none();
              innerNode["category"] =
                  inner.featureType == FeatureType::Categorical
                      ? py::cast(inner.rawFeature)
                      : py::none();
              innerNode["left"] = inner.left;
              innerNode["right"] = inner.right;
              innerNode["missing"] = inner.missing;
            }
            innerTree.append(innerNode);
          }
          d["pair_inner_tree"] = innerTree;
          d["pair_leaf_bins"] = leafBins;
        }
        const bool isCategorical =
            isCategoricalInnerDiscretizer(*n.innerDiscretizer);
        d["is_categorical"] = isCategorical;
        d["thresholds"] = innerThresholdsPy(n);
        if (isCategorical)
          d["bin_categories"] = categoricalBinCategoriesPy(n);
        py::list b2p;
        for (size_t p : n.binToPartition) b2p.append(p);
        d["bin_to_partition"] = b2p;
        py::list bc;
        for (const auto &bin : n.splitLeafStats) {
          py::list outputs;
          for (const auto &moments : bin) {
            py::list row;
            for (double v : moments) row.append(v);
            outputs.append(row);
          }
          bc.append(outputs);
        }
        d["bin_counts"] = bc;
        py::list bw;
        for (double w : n.splitBinWeights) bw.append(w);
        d["bin_weights"] = bw;
        py::list bsc;
        for (size_t v : n.binSampleCounts) bsc.append(v);
        d["bin_sample_counts"] = bsc;
        if (!pairDisc && !taoPairDisc)
          d["nan_prediction_partition"] =
              n.binToPartition.empty() ? 0 : n.binToPartition.back();
        py::list ch;
        if (i < childIdx.size()) {
          for (size_t c : childIdx[i]) ch.append(c);
        }
        d["children"] = ch;
      }
      nodes_list.append(d);
    }
    out["nodes"] = nodes_list;
    return out;
  }

private:
  std::unique_ptr<RegressionShapeGeneralizedTree> impl_;
  std::string criterionStr_;
};

} // namespace sgt::bindings
