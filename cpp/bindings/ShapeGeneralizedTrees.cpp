/**
 * @file ShapeGeneralizedTrees.cpp
 * @brief pybind11 module ``ShapeGeneralizedTrees`` exposing ``ClassificationShapeGeneralizedTree``.
 *
 * Inputs follow the sklearn convention (X is (n_samples, n_features),
 * C-contiguous). The `_arma_bridge.h` helper turns those into Armadillo
 * matrices with zero copies in the canonical case.
 */

#include "_arma_bridge.h"

#include "Domain/LearningCriterion.h"
#include "Estimators/ClassificationShapeGeneralizedTree.h"
#include "Estimators/RegressionShapeGeneralizedTree.h"
#include "algorithms/FeatureBagging.h"

#include <algorithm>
#include <armadillo>
#include <cctype>
#include <memory>
#include <pybind11/numpy.h>
#include <span>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

namespace py = pybind11;
namespace bridge = sgt::bindings;

namespace {

std::string normalizeCriterion(std::string s) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

LearningCriterion parseClassificationCriterion(const std::string &raw) {
  const std::string s = normalizeCriterion(raw);
  if (s == "gini")
    return LearningCriterion::Gini;
  if (s == "entropy" || s == "log_loss")
    return LearningCriterion::Entropy;
  throw std::invalid_argument(
      "criterion must be 'gini', 'entropy', or 'log_loss'; got '" + raw + "'");
}

LearningCriterion parseRegressionCriterion(const std::string &raw) {
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

FeatureBaggingPickFn parseMaxFeaturesPy(py::handle mf_h) {
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
  if (py::cast<bool>(py::isinstance(mf, numbers.attr("Integral")))) {
    const long long vll = py::cast<long long>(mf);
    if (vll < 0)
      throw std::invalid_argument("max_features int must be non-negative");
    if (vll == 0)
      return makeFeatureBaggingAll();
    return makeFeatureBaggingCap(static_cast<size_t>(vll));
  }
  if (py::cast<bool>(py::isinstance(mf, numbers.attr("Real")))) {
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

/**
 * Thin Python adapter around `ClassificationShapeGeneralizedTree`. Owns the
 * C++ implementation and handles NumPy <-> Armadillo plumbing.
 */
class ClassificationShapeGeneralizedTreePy {
public:
  ClassificationShapeGeneralizedTreePy(
      std::string criterion, size_t numClasses, size_t numPartitions,
      size_t outerMinLeafSize, double outerMinGainSplit, size_t outerMaxDepth,
      size_t outerMaxLeafNodes, size_t innerMinLeafSize,
      double innerMinGainSplit, size_t innerMaxDepth, size_t innerMaxLeafNodes,
      size_t coordinateDescentMaxIters, size_t coordinateDescentPatience,
      bool coordinateDescentSmartInit, uint64_t random_state,
      py::object max_features = py::none()) {
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
        crit, numClasses, numPartitions, outer, inner, cd, random_state,
        parseMaxFeaturesPy(max_features));
  }

  void fit(const py::array &X, const py::array &y) {
    auto Xb = bridge::asSamplesByFeatures<float>(X, "X");
    /** Owning copy: zero-copy Row views from NumPy often fail Armadillo strict
     *  checks on some dtypes / strides; C++ expects `arma::Row<size_t>`. */
    arma::Col<size_t> y_col = bridge::as1DColOwning<size_t>(y, "y");

    if (y_col.n_elem != Xb.view().n_cols)
      throw std::invalid_argument(
          "y.shape[0] must equal X.shape[0] (number of samples)");

    arma::Row<size_t> y_row(y_col.n_elem);
    for (arma::uword i = 0; i < y_col.n_elem; ++i)
      y_row(i) = y_col(i);

    // The C++ trainer can be long-running; release the GIL so Python
    // threads make progress while we fit.
    py::gil_scoped_release release;
    impl_->fit(Xb.view(), y_row);
  }

  py::array_t<size_t> predict(const py::array &X) {
    auto Xb = bridge::asSamplesByFeatures<float>(X, "X");
    arma::Row<size_t> preds;
    {
      py::gil_scoped_release release;
      preds = impl_->predict(Xb.view());
    }
    return bridge::rowToNumpy(preds);
  }

  py::array_t<float> predictProba(const py::array &X) {
    auto Xb = bridge::asSamplesByFeatures<float>(X, "X");
    arma::fmat proba;
    {
      py::gil_scoped_release release;
      proba = impl_->predictProba(Xb.view());
    }
    // arma (numClasses, numSamples) -> numpy (numSamples, numClasses)
    return bridge::samplesByFeaturesToNumpy(proba);
  }

  size_t numLeaves() const { return impl_->numLeaves(); }
  size_t numNodes() const { return impl_->numNodes(); }
  bool isFitted() const { return impl_->isFitted(); }

private:
  std::unique_ptr<ClassificationShapeGeneralizedTree> impl_;
};

class RegressionShapeGeneralizedTreePy {
public:
  RegressionShapeGeneralizedTreePy(
      std::string criterion, size_t numPartitions, size_t outerMinLeafSize,
      double outerMinGainSplit, size_t outerMaxDepth, size_t outerMaxLeafNodes,
      size_t innerMinLeafSize, double innerMinGainSplit, size_t innerMaxDepth,
      size_t innerMaxLeafNodes, size_t coordinateDescentMaxIters,
      size_t coordinateDescentPatience, bool /* coordinateDescentSmartInit */,
      uint64_t random_state, py::object max_features = py::none()) {
    const LearningCriterion crit = parseRegressionCriterion(criterion);
    const TreeBuildingParams outer{outerMinLeafSize, outerMinGainSplit,
                                   outerMaxDepth, outerMaxLeafNodes};
    const TreeBuildingParams inner{innerMinLeafSize, innerMinGainSplit,
                                   innerMaxDepth, innerMaxLeafNodes};
    CoordinateDescentParams cd;
    cd.maxIters = coordinateDescentMaxIters;
    cd.patience = coordinateDescentPatience;
    cd.smartInit = false;
    impl_ = std::make_unique<RegressionShapeGeneralizedTree>(
        crit, numPartitions, outer, inner, cd, random_state,
        parseMaxFeaturesPy(max_features));
  }

  void fit(const py::array &X, const py::array &y) {
    auto Xb = bridge::asSamplesByFeatures<float>(X, "X");
    auto yb = bridge::as1DRow<float>(y, "y");
    if (yb.view().n_elem != Xb.view().n_cols)
      throw std::invalid_argument(
          "y.shape[0] must equal X.shape[0] (number of samples)");
    py::gil_scoped_release release;
    impl_->fit(Xb.view(), yb.view());
  }

  py::array_t<float> predict(const py::array &X) {
    auto Xb = bridge::asSamplesByFeatures<float>(X, "X");
    arma::Row<float> preds;
    {
      py::gil_scoped_release release;
      preds = impl_->predict(Xb.view());
    }
    return bridge::rowToNumpy(preds);
  }

  size_t numLeaves() const { return impl_->numLeaves(); }
  size_t numNodes() const { return impl_->numNodes(); }
  bool isFitted() const { return impl_->isFitted(); }

  std::vector<std::vector<float>> leafRegressionStats() const {
    return impl_->leafRegressionStats;
  }

  std::vector<size_t> leafNumSamples() const { return impl_->leafNumSamples; }

private:
  std::unique_ptr<RegressionShapeGeneralizedTree> impl_;
};

} // namespace

PYBIND11_MODULE(ShapeGeneralizedTrees, m) {
  m.doc() =
      "Shape-Generalized Tree bindings (classification and regression).";

  py::class_<ClassificationShapeGeneralizedTreePy>(
      m, "ClassificationShapeGeneralizedTree")
      .def(py::init<std::string, size_t, size_t, size_t, double, size_t,
                    size_t, size_t, double, size_t, size_t, size_t, size_t,
                    bool, uint64_t, py::object>(),
           py::arg("criterion") = "gini", py::arg("num_classes"),
           py::arg("num_partitions") = 2,
           py::arg("outer_min_leaf_size") = 1,
           py::arg("outer_min_gain_split") = 1e-7,
           py::arg("outer_max_depth") = 0,
           py::arg("outer_max_leaf_nodes") = 0,
           py::arg("inner_min_leaf_size") = 1,
           py::arg("inner_min_gain_split") = 1e-7,
           py::arg("inner_max_depth") = 0,
           py::arg("inner_max_leaf_nodes") = 0,
           py::arg("coordinate_descent_max_iters") = 10,
           py::arg("coordinate_descent_patience") = 5,
           py::arg("coordinate_descent_smart_init") = true,
           py::arg("random_state") = 42,
           py::arg("max_features") = py::none())
      .def("fit", &ClassificationShapeGeneralizedTreePy::fit, py::arg("X"),
           py::arg("y"),
           "Fit the routing tree. X is (n_samples, n_features) float32; y is "
           "1-D uint class labels. All feature columns are routing candidates.")
      .def("predict", &ClassificationShapeGeneralizedTreePy::predict,
           py::arg("X"),
           "Predict class labels for X (shape (n_samples, n_features)).")
      .def("predict_proba", &ClassificationShapeGeneralizedTreePy::predictProba,
           py::arg("X"),
           "Predict class probabilities for X; output shape "
           "(n_samples, n_classes).")
      .def_property_readonly(
          "num_leaves", &ClassificationShapeGeneralizedTreePy::numLeaves)
      .def_property_readonly(
          "num_nodes", &ClassificationShapeGeneralizedTreePy::numNodes)
      .def_property_readonly(
          "is_fitted", &ClassificationShapeGeneralizedTreePy::isFitted);

  py::class_<RegressionShapeGeneralizedTreePy>(
      m, "RegressionShapeGeneralizedTree")
      .def(py::init<std::string, size_t, size_t, double, size_t, size_t, size_t,
                    double, size_t, size_t, size_t, size_t, bool, uint64_t,
                    py::object>(),
           py::arg("criterion") = "squared_error",
           py::arg("num_partitions") = 2, py::arg("outer_min_leaf_size") = 1,
           py::arg("outer_min_gain_split") = 1e-7, py::arg("outer_max_depth") = 0,
           py::arg("outer_max_leaf_nodes") = 0,
           py::arg("inner_min_leaf_size") = 1,
           py::arg("inner_min_gain_split") = 1e-7, py::arg("inner_max_depth") = 0,
           py::arg("inner_max_leaf_nodes") = 0,
           py::arg("coordinate_descent_max_iters") = 10,
           py::arg("coordinate_descent_patience") = 5,
           py::arg("coordinate_descent_smart_init") = true,
           py::arg("random_state") = 42, py::arg("max_features") = py::none(),
           R"(Regression tree: inner bins are round-robin seeded. ``squared_error`` runs
coordinate descent and keeps the map only if branch MSE improves clearly vs the seed;
otherwise the snapshot is restored and the branch objective is rebuilt.
``absolute_error`` / ``mae`` skip coordinate descent. coordinate_descent_smart_init
is accepted for API parity with ClassificationShapeGeneralizedTree but ignored.)")
      .def("fit", &RegressionShapeGeneralizedTreePy::fit, py::arg("X"),
           py::arg("y"),
           "Fit the routing tree. X is (n_samples, n_features) float32; y is "
           "1-D float32 targets.")
      .def("predict", &RegressionShapeGeneralizedTreePy::predict, py::arg("X"),
           "Predict scalar targets for X (shape (n_samples,)).")
      .def_property_readonly("num_leaves",
                             &RegressionShapeGeneralizedTreePy::numLeaves)
      .def_property_readonly("num_nodes",
                             &RegressionShapeGeneralizedTreePy::numNodes)
      .def_property_readonly("is_fitted",
                             &RegressionShapeGeneralizedTreePy::isFitted)
      .def_property_readonly(
          "leaf_regression_stats",
          &RegressionShapeGeneralizedTreePy::leafRegressionStats)
      .def_property_readonly("leaf_num_samples",
                             &RegressionShapeGeneralizedTreePy::leafNumSamples);
}
