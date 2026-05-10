/**
 * Python bindings for the Shape-Generalized Tree family.
 *
 * Module: ShapeGeneralizedTrees
 * Classes:
 *   - ClassificationShapeGeneralizedTree
 *
 * Inputs follow the sklearn convention (X is (n_samples, n_features),
 * C-contiguous). The `_arma_bridge.h` helper turns those into Armadillo
 * matrices with zero copies in the canonical case.
 */

#include "_arma_bridge.h"

#include "Domain/LearningCriterion.h"
#include "algorithms/ClassificationShapeGeneralizedTree.h"

#include <algorithm>
#include <armadillo>
#include <cctype>
#include <memory>
#include <pybind11/numpy.h>
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
      bool coordinateDescentSmartInit, size_t coordinateDescentSeed) {
    const LearningCriterion crit = parseClassificationCriterion(criterion);
    const TreeBuildingParams outer{outerMinLeafSize, outerMinGainSplit,
                                   outerMaxDepth, outerMaxLeafNodes};
    const TreeBuildingParams inner{innerMinLeafSize, innerMinGainSplit,
                                   innerMaxDepth, innerMaxLeafNodes};
    CoordinateDescentParams cd;
    cd.maxIters = coordinateDescentMaxIters;
    cd.patience = coordinateDescentPatience;
    cd.seed = coordinateDescentSeed;
    cd.smartInit = coordinateDescentSmartInit;
    impl_ = std::make_unique<ClassificationShapeGeneralizedTree>(
        crit, numClasses, numPartitions, outer, inner, cd);
  }

  void fit(const py::array &X, const py::array &features, const py::array &y) {
    auto Xb = bridge::asSamplesByFeatures<float>(X, "X");
    /** Owning copy: zero-copy Row views from NumPy often fail Armadillo strict
     *  checks on some dtypes / strides; C++ expects `arma::Row<size_t>`. */
    arma::Col<size_t> y_col = bridge::as1DColOwning<size_t>(y, "y");
    auto featuresArma =
        bridge::as1DColOwning<arma::uword>(features, "features");

    if (y_col.n_elem != Xb.view().n_cols)
      throw std::invalid_argument(
          "y.shape[0] must equal X.shape[0] (number of samples)");
    if (!featuresArma.is_empty() &&
        featuresArma.max() >= Xb.view().n_rows)
      throw std::invalid_argument(
          "features contains an index >= X.shape[1]");

    arma::Row<size_t> y_row(y_col.n_elem);
    for (arma::uword i = 0; i < y_col.n_elem; ++i)
      y_row(i) = y_col(i);

    // The C++ trainer can be long-running; release the GIL so Python
    // threads make progress while we fit.
    py::gil_scoped_release release;
    impl_->fit(Xb.view(), featuresArma, y_row);
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

} // namespace

PYBIND11_MODULE(ShapeGeneralizedTrees, m) {
  m.doc() = "Shape-Generalized Tree bindings (classification family).";

  py::class_<ClassificationShapeGeneralizedTreePy>(
      m, "ClassificationShapeGeneralizedTree")
      .def(py::init<std::string, size_t, size_t, size_t, double, size_t,
                    size_t, size_t, double, size_t, size_t, size_t, size_t,
                    bool, size_t>(),
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
           py::arg("coordinate_descent_seed") = 42)
      .def("fit", &ClassificationShapeGeneralizedTreePy::fit, py::arg("X"),
           py::arg("features"), py::arg("y"),
           "Fit the routing tree. X is (n_samples, n_features) float32, "
           "features is 1-D uint, y is 1-D uint class labels.")
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
}
