/**
 * @file TreeAlternatingOptimization.cpp
 * @brief Standalone pybind11 module ``TreeAlternatingOptimization`` exposing
 *        Tree-Alternating Optimization (TAO) refinement for fitted
 *        shape-generalized trees.
 *
 * TAO is intentionally decoupled from the estimator wrappers: instead of being
 * a method on ``ClassificationShapeGeneralizedTree``, it is a free function in
 * its own module that receives an already-registered estimator object and
 * refines its underlying C++ tree in place. The wrapper type is shared via
 * ``_sgt_estimators.h``; pybind11's cross-module type registry resolves the
 * argument as long as ``ShapeGeneralizedTrees`` has been imported first.
 */

#include "_arma_bridge.h"
#include "_sgt_estimators.h"

#include "Estimators/ClassificationShapeGeneralizedTree.h"
#include "Estimators/RegressionShapeGeneralizedTree.h"
#include "algorithms/TreeAlternatingOptimization.h"

#include <armadillo>
#include <cstddef>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;
namespace bridge = sgt::bindings;

namespace {

/**
 * Refine a fitted classification tree in place with TAO.
 *
 * Runs up to ``n_runs`` bottom-up sweeps over the internal nodes, replacing
 * each node's routing rule when doing so routes more samples toward a
 * correctly-classifying subtree. The training data must be supplied again
 * because sample partitions are not retained after ``fit``.
 */
void optimizeClassification(bridge::ClassificationShapeGeneralizedTreePy &tree,
                            const py::array &X, const py::array &y,
                            py::object sample_weight = py::none(),
                            size_t n_runs = 10, double lambda_ = 0.0) {
  if (!tree.isFitted())
    throw std::logic_error("optimize_classification: model is not fitted");

  auto Xb = bridge::asSamplesByFeatures<float>(X, "X");
  arma::Col<size_t> y_col = bridge::as1DColOwning<size_t>(y, "y");
  if (y_col.n_elem != Xb.view().n_cols)
    throw std::invalid_argument(
        "y.shape[0] must equal X.shape[0] (number of samples)");

  arma::Row<size_t> y_row(y_col.n_elem);
  for (arma::uword i = 0; i < y_col.n_elem; ++i)
    y_row(i) = y_col(i);

  const arma::Row<float> w_row =
      bridge::sampleWeightRowFromPy(sample_weight, Xb.view().n_cols);

  py::gil_scoped_release release;
  tao::optimizeClassification(tree.impl(), Xb.view(), y_row, w_row, n_runs,
                              lambda_);
}

/**
 * Refine a fitted regression tree in place with TAO.
 *
 * Identical sweep to ``optimizeClassification``; the only task-specific
 * differences (care set + pseudolabels and the leaf-statistic refresh) live in
 * the C++ ``tao::RegressionTask`` policy. The training data must be supplied
 * again because sample partitions are not retained after ``fit``.
 */
void optimizeRegression(bridge::RegressionShapeGeneralizedTreePy &tree,
                        const py::array &X, const py::array &y,
                        py::object sample_weight = py::none(),
                        size_t n_runs = 10, double lambda_ = 0.0) {
  if (!tree.isFitted())
    throw std::logic_error("optimize_regression: model is not fitted");

  auto Xb = bridge::asSamplesByFeatures<float>(X, "X");
  auto yb = bridge::as1DRow<float>(y, "y");
  if (yb.view().n_elem != Xb.view().n_cols)
    throw std::invalid_argument(
        "y.shape[0] must equal X.shape[0] (number of samples)");

  const arma::Row<float> w_row =
      bridge::sampleWeightRowFromPy(sample_weight, Xb.view().n_cols);

  py::gil_scoped_release release;
  tao::optimizeRegression(tree.impl(), Xb.view(), yb.view(), w_row, n_runs,
                          lambda_);
}

} // namespace

PYBIND11_MODULE(TreeAlternatingOptimization, m) {
  // CARMA's allocator backs Armadillo's heap and lazily calls _import_array()
  // on its first free, which must happen with the GIL held. The refinement
  // releases the GIL and destroys Armadillo temporaries afterwards, so prime
  // the numpy C-API table here at module load to keep those frees safe.
  if (_import_array() < 0) {
    PyErr_Clear();
    throw std::runtime_error(
        "TreeAlternatingOptimization: numpy.core.multiarray failed to import; "
        "ensure numpy is installed and importable before importing this module");
  }

  m.doc() = "Tree-Alternating Optimization (TAO) refinement for fitted "
            "shape-generalized trees.";

  m.def("optimize_classification", &optimizeClassification, py::arg("tree"),
        py::arg("X"), py::arg("y"), py::arg("sample_weight") = py::none(),
        py::arg("n_runs") = 10, py::arg("lambda_") = 0.0,
        "Refine a fitted ClassificationShapeGeneralizedTree in place with "
        "Tree-Alternating Optimization. X is (n_samples, n_features) float32; "
        "y is 1-D uint class labels (same data used for fit). Runs up to "
        "n_runs bottom-up sweeps; lambda_ penalizes per-split routing "
        "accuracy.");

  m.def("optimize_regression", &optimizeRegression, py::arg("tree"),
        py::arg("X"), py::arg("y"), py::arg("sample_weight") = py::none(),
        py::arg("n_runs") = 10, py::arg("lambda_") = 0.0,
        "Refine a fitted RegressionShapeGeneralizedTree in place with "
        "Tree-Alternating Optimization. X is (n_samples, n_features) float32; "
        "y is 1-D float32 targets (same data used for fit). Runs up to "
        "n_runs bottom-up sweeps; lambda_ penalizes per-split routing "
        "reward.");
}
