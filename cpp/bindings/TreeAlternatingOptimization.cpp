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
#include "algorithms/TAO/ClassificationTaoAdapter.h"
#include "algorithms/TAO/RegressionTaoAdapter.h"
#include "algorithms/TAO/TaoAdapter.h"
#include "algorithms/TAO/TreeAlternatingOptimization.h"

#include <armadillo>
#include <cstddef>
#include <memory>
#include <optional>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <utility>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;
namespace bridge = sgt::bindings;

namespace {

/** Owns parsed training data and the task-specific ``TaoAdapter`` for one run. */
class TaoRunContext {
public:
  static TaoRunContext make(py::object tree, const py::array &X,
                            const py::array &y, py::object sample_weight);

  bool isFitted() const { return fitted_; }

  tao::TaoAdapter &adapter() { return *adapter_; }

private:
  TaoRunContext(
      bool fitted, bridge::ArmaMatBridge<float> Xb, arma::Row<float> sampleWeights,
      arma::Row<size_t> yClassification,
      ClassificationShapeGeneralizedTree &tree);

  TaoRunContext(
      bool fitted, bridge::ArmaMatBridge<float> Xb, arma::Row<float> sampleWeights,
      bridge::ArmaRowBridge<float> yRegression, RegressionShapeGeneralizedTree &tree);

  bool fitted_ = false;
  bridge::ArmaMatBridge<float> Xb_;
  arma::Row<float> sampleWeights_;
  arma::Row<size_t> yClassification_;
  std::optional<bridge::ArmaRowBridge<float>> yRegression_;
  std::unique_ptr<tao::TaoAdapter> adapter_;
};

TaoRunContext::TaoRunContext(
    bool fitted, bridge::ArmaMatBridge<float> Xb, arma::Row<float> sampleWeights,
    arma::Row<size_t> yClassification, ClassificationShapeGeneralizedTree &tree)
    : fitted_(fitted), Xb_(std::move(Xb)),
      sampleWeights_(std::move(sampleWeights)),
      yClassification_(std::move(yClassification)),
      adapter_(std::make_unique<tao::ClassificationTaoAdapter>(
          tree, Xb_.view(), yClassification_, sampleWeights_)) {}

TaoRunContext::TaoRunContext(
    bool fitted, bridge::ArmaMatBridge<float> Xb, arma::Row<float> sampleWeights,
    bridge::ArmaRowBridge<float> yRegression, RegressionShapeGeneralizedTree &tree)
    : fitted_(fitted), Xb_(std::move(Xb)),
      sampleWeights_(std::move(sampleWeights)),
      yRegression_(std::move(yRegression)),
      adapter_(std::make_unique<tao::RegressionTaoAdapter>(
          tree, Xb_.view(), yRegression_->view(), sampleWeights_)) {}

TaoRunContext TaoRunContext::make(py::object tree, const py::array &X,
                                  const py::array &y, py::object sample_weight) {
  auto Xb = bridge::asSamplesByFeatures<float>(X, "X");
  arma::Row<float> w_row =
      bridge::sampleWeightRowFromPy(sample_weight, Xb.view().n_cols);

  if (py::isinstance<bridge::ClassificationShapeGeneralizedTreePy>(tree)) {
    auto &clsTree = tree.cast<bridge::ClassificationShapeGeneralizedTreePy &>();
    arma::Col<size_t> y_col = bridge::as1DColOwning<size_t>(y, "y");
    if (y_col.n_elem != Xb.view().n_cols)
      throw std::invalid_argument(
          "y.shape[0] must equal X.shape[0] (number of samples)");
    arma::Row<size_t> y_row(y_col.n_elem);
    for (arma::uword i = 0; i < y_col.n_elem; ++i)
      y_row(i) = y_col(i);

    return TaoRunContext(clsTree.isFitted(), std::move(Xb), std::move(w_row),
                         std::move(y_row), clsTree.impl());
  }

  if (py::isinstance<bridge::RegressionShapeGeneralizedTreePy>(tree)) {
    auto &regTree = tree.cast<bridge::RegressionShapeGeneralizedTreePy &>();
    auto yb = bridge::as1DRow<float>(y, "y");
    if (yb.view().n_elem != Xb.view().n_cols)
      throw std::invalid_argument(
          "y.shape[0] must equal X.shape[0] (number of samples)");

    return TaoRunContext(regTree.isFitted(), std::move(Xb), std::move(w_row),
                         std::move(yb), regTree.impl());
  }

  throw std::runtime_error(
      "TreeAlternatingOptimization: tree type is not supported; expected a "
      "ClassificationShapeGeneralizedTree or RegressionShapeGeneralizedTree "
      "from ShapeGeneralizedTrees");
}

/**
 * Refine a fitted shape-generalized tree in place with TAO.
 *
 * Dispatches to a task-specific ``TaoAdapter`` via ``TaoRunContext::make``.
 * Training data must be supplied again because sample partitions are not
 * retained after ``fit``.
 */
void TreeAlternatingOptimization(py::object tree, const py::array &X,
                                 const py::array &y,
                                 py::object sample_weight = py::none(),
                                 size_t n_runs = 10, double lambda_ = 0.0) {
  TaoRunContext ctx = TaoRunContext::make(tree, X, y, sample_weight);
  if (!ctx.isFitted())
    throw std::logic_error("TreeAlternatingOptimization: model is not fitted");

  py::gil_scoped_release release;
  tao::optimize(ctx.adapter(), n_runs, lambda_);
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

  m.def("TreeAlternatingOptimization", &TreeAlternatingOptimization,
        py::arg("tree"), py::arg("X"), py::arg("y"),
        py::arg("sample_weight") = py::none(), py::arg("n_runs") = 10,
        py::arg("lambda_") = 0.0,
        "Refine a fitted ClassificationShapeGeneralizedTree or "
        "RegressionShapeGeneralizedTree in place. X is "
        "(n_samples, n_features) float32; y is 1-D class labels (uint) or "
        "float targets matching the tree type. Runs up to n_runs bottom-up "
        "sweeps; lambda_ penalizes non-constant routing splits by "
        "lambda_ * totalSampleWeight in weighted reward units.");
}
