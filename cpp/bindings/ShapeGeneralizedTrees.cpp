/**
 * @file ShapeGeneralizedTrees.cpp
 * @brief pybind11 module ``ShapeGeneralizedTrees`` exposing the classification
 *        and regression shape-generalized tree estimators.
 *
 * Inputs follow the sklearn convention (X is (n_samples, n_features),
 * C-contiguous). The `_arma_bridge.h` helper turns those into Armadillo
 * matrices with zero copies in the canonical case. The wrapper classes
 * themselves live in `_sgt_estimators.h` so sibling modules (e.g.
 * ``TreeAlternatingOptimization``) can operate on the same estimator types.
 */

#include "_sgt_estimators.h"

#include <cstdint>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

namespace py = pybind11;
using sgt::bindings::ClassificationShapeGeneralizedTreePy;
using sgt::bindings::RegressionShapeGeneralizedTreePy;

PYBIND11_MODULE(ShapeGeneralizedTrees, m) {
  // CARMA's allocator may lazily call _import_array() on its first use, which
  // must happen with the GIL held. The trainer releases the GIL during fit, so
  // a small first-allocation there can segfault on numpy 2.x. Prime the C-API
  // table here at module load (the GIL is held) so CARMA's later calls are no-ops.
  if (_import_array() < 0) {
    PyErr_Clear();
    throw std::runtime_error(
        "ShapeGeneralizedTrees: numpy.core.multiarray failed to import; "
        "ensure numpy is installed and importable before importing this module");
  }

  m.doc() =
      "Shape-Generalized Tree bindings (classification and regression).";

  py::class_<ClassificationShapeGeneralizedTreePy>(
      m, "ClassificationShapeGeneralizedTree")
      .def(py::init([](std::string criterion, py::object num_classes,
                       size_t num_partitions, size_t outer_min_leaf_size,
                       double outer_min_gain_split, size_t outer_max_depth,
                       size_t outer_max_leaf_nodes, size_t inner_min_leaf_size,
                       double inner_min_gain_split, size_t inner_max_depth,
                       size_t inner_max_leaf_nodes,
                       size_t coordinate_descent_max_iters,
                       size_t coordinate_descent_patience,
                       bool coordinate_descent_smart_init, uint64_t random_state,
                       py::object max_features, size_t pairwise_candidates,
                       double pairwise_penalty) {
             return ClassificationShapeGeneralizedTreePy(
                 std::move(criterion), std::move(num_classes), num_partitions,
                 outer_min_leaf_size, outer_min_gain_split, outer_max_depth,
                 outer_max_leaf_nodes, inner_min_leaf_size, inner_min_gain_split,
                 inner_max_depth, inner_max_leaf_nodes,
                 coordinate_descent_max_iters, coordinate_descent_patience,
                 coordinate_descent_smart_init, random_state,
                 std::move(max_features), pairwise_candidates,
                 pairwise_penalty);
           }),
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
           py::arg("max_features") = py::none(),
           py::arg("pairwise_candidates") = 0,
           py::arg("pairwise_penalty") = 0.0)
      .def("fit", &ClassificationShapeGeneralizedTreePy::fit, py::arg("X"),
           py::arg("y"), py::arg("sample_weight") = py::none(),
           py::arg("features"),
           "Fit the routing tree. X is (n_samples, n_features) float32; y is "
           "uint class labels, 1-D (n_samples,) or 2-D (n_samples, n_outputs). "
           "Optional sample_weight is 1-D float32.")
      .def("predict", &ClassificationShapeGeneralizedTreePy::predict,
           py::arg("X"),
           "Predict class labels for X (shape (n_samples, n_features)). Returns "
           "(n_samples,) for a single output or (n_samples, n_outputs) for "
           "multi-output.")
      .def("predict_proba", &ClassificationShapeGeneralizedTreePy::predictProba,
           py::arg("X"),
           "Predict class probabilities for X. Single output: a "
           "(n_samples, n_classes) array. Multi-output: a list of such arrays, "
           "one per output.")
      .def_property_readonly(
          "num_leaves", &ClassificationShapeGeneralizedTreePy::numLeaves)
      .def_property_readonly(
          "num_nodes", &ClassificationShapeGeneralizedTreePy::numNodes)
      .def_property_readonly(
          "n_outputs", &ClassificationShapeGeneralizedTreePy::nOutputs)
      .def_property_readonly(
          "classes_per_output",
          &ClassificationShapeGeneralizedTreePy::classesPerOutput)
      .def_property_readonly(
          "is_fitted", &ClassificationShapeGeneralizedTreePy::isFitted)
      .def_property_readonly(
          "has_pair_nodes", &ClassificationShapeGeneralizedTreePy::hasPairNodes)
      .def_property_readonly(
          "feature_importance",
          &ClassificationShapeGeneralizedTreePy::featureImportance,
          "Normalized feature importances aligned with the features sequence "
          "passed to fit. Available only after training.")
      .def("tree_export", &ClassificationShapeGeneralizedTreePy::tree_export,
           "Return a flat snapshot of the fitted tree as a Python dict.");

  py::class_<RegressionShapeGeneralizedTreePy>(
      m, "RegressionShapeGeneralizedTree")
      .def(py::init([](std::string criterion, size_t num_partitions,
                       size_t outer_min_leaf_size, double outer_min_gain_split,
                       size_t outer_max_depth, size_t outer_max_leaf_nodes,
                       size_t inner_min_leaf_size, double inner_min_gain_split,
                       size_t inner_max_depth, size_t inner_max_leaf_nodes,
                       size_t coordinate_descent_max_iters,
                       size_t coordinate_descent_patience,
                       bool coordinate_descent_smart_init, uint64_t random_state,
                       py::object max_features, size_t pairwise_candidates,
                       double pairwise_penalty) {
             return RegressionShapeGeneralizedTreePy(
                 std::move(criterion), num_partitions, outer_min_leaf_size,
                 outer_min_gain_split, outer_max_depth, outer_max_leaf_nodes,
                 inner_min_leaf_size, inner_min_gain_split, inner_max_depth,
                 inner_max_leaf_nodes, coordinate_descent_max_iters,
                 coordinate_descent_patience, coordinate_descent_smart_init,
                 random_state, std::move(max_features), pairwise_candidates,
                 pairwise_penalty);
           }),
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
           py::arg("pairwise_candidates") = 0,
           py::arg("pairwise_penalty") = 0.0,
           R"(Regression tree: inner bins are round-robin seeded. ``squared_error`` runs
coordinate descent and keeps the map only if branch MSE improves clearly vs the seed;
otherwise the snapshot is restored and the branch objective is rebuilt.
``absolute_error`` / ``mae`` skip coordinate descent. coordinate_descent_smart_init
is accepted for API parity with ClassificationShapeGeneralizedTree but ignored.)")
      .def("fit", &RegressionShapeGeneralizedTreePy::fit, py::arg("X"),
           py::arg("y"), py::arg("sample_weight") = py::none(),
           py::arg("features"),
           "Fit the routing tree. X is (n_samples, n_features) float32; y is "
           "float32 targets, 1-D (n_samples,) or 2-D (n_samples, n_outputs). "
           "Optional sample_weight is 1-D float32.")
      .def("predict", &RegressionShapeGeneralizedTreePy::predict, py::arg("X"),
           "Predict targets for X. Returns (n_samples,) for a single output or "
           "(n_samples, n_outputs) for multi-output.")
      .def_property_readonly("num_leaves",
                             &RegressionShapeGeneralizedTreePy::numLeaves)
      .def_property_readonly("num_nodes",
                             &RegressionShapeGeneralizedTreePy::numNodes)
      .def_property_readonly("n_outputs",
                             &RegressionShapeGeneralizedTreePy::nOutputs)
      .def_property_readonly("is_fitted",
                             &RegressionShapeGeneralizedTreePy::isFitted)
      .def_property_readonly(
          "has_pair_nodes", &RegressionShapeGeneralizedTreePy::hasPairNodes)
      .def_property_readonly(
          "feature_importance",
          &RegressionShapeGeneralizedTreePy::featureImportance,
          "Normalized feature importances aligned with the features sequence "
          "passed to fit. Available only after training.")
      .def_property_readonly(
          "leaf_regression_stats",
          &RegressionShapeGeneralizedTreePy::leafRegressionStats)
      .def_property_readonly("leaf_num_samples",
                             &RegressionShapeGeneralizedTreePy::leafNumSamples)
      .def("tree_export", &RegressionShapeGeneralizedTreePy::tree_export,
           "Return a flat snapshot of the fitted tree as a Python dict.");
}
