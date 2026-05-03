#include <carma>

#include <algorithm>
#include <armadillo>
#include <cctype>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <stdexcept>
#include <string>
#include <variant>

#include "UnivariateClassificationDiscretizer.h"
#include "UnivariateRegressionDiscretizer.h"

namespace py = pybind11;

namespace {

std::string normalize_criterion(std::string s) {
  const auto not_space = [](unsigned char c) { return !std::isspace(c); };
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), not_space));
  s.erase(std::find_if(s.rbegin(), s.rend(), not_space).base(), s.end());
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return s;
}

using GiniDisc = UnivariateClassificationDiscretizer<GiniSplitter>;
using EntropyDisc = UnivariateClassificationDiscretizer<EntropySplitter>;
using SqErrRegDisc = UnivariateRegressionDiscretizer<SquaredErrorSplitter>;

/** Armadillo Row is 1×N; NumPy indexing expects a 1-D array of length N. */
py::array_t<size_t> row_to_numpy_1d(const arma::Row<size_t> &row) {
  py::array_t<size_t> out({static_cast<py::ssize_t>(row.n_elem)});
  auto buf = out.mutable_unchecked<1>();
  for (size_t i = 0; i < row.n_elem; ++i)
    buf(static_cast<py::ssize_t>(i)) = row(i);
  return out;
}

py::array_t<size_t> col_to_numpy_1d(const arma::Col<size_t> &col) {
  py::array_t<size_t> out({static_cast<py::ssize_t>(col.n_elem)});
  auto buf = out.mutable_unchecked<1>();
  for (size_t i = 0; i < col.n_elem; ++i)
    buf(static_cast<py::ssize_t>(i)) = col(i);
  return out;
}

py::array_t<float> vector_float_to_numpy_1d(const std::vector<float> &v) {
  py::array_t<float> out({static_cast<py::ssize_t>(v.size())});
  auto buf = out.mutable_unchecked<1>();
  for (size_t i = 0; i < v.size(); ++i)
    buf(static_cast<py::ssize_t>(i)) = v[i];
  return out;
}

std::string normalize_regression_criterion(std::string s) {
  s = normalize_criterion(std::move(s));
  if (s == "mse")
    return "squared_error";
  return s;
}

void validate_regression_criterion(const std::string &criterion) {
  if (criterion == "absolute_error" || criterion == "mae")
    throw std::invalid_argument(
        "criterion 'absolute_error' / 'mae' is not implemented yet");
  if (criterion != "squared_error" && criterion != "friedman_mse")
    throw std::invalid_argument(
        "criterion must be 'squared_error', 'mse' (alias), or "
        "'friedman_mse' (got '" +
        criterion + "')");
}

py::list
VectorOfVectorsToNumpyList(const std::vector<std::vector<size_t>> &bins) {
  py::list out;
  for (const auto &bin : bins) {
    arma::Col<size_t> col(bin.size());
    for (size_t i = 0; i < bin.size(); ++i)
      col(i) = bin[i];
    out.append(col_to_numpy_1d(col));
  }
  return out;
}

class UnivariateClassificationDiscretizerPy {
  std::variant<GiniDisc, EntropyDisc> impl_;

public:
  explicit UnivariateClassificationDiscretizerPy(std::string criterion) {
    criterion = normalize_criterion(std::move(criterion));
    if (criterion == "gini")
      impl_.emplace<0>();
    else if (criterion == "entropy" || criterion == "log_loss")
      impl_.emplace<1>();
    else
      throw std::invalid_argument(
          "criterion must be 'gini', 'entropy', or 'log_loss' (got '" +
          criterion + "')");
  }

  void Train(const py::array_t<float> &X, const py::array_t<size_t> &features,
             const py::array_t<size_t> &y, size_t numClasses, size_t minLeafSize,
             double minGainSplit, size_t maxDepth, size_t maxLeafNodes) {
    py::array_t<float> Xcopy = py::array_t<float>::ensure(X);
    if (Xcopy.ndim() != 2)
      throw std::invalid_argument(
          "X must be a 2D numpy array with shape (N_samples, N_features)");
    const arma::fmat xArma = arma::fmat(carma::arr_to_mat<float>(Xcopy, true).t());

    py::array_t<size_t> fcopy = py::array_t<size_t>::ensure(features);
    if (fcopy.ndim() != 1)
      throw std::invalid_argument("features must be a 1D numpy array");
    const arma::uvec featuresArma = arma::conv_to<arma::uvec>::from(
        carma::arr_to_col<size_t>(fcopy, true));

    py::array_t<size_t> ycopy = py::array_t<size_t>::ensure(y);
    if (ycopy.ndim() != 1)
      throw std::invalid_argument("y must be a 1D numpy array");
    arma::Row<size_t> yArma = carma::arr_to_row<size_t>(ycopy, true);
    if (yArma.n_elem != xArma.n_cols)
      throw std::invalid_argument("y length must match X.shape[0]");

    arma::uvec featuresMut = featuresArma;
    std::visit(
        [&](auto &d) {
          d.Train(xArma, featuresMut, yArma, numClasses, minLeafSize,
                  minGainSplit, maxDepth, maxLeafNodes);
        },
        impl_);
  }

  py::array_t<size_t> transform(const py::array_t<float> &X) {
    py::array_t<float> Xcopy = py::array_t<float>::ensure(X);
    if (Xcopy.ndim() != 2)
      throw std::invalid_argument(
          "X must be a 2D numpy array with shape (N_samples, N_features)");
    const arma::fmat xArma = arma::fmat(carma::arr_to_mat<float>(Xcopy, true).t());
    arma::Row<size_t> bins;
    std::visit([&](auto &d) { d.transform(xArma, bins); }, impl_);
    return row_to_numpy_1d(bins);
  }

  py::list getInSampleDiscretizations() {
    return std::visit(
        [](auto &d) {
          return VectorOfVectorsToNumpyList(d.getInSampleDiscretizations());
        },
        impl_);
  }

  py::array_t<size_t> getBinPredictions() {
    return std::visit(
        [](auto &d) {
          const auto &preds = d.getBinPredictions();
          arma::Col<size_t> col(preds.size());
          for (size_t i = 0; i < preds.size(); ++i)
            col(i) = preds[i];
          return col_to_numpy_1d(col);
        },
        impl_);
  }

  size_t getNumLeaves() const {
    return std::visit([](const auto &d) { return d.numLeaves; }, impl_);
  }

  void setNumLeaves(size_t v) {
    std::visit([v](auto &d) { d.numLeaves = v; }, impl_);
  }
};

class UnivariateRegressionDiscretizerPy {
  SqErrRegDisc impl_;

public:
  explicit UnivariateRegressionDiscretizerPy(std::string criterion) {
    criterion = normalize_regression_criterion(std::move(criterion));
    validate_regression_criterion(criterion);
    (void)criterion;
  }

  void Train(const py::array_t<float> &X, const py::array_t<size_t> &features,
             const py::array_t<float> &y, size_t minLeafSize, double minGainSplit,
             size_t maxDepth, size_t maxLeafNodes) {
    py::array_t<float> Xcopy = py::array_t<float>::ensure(X);
    if (Xcopy.ndim() != 2)
      throw std::invalid_argument(
          "X must be a 2D numpy array with shape (N_samples, N_features)");
    const arma::fmat xArma = arma::fmat(carma::arr_to_mat<float>(Xcopy, true).t());

    py::array_t<size_t> fcopy = py::array_t<size_t>::ensure(features);
    if (fcopy.ndim() != 1)
      throw std::invalid_argument("features must be a 1D numpy array");
    const arma::uvec featuresArma = arma::conv_to<arma::uvec>::from(
        carma::arr_to_col<size_t>(fcopy, true));

    py::array_t<float> ycopy = py::array_t<float>::ensure(y);
    if (ycopy.ndim() != 1)
      throw std::invalid_argument("y must be a 1D numpy array");
    arma::Row<float> yArma = carma::arr_to_row<float>(ycopy, true);
    if (yArma.n_elem != xArma.n_cols)
      throw std::invalid_argument("y length must match X.shape[0]");

    arma::uvec featuresMut = featuresArma;
    impl_.Train(xArma, featuresMut, yArma, minLeafSize, minGainSplit, maxDepth,
                maxLeafNodes);
  }

  py::array_t<size_t> transform(const py::array_t<float> &X) {
    py::array_t<float> Xcopy = py::array_t<float>::ensure(X);
    if (Xcopy.ndim() != 2)
      throw std::invalid_argument(
          "X must be a 2D numpy array with shape (N_samples, N_features)");
    const arma::fmat xArma = arma::fmat(carma::arr_to_mat<float>(Xcopy, true).t());
    arma::Row<size_t> bins;
    impl_.transform(xArma, bins);
    return row_to_numpy_1d(bins);
  }

  py::list getInSampleDiscretizations() {
    return VectorOfVectorsToNumpyList(impl_.getInSampleDiscretizations());
  }

  py::array_t<float> getBinPredictions() {
    return vector_float_to_numpy_1d(impl_.getBinPredictions());
  }

  size_t getNumLeaves() const { return impl_.numLeaves; }

  void setNumLeaves(size_t v) { impl_.numLeaves = v; }
};

} // namespace

PYBIND11_MODULE(Discretizers, m) {
  py::class_<UnivariateClassificationDiscretizerPy>(m,
                                                      "UnivariateClassificationDiscretizer")
      .def(py::init<std::string>(), py::arg("criterion") = "gini")
      .def(
          "Train",
          &UnivariateClassificationDiscretizerPy::Train, py::arg("X"),
          py::arg("features"), py::arg("y"), py::arg("numClasses"),
          py::arg("minLeafSize") = 1, py::arg("minGainSplit") = 1e-7,
          py::arg("maxDepth") = 0, py::arg("maxLeafNodes") = 0)
      .def("transform", &UnivariateClassificationDiscretizerPy::transform,
           py::arg("X"))
      .def("getInSampleDiscretizations",
           &UnivariateClassificationDiscretizerPy::getInSampleDiscretizations)
      .def("getBinPredictions",
           &UnivariateClassificationDiscretizerPy::getBinPredictions)
      .def_property("numLeaves", &UnivariateClassificationDiscretizerPy::getNumLeaves,
                    &UnivariateClassificationDiscretizerPy::setNumLeaves);

  py::class_<UnivariateRegressionDiscretizerPy>(m, "UnivariateRegressionDiscretizer")
      .def(py::init<std::string>(), py::arg("criterion") = "squared_error")
      .def(
          "Train",
          &UnivariateRegressionDiscretizerPy::Train, py::arg("X"),
          py::arg("features"), py::arg("y"), py::arg("minLeafSize") = 1,
          py::arg("minGainSplit") = 1e-7, py::arg("maxDepth") = 0,
          py::arg("maxLeafNodes") = 0)
      .def("transform", &UnivariateRegressionDiscretizerPy::transform,
           py::arg("X"))
      .def("getInSampleDiscretizations",
           &UnivariateRegressionDiscretizerPy::getInSampleDiscretizations)
      .def("getBinPredictions",
           &UnivariateRegressionDiscretizerPy::getBinPredictions)
      .def_property("numLeaves", &UnivariateRegressionDiscretizerPy::getNumLeaves,
                    &UnivariateRegressionDiscretizerPy::setNumLeaves);
}
