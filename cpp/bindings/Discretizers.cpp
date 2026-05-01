#include <armadillo>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "UnivariateClassificationDiscretizer.h"

namespace py = pybind11;

namespace {

arma::fmat NumpyToArmaFeaturesBySamples(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &X) {
  const auto xbuf = X.request();
  if (xbuf.ndim != 2) {
    throw std::invalid_argument(
        "X must be a 2D numpy array with shape (N_samples, N_features)");
  }

  const auto nSamples = static_cast<size_t>(xbuf.shape[0]);
  const auto nFeatures = static_cast<size_t>(xbuf.shape[1]);
  const auto *xptr = static_cast<const float *>(xbuf.ptr);

  arma::fmat xArma(nFeatures, nSamples);
  for (size_t i = 0; i < nSamples; ++i) {
    for (size_t j = 0; j < nFeatures; ++j) {
      xArma(j, i) = xptr[i * nFeatures + j];
    }
  }
  return xArma;
}

arma::uvec NumpyToArmaUvec(
    const py::array_t<size_t, py::array::c_style | py::array::forcecast>
        &values,
    const char *argName) {
  const auto vbuf = values.request();
  if (vbuf.ndim != 1) {
    throw std::invalid_argument(std::string(argName) +
                                " must be a 1D numpy array");
  }

  const auto n = static_cast<size_t>(vbuf.shape[0]);
  const auto *ptr = static_cast<const size_t *>(vbuf.ptr);
  arma::uvec out(n);
  for (size_t i = 0; i < n; ++i) {
    out(i) = ptr[i];
  }
  return out;
}

arma::Row<size_t> NumpyToArmaTargetRow(
    const py::array_t<size_t, py::array::c_style | py::array::forcecast> &y,
    size_t expectedLength) {
  const auto lbuf = y.request();
  if (lbuf.ndim != 1) {
    throw std::invalid_argument("y must be a 1D numpy array");
  }

  const auto n = static_cast<size_t>(lbuf.shape[0]);
  if (n != expectedLength) {
    throw std::invalid_argument("y length must match X.shape[0]");
  }

  const auto *ptr = static_cast<const size_t *>(lbuf.ptr);
  arma::Row<size_t> out(n);
  for (size_t i = 0; i < n; ++i) {
    out(i) = ptr[i];
  }
  return out;
}

py::array_t<size_t> ArmaRowToNumpy(const arma::Row<size_t> &row) {
  py::array_t<size_t> out(row.n_elem);
  auto outBuf = out.request();
  auto *outPtr = static_cast<size_t *>(outBuf.ptr);
  for (size_t i = 0; i < row.n_elem; ++i) {
    outPtr[i] = row(i);
  }
  return out;
}

py::list
VectorOfVectorsToNumpyList(const std::vector<std::vector<size_t>> &bins) {
  py::list out;
  for (const auto &bin : bins) {
    py::array_t<size_t> arr(bin.size());
    auto arrBuf = arr.request();
    auto *arrPtr = static_cast<size_t *>(arrBuf.ptr);
    for (size_t i = 0; i < bin.size(); ++i) {
      arrPtr[i] = bin[i];
    }
    out.append(arr);
  }
  return out;
}

py::array_t<size_t> VectorToNumpy(const std::vector<size_t> &values) {
  py::array_t<size_t> out(values.size());
  auto outBuf = out.request();
  auto *outPtr = static_cast<size_t *>(outBuf.ptr);
  for (size_t i = 0; i < values.size(); ++i) {
    outPtr[i] = values[i];
  }
  return out;
}

py::array_t<float> VectorFloatToNumpy(const std::vector<float> &values) {
  py::array_t<float> out(values.size());
  auto outBuf = out.request();
  auto *outPtr = static_cast<float *>(outBuf.ptr);
  for (size_t i = 0; i < values.size(); ++i) {
    outPtr[i] = values[i];
  }
  return out;
}

arma::frowvec NumpyToArmaFloatTargetRow(
    const py::array_t<float, py::array::c_style | py::array::forcecast> &y,
    size_t expectedLength) {
  const auto ybuf = y.request();
  if (ybuf.ndim != 1) {
    throw std::invalid_argument("y must be a 1D numpy array");
  }
  const auto n = static_cast<size_t>(ybuf.shape[0]);
  if (n != expectedLength) {
    throw std::invalid_argument("y length must match X.shape[0]");
  }
  const auto *ptr = static_cast<const float *>(ybuf.ptr);
  arma::frowvec out(n);
  for (size_t i = 0; i < n; ++i) {
    out(i) = ptr[i];
  }
  return out;
}

} // namespace

PYBIND11_MODULE(Discretizers, m) {

  // region UnivariateClassificationDiscretizer
  py::class_<UnivariateClassificationDiscretizer>(
      m, "UnivariateClassificationDiscretizer")
      .def(py::init<>())
      .def(
          "Train",
          [](UnivariateClassificationDiscretizer &self,
             const py::array_t<float, py::array::c_style | py::array::forcecast>
                 &X,
             const py::array_t<size_t, py::array::c_style |
                                           py::array::forcecast> &features,
             const py::array_t<size_t,
                               py::array::c_style | py::array::forcecast> &y,
             size_t numClasses, size_t minLeafSize, double minGainSplit,
             size_t maxDepth, size_t maxLeafNodes) {
            arma::fmat xArma = NumpyToArmaFeaturesBySamples(X);
            arma::uvec featuresArma = NumpyToArmaUvec(features, "features");
            arma::Row<size_t> yArma = NumpyToArmaTargetRow(y, xArma.n_cols);
            self.Train(xArma, featuresArma, yArma, numClasses, minLeafSize,
                       minGainSplit, maxDepth, maxLeafNodes);
          },
          py::arg("X"), py::arg("features"), py::arg("y"),
          py::arg("numClasses"), py::arg("minLeafSize") = 1,
          py::arg("minGainSplit") = 1e-7, py::arg("maxDepth") = 0,
          py::arg("maxLeafNodes") = 0)
      .def(
          "transform",
          [](UnivariateClassificationDiscretizer &self,
             const py::array_t<float, py::array::c_style | py::array::forcecast>
                 &X) {
            arma::fmat xArma = NumpyToArmaFeaturesBySamples(X);
            arma::Row<size_t> bins;
            self.transform(xArma, bins);
            return ArmaRowToNumpy(bins);
          },
          py::arg("X"))
      .def("getInSampleDiscretizations",
           [](UnivariateClassificationDiscretizer &self) {
             return VectorOfVectorsToNumpyList(
                 self.getInSampleDiscretizations());
           })
      .def("getBinPredictions",
           [](UnivariateClassificationDiscretizer &self) {
             return VectorToNumpy(self.getBinPredictions());
           })
      .def_readwrite("numClasses",
                     &UnivariateClassificationDiscretizer::numClasses)
      .def_readwrite("minLeafSize",
                     &UnivariateClassificationDiscretizer::minLeafSize)
      .def_readwrite("minGainSplit",
                     &UnivariateClassificationDiscretizer::minGainSplit)
      .def_readwrite("maxDepth", &UnivariateClassificationDiscretizer::maxDepth)
      .def_readwrite("maxLeafNodes",
                     &UnivariateClassificationDiscretizer::maxLeafNodes)
      .def_readwrite("depth", &UnivariateClassificationDiscretizer::depth)
      .def_readwrite("numLeaves",
                     &UnivariateClassificationDiscretizer::numLeaves)
      .def_readwrite("numNodes",
                     &UnivariateClassificationDiscretizer::numNodes);
  // endregion UnivariateClassificationDiscretizer
}
