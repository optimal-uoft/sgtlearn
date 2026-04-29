#include <armadillo>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "UnivariateDiscretizer.h"

namespace py = pybind11;

namespace {

arma::fmat NumpyToArmaFeaturesBySamples(const py::array_t<float, py::array::c_style | py::array::forcecast>& X) {
    const auto xbuf = X.request();
    if (xbuf.ndim != 2) {
        throw std::invalid_argument("X must be a 2D numpy array with shape (N_samples, N_features)");
    }

    const auto nSamples = static_cast<size_t>(xbuf.shape[0]);
    const auto nFeatures = static_cast<size_t>(xbuf.shape[1]);
    const auto* xptr = static_cast<const float*>(xbuf.ptr);

    arma::fmat xArma(nFeatures, nSamples);
    for (size_t i = 0; i < nSamples; ++i) {
        for (size_t j = 0; j < nFeatures; ++j) {
            xArma(j, i) = xptr[i * nFeatures + j];
        }
    }
    return xArma;
}

arma::uvec NumpyToArmaUvec(
    const py::array_t<size_t, py::array::c_style | py::array::forcecast>& values,
    const char* argName
) {
    const auto vbuf = values.request();
    if (vbuf.ndim != 1) {
        throw std::invalid_argument(std::string(argName) + " must be a 1D numpy array");
    }

    const auto n = static_cast<size_t>(vbuf.shape[0]);
    const auto* ptr = static_cast<const size_t*>(vbuf.ptr);
    arma::uvec out(n);
    for (size_t i = 0; i < n; ++i) {
        out(i) = ptr[i];
    }
    return out;
}

arma::Row<size_t> NumpyToArmaLabelRow(
    const py::array_t<size_t, py::array::c_style | py::array::forcecast>& labels,
    size_t expectedLength
) {
    const auto lbuf = labels.request();
    if (lbuf.ndim != 1) {
        throw std::invalid_argument("labels must be a 1D numpy array");
    }

    const auto n = static_cast<size_t>(lbuf.shape[0]);
    if (n != expectedLength) {
        throw std::invalid_argument("labels length must match X.shape[0]");
    }

    const auto* ptr = static_cast<const size_t*>(lbuf.ptr);
    arma::Row<size_t> out(n);
    for (size_t i = 0; i < n; ++i) {
        out(i) = ptr[i];
    }
    return out;
}

arma::frowvec NumpyToArmaResponseRow(
    const py::array_t<float, py::array::c_style | py::array::forcecast>& responses,
    size_t expectedLength
) {
    const auto rbuf = responses.request();
    if (rbuf.ndim != 1) {
        throw std::invalid_argument("responses must be a 1D numpy array");
    }

    const auto n = static_cast<size_t>(rbuf.shape[0]);
    if (n != expectedLength) {
        throw std::invalid_argument("responses length must match X.shape[0]");
    }

    const auto* ptr = static_cast<const float*>(rbuf.ptr);
    arma::frowvec out(n);
    for (size_t i = 0; i < n; ++i) {
        out(i) = ptr[i];
    }
    return out;
}

py::array_t<size_t> ArmaRowToNumpy(const arma::Row<size_t>& row) {
    py::array_t<size_t> out(row.n_elem);
    auto outBuf = out.request();
    auto* outPtr = static_cast<size_t*>(outBuf.ptr);
    for (size_t i = 0; i < row.n_elem; ++i) {
        outPtr[i] = row(i);
    }
    return out;
}

py::list VectorOfVectorsToNumpyList(const std::vector<std::vector<size_t>>& bins) {
    py::list out;
    for (const auto& bin : bins) {
        py::array_t<size_t> arr(bin.size());
        auto arrBuf = arr.request();
        auto* arrPtr = static_cast<size_t*>(arrBuf.ptr);
        for (size_t i = 0; i < bin.size(); ++i) {
            arrPtr[i] = bin[i];
        }
        out.append(arr);
    }
    return out;
}

py::array_t<size_t> VectorToNumpy(const std::vector<size_t>& values) {
    py::array_t<size_t> out(values.size());
    auto outBuf = out.request();
    auto* outPtr = static_cast<size_t*>(outBuf.ptr);
    for (size_t i = 0; i < values.size(); ++i) {
        outPtr[i] = values[i];
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(Discretizers, m) {

    // region UnivariateDiscretizer
    py::class_<UnivariateDiscretizer>(m, "UnivariateDiscretizer")
        .def(py::init<>())
        .def(
            "Train",
            [](UnivariateDiscretizer& self,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& X,
               const py::array_t<size_t, py::array::c_style | py::array::forcecast>& features,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& responses,
               size_t minLeafSize,
               double minGainSplit,
               size_t maxDepth,
               size_t maxLeafNodes) {
                arma::fmat xArma = NumpyToArmaFeaturesBySamples(X);
                arma::uvec featuresArma = NumpyToArmaUvec(features, "features");
                arma::frowvec responsesArma = NumpyToArmaResponseRow(responses, xArma.n_cols);
                self.Train(xArma, featuresArma, responsesArma, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
            },
            py::arg("X"),
            py::arg("features"),
            py::arg("responses"),
            py::arg("minLeafSize") = 1,
            py::arg("minGainSplit") = 1e-7,
            py::arg("maxDepth") = 0,
            py::arg("maxLeafNodes") = 0
        )
        .def(
            "Train",
            [](UnivariateDiscretizer& self,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& X,
               const py::array_t<size_t, py::array::c_style | py::array::forcecast>& features,
               const py::array_t<size_t, py::array::c_style | py::array::forcecast>& labels,
               size_t numClasses,
               size_t minLeafSize,
               double minGainSplit,
               size_t maxDepth,
               size_t maxLeafNodes) {
                arma::fmat xArma = NumpyToArmaFeaturesBySamples(X);
                arma::uvec featuresArma = NumpyToArmaUvec(features, "features");
                arma::Row<size_t> labelsArma = NumpyToArmaLabelRow(labels, xArma.n_cols);
                self.Train(xArma, featuresArma, labelsArma, numClasses, minLeafSize, minGainSplit, maxDepth, maxLeafNodes);
            },
            py::arg("X"),
            py::arg("features"),
            py::arg("labels"),
            py::arg("numClasses"),
            py::arg("minLeafSize") = 1,
            py::arg("minGainSplit") = 1e-7,
            py::arg("maxDepth") = 0,
            py::arg("maxLeafNodes") = 0
        )
        .def(
            "transform",
            [](UnivariateDiscretizer& self,
               const py::array_t<float, py::array::c_style | py::array::forcecast>& X) {
                arma::fmat xArma = NumpyToArmaFeaturesBySamples(X);
                arma::Row<size_t> bins;
                self.transform(xArma, bins);
                return ArmaRowToNumpy(bins);
            },
            py::arg("X")
        )
        .def(
            "getInSampleDiscretizations",
            [](UnivariateDiscretizer& self) {
                return VectorOfVectorsToNumpyList(self.getInSampleDiscretizations());
            }
        )
        .def(
            "getBinPredictions",
            [](UnivariateDiscretizer& self) {
                return VectorToNumpy(self.getBinPredictions());
            }
        )
        .def_readwrite("numClasses", &UnivariateDiscretizer::numClasses)
        .def_readwrite("minLeafSize", &UnivariateDiscretizer::minLeafSize)
        .def_readwrite("minGainSplit", &UnivariateDiscretizer::minGainSplit)
        .def_readwrite("maxDepth", &UnivariateDiscretizer::maxDepth)
        .def_readwrite("maxLeafNodes", &UnivariateDiscretizer::maxLeafNodes)
        .def_readwrite("depth", &UnivariateDiscretizer::depth)
        .def_readwrite("numLeaves", &UnivariateDiscretizer::numLeaves)
        .def_readwrite("numNodes", &UnivariateDiscretizer::numNodes);
    // endregion UnivariateDiscretizer
}
