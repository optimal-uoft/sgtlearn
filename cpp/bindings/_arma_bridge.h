#pragma once

/**
 * @file _arma_bridge.h
 * @brief Zero-copy(ish) NumPy ↔ Armadillo bridge for sgtlearn's pybind11 modules.
 *
 * Layout convention
 * -----------------
 * Python-facing code uses the sklearn convention: 2-D arrays of shape
 *   (n_samples, n_features), C-contiguous.
 *
 * The C++ side uses Armadillo's column-major convention: 2-D matrices of
 *   shape (n_features, n_samples), F-contiguous.
 *
 * The two are *byte-identical*: feature j of sample i lives at offset
 *   i * n_features + j  in both layouts. The bridge therefore reinterprets
 * the NumPy buffer as an Armadillo matrix with swapped (rows, cols) and
 * borrows the storage. No transpose, no copy.
 *
 * Ownership / lifetime
 * --------------------
 * The bridge structs own a `py::array_t<T>` (which owns / refs the buffer)
 * and hold a non-owning `arma::Mat<T>` / `arma::Row<T>` / `arma::Col<T>`
 * view over it. The view is valid for the lifetime of the bridge. Bridges
 * are move-only to make accidental copies and dangling views impossible.
 *
 * Copy policy
 * -----------
 * pybind11's `c_style | forcecast` will copy iff the input is the wrong
 * dtype, not C-contiguous, or otherwise non-conformant. So:
 *   - Best case  (caller passed canonical sklearn input): zero copies.
 *   - Worst case (dtype mismatch / F-contig / strided):    one copy.
 */

#include <armadillo>
#include <cstddef>
#include <cstring>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace sgt::bindings {

namespace py = pybind11;

// =====================================================================
//                            Input bridges
// =====================================================================

/**
 * Owning handle that pairs a NumPy buffer with a non-owning Armadillo
 * matrix/vector view. Move-only.
 */
template <typename T, typename ArmaView> class ArmaBridge {
  static_assert(std::is_arithmetic_v<T>, "T must be a numeric type");

public:
  ArmaBridge(ArmaBridge &&) noexcept = default;
  ArmaBridge &operator=(ArmaBridge &&) noexcept = default;
  ArmaBridge(const ArmaBridge &) = delete;
  ArmaBridge &operator=(const ArmaBridge &) = delete;

  const ArmaView &view() const noexcept { return view_; }
  ArmaView &view() noexcept { return view_; }

  ArmaBridge(py::array_t<T, py::array::c_style | py::array::forcecast> owner,
             ArmaView view)
      : owner_(std::move(owner)), view_(std::move(view)) {}

private:
  py::array_t<T, py::array::c_style | py::array::forcecast> owner_;
  ArmaView view_;
};

template <typename T> using ArmaMatBridge = ArmaBridge<T, arma::Mat<T>>;
template <typename T> using ArmaRowBridge = ArmaBridge<T, arma::Row<T>>;

namespace detail {

template <typename T>
py::array_t<T, py::array::c_style | py::array::forcecast>
acquireCStyle(const py::array &a) {
  return py::array_t<T, py::array::c_style | py::array::forcecast>(a);
}

[[noreturn]] inline void throwShape(const char *name, const char *expected,
                                    py::ssize_t got) {
  throw std::invalid_argument(std::string(name) + " must be " + expected +
                              "; got ndim=" + std::to_string(got));
}

} // namespace detail

/**
 * Wrap a NumPy `(n_samples, n_features)` C-contiguous array as an Armadillo
 * `(n_features, n_samples)` F-contiguous matrix view. Zero copies in the
 * canonical case.
 */
template <typename T>
ArmaMatBridge<T> asSamplesByFeatures(const py::array &X,
                                     const char *name = "X") {
  auto owner = detail::acquireCStyle<T>(X);
  if (owner.ndim() != 2)
    detail::throwShape(name, "2-D (n_samples, n_features)", owner.ndim());

  const auto nSamples = static_cast<arma::uword>(owner.shape(0));
  const auto nFeatures = static_cast<arma::uword>(owner.shape(1));
  arma::Mat<T> view(static_cast<T *>(owner.mutable_data()), nFeatures,
                    nSamples, /*copy_aux_mem=*/false, /*strict=*/true);
  return ArmaMatBridge<T>(std::move(owner), std::move(view));
}

/**
 * Wrap a NumPy target array as an Armadillo `(n_outputs, n_samples)` matrix
 * view, mirroring the sklearn ``y`` convention:
 *   - 1-D ``(n_samples,)``            -> Mat ``(1, n_samples)``
 *   - 2-D ``(n_samples, n_outputs)``  -> Mat ``(n_outputs, n_samples)``
 *
 * Uses the same byte-identity trick as :func:`asSamplesByFeatures`: a
 * C-contiguous ``(n_samples, n_outputs)`` buffer is bytewise identical to an
 * F-contiguous ``(n_outputs, n_samples)`` matrix, so the view borrows storage
 * with zero copies in the canonical case.
 */
template <typename T>
ArmaMatBridge<T> asSamplesByOutputs(const py::array &y,
                                    const char *name = "y") {
  auto owner = detail::acquireCStyle<T>(y);
  arma::uword nSamples = 0;
  arma::uword nOutputs = 0;
  if (owner.ndim() == 1) {
    nOutputs = 1;
    nSamples = static_cast<arma::uword>(owner.shape(0));
  } else if (owner.ndim() == 2) {
    nSamples = static_cast<arma::uword>(owner.shape(0));
    nOutputs = static_cast<arma::uword>(owner.shape(1));
  } else {
    detail::throwShape(name, "1-D (n_samples,) or 2-D (n_samples, n_outputs)",
                       owner.ndim());
  }
  arma::Mat<T> view(static_cast<T *>(owner.mutable_data()), nOutputs, nSamples,
                    /*copy_aux_mem=*/false, /*strict=*/true);
  return ArmaMatBridge<T>(std::move(owner), std::move(view));
}

/**
 * Convert a NumPy target array to an OWNING Armadillo
 * ``(n_outputs, n_samples)`` matrix. Prefer this for integer label matrices
 * where zero-copy views can trip Armadillo strict checks and where the C++
 * side expects ``arma::Mat<size_t>``.
 */
template <typename T>
arma::Mat<T> asSamplesByOutputsOwning(const py::array &y,
                                      const char *name = "y") {
  auto owner = detail::acquireCStyle<T>(y);
  arma::uword nSamples = 0;
  arma::uword nOutputs = 0;
  if (owner.ndim() == 1) {
    nOutputs = 1;
    nSamples = static_cast<arma::uword>(owner.shape(0));
  } else if (owner.ndim() == 2) {
    nSamples = static_cast<arma::uword>(owner.shape(0));
    nOutputs = static_cast<arma::uword>(owner.shape(1));
  } else {
    detail::throwShape(name, "1-D (n_samples,) or 2-D (n_samples, n_outputs)",
                       owner.ndim());
  }
  arma::Mat<T> out(nOutputs, nSamples);
  if (out.n_elem)
    std::memcpy(out.memptr(), owner.data(), out.n_elem * sizeof(T));
  return out;
}

/** Wrap a 1-D NumPy array as an `arma::Row<T>` view. */
template <typename T>
ArmaRowBridge<T> as1DRow(const py::array &v, const char *name = "v") {
  auto owner = detail::acquireCStyle<T>(v);
  if (owner.ndim() != 1)
    detail::throwShape(name, "1-D", owner.ndim());
  arma::Row<T> view(static_cast<T *>(owner.mutable_data()),
                    static_cast<arma::uword>(owner.shape(0)),
                    /*copy_aux_mem=*/false, /*strict=*/true);
  return ArmaRowBridge<T>(std::move(owner), std::move(view));
}

// =====================================================================
//                            Output builders
// =====================================================================

/** Copy an `arma::Col<T>` (length N) into a fresh NumPy 1-D array. */
template <typename T> py::array_t<T> colToNumpy(const arma::Col<T> &c) {
  py::array_t<T> out({static_cast<py::ssize_t>(c.n_elem)});
  if (c.n_elem)
    std::memcpy(out.mutable_data(), c.memptr(), c.n_elem * sizeof(T));
  return out;
}

/**
 * Copy an Armadillo `(n_features, n_samples)` matrix into a NumPy
 * `(n_samples, n_features)` array. Single memcpy thanks to the layout
 * coincidence described in the module header.
 */
template <typename T>
py::array_t<T> samplesByFeaturesToNumpy(const arma::Mat<T> &m) {
  py::array_t<T> out({static_cast<py::ssize_t>(m.n_cols),
                      static_cast<py::ssize_t>(m.n_rows)});
  if (m.n_elem)
    std::memcpy(out.mutable_data(), m.memptr(), m.n_elem * sizeof(T));
  return out;
}

/**
 * Convert an Armadillo ``(n_outputs, n_samples)`` prediction matrix into a
 * NumPy array following the sklearn ``y`` convention:
 *   - ``n_outputs == 1`` -> 1-D ``(n_samples,)``
 *   - ``n_outputs  > 1`` -> 2-D ``(n_samples, n_outputs)``
 *
 * Single memcpy thanks to the layout coincidence described in the module
 * header (F-contiguous ``(n_outputs, n_samples)`` == C-contiguous
 * ``(n_samples, n_outputs)``).
 */
template <typename T>
py::array_t<T> samplesByOutputsToNumpy(const arma::Mat<T> &m) {
  const arma::uword nOutputs = m.n_rows;
  const arma::uword nSamples = m.n_cols;
  if (nOutputs == 1) {
    py::array_t<T> out({static_cast<py::ssize_t>(nSamples)});
    if (m.n_elem)
      std::memcpy(out.mutable_data(), m.memptr(), m.n_elem * sizeof(T));
    return out;
  }
  py::array_t<T> out({static_cast<py::ssize_t>(nSamples),
                      static_cast<py::ssize_t>(nOutputs)});
  if (m.n_elem)
    std::memcpy(out.mutable_data(), m.memptr(), m.n_elem * sizeof(T));
  return out;
}

} // namespace sgt::bindings
