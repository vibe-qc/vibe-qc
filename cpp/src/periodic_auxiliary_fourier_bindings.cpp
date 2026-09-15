// Tiny internal diagnostics for the native auxiliary-Gaussian Fourier panel.
// Production periodic code calls the C++ API directly.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vibeqc/periodic_auxiliary_fourier.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {

constexpr std::size_t kAuxiliaryFourierDiagnosticMaxVectors = 4096U;
constexpr std::size_t kAuxiliaryFourierDiagnosticMaxElements = 65536U;

void require_tiny_auxiliary_fourier_diagnostic(std::size_t n_auxiliary,
                                                std::size_t n_vectors) {
    if (n_vectors > kAuxiliaryFourierDiagnosticMaxVectors
        || (n_vectors != 0U
            && n_auxiliary
                > kAuxiliaryFourierDiagnosticMaxElements / n_vectors)) {
        throw std::length_error(
            "internal auxiliary Fourier diagnostic is limited to 4096 "
            "vectors and 65536 output elements");
    }
}

py::array_t<std::complex<double>> auxiliary_fourier_array(
    vibeqc::AuxiliaryFourierPanel panel) {
    const auto n_auxiliary = static_cast<py::ssize_t>(panel.n_auxiliary);
    const auto n_vectors = static_cast<py::ssize_t>(panel.n_vectors);
    using Complex = std::complex<double>;
    auto* owner = new std::vector<Complex>(std::move(panel.data));
    py::capsule keep_alive(owner, [](void* pointer) {
        delete static_cast<std::vector<Complex>*>(pointer);
    });
    return py::array_t<Complex>(
        {n_auxiliary, n_vectors},
        {static_cast<py::ssize_t>(n_vectors * sizeof(Complex)),
         static_cast<py::ssize_t>(sizeof(Complex))},
        owner->data(),
        keep_alive);
}

}  // namespace

void bind_periodic_auxiliary_fourier(py::module_& m) {
    m.attr("_AUXILIARY_BASIS_CONTENT_DIGEST_VERSION") =
        py::int_(vibeqc::kAuxiliaryBasisContentDigestVersion);
    m.attr("_PERIODIC_AUXILIARY_FOURIER_DIAGNOSTIC_MAX_VECTORS") =
        py::int_(kAuxiliaryFourierDiagnosticMaxVectors);
    m.attr("_PERIODIC_AUXILIARY_FOURIER_DIAGNOSTIC_MAX_ELEMENTS") =
        py::int_(kAuxiliaryFourierDiagnosticMaxElements);

    m.def(
        "_auxiliary_basis_content_identity_sha256",
        &vibeqc::auxiliary_basis_content_identity_sha256,
        py::arg("basis"),
        "Internal diagnostic for the versioned native auxiliary-basis "
        "content identity. The basis display name is excluded.");

    m.def(
        "_periodic_auxiliary_gaussian_fourier_panel",
        [](const vibeqc::BasisSet& basis,
           const py::array_t<double, py::array::c_style>& vectors,
           std::uint64_t output_byte_cap) {
            if (vectors.ndim() != 2 || vectors.shape(1) != 3) {
                throw std::invalid_argument(
                    "vectors must have shape (n_vectors, 3)");
            }
            const auto n_vectors =
                static_cast<std::size_t>(vectors.shape(0));
            require_tiny_auxiliary_fourier_diagnostic(
                basis.nbasis(), n_vectors);
            Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3,
                                           Eigen::RowMajor>>
                vector_map(
                    vectors.data(),
                    static_cast<Eigen::Index>(n_vectors),
                    3);
            vibeqc::AuxiliaryFourierPanel panel;
            {
                py::gil_scoped_release release;
                panel = vibeqc::auxiliary_gaussian_fourier_panel(
                    basis, vector_map, output_byte_cap);
            }
            return auxiliary_fourier_array(std::move(panel));
        },
        py::arg("basis"),
        py::arg("vectors").noconvert(),
        py::arg("output_byte_cap"),
        "Tiny internal diagnostic for the row-major native auxiliary "
        "Fourier-panel overload. Exact C-contiguous float64 input only.");

    m.def(
        "_periodic_auxiliary_gaussian_fourier_panel_soa",
        [](const vibeqc::BasisSet& basis,
           const py::array_t<double, py::array::c_style>& x,
           const py::array_t<double, py::array::c_style>& y,
           const py::array_t<double, py::array::c_style>& z,
           std::uint64_t output_byte_cap) {
            if (x.ndim() != 1 || y.ndim() != 1 || z.ndim() != 1) {
                throw std::invalid_argument(
                    "x, y, and z must each have shape (n_vectors,)");
            }
            if (x.shape(0) != y.shape(0) || x.shape(0) != z.shape(0)) {
                throw std::invalid_argument(
                    "x, y, and z vector lanes must have equal lengths");
            }
            const auto n_vectors = static_cast<std::size_t>(x.shape(0));
            require_tiny_auxiliary_fourier_diagnostic(
                basis.nbasis(), n_vectors);
            vibeqc::AuxiliaryFourierVectorView view;
            view.count = n_vectors;
            if (n_vectors > 0U) {
                view.x = x.data();
                view.y = y.data();
                view.z = z.data();
            }
            vibeqc::AuxiliaryFourierPanel panel;
            {
                py::gil_scoped_release release;
                panel = vibeqc::auxiliary_gaussian_fourier_panel(
                    basis, view, output_byte_cap);
            }
            return auxiliary_fourier_array(std::move(panel));
        },
        py::arg("basis"),
        py::arg("x").noconvert(),
        py::arg("y").noconvert(),
        py::arg("z").noconvert(),
        py::arg("output_byte_cap"),
        "Tiny internal diagnostic for the zero-copy SoA native auxiliary "
        "Fourier-panel entry point. Exact C-contiguous float64 lanes only.");
}
