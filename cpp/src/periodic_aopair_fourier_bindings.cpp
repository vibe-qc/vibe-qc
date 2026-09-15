// Tiny diagnostics for the bounded native AO-pair Fourier panel.
// Included by bindings.cpp; not a standalone translation unit.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vibeqc/aopair_ft.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {

py::dict pair_panel_diagnostic_result(vibeqc::AOPairFourierPanel panel) {
    using Complex = std::complex<double>;
    const auto rows = static_cast<py::ssize_t>(panel.n_pairs);
    const auto columns = static_cast<py::ssize_t>(panel.n_vectors);
    auto* owner = new std::vector<Complex>(std::move(panel.data));
    py::capsule lifetime(owner, [](void* pointer) {
        delete static_cast<std::vector<Complex>*>(pointer);
    });
    py::dict result;
    result["values"] = py::array_t<Complex>(
        {rows, columns},
        {columns * static_cast<py::ssize_t>(sizeof(Complex)),
         static_cast<py::ssize_t>(sizeof(Complex))}, owner->data(), lifetime);
    result["pair_begin"] = panel.pair_begin;
    result["output_bytes"] = panel.output_bytes;
    result["image_candidate_count"] = panel.image_candidate_count;
    result["retained_pair_image_count"] = panel.retained_pair_image_count;
    result["fixed_numeric_workspace_bytes"] = panel.fixed_numeric_workspace_bytes;
    return result;
}

void require_tiny_pair_panel_diagnostic(
    const vibeqc::BasisSet& basis, std::uint64_t pairs, std::size_t vectors,
    std::uint64_t candidate_cap) {
    if (basis.nbasis() > 64 || pairs > 64 || vectors > 64
        || candidate_cap > 65536) {
        throw std::length_error("AO-pair Fourier diagnostic exceeds tiny shape or source cap");
    }
    std::uint64_t max_primitives = 0;
    for (const auto& shell : basis.libint()) {
        max_primitives = std::max(max_primitives,
            static_cast<std::uint64_t>(shell.alpha.size()));
    }
    if (max_primitives > 16) {
        throw std::length_error("AO-pair Fourier diagnostic allows at most 16 primitives per shell");
    }
    // All factors have already been bounded above, so this product is safe.
    if (candidate_cap * vectors * max_primitives * max_primitives > 2000000) {
        throw std::length_error("AO-pair Fourier diagnostic exceeds candidate-vector-primitive work cap");
    }
}

}  // namespace

void bind_periodic_aopair_fourier(py::module_& m) {
    m.attr("_PERIODIC_AOPAIR_FOURIER_FIXED_NUMERIC_WORKSPACE_BYTES") =
        py::int_(vibeqc::ao_pair_fourier_fixed_numeric_workspace_bytes());
    m.def(
        "_periodic_ao_pair_gaussian_fourier_panel",
        [](const vibeqc::BasisSet& basis, const vibeqc::PeriodicSystem& system,
           const py::array_t<double, py::array::c_style>& vectors,
           const py::array_t<double, py::array::c_style>& k_ket,
           std::uint64_t pair_begin, std::uint64_t pair_count, double cutoff,
           std::uint64_t maximum_image_candidates, std::uint64_t output_byte_cap) {
            if (vectors.ndim() != 2 || vectors.shape(1) != 3
                || k_ket.ndim() != 1 || k_ket.shape(0) != 3) {
                throw std::invalid_argument("vectors must be (n,3) and k_ket must be (3,)");
            }
            const auto count = static_cast<std::size_t>(vectors.shape(0));
            require_tiny_pair_panel_diagnostic(basis, pair_count, count,
                                               maximum_image_candidates);
            vibeqc::AuxiliaryFourierVectorView view;
            view.count = count;
            view.x_stride = view.y_stride = view.z_stride = 3;
            if (count != 0) {
                view.x = vectors.data();
                view.y = vectors.data() + 1;
                view.z = vectors.data() + 2;
            }
            const Eigen::Vector3d k(k_ket.data()[0], k_ket.data()[1], k_ket.data()[2]);
            vibeqc::AOPairFourierPanel panel;
            {
                py::gil_scoped_release release;
                panel = vibeqc::ao_pair_gaussian_fourier_panel(
                    basis, system, view, k, pair_begin, pair_count, cutoff,
                    maximum_image_candidates, output_byte_cap);
            }
            return pair_panel_diagnostic_result(std::move(panel));
        },
        py::arg("basis"), py::arg("system"), py::arg("vectors").noconvert(),
        py::arg("k_ket").noconvert(), py::arg("pair_begin"), py::arg("pair_count"),
        py::arg("image_cutoff_bohr"), py::arg("maximum_image_candidates"),
        py::arg("output_byte_cap"),
        "Tiny finite-image reference panel with native AO normalization. "
        "The enumeration is not a certified correlation-factor source.");
}
