// Tiny two-basis diagnostics; finite images, not an HF overlap certificate.
// Included by bindings.cpp; not a standalone translation unit.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vibeqc/aopair_ft.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

namespace {

py::dict cross_pair_panel_diagnostic_result(vibeqc::AOPairFourierPanel panel) {
    using Complex = std::complex<double>;
    const auto rows = static_cast<py::ssize_t>(panel.n_pairs);
    const auto columns = static_cast<py::ssize_t>(panel.n_vectors);
    auto owner = std::make_unique<std::vector<Complex>>(std::move(panel.data));
    auto* data = owner->data();
    py::capsule lifetime(owner.get(), [](void* pointer) {
        delete static_cast<std::vector<Complex>*>(pointer);
    });
    owner.release();
    py::dict result;
    result["values"] = py::array_t<Complex>(
        {rows, columns},
        {columns * static_cast<py::ssize_t>(sizeof(Complex)),
         static_cast<py::ssize_t>(sizeof(Complex))}, data, lifetime);
    result["pair_begin"] = panel.pair_begin;
    result["output_bytes"] = panel.output_bytes;
    result["image_candidate_count"] = panel.image_candidate_count;
    result["retained_pair_image_count"] = panel.retained_pair_image_count;
    result["fixed_numeric_workspace_bytes"] = panel.fixed_numeric_workspace_bytes;
    return result;
}

void require_tiny_cross_pair_panel_diagnostic(
    const vibeqc::BasisSet& bra, const vibeqc::BasisSet& ket,
    std::uint64_t pairs, std::size_t vectors, std::uint64_t candidate_cap) {
    if (bra.nbasis() > 64 || ket.nbasis() > 64 || pairs > 64
        || vectors > 64 || candidate_cap > 65536) {
        throw std::length_error("Cross AO-pair Fourier diagnostic exceeds tiny shape or source cap");
    }
    const auto primitive_count = [](const vibeqc::BasisSet& basis) {
        std::uint64_t maximum = 0;
        for (const auto& shell : basis.libint()) {
            maximum = std::max(maximum, static_cast<std::uint64_t>(shell.alpha.size()));
        }
        if (maximum > 16) {
            throw std::length_error("Cross AO-pair Fourier diagnostic allows at most 16 primitives per shell");
        }
        return maximum;
    };
    const auto bra_primitives = primitive_count(bra);
    const auto ket_primitives = primitive_count(ket);
    // All factors bounded above before multiplication; no primitive cross
    // array, merged BasisSet or candidate-image list is materialized.
    if (candidate_cap * vectors * bra_primitives * ket_primitives > 2000000) {
        throw std::length_error("Cross AO-pair Fourier diagnostic exceeds candidate-vector-primitive work cap");
    }
}

}  // namespace

void bind_periodic_aopair_cross_fourier(py::module_& m) {
    m.def(
        "_periodic_ao_pair_gaussian_cross_fourier_panel",
        [](const vibeqc::BasisSet& bra, const vibeqc::BasisSet& ket,
           const vibeqc::PeriodicSystem& system, const py::array& vectors,
           const py::array& k_ket, std::uint64_t pair_begin, std::uint64_t pair_count,
           double cutoff, std::uint64_t maximum_image_candidates, std::uint64_t output_byte_cap) {
            if (!vectors.dtype().is(py::dtype::of<double>()) || vectors.ndim() != 2
                || vectors.shape(1) != 3 || !(vectors.flags() & py::array::c_style)
                || !k_ket.dtype().is(py::dtype::of<double>()) || k_ket.ndim() != 1
                || k_ket.shape(0) != 3 || !(k_ket.flags() & py::array::c_style)) {
                throw std::invalid_argument("Cross Fourier vectors and k_ket must be existing C-contiguous float64 (n,3) and (3,) arrays");
            }
            if (reinterpret_cast<std::uintptr_t>(vectors.data()) % alignof(double)
                || reinterpret_cast<std::uintptr_t>(k_ket.data()) % alignof(double)) {
                throw std::invalid_argument("Cross Fourier vector and k_ket arrays must be aligned");
            }
            const auto count = static_cast<std::size_t>(vectors.shape(0));
            require_tiny_cross_pair_panel_diagnostic(bra, ket, pair_count, count,
                                                     maximum_image_candidates);
            vibeqc::AuxiliaryFourierVectorView view;
            view.count = count;
            view.x_stride = view.y_stride = view.z_stride = 3;
            if (count != 0) {
                view.x = static_cast<const double*>(vectors.data());
                view.y = view.x + 1;
                view.z = view.x + 2;
            }
            const auto* k = static_cast<const double*>(k_ket.data());
            const Eigen::Vector3d momentum(k[0], k[1], k[2]);
            // Keep the GIL for this capped diagnostic so mutable Python
            // geometry/input wrappers cannot be changed during native work.
            return cross_pair_panel_diagnostic_result(vibeqc::ao_pair_gaussian_fourier_panel(
                bra, ket, system, view, momentum, pair_begin, pair_count, cutoff,
                maximum_image_candidates, output_byte_cap));
        },
        py::arg("bra_basis"), py::arg("ket_basis"), py::arg("system"),
        py::arg("vectors").noconvert(), py::arg("k_ket").noconvert(),
        py::arg("pair_begin"), py::arg("pair_count"), py::arg("image_cutoff_bohr"),
        py::arg("maximum_image_candidates"), py::arg("output_byte_cap"),
        "Tiny rectangular finite-image Fourier panel, mu*n_ket+nu ordering. "
        "No omitted-tail, HF-overlap-source or minimal-basis certification.");
}
