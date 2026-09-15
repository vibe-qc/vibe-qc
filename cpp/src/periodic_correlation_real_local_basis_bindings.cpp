// Tiny inspection bindings for the native real-local basis certificate.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <stdexcept>
#include "vibeqc/periodic_correlation_real_local_basis.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_real_local_basis(py::module_& m) {
    using namespace vibeqc;
    using Options = PeriodicCorrelationRealLocalBasisOptions;
    using Caps = PeriodicCorrelationRealLocalBasisCaps;
    using Memory = PeriodicCorrelationRealLocalBasisMemoryPlan;
    using Diagnostics = PeriodicCorrelationRealLocalBasisDiagnostics;
    using Basis = PeriodicCorrelationRealLocalBasis;
    m.attr("_PERIODIC_CORRELATION_REAL_LOCAL_BASIS_CONTRACT_VERSION") = py::int_(1);
#define REAL_BASIS_OPTION(name) .def_readwrite(#name, &Options::name)
    py::class_<Options>(m, "_PeriodicCorrelationRealLocalBasisOptions").def(py::init<>())
        REAL_BASIS_OPTION(coefficient_tr_absolute_tolerance) REAL_BASIS_OPTION(coefficient_tr_relative_tolerance)
        REAL_BASIS_OPTION(orthonormality_absolute_tolerance) REAL_BASIS_OPTION(orthonormality_relative_tolerance)
        REAL_BASIS_OPTION(fock_absolute_tolerance) REAL_BASIS_OPTION(fock_relative_tolerance)
        REAL_BASIS_OPTION(maximum_fock_projection_error);
#undef REAL_BASIS_OPTION
    py::class_<Caps>(m, "_PeriodicCorrelationRealLocalBasisCaps").def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_work_units", &Caps::maximum_work_units);
#define REAL_BASIS_MEMORY(name) .def_readonly(#name, &Memory::name)
    py::class_<Memory>(m, "_PeriodicCorrelationRealLocalBasisMemoryPlan")
        REAL_BASIS_MEMORY(n_cells) REAL_BASIS_MEMORY(n_basis) REAL_BASIS_MEMORY(occupied_count)
        REAL_BASIS_MEMORY(virtual_count) REAL_BASIS_MEMORY(orbital_count)
        REAL_BASIS_MEMORY(retained_index_bytes) REAL_BASIS_MEMORY(caller_index_bytes) REAL_BASIS_MEMORY(retained_fock_bytes)
        REAL_BASIS_MEMORY(retained_output_bytes) REAL_BASIS_MEMORY(projection_matrix_bytes)
        REAL_BASIS_MEMORY(coefficient_panel_bytes) REAL_BASIS_MEMORY(coefficient_scratch_bytes)
        REAL_BASIS_MEMORY(peak_owned_numerical_bytes) REAL_BASIS_MEMORY(caller_gauge_bytes)
        REAL_BASIS_MEMORY(live_wannier_bytes) REAL_BASIS_MEMORY(live_domain_bytes) REAL_BASIS_MEMORY(live_space_bytes)
        REAL_BASIS_MEMORY(work_units) REAL_BASIS_MEMORY(required_node_memory_bytes);
#undef REAL_BASIS_MEMORY
#define REAL_BASIS_DIAGNOSTIC(name) .def_readonly(#name, &Diagnostics::name)
    py::class_<Diagnostics>(m, "_PeriodicCorrelationRealLocalBasisDiagnostics")
        REAL_BASIS_DIAGNOSTIC(inspected_kpoints) REAL_BASIS_DIAGNOSTIC(inspected_trim_points)
        REAL_BASIS_DIAGNOSTIC(maximum_coefficient_tr_error) REAL_BASIS_DIAGNOSTIC(coefficient_tr_frobenius_upper_bound)
        REAL_BASIS_DIAGNOSTIC(maximum_orthonormality_error) REAL_BASIS_DIAGNOSTIC(orthonormality_frobenius_upper_bound)
        REAL_BASIS_DIAGNOSTIC(maximum_raw_fock_imaginary_magnitude) REAL_BASIS_DIAGNOSTIC(maximum_raw_fock_hermitian_error)
        REAL_BASIS_DIAGNOSTIC(maximum_fock_projection_error) REAL_BASIS_DIAGNOSTIC(fock_projection_frobenius_upper_bound);
#undef REAL_BASIS_DIAGNOSTIC
    py::class_<Basis>(m, "_PeriodicCorrelationRealLocalBasis")
        .def_property_readonly("contract_version", &Basis::contract_version)
        .def_property_readonly("memory", [](const Basis& b) { return b.memory(); })
        .def_property_readonly("diagnostics", [](const Basis& b) { return b.diagnostics(); })
        .def_property_readonly("options", [](const Basis& b) { return b.options(); })
        .def_property_readonly("virtual_selection", &Basis::virtual_selection)
        .def_property_readonly("identity_sha256", &Basis::identity_sha256)
        .def_property_readonly("payload_sha256", &Basis::payload_sha256)
        .def_property_readonly("local_basis_identity_sha256", &Basis::local_basis_identity_sha256)
        .def_property_readonly("allocation_identity", &Basis::allocation_identity)
        .def_property_readonly("hf_hamiltonian_match_certified", &Basis::hf_hamiltonian_match_certified)
        .def("occupied", [](const Basis& b, std::size_t i) {
            const auto label = b.occupied(i); return py::make_tuple(label.occupied_index, label.cell);
        })
        .def("fock", &Basis::fock)
        .def("fock_copy", [](const Basis& b) {
            const auto n = b.memory().orbital_count;
            if (n > 16) throw std::length_error("real-local basis diagnostic exceeds tiny shape cap");
            py::array_t<double> out({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
            for (std::size_t i = 0; i < n; ++i) for (std::size_t j = 0; j < n; ++j)
                out.mutable_data()[i*n+j] = b.fock(i,j);
            return out;
        });
    m.def("_plan_periodic_correlation_real_local_basis", &plan_periodic_correlation_real_local_basis,
        py::arg("reference"), py::arg("wannier"), py::arg("domain"), py::arg("space"),
        py::arg("occupied_count"), py::arg("virtual_selection"));
    m.def("_make_periodic_correlation_real_local_basis", [](
        const PeriodicCorrelationAdmittedReference& ref, const PeriodicCorrelationWannier& wannier,
        const py::array& gauges, const PeriodicCorrelationPAODomain& domain, const PeriodicCorrelationPAOSpace& space,
        const py::array& indices, const PeriodicCorrelationVirtualBlockSelection& selected,
        const Options& options, const Caps& caps) {
        using Complex = std::complex<double>;
        if (!indices.dtype().is(py::dtype::of<std::uint64_t>()) || indices.ndim() != 2
            || indices.shape(1) != 2 || !(indices.flags() & py::array::c_style))
            throw std::invalid_argument("real-local basis indices require C-contiguous uint64 [count,2]");
        const auto& state = ref.state();
        if (state.n_kpoints() > 8 || state.n_basis() > 8 || state.n_correlated_occupied() > 4
            || domain.domain_dimension() > 32 || indices.shape(0) > 16 || selected.count > 16
            || static_cast<std::uint64_t>(indices.shape(0)) + selected.count > 16
            || caps.maximum_owned_numerical_bytes > 1048576 || caps.maximum_work_units > 100000000)
            throw std::length_error("real-local basis diagnostic exceeds tiny shape or work cap");
        if (!gauges.dtype().is(py::dtype::of<Complex>()) || gauges.ndim() != 3 || !(gauges.flags() & py::array::c_style)
            || gauges.shape(0) != static_cast<py::ssize_t>(state.n_kpoints())
            || gauges.shape(1) != static_cast<py::ssize_t>(state.n_correlated_occupied())
            || gauges.shape(2) != static_cast<py::ssize_t>(state.n_correlated_occupied()))
            throw std::invalid_argument("real-local basis gauges require C-contiguous complex128 [Nk,nactive,nactive]");
        const auto* g = static_cast<const Complex*>(gauges.data()); const auto ng = static_cast<std::size_t>(gauges.size());
        const auto* labels = static_cast<const std::uint64_t*>(indices.data());
        const auto accessible = static_cast<std::size_t>(indices.size()), occupied = static_cast<std::size_t>(indices.shape(0));
        const auto selection = selected; const auto settings = options; const auto limits = caps;
        py::gil_scoped_release release;
        return make_periodic_correlation_real_local_basis(ref, wannier, g, ng, domain, space,
            labels, accessible, occupied, selection, settings, limits);
    }, py::arg("reference"), py::arg("wannier"), py::arg("gauges").noconvert(), py::arg("domain"), py::arg("space"),
       py::arg("occupied_indices").noconvert(), py::arg("virtual_selection"), py::arg("options"), py::arg("caps"));
}
