// Tiny diagnostic surface; native verified-store input, never factor arrays.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstring>
#include <stdexcept>

#include "vibeqc/periodic_correlation_local_orbital_factors.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_local_orbital_factors(py::module_& m) {
    using namespace vibeqc;
    using Complex = std::complex<double>;
    using Reference = PeriodicCorrelationAdmittedReference;
    using Schedule = PeriodicCorrelationFactorStreamSchedule;
    using Reader = PeriodicCorrelationPrivateFactorReader;
    using Wannier = PeriodicCorrelationWannier;
    using Domain = PeriodicCorrelationPAODomain;
    using Space = PeriodicCorrelationPAOSpace;
    using Virtual = PeriodicCorrelationVirtualBlockSelection;
    using Caps = PeriodicCorrelationLocalFactorCaps;
    using Memory = PeriodicCorrelationLocalOrbitalFactorMemoryPlan;
    using Panel = PeriodicCorrelationLocalOrbitalFactorPanel;
    m.attr("_PERIODIC_CORRELATION_LOCAL_ORBITAL_FACTOR_CONTRACT_VERSION") = py::int_(1);
#define LOCAL_ORBITAL_MEMORY(name) .def_readonly(#name, &Memory::name)
    py::class_<Memory>(m, "_PeriodicCorrelationLocalOrbitalFactorMemoryPlan")
        LOCAL_ORBITAL_MEMORY(n_cells) LOCAL_ORBITAL_MEMORY(n_basis)
        LOCAL_ORBITAL_MEMORY(occupied_count) LOCAL_ORBITAL_MEMORY(virtual_count)
        LOCAL_ORBITAL_MEMORY(orbital_count) LOCAL_ORBITAL_MEMORY(auxiliary_count)
        LOCAL_ORBITAL_MEMORY(retained_output_bytes) LOCAL_ORBITAL_MEMORY(compensation_bytes)
        LOCAL_ORBITAL_MEMORY(coefficient_panel_bytes) LOCAL_ORBITAL_MEMORY(coefficient_scratch_bytes)
        LOCAL_ORBITAL_MEMORY(retained_index_bytes) LOCAL_ORBITAL_MEMORY(caller_index_bytes)
        LOCAL_ORBITAL_MEMORY(peak_owned_numerical_bytes) LOCAL_ORBITAL_MEMORY(caller_gauge_bytes)
        LOCAL_ORBITAL_MEMORY(live_wannier_bytes) LOCAL_ORBITAL_MEMORY(live_domain_bytes) LOCAL_ORBITAL_MEMORY(live_space_bytes)
        LOCAL_ORBITAL_MEMORY(live_reader_numeric_bytes) LOCAL_ORBITAL_MEMORY(live_reader_control_bytes)
        LOCAL_ORBITAL_MEMORY(maximum_reader_tile_bytes) LOCAL_ORBITAL_MEMORY(reader_payload_bytes)
        LOCAL_ORBITAL_MEMORY(tile_visits) LOCAL_ORBITAL_MEMORY(work_units) LOCAL_ORBITAL_MEMORY(required_node_memory_bytes);
#undef LOCAL_ORBITAL_MEMORY
    py::class_<Panel>(m, "_PeriodicCorrelationLocalOrbitalFactorPanel")
        .def_property_readonly("contract_version", &Panel::contract_version)
        .def_property_readonly("memory", [](const Panel& p) { return p.memory(); })
        .def_property_readonly("q_index", &Panel::q_index)
        .def_property_readonly("auxiliary_begin", &Panel::auxiliary_begin)
        .def_property_readonly("virtual_selection", &Panel::virtual_selection)
        .def_property_readonly("finite_image_reference", &Panel::finite_image_reference)
        .def_property_readonly("ao_image_source_certified", &Panel::ao_image_source_certified)
        .def_property_readonly("density_symmetry_certified", &Panel::density_symmetry_certified)
        .def_property_readonly("identity_sha256", &Panel::identity_sha256)
        .def_property_readonly("payload_sha256", &Panel::payload_sha256)
        .def_property_readonly("selection_identity_sha256", &Panel::selection_identity_sha256)
        .def_property_readonly("local_basis_identity_sha256", &Panel::local_basis_identity_sha256)
        .def_property_readonly("ao_basis_identity_sha256", &Panel::ao_basis_identity_sha256)
        .def_property_readonly("auxiliary_basis_identity_sha256", &Panel::auxiliary_basis_identity_sha256)
        .def_property_readonly("source_identity_sha256", &Panel::source_identity_sha256)
        .def_property_readonly("whitener_payload_identity_sha256", &Panel::whitener_payload_identity_sha256)
        .def_property_readonly("store_identity_sha256", &Panel::store_identity_sha256)
        .def_property_readonly("consumed_tiles_identity_sha256", &Panel::consumed_tiles_identity_sha256)
        .def("occupied", [](const Panel& p, std::size_t i) {
            const auto label = p.occupied(i); return py::make_tuple(label.occupied_index, label.cell);
        })
        .def("element", &Panel::element)
        .def("tensor_copy", [](const Panel& p) {
            const auto& memory = p.memory();
            if (memory.auxiliary_count > 8 || memory.orbital_count > 16)
                throw std::length_error("local orbital factor diagnostic copy exceeds tiny shape cap");
            const auto* source = p.data();
            py::array_t<Complex> output({static_cast<py::ssize_t>(memory.auxiliary_count),
                static_cast<py::ssize_t>(memory.orbital_count), static_cast<py::ssize_t>(memory.orbital_count)});
            std::memcpy(output.mutable_data(), source, memory.retained_output_bytes);
            return output;
        });
    m.def("_plan_periodic_correlation_local_orbital_factor_panel", &plan_periodic_correlation_local_orbital_factor_panel,
        py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"),
        py::arg("domain"), py::arg("space"), py::arg("occupied_count"), py::arg("virtual_selection"),
        py::arg("q_index"), py::arg("auxiliary_begin"), py::arg("auxiliary_count"));
    m.def("_build_periodic_correlation_local_orbital_factor_panel", [](
        const Reference& reference, const Schedule& schedule, const Reader& reader, const Wannier& wannier,
        const py::array& gauges, const Domain& domain, const Space& space, const py::array& indices,
        const Virtual& selected, std::uint64_t q, std::uint64_t begin, std::uint64_t count, const Caps& caps) {
        if (!indices.dtype().is(py::dtype::of<std::uint64_t>()) || indices.ndim() != 2
            || indices.shape(1) != 2 || !(indices.flags() & py::array::c_style))
            throw std::invalid_argument("local orbital factor labels require C-contiguous uint64 [count,2]");
        const auto& state = reference.state();
        if (state.n_kpoints() > 8 || state.n_basis() > 8 || state.n_correlated_occupied() > 4
            || schedule.shape().n_auxiliary > 8 || schedule.shape().tile_count > 128
            || reader.file_bytes() > 1048576 || domain.domain_dimension() > 32
            || indices.shape(0) > 16 || selected.count > 16
            || static_cast<std::uint64_t>(indices.shape(0)) + selected.count > 16
            || caps.maximum_owned_numerical_bytes > 1048576 || caps.maximum_work_units > 100000000
            || caps.maximum_tile_visits > 128 || caps.maximum_reader_tile_bytes > 8192)
            throw std::length_error("local orbital factor diagnostic exceeds tiny shape or work cap");
        if (!gauges.dtype().is(py::dtype::of<Complex>()) || gauges.ndim() != 3
            || !(gauges.flags() & py::array::c_style)
            || gauges.shape(0) != static_cast<py::ssize_t>(state.n_kpoints())
            || gauges.shape(1) != static_cast<py::ssize_t>(state.n_correlated_occupied())
            || gauges.shape(2) != static_cast<py::ssize_t>(state.n_correlated_occupied()))
            throw std::invalid_argument("local orbital factor gauges require C-contiguous complex128 [Nk,nactive,nactive]");
        const auto* g = static_cast<const Complex*>(gauges.data());
        const auto ng = static_cast<std::size_t>(gauges.size());
        const auto* labels = static_cast<const std::uint64_t*>(indices.data());
        const auto accessible = static_cast<std::size_t>(indices.size());
        const auto occupied = static_cast<std::size_t>(indices.shape(0));
        const Virtual selection = selected;
        const Caps controls = caps;
        py::gil_scoped_release release;
        return build_periodic_correlation_local_orbital_factor_panel(reference, schedule, reader, wannier,
            g, ng, domain, space, labels, accessible, occupied, selection, q, begin, count, controls);
    }, py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"),
       py::arg("gauges").noconvert(), py::arg("domain"), py::arg("space"), py::arg("occupied_indices").noconvert(),
       py::arg("virtual_selection"), py::arg("q_index"), py::arg("auxiliary_begin"), py::arg("auxiliary_count"), py::arg("caps"));
}
