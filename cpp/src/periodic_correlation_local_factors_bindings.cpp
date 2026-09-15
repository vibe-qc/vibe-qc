// Tiny diagnostic surface only. Native consumers use immutable block owners.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "vibeqc/periodic_correlation_local_factors.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_local_factors(py::module_& m) {
    using namespace vibeqc;
    using Complex = std::complex<double>;
    using Reference = PeriodicCorrelationAdmittedReference;
    using Wannier = PeriodicCorrelationWannier;
    using Domain = PeriodicCorrelationPAODomain;
    using Space = PeriodicCorrelationPAOSpace;
    using Schedule = PeriodicCorrelationFactorStreamSchedule;
    using Reader = PeriodicCorrelationPrivateFactorReader;
    using Selection = PeriodicCorrelationLocalOrbitalSelection;
    using Caps = PeriodicCorrelationLocalFactorCaps;
    using Memory = PeriodicCorrelationLocalFactorMemoryPlan;
    using Orientation = PeriodicCorrelationLocalFactorOrientation;
    using Panel = PeriodicCorrelationLocalCoefficientPanel;
    using Block = PeriodicCorrelationLocalFactorBlock;
    m.attr("_PERIODIC_CORRELATION_LOCAL_FACTOR_CONTRACT_VERSION") = py::int_(1);
    py::enum_<Orientation>(m, "_PeriodicCorrelationLocalFactorOrientation")
        .value("OCCUPIED_VIRTUAL", Orientation::OccupiedVirtual)
        .value("VIRTUAL_OCCUPIED", Orientation::VirtualOccupied);
    py::class_<Selection>(m, "_PeriodicCorrelationLocalOrbitalSelection")
        .def(py::init<>())
        .def_readwrite("occupied_index", &Selection::occupied_index)
        .def_readwrite("occupied_cell", &Selection::occupied_cell)
        .def_readwrite("virtual_begin", &Selection::virtual_begin)
        .def_readwrite("virtual_count", &Selection::virtual_count)
        .def_readwrite("virtual_translation_cell", &Selection::virtual_translation_cell);
    py::class_<Caps>(m, "_PeriodicCorrelationLocalFactorCaps")
        .def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_work_units", &Caps::maximum_work_units)
        .def_readwrite("maximum_tile_visits", &Caps::maximum_tile_visits)
        .def_readwrite("maximum_reader_tile_bytes", &Caps::maximum_reader_tile_bytes);
#define LOCAL_MEMORY_FIELD(name) .def_readonly(#name, &Memory::name)
    py::class_<Memory>(m, "_PeriodicCorrelationLocalFactorMemoryPlan")
        LOCAL_MEMORY_FIELD(n_cells)
        LOCAL_MEMORY_FIELD(n_basis)
        LOCAL_MEMORY_FIELD(n_home_occupied)
        LOCAL_MEMORY_FIELD(domain_dimension)
        LOCAL_MEMORY_FIELD(retained_virtual_dimension)
        LOCAL_MEMORY_FIELD(virtual_count)
        LOCAL_MEMORY_FIELD(auxiliary_count)
        LOCAL_MEMORY_FIELD(coefficient_panel_bytes)
        LOCAL_MEMORY_FIELD(coefficient_scratch_bytes)
        LOCAL_MEMORY_FIELD(retained_output_bytes)
        LOCAL_MEMORY_FIELD(compensation_bytes)
        LOCAL_MEMORY_FIELD(peak_owned_numerical_bytes)
        LOCAL_MEMORY_FIELD(caller_gauge_bytes)
        LOCAL_MEMORY_FIELD(live_wannier_bytes)
        LOCAL_MEMORY_FIELD(live_domain_bytes)
        LOCAL_MEMORY_FIELD(live_space_bytes)
        LOCAL_MEMORY_FIELD(live_reader_numeric_bytes)
        LOCAL_MEMORY_FIELD(live_reader_control_bytes)
        LOCAL_MEMORY_FIELD(maximum_reader_tile_bytes)
        LOCAL_MEMORY_FIELD(tile_visits)
        LOCAL_MEMORY_FIELD(reader_payload_bytes)
        LOCAL_MEMORY_FIELD(work_units)
        LOCAL_MEMORY_FIELD(required_node_memory_bytes);
#undef LOCAL_MEMORY_FIELD
    py::class_<Panel>(m, "_PeriodicCorrelationLocalCoefficientPanel")
        .def_property_readonly("memory", [](const Panel& p) { return p.memory(); })
        .def_property_readonly("selection", [](const Panel& p) { return p.selection(); })
        .def_property_readonly("occupied_k_index", &Panel::occupied_k_index)
        .def_property_readonly("virtual_k_index", &Panel::virtual_k_index)
        .def_property_readonly("identity_sha256", &Panel::identity_sha256)
        .def_property_readonly("payload_sha256", &Panel::payload_sha256)
        .def("coefficient", &Panel::coefficient)
        .def("coefficients_copy", [](const Panel& p) {
            const auto& shape = p.memory();
            if (shape.n_basis > 8 || shape.virtual_count > 16)
                throw std::length_error("local coefficient diagnostic copy exceeds tiny shape cap");
            (void) p.data();
            py::array_t<Complex> result({static_cast<py::ssize_t>(shape.n_basis),
                                         static_cast<py::ssize_t>(shape.virtual_count + 1)});
            for (std::size_t mu = 0; mu < shape.n_basis; ++mu)
                for (std::size_t a = 0; a <= shape.virtual_count; ++a)
                    result.mutable_data()[mu * (shape.virtual_count + 1) + a] = p.coefficient(mu, a);
            return result;
        });
    py::class_<Block>(m, "_PeriodicCorrelationLocalFactorBlock")
        .def_property_readonly("memory", [](const Block& b) { return b.memory(); })
        .def_property_readonly("selection", [](const Block& b) { return b.selection(); })
        .def_property_readonly("q_index", &Block::q_index)
        .def_property_readonly("auxiliary_begin", &Block::auxiliary_begin)
        .def_property_readonly("orientation", &Block::orientation)
        .def_property_readonly("finite_image_reference", &Block::finite_image_reference)
        .def_property_readonly("ao_image_source_certified", &Block::ao_image_source_certified)
        .def_property_readonly("identity_sha256", &Block::identity_sha256)
        .def_property_readonly("payload_sha256", &Block::payload_sha256)
        .def_property_readonly("source_identity_sha256", &Block::source_identity_sha256)
        .def_property_readonly("whitener_payload_identity_sha256", &Block::whitener_payload_identity_sha256)
        .def_property_readonly("store_identity_sha256", &Block::store_identity_sha256)
        .def_property_readonly("consumed_tiles_identity_sha256", &Block::consumed_tiles_identity_sha256)
        .def("element", &Block::element)
        .def("matrix_copy", [](const Block& b) {
            const auto& shape = b.memory();
            if (shape.auxiliary_count > 8 || shape.virtual_count > 16)
                throw std::length_error("local factor diagnostic copy exceeds tiny shape cap");
            const auto* data = b.data();
            py::array_t<Complex> result({static_cast<py::ssize_t>(shape.auxiliary_count),
                                         static_cast<py::ssize_t>(shape.virtual_count)});
            std::memcpy(result.mutable_data(), data, shape.retained_output_bytes);
            return result;
        });
    const auto view = [](const Reference& reference, const py::array& gauges,
                         const Domain& domain, const Selection& selection, const Caps& caps) {
        const auto& state = reference.state();
        if (state.n_kpoints() > 8 || state.n_basis() > 8 || state.n_correlated_occupied() > 4
            || domain.domain_dimension() > 32 || selection.virtual_count > 16
            || caps.maximum_work_units > 100000000 || caps.maximum_owned_numerical_bytes > 1048576
            || caps.maximum_tile_visits > 128 || caps.maximum_reader_tile_bytes > 8192)
            throw std::length_error("local factor diagnostic exceeds tiny shape or work cap");
        if (!gauges.dtype().is(py::dtype::of<Complex>()) || gauges.ndim() != 3
            || !(gauges.flags() & py::array::c_style)
            || gauges.shape(0) != static_cast<py::ssize_t>(state.n_kpoints())
            || gauges.shape(1) != static_cast<py::ssize_t>(state.n_correlated_occupied())
            || gauges.shape(2) != static_cast<py::ssize_t>(state.n_correlated_occupied()))
            throw std::invalid_argument("local factor gauges require existing C-contiguous complex128 [Nk,nactive,nactive]");
        return std::make_pair(static_cast<const Complex*>(gauges.data()), static_cast<std::size_t>(gauges.size()));
    };
    m.def("_plan_periodic_correlation_local_coefficient_panel", &plan_periodic_correlation_local_coefficient_panel,
          py::arg("reference"), py::arg("wannier"), py::arg("domain"), py::arg("space"), py::arg("selection"));
    m.def("_plan_periodic_correlation_local_factor_block", &plan_periodic_correlation_local_factor_block,
          py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"), py::arg("domain"),
          py::arg("space"), py::arg("selection"), py::arg("q_index"), py::arg("auxiliary_begin"), py::arg("auxiliary_count"));
    m.def("_make_periodic_correlation_local_coefficient_panel", [view](
        const Reference& reference, const Wannier& wannier, const py::array& gauges,
        const Domain& domain, const Space& space, const Selection& selection,
        std::uint64_t ko, std::uint64_t kv, const Caps& caps) {
        const auto array = view(reference, gauges, domain, selection, caps);
        const Selection selected = selection;
        const Caps controls = caps;
        py::gil_scoped_release release;
        return make_periodic_correlation_local_coefficient_panel(reference, wannier, array.first, array.second,
            domain, space, selected, ko, kv, controls);
    }, py::arg("reference"), py::arg("wannier"), py::arg("gauges").noconvert(), py::arg("domain"),
       py::arg("space"), py::arg("selection"), py::arg("occupied_k_index"), py::arg("virtual_k_index"), py::arg("caps"));
    m.def("_build_periodic_correlation_local_factor_block", [view](
        const Reference& reference, const Schedule& schedule, const Reader& reader,
        const Wannier& wannier, const py::array& gauges, const Domain& domain, const Space& space,
        const Selection& selection, std::uint64_t q, std::uint64_t begin, std::uint64_t count,
        Orientation orientation, const Caps& caps) {
        const auto array = view(reference, gauges, domain, selection, caps);
        if (schedule.shape().tile_count > 128 || schedule.shape().n_auxiliary > 8
            || reader.file_bytes() > 1048576)
            throw std::length_error("local factor diagnostic requires a tiny native store");
        const Selection selected = selection;
        const Caps controls = caps;
        py::gil_scoped_release release;
        return build_periodic_correlation_local_factor_block(reference, schedule, reader, wannier,
            array.first, array.second, domain, space, selected, q, begin, count, orientation, controls);
    }, py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"),
       py::arg("gauges").noconvert(), py::arg("domain"), py::arg("space"), py::arg("selection"),
       py::arg("q_index"), py::arg("auxiliary_begin"), py::arg("auxiliary_count"), py::arg("orientation"), py::arg("caps"),
       "Finite-source global-auxiliary RI diagnostic. Input gauge storage must remain immutable.");
}
