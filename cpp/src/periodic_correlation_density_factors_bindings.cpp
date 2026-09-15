// Tiny diagnostic bindings; no caller-labelled factors or complete tensors.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstring>
#include <stdexcept>

#include "vibeqc/periodic_correlation_density_factors.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_density_factors(py::module_& m) {
    using namespace vibeqc;
    using Complex = std::complex<double>;
    using Reference = PeriodicCorrelationAdmittedReference;
    using Schedule = PeriodicCorrelationFactorStreamSchedule;
    using Reader = PeriodicCorrelationPrivateFactorReader;
    using Wannier = PeriodicCorrelationWannier;
    using Domain = PeriodicCorrelationPAODomain;
    using Space = PeriodicCorrelationPAOSpace;
    using Caps = PeriodicCorrelationLocalFactorCaps;
    using Memory = PeriodicCorrelationDensityFactorMemoryPlan;
    using Kind = PeriodicCorrelationDensityFactorKind;
    using Virtual = PeriodicCorrelationVirtualBlockSelection;
    using Block = PeriodicCorrelationDensityFactorBlock;
    m.attr("_PERIODIC_CORRELATION_DENSITY_FACTOR_CONTRACT_VERSION") = py::int_(1);
    py::enum_<Kind>(m, "_PeriodicCorrelationDensityFactorKind")
        .value("OCCUPIED_OCCUPIED", Kind::OccupiedOccupied).value("VIRTUAL_VIRTUAL", Kind::VirtualVirtual);
    py::class_<Virtual>(m, "_PeriodicCorrelationVirtualBlockSelection")
        .def(py::init<>()).def_readwrite("begin", &Virtual::begin)
        .def_readwrite("count", &Virtual::count).def_readwrite("translation_cell", &Virtual::translation_cell);
#define DENSITY_MEMORY(name) .def_readonly(#name, &Memory::name)
    py::class_<Memory>(m, "_PeriodicCorrelationDensityFactorMemoryPlan")
        DENSITY_MEMORY(n_cells) DENSITY_MEMORY(n_basis)
        DENSITY_MEMORY(left_count) DENSITY_MEMORY(right_count) DENSITY_MEMORY(auxiliary_count)
        DENSITY_MEMORY(retained_output_bytes) DENSITY_MEMORY(compensation_bytes)
        DENSITY_MEMORY(coefficient_panel_bytes) DENSITY_MEMORY(coefficient_scratch_bytes)
        DENSITY_MEMORY(retained_index_bytes) DENSITY_MEMORY(caller_index_bytes)
        DENSITY_MEMORY(peak_owned_numerical_bytes) DENSITY_MEMORY(caller_gauge_bytes)
        DENSITY_MEMORY(live_wannier_bytes) DENSITY_MEMORY(live_domain_bytes) DENSITY_MEMORY(live_space_bytes)
        DENSITY_MEMORY(live_reader_numeric_bytes) DENSITY_MEMORY(live_reader_control_bytes)
        DENSITY_MEMORY(maximum_reader_tile_bytes) DENSITY_MEMORY(reader_payload_bytes)
        DENSITY_MEMORY(tile_visits) DENSITY_MEMORY(work_units) DENSITY_MEMORY(required_node_memory_bytes);
#undef DENSITY_MEMORY
    py::class_<Block>(m, "_PeriodicCorrelationDensityFactorBlock")
        .def_property_readonly("kind", &Block::kind)
        .def_property_readonly("memory", [](const Block& b) { return b.memory(); })
        .def_property_readonly("q_index", &Block::q_index)
        .def_property_readonly("auxiliary_begin", &Block::auxiliary_begin)
        .def_property_readonly("finite_image_reference", &Block::finite_image_reference)
        .def_property_readonly("ao_image_source_certified", &Block::ao_image_source_certified)
        .def_property_readonly("identity_sha256", &Block::identity_sha256)
        .def_property_readonly("payload_sha256", &Block::payload_sha256)
        .def_property_readonly("selection_identity_sha256", &Block::selection_identity_sha256)
        .def_property_readonly("source_identity_sha256", &Block::source_identity_sha256)
        .def_property_readonly("whitener_payload_identity_sha256", &Block::whitener_payload_identity_sha256)
        .def_property_readonly("store_identity_sha256", &Block::store_identity_sha256)
        .def_property_readonly("consumed_tiles_identity_sha256", &Block::consumed_tiles_identity_sha256)
        .def("left_occupied", [](const Block& b, std::size_t i) {
            const auto row = b.left_occupied(i); return py::make_tuple(row.occupied_index, row.cell);
        })
        .def("right_occupied", [](const Block& b, std::size_t i) {
            const auto row = b.right_occupied(i); return py::make_tuple(row.occupied_index, row.cell);
        })
        .def("left_virtual", &Block::left_virtual).def("right_virtual", &Block::right_virtual)
        .def("element", &Block::element)
        .def("tensor_copy", [](const Block& b) {
            const auto& p = b.memory();
            if (p.auxiliary_count > 8 || p.left_count > 16 || p.right_count > 16)
                throw std::length_error("density factor diagnostic copy exceeds tiny shape cap");
            const auto* source = b.data();
            py::array_t<Complex> output({static_cast<py::ssize_t>(p.auxiliary_count),
                static_cast<py::ssize_t>(p.left_count), static_cast<py::ssize_t>(p.right_count)});
            std::memcpy(output.mutable_data(), source, p.retained_output_bytes);
            return output;
        });
    const auto tiny = [](const Reference& ref, const Schedule& schedule, const Reader& reader,
                         std::uint64_t left, std::uint64_t right, const Caps& caps) {
        if (ref.state().n_kpoints() > 8 || ref.state().n_basis() > 8
            || ref.state().n_correlated_occupied() > 4 || schedule.shape().n_auxiliary > 8
            || schedule.shape().tile_count > 128 || reader.file_bytes() > 1048576
            || left > 16 || right > 16 || caps.maximum_owned_numerical_bytes > 1048576
            || caps.maximum_work_units > 100000000 || caps.maximum_tile_visits > 128
            || caps.maximum_reader_tile_bytes > 8192)
            throw std::length_error("density factor diagnostic exceeds tiny shape or work cap");
    };
    m.def("_plan_periodic_correlation_occupied_density_factor_block", &plan_periodic_correlation_occupied_density_factor_block,
        py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"),
        py::arg("left_count"), py::arg("right_count"), py::arg("q_index"),
        py::arg("auxiliary_begin"), py::arg("auxiliary_count"));
    m.def("_plan_periodic_correlation_virtual_density_factor_block", &plan_periodic_correlation_virtual_density_factor_block,
        py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("domain"), py::arg("space"),
        py::arg("left"), py::arg("right"), py::arg("q_index"), py::arg("auxiliary_begin"), py::arg("auxiliary_count"));
    m.def("_build_periodic_correlation_occupied_density_factor_block", [tiny](
        const Reference& reference, const Schedule& schedule, const Reader& reader, const Wannier& wannier,
        const py::array& gauges, const py::array& left, const py::array& right,
        std::uint64_t q, std::uint64_t begin, std::uint64_t count, const Caps& caps) {
        for (const auto* array : {&left, &right}) {
            if (!array->dtype().is(py::dtype::of<std::uint64_t>()) || array->ndim() != 2
                || array->shape(1) != 2 || !(array->flags() & py::array::c_style))
                throw std::invalid_argument("density factor occupied lists require C-contiguous uint64 [count,2]");
        }
        tiny(reference, schedule, reader, left.shape(0), right.shape(0), caps);
        const auto& state = reference.state();
        if (!gauges.dtype().is(py::dtype::of<Complex>()) || gauges.ndim() != 3
            || !(gauges.flags() & py::array::c_style)
            || gauges.shape(0) != static_cast<py::ssize_t>(state.n_kpoints())
            || gauges.shape(1) != static_cast<py::ssize_t>(state.n_correlated_occupied())
            || gauges.shape(2) != static_cast<py::ssize_t>(state.n_correlated_occupied()))
            throw std::invalid_argument("density factor gauges require C-contiguous complex128 [Nk,nactive,nactive]");
        const auto* g = static_cast<const Complex*>(gauges.data());
        const auto ng = static_cast<std::size_t>(gauges.size());
        const auto* l = static_cast<const std::uint64_t*>(left.data());
        const auto* r = static_cast<const std::uint64_t*>(right.data());
        const auto lc = static_cast<std::size_t>(left.shape(0)), rc = static_cast<std::size_t>(right.shape(0));
        const auto ls = static_cast<std::size_t>(left.size()), rs = static_cast<std::size_t>(right.size());
        const Caps controls = caps;
        py::gil_scoped_release release;
        return build_periodic_correlation_occupied_density_factor_block(reference, schedule, reader, wannier,
            g, ng, l, ls, lc, r, rs, rc, q, begin, count, controls);
    }, py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"),
       py::arg("gauges").noconvert(), py::arg("left_indices").noconvert(), py::arg("right_indices").noconvert(),
       py::arg("q_index"), py::arg("auxiliary_begin"), py::arg("auxiliary_count"), py::arg("caps"));
    m.def("_build_periodic_correlation_virtual_density_factor_block", [tiny](
        const Reference& reference, const Schedule& schedule, const Reader& reader, const Domain& domain,
        const Space& space, const Virtual& left, const Virtual& right, std::uint64_t q,
        std::uint64_t begin, std::uint64_t count, const Caps& caps) {
        tiny(reference, schedule, reader, left.count, right.count, caps);
        if (domain.domain_dimension() > 32)
            throw std::length_error("density factor diagnostic requires a tiny PAO domain");
        const Virtual l = left, r = right;
        const Caps controls = caps;
        py::gil_scoped_release release;
        return build_periodic_correlation_virtual_density_factor_block(reference, schedule, reader, domain,
            space, l, r, q, begin, count, controls);
    }, py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("domain"), py::arg("space"),
       py::arg("left"), py::arg("right"), py::arg("q_index"), py::arg("auxiliary_begin"), py::arg("auxiliary_count"), py::arg("caps"));
}
