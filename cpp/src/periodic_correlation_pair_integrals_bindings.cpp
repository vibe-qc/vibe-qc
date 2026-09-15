// Tiny diagnostics only; production consumes native move-only block owners.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <cstring>
#include <stdexcept>
#include "vibeqc/periodic_correlation_pair_integrals.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_pair_integrals(py::module_& m) {
    using namespace vibeqc;
    using S = PeriodicCorrelationPairIntegralSelection;
    using C = PeriodicCorrelationPairIntegralCaps;
    using P = PeriodicCorrelationPairIntegralMemoryPlan;
    using B = PeriodicCorrelationPairIntegralBlock;
    using Complex = std::complex<double>;
    py::class_<S>(m, "_PeriodicCorrelationPairIntegralSelection")
        .def(py::init<>())
#define PAIR_SELECTION(name) .def_readwrite(#name, &S::name)
        PAIR_SELECTION(occupied_i) PAIR_SELECTION(cell_i)
        PAIR_SELECTION(occupied_j) PAIR_SELECTION(cell_j)
        PAIR_SELECTION(virtual_begin) PAIR_SELECTION(virtual_count)
        PAIR_SELECTION(virtual_translation_cell) PAIR_SELECTION(virtual_block);
#undef PAIR_SELECTION
    py::class_<C>(m, "_PeriodicCorrelationPairIntegralCaps")
        .def(py::init<>())
        .def_readwrite("maximum_owned_numerical_bytes", &C::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_work_units", &C::maximum_work_units)
        .def_readwrite("maximum_factor_builds", &C::maximum_factor_builds)
        .def_readwrite("maximum_tile_visits", &C::maximum_tile_visits);
    py::class_<P>(m, "_PeriodicCorrelationPairIntegralMemoryPlan")
#define PAIR_PLAN(name) .def_readonly(#name, &P::name)
        PAIR_PLAN(n_virtual) PAIR_PLAN(retained_output_bytes) PAIR_PLAN(compensation_bytes)
        PAIR_PLAN(retained_left_factor_bytes) PAIR_PLAN(maximum_factor_owned_bytes)
        PAIR_PLAN(peak_owned_numerical_bytes) PAIR_PLAN(factor_builds)
        PAIR_PLAN(tile_visits_upper_bound) PAIR_PLAN(work_units_upper_bound)
        PAIR_PLAN(required_node_memory_bytes) PAIR_PLAN(maximum_factor);
#undef PAIR_PLAN
    py::class_<B>(m, "_PeriodicCorrelationPairIntegralBlock")
        .def_property_readonly("memory", [](const B& b) { return b.memory(); })
        .def_property_readonly("selection", [](const B& b) { return b.selection(); })
        .def_property_readonly("identity_sha256", &B::identity_sha256)
        .def_property_readonly("payload_sha256", &B::payload_sha256)
        .def_property_readonly("consumed_factors_sha256", &B::consumed_factors_sha256)
        .def_property_readonly("store_identity_sha256", &B::store_identity_sha256)
        .def_property_readonly("factor_builds", &B::factor_builds)
        .def_property_readonly("tile_visits", &B::tile_visits)
        .def_property_readonly("finite_image_reference", &B::finite_image_reference)
        .def_property_readonly("real_orbital_integrals_certified", &B::real_orbital_integrals_certified)
        .def("matrix_copy", [](const B& b) {
            const auto n = b.memory().n_virtual;
            if (n > 8) throw std::length_error("pair integral copy requires tiny diagnostic dimensions");
            const auto* data = b.data();
            py::array_t<Complex> result({static_cast<py::ssize_t>(n), static_cast<py::ssize_t>(n)});
            std::memcpy(result.mutable_data(), data, b.memory().retained_output_bytes);
            return result;
        });
    m.def("_plan_periodic_correlation_pair_integral_block", &plan_periodic_correlation_pair_integral_block,
        py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"),
        py::arg("domain"), py::arg("space"), py::arg("selection"));
    m.def("_build_periodic_correlation_pair_integral_block", [](
        const PeriodicCorrelationAdmittedReference& reference,
        const PeriodicCorrelationFactorStreamSchedule& schedule,
        const PeriodicCorrelationPrivateFactorReader& reader,
        const PeriodicCorrelationWannier& wannier, const py::array& gauges,
        const PeriodicCorrelationPAODomain& domain, const PeriodicCorrelationPAOSpace& space,
        const S& selection, const C& caps) {
        const auto& state = reference.state();
        if (state.n_basis() > 8 || state.n_kpoints() > 8 || state.n_correlated_occupied() > 4
            || domain.domain_dimension() > 8 || selection.virtual_count > 8
            || schedule.shape().tile_count > 128 || reader.file_bytes() > 1048576
            || caps.maximum_owned_numerical_bytes > 1048576 || caps.maximum_factor_builds > 4096
            || caps.maximum_tile_visits > 65536 || caps.maximum_work_units > 100000000)
            throw std::length_error("pair integral diagnostic exceeds tiny shape or work limits");
        if (!gauges.dtype().is(py::dtype::of<Complex>()) || gauges.ndim() != 3
            || !(gauges.flags() & py::array::c_style)
            || reinterpret_cast<std::uintptr_t>(gauges.data()) % alignof(Complex)
            || gauges.shape(0) != static_cast<py::ssize_t>(state.n_kpoints())
            || gauges.shape(1) != static_cast<py::ssize_t>(state.n_correlated_occupied())
            || gauges.shape(2) != static_cast<py::ssize_t>(state.n_correlated_occupied()))
            throw std::invalid_argument("pair integral diagnostic requires aligned contiguous complex128 gauges");
        const S selected = selection; const C controls = caps;
        const auto* values = static_cast<const Complex*>(gauges.data());
        const auto count = static_cast<std::size_t>(gauges.size());
        py::gil_scoped_release release;
        return build_periodic_correlation_pair_integral_block(reference, schedule, reader, wannier,
            values, count, domain, space, selected, controls);
    }, py::arg("reference"), py::arg("schedule"), py::arg("reader"), py::arg("wannier"),
       py::arg("gauges").noconvert(), py::arg("domain"), py::arg("space"), py::arg("selection"), py::arg("caps"));
}
