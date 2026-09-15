// Tiny diagnostic bindings for native cross-PAO-space overlaps.
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <complex>
#include <cstring>
#include <stdexcept>

#include "vibeqc/periodic_correlation_pao_overlap.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_pao_overlap(py::module_& m) {
    using namespace vibeqc;
    using Reference = PeriodicCorrelationAdmittedReference;
    using Domain = PeriodicCorrelationPAODomain;
    using Space = PeriodicCorrelationPAOSpace;
    using Selection = PeriodicCorrelationVirtualBlockSelection;
    using Caps = PeriodicCorrelationPAOOverlapCaps;
    using Memory = PeriodicCorrelationPAOOverlapMemoryPlan;
    using Overlap = PeriodicCorrelationPAOSpaceOverlap;
    using Complex = std::complex<double>;
    m.attr("_PERIODIC_CORRELATION_PAO_OVERLAP_CONTRACT_VERSION") = py::int_(1);
    py::class_<Caps>(m, "_PeriodicCorrelationPAOOverlapCaps")
        .def(py::init<>()).def_readwrite("maximum_owned_numerical_bytes", &Caps::maximum_owned_numerical_bytes)
        .def_readwrite("maximum_work_units", &Caps::maximum_work_units);
#define PAO_OVERLAP_MEMORY(name) .def_readonly(#name, &Memory::name)
    py::class_<Memory>(m, "_PeriodicCorrelationPAOOverlapMemoryPlan")
        PAO_OVERLAP_MEMORY(n_cells) PAO_OVERLAP_MEMORY(n_basis)
        PAO_OVERLAP_MEMORY(left_count) PAO_OVERLAP_MEMORY(right_count)
        PAO_OVERLAP_MEMORY(output_bytes) PAO_OVERLAP_MEMORY(compensation_bytes)
        PAO_OVERLAP_MEMORY(coefficient_panel_bytes) PAO_OVERLAP_MEMORY(scratch_bytes)
        PAO_OVERLAP_MEMORY(peak_owned_numerical_bytes)
        PAO_OVERLAP_MEMORY(unique_domain_owners) PAO_OVERLAP_MEMORY(unique_space_owners)
        PAO_OVERLAP_MEMORY(live_domain_bytes) PAO_OVERLAP_MEMORY(live_space_bytes)
        PAO_OVERLAP_MEMORY(work_units) PAO_OVERLAP_MEMORY(required_node_memory_bytes);
#undef PAO_OVERLAP_MEMORY
    py::class_<Overlap>(m, "_PeriodicCorrelationPAOSpaceOverlap")
        .def_property_readonly("memory", [](const Overlap& o) { return o.memory(); })
        .def_property_readonly("left_selection", &Overlap::left_selection)
        .def_property_readonly("right_selection", &Overlap::right_selection)
        .def_property_readonly("identity_sha256", &Overlap::identity_sha256)
        .def_property_readonly("payload_sha256", &Overlap::payload_sha256)
        .def("element", &Overlap::element)
        .def("matrix_copy", [](const Overlap& o) {
            const auto& p = o.memory();
            if (p.left_count > 32 || p.right_count > 32)
                throw std::length_error("PAO overlap diagnostic copy exceeds tiny shape cap");
            const auto* source = o.data();
            py::array_t<Complex> output({static_cast<py::ssize_t>(p.left_count), static_cast<py::ssize_t>(p.right_count)});
            std::memcpy(output.mutable_data(), source, p.output_bytes);
            return output;
        });
    m.def("_plan_periodic_correlation_pao_space_overlap", &plan_periodic_correlation_pao_space_overlap,
        py::arg("reference"), py::arg("left_domain"), py::arg("left_space"), py::arg("left_selection"),
        py::arg("right_domain"), py::arg("right_space"), py::arg("right_selection"));
    m.def("_make_periodic_correlation_pao_space_overlap", [](
        const Reference& reference, const Domain& ld, const Space& ls, const Selection& left,
        const Domain& rd, const Space& rs, const Selection& right, const Caps& caps) {
        if (reference.state().n_kpoints() > 64 || reference.state().n_basis() > 8
            || ld.domain_dimension() > 32 || rd.domain_dimension() > 32 || left.count > 32 || right.count > 32
            || caps.maximum_owned_numerical_bytes > 1048576 || caps.maximum_work_units > 100000000)
            throw std::length_error("PAO overlap diagnostic exceeds tiny shape or work cap");
        const Selection l = left, r = right;
        const Caps controls = caps;
        py::gil_scoped_release release;
        return make_periodic_correlation_pao_space_overlap(reference, ld, ls, l, rd, rs, r, controls);
    }, py::arg("reference"), py::arg("left_domain"), py::arg("left_space"), py::arg("left_selection"),
       py::arg("right_domain"), py::arg("right_space"), py::arg("right_selection"), py::arg("caps"));
}
