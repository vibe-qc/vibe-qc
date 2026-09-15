// Internal diagnostic bindings for the native admitted-reference boundary.

#include <pybind11/pybind11.h>

#include <memory>
#include <utility>

#include "vibeqc/periodic_correlation_admitted_reference.hpp"

#ifndef VIBEQC_BINDINGS_HAS_PY_ALIAS
namespace py = pybind11;
#endif

void bind_periodic_correlation_admitted_reference(py::module_& m) {
    using State = vibeqc::PeriodicRestrictedMeanFieldState;
    using Reference = vibeqc::PeriodicCorrelationAdmittedReference;

    m.attr("_PERIODIC_CORRELATION_ADMITTED_REFERENCE_CONTRACT_VERSION") =
        py::int_(
            vibeqc::kPeriodicCorrelationAdmittedReferenceContractVersion);

    py::class_<Reference>(m, "_PeriodicCorrelationAdmittedReference")
        .def_property_readonly(
            "contract_version", &Reference::contract_version)
        .def_property_readonly(
            "state",
            [](const Reference& reference) -> std::shared_ptr<State> {
                return std::const_pointer_cast<State>(
                    reference.state_handle());
            })
        .def_property_readonly(
            "dimensions",
            [](const Reference& reference) {
                return reference.dimensions();
            })
        .def_property_readonly(
            "budget",
            [](const Reference& reference) { return reference.budget(); })
        .def_property_readonly(
            "plan",
            [](const Reference& reference) { return reference.plan(); })
        .def_property_readonly(
            "state_resident_bytes", &Reference::state_resident_bytes);

    m.def(
        "_make_periodic_correlation_admitted_reference",
        [](std::shared_ptr<State> state,
           vibeqc::PeriodicCorrelationStaticDimensions dimensions,
           vibeqc::PeriodicCorrelationResourceBudget budget) {
            std::shared_ptr<const State> immutable_state = std::move(state);
            return vibeqc::make_periodic_correlation_admitted_reference(
                std::move(immutable_state),
                std::move(dimensions),
                std::move(budget));
        },
        py::arg("state"),
        py::arg("dimensions"),
        py::arg("budget"));
}
