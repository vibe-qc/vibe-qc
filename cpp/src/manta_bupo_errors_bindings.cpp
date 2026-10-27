#include "vibeqc/manta_bupo_errors.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_manta_bupo_errors(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<BubbleErrorPlan>(m,"BubbleErrorPlan")
        .def_readonly("order",&BubbleErrorPlan::order)
        .def_readonly("nodes",&BubbleErrorPlan::nodes)
        .def_readonly("moments",&BubbleErrorPlan::moments)
        .def_readonly("scalar_terms",&BubbleErrorPlan::scalar_terms)
        .def_readonly("kernel_terms",&BubbleErrorPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&BubbleErrorPlan::modeled_peak_bytes);
    py::class_<BubbleEnvelope>(m,"BubbleEnvelope")
        .def_readonly("full_moments",&BubbleEnvelope::full_moments)
        .def_readonly("tail_moments",&BubbleEnvelope::tail_moments)
        .def_readonly("potential",&BubbleEnvelope::potential);
    py::class_<BubbleErrorBound>(m,"BubbleErrorBound")
        .def_readonly("compact_remainder",&BubbleErrorBound::compact_remainder)
        .def_readonly("tail_penetration",&BubbleErrorBound::tail_penetration)
        .def_readonly("full_moment_tail",&BubbleErrorBound::full_moment_tail)
        .def_readonly("total",&BubbleErrorBound::total)
        .def_readonly("source_identity",&BubbleErrorBound::source_identity);
    py::class_<BubbleInteraction>(m,"BubbleInteraction")
        .def_readonly("matrix",&BubbleInteraction::matrix)
        .def_readonly("error",&BubbleInteraction::error);
    py::class_<BubbleErrorModel,std::shared_ptr<BubbleErrorModel>>(m,"BubbleErrorModel")
        .def(py::init<std::shared_ptr<const BubbleTree>,int,const ResourceBudget&>(),py::arg("tree"),py::arg("order"),py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",[](const BubbleErrorModel& m) { return BubbleErrorPlan(m.plan()); })
        .def_property_readonly("envelopes",[](const BubbleErrorModel& m) { return std::vector<BubbleEnvelope>(m.envelopes()); })
        .def_property_readonly("source_identity",&BubbleErrorModel::source_identity)
        .def("interaction",&BubbleErrorModel::interaction,py::arg("target"),py::arg("source"),py::call_guard<py::gil_scoped_release>())
        .def("auxiliary_interaction",&BubbleErrorModel::auxiliary_interaction,py::arg("target"),py::arg("source"),py::call_guard<py::gil_scoped_release>())
        .def("bound",&BubbleErrorModel::bound,py::arg("target"),py::arg("source"),py::call_guard<py::gil_scoped_release>());
    m.def("plan_bubble_errors",&plan_bubble_errors,py::arg("tree"),py::arg("order"),py::arg("budget")=ResourceBudget{});
}
