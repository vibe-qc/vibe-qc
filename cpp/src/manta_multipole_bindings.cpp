#include "vibeqc/manta_multipole.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_manta_multipole(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<MultipolePlan>(m,"MultipolePlan")
        .def_readonly("order",&MultipolePlan::order)
        .def_readonly("moments",&MultipolePlan::moments)
        .def_readonly("scalar_terms",&MultipolePlan::scalar_terms)
        .def_readonly("kernel_terms",&MultipolePlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&MultipolePlan::modeled_peak_bytes);
    py::class_<MultipoleMatrix>(m,"MultipoleMatrix")
        .def_property_readonly("values",[](const MultipoleMatrix& r) { return Eigen::MatrixXd(r.values); })
        .def_readonly("indices",&MultipoleMatrix::indices)
        .def_readonly("operation",&MultipoleMatrix::operation)
        .def_readonly("source_identity",&MultipoleMatrix::source_identity)
        .def_readonly("normalization",&MultipoleMatrix::normalization)
        .def_readonly("coefficients_evaluated",&MultipoleMatrix::coefficients_evaluated)
        .def_readonly("modeled_peak_bytes",&MultipoleMatrix::modeled_peak_bytes);
    m.def("plan_multipoles",&plan_multipoles,py::arg("order"),py::arg("budget")=ResourceBudget{});
    m.def("multipole_translation",&multipole_translation,py::arg("source_minus_target"),py::arg("order"),py::arg("budget")=ResourceBudget{},py::call_guard<py::gil_scoped_release>());
    m.def("multipole_coulomb",&multipole_coulomb,py::arg("target_minus_source"),py::arg("order"),py::arg("budget")=ResourceBudget{},py::call_guard<py::gil_scoped_release>());
    m.def("compact_multipole_remainder",&compact_multipole_remainder,py::arg("distance"),py::arg("target_radius"),py::arg("source_radius"),py::arg("target_norm"),py::arg("source_norm"),py::arg("order"));
}
