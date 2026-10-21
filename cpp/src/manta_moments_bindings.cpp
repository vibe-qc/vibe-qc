#include "vibeqc/manta_moments.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_manta_moments(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<PairMomentPlan>(m,"PairMomentPlan")
        .def_readonly("shape",&PairMomentPlan::shape)
        .def_readonly("order",&PairMomentPlan::order)
        .def_readonly("moments",&PairMomentPlan::moments)
        .def_readonly("primitive_pairs",&PairMomentPlan::primitive_pairs)
        .def_readonly("scalar_terms",&PairMomentPlan::scalar_terms)
        .def_readonly("kernel_terms",&PairMomentPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&PairMomentPlan::modeled_peak_bytes);
    py::class_<PairMomentReceipt>(m,"PairMomentReceipt")
        .def_readonly("source_identity",&PairMomentReceipt::source_identity)
        .def_readonly("normalization",&PairMomentReceipt::normalization)
        .def_readonly("shell_calls",&PairMomentReceipt::shell_calls)
        .def_readonly("primitive_pairs",&PairMomentReceipt::primitive_pairs)
        .def_readonly("primitive_expansions",&PairMomentReceipt::primitive_expansions)
        .def_readonly("primitive_moment_products",&PairMomentReceipt::primitive_moment_products)
        .def_readonly("modeled_peak_bytes",&PairMomentReceipt::modeled_peak_bytes);
    py::class_<PairMomentResult>(m,"PairMomentResult")
        .def_property_readonly("values",[](const PairMomentResult& r) { return md::Matrix(r.values); })
        .def_readonly("indices",&PairMomentResult::indices)
        .def_readonly("ao_shape",&PairMomentResult::ao_shape)
        .def_readonly("receipt",&PairMomentResult::receipt);
    m.def("plan_pair_moments",&plan_pair_moments,py::arg("basis"),py::arg("shells"),py::arg("order"),py::arg("budget")=ResourceBudget{});
    m.def("compute_pair_moments",&compute_pair_moments,py::arg("basis"),py::arg("shells"),py::arg("origin"),py::arg("order"),py::arg("budget")=ResourceBudget{},py::call_guard<py::gil_scoped_release>());
    py::class_<AuxiliaryMomentPlan>(m,"AuxiliaryMomentPlan")
        .def_readonly("shape",&AuxiliaryMomentPlan::shape)
        .def_readonly("order",&AuxiliaryMomentPlan::order)
        .def_readonly("moments",&AuxiliaryMomentPlan::moments)
        .def_readonly("primitives",&AuxiliaryMomentPlan::primitives)
        .def_readonly("scalar_terms",&AuxiliaryMomentPlan::scalar_terms)
        .def_readonly("kernel_terms",&AuxiliaryMomentPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&AuxiliaryMomentPlan::modeled_peak_bytes);
    py::class_<AuxiliaryMomentReceipt>(m,"AuxiliaryMomentReceipt")
        .def_readonly("source_identity",&AuxiliaryMomentReceipt::source_identity)
        .def_readonly("normalization",&AuxiliaryMomentReceipt::normalization)
        .def_readonly("shell_calls",&AuxiliaryMomentReceipt::shell_calls)
        .def_readonly("primitives",&AuxiliaryMomentReceipt::primitives)
        .def_readonly("primitive_expansions",&AuxiliaryMomentReceipt::primitive_expansions)
        .def_readonly("primitive_moment_products",&AuxiliaryMomentReceipt::primitive_moment_products)
        .def_readonly("modeled_peak_bytes",&AuxiliaryMomentReceipt::modeled_peak_bytes);
    py::class_<AuxiliaryMomentResult>(m,"AuxiliaryMomentResult")
        .def_property_readonly("values",[](const AuxiliaryMomentResult& r) { return md::Matrix(r.values); })
        .def_readonly("indices",&AuxiliaryMomentResult::indices)
        .def_readonly("receipt",&AuxiliaryMomentResult::receipt);
    m.def("plan_auxiliary_moments",&plan_auxiliary_moments,py::arg("basis"),py::arg("shell"),py::arg("order"),py::arg("budget")=ResourceBudget{});
    m.def("compute_auxiliary_moments",&compute_auxiliary_moments,py::arg("basis"),py::arg("shell"),py::arg("origin"),py::arg("order"),py::arg("budget")=ResourceBudget{},py::call_guard<py::gil_scoped_release>());
}
