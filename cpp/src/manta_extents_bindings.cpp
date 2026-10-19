#include "vibeqc/manta_extents.hpp"
#include "vibeqc/manta_gaussian_majorants.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py=pybind11;
void bind_manta_extents(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<GaussianMajorantTerm>(m,"GaussianMajorantTerm")
        .def_readonly("center",&GaussianMajorantTerm::center)
        .def_readonly("exponent",&GaussianMajorantTerm::exponent)
        .def_readonly("charge_bound",&GaussianMajorantTerm::charge_bound);
    auto gp=py::class_<GaussianMajorantPlan>(m,"GaussianMajorantPlan");
#define P(name) gp.def_readonly(#name,&GaussianMajorantPlan::name)
    P(maximum_terms); P(shell_calls); P(primitive_pairs); P(scalar_terms); P(kernel_terms); P(modeled_peak_bytes);
#undef P
    py::class_<GaussianMajorant>(m,"GaussianMajorant")
        .def_property_readonly("terms",[](const GaussianMajorant& g) { return std::vector<GaussianMajorantTerm>(g.terms()); })
        .def_property_readonly("plan",[](const GaussianMajorant& g) { return GaussianMajorantPlan(g.plan()); })
        .def_property_readonly("source_identity",&GaussianMajorant::source_identity)
        .def_property_readonly("basis_identity",&GaussianMajorant::basis_identity)
        .def_property_readonly("kind",&GaussianMajorant::kind)
        .def_property_readonly("charge_bound",&GaussianMajorant::charge_bound);
    m.def("plan_pair_gaussian_majorant",&plan_pair_gaussian_majorant,py::arg("basis"),py::arg("shells"),py::arg("budget")=ResourceBudget{});
    m.def("plan_auxiliary_gaussian_majorant",&plan_auxiliary_gaussian_majorant,py::arg("basis"),py::arg("shell"),py::arg("budget")=ResourceBudget{});
    m.def("compute_pair_gaussian_majorant",&compute_pair_gaussian_majorant,py::arg("basis"),py::arg("shells"),
        py::arg("ket_translation")=md::Point{},py::arg("budget")=ResourceBudget{});
    m.def("compute_auxiliary_gaussian_majorant",&compute_auxiliary_gaussian_majorant,py::arg("basis"),py::arg("shell"),py::arg("budget")=ResourceBudget{});
    py::class_<ExtentOptions>(m,"ExtentOptions")
        .def(py::init<>())
        .def_readwrite("pointwise_tolerance",&ExtentOptions::pointwise_tolerance)
        .def_readwrite("tail_mass_tolerance",&ExtentOptions::tail_mass_tolerance)
        .def_readwrite("minimum_radius",&ExtentOptions::minimum_radius)
        .def_readwrite("maximum_radius",&ExtentOptions::maximum_radius)
        .def_readwrite("moment_order",&ExtentOptions::moment_order)
        .def_readwrite("retain_full_moments",&ExtentOptions::retain_full_moments);
    py::class_<ExtentPlan>(m,"ExtentPlan")
        .def_readonly("shell_pairs",&ExtentPlan::shell_pairs)
        .def_readonly("primitive_pairs",&ExtentPlan::primitive_pairs)
        .def_readonly("maximum_pair_primitives",&ExtentPlan::maximum_pair_primitives)
        .def_readonly("scalar_terms",&ExtentPlan::scalar_terms)
        .def_readonly("kernel_terms",&ExtentPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&ExtentPlan::modeled_peak_bytes);
    py::class_<PairExtent>(m,"PairExtent")
        .def_readonly("shells",&PairExtent::shells)
        .def_readonly("center",&PairExtent::center)
        .def_readonly("radius",&PairExtent::radius)
        .def_readonly("pointwise_bound",&PairExtent::pointwise_bound)
        .def_readonly("tail_mass_bound",&PairExtent::tail_mass_bound)
        .def_readonly("l1_norm_bound",&PairExtent::l1_norm_bound)
        .def_readonly("linfinity_bound",&PairExtent::linfinity_bound)
        .def_readonly("potential_bound",&PairExtent::potential_bound)
        .def_readonly("tail_moment_bounds",&PairExtent::tail_moment_bounds)
        .def_readonly("full_moment_bounds",&PairExtent::full_moment_bounds)
        .def_readonly("always_near",&PairExtent::always_near)
        .def_readonly("reason",&PairExtent::reason);
    py::class_<ExtentReceipt>(m,"ExtentReceipt")
        .def_readonly("source_identity",&ExtentReceipt::source_identity)
        .def_readonly("center_policy",&ExtentReceipt::center_policy)
        .def_readonly("envelope",&ExtentReceipt::envelope)
        .def_readonly("shell_pairs",&ExtentReceipt::shell_pairs)
        .def_readonly("primitive_pairs",&ExtentReceipt::primitive_pairs)
        .def_readonly("envelope_evaluations",&ExtentReceipt::envelope_evaluations)
        .def_readonly("always_near_pairs",&ExtentReceipt::always_near_pairs)
        .def_readonly("modeled_peak_bytes",&ExtentReceipt::modeled_peak_bytes);
    py::class_<PairExtents>(m,"PairExtents")
        .def_readonly("pairs",&PairExtents::pairs)
        .def_readonly("receipt",&PairExtents::receipt);
    m.def("plan_pair_extents",&plan_pair_extents,py::arg("basis"),py::arg("options")=ExtentOptions{},py::arg("budget")=ResourceBudget{});
    m.def("compute_pair_extents",&compute_pair_extents,py::arg("basis"),py::arg("options")=ExtentOptions{},py::arg("budget")=ResourceBudget{},py::call_guard<py::gil_scoped_release>());
    py::class_<AuxiliaryExtentPlan>(m,"AuxiliaryExtentPlan")
        .def_readonly("shells",&AuxiliaryExtentPlan::shells)
        .def_readonly("primitives",&AuxiliaryExtentPlan::primitives)
        .def_readonly("maximum_shell_primitives",&AuxiliaryExtentPlan::maximum_shell_primitives)
        .def_readonly("scalar_terms",&AuxiliaryExtentPlan::scalar_terms)
        .def_readonly("kernel_terms",&AuxiliaryExtentPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&AuxiliaryExtentPlan::modeled_peak_bytes);

    py::class_<AuxiliaryExtent>(m,"AuxiliaryExtent")
        .def_readonly("shell",&AuxiliaryExtent::shell)
        .def_readonly("center",&AuxiliaryExtent::center)
        .def_readonly("radius",&AuxiliaryExtent::radius)
        .def_readonly("pointwise_bound",&AuxiliaryExtent::pointwise_bound)
        .def_readonly("tail_mass_bound",&AuxiliaryExtent::tail_mass_bound)
        .def_readonly("l1_norm_bound",&AuxiliaryExtent::l1_norm_bound)
        .def_readonly("linfinity_bound",&AuxiliaryExtent::linfinity_bound)
        .def_readonly("potential_bound",&AuxiliaryExtent::potential_bound)
        .def_readonly("tail_moment_bounds",&AuxiliaryExtent::tail_moment_bounds)
        .def_readonly("full_moment_bounds",&AuxiliaryExtent::full_moment_bounds)
        .def_readonly("always_near",&AuxiliaryExtent::always_near)
        .def_readonly("reason",&AuxiliaryExtent::reason);

    py::class_<AuxiliaryExtentReceipt>(m,"AuxiliaryExtentReceipt")
        .def_readonly("source_identity",&AuxiliaryExtentReceipt::source_identity)
        .def_readonly("center_policy",&AuxiliaryExtentReceipt::center_policy)
        .def_readonly("envelope",&AuxiliaryExtentReceipt::envelope)
        .def_readonly("shells",&AuxiliaryExtentReceipt::shells)
        .def_readonly("primitives",&AuxiliaryExtentReceipt::primitives)
        .def_readonly("envelope_evaluations",&AuxiliaryExtentReceipt::envelope_evaluations)
        .def_readonly("always_near_shells",&AuxiliaryExtentReceipt::always_near_shells)
        .def_readonly("modeled_peak_bytes",&AuxiliaryExtentReceipt::modeled_peak_bytes);

    py::class_<AuxiliaryExtents>(m,"AuxiliaryExtents")
        .def_readonly("shells",&AuxiliaryExtents::shells)
        .def_readonly("receipt",&AuxiliaryExtents::receipt);
    m.def("plan_auxiliary_extents",&plan_auxiliary_extents,py::arg("basis"),py::arg("options")=ExtentOptions{},py::arg("budget")=ResourceBudget{});
    m.def("compute_auxiliary_extents",&compute_auxiliary_extents,py::arg("basis"),py::arg("options")=ExtentOptions{},py::arg("budget")=ResourceBudget{},py::call_guard<py::gil_scoped_release>());

}
