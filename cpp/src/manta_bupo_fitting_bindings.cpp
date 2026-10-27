#include "vibeqc/manta_bupo_fitting.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace {
void check_array(const py::array& a) {
    if(!a.dtype().is(py::dtype::of<double>()) || !(a.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA BUPO fitting input must be contiguous float64");
}
}
void bind_manta_bupo_fitting(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<BupoFittingOptions>(m,"BupoFittingOptions")
        .def(py::init<>())
        .def_readwrite("order",&BupoFittingOptions::order)
        .def_readwrite("opening",&BupoFittingOptions::opening)
        .def_readwrite("coupling_tolerance",&BupoFittingOptions::coupling_tolerance)
        .def_readwrite("extents",&BupoFittingOptions::extents)
        .def_readwrite("bubbles",&BupoFittingOptions::bubbles);
    py::class_<BupoFittingPlan>(m,"BupoFittingPlan")
        .def_readonly("n_basis",&BupoFittingPlan::n_basis)
        .def_readonly("n_auxiliary",&BupoFittingPlan::n_auxiliary)
        .def_readonly("shell_pairs",&BupoFittingPlan::shell_pairs)
        .def_readonly("nodes",&BupoFittingPlan::nodes)
        .def_readonly("moments",&BupoFittingPlan::moments)
        .def_readonly("near_shell_calls",&BupoFittingPlan::near_shell_calls)
        .def_readonly("near_primitive_products",&BupoFittingPlan::near_primitive_products)
        .def_readonly("far_blocks",&BupoFittingPlan::far_blocks)
        .def_readonly("traversal_visits",&BupoFittingPlan::traversal_visits)
        .def_readonly("pair_moment_panels",&BupoFittingPlan::pair_moment_panels)
        .def_readonly("auxiliary_moment_panels",&BupoFittingPlan::auxiliary_moment_panels)
        .def_readonly("moment_primitive_products",&BupoFittingPlan::moment_primitive_products)
        .def_readonly("shell_calls",&BupoFittingPlan::shell_calls)
        .def_readonly("primitive_products",&BupoFittingPlan::primitive_products)
        .def_readonly("scalar_terms",&BupoFittingPlan::scalar_terms)
        .def_readonly("kernel_terms",&BupoFittingPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&BupoFittingPlan::modeled_peak_bytes)
        .def_readonly("maximum_exact_coupling",&BupoFittingPlan::maximum_exact_coupling)
        .def_readonly("maximum_coupling_error",&BupoFittingPlan::maximum_coupling_error)
        .def_readonly("maximum_compact_error",&BupoFittingPlan::maximum_compact_error)
        .def_readonly("maximum_penetration_error",&BupoFittingPlan::maximum_penetration_error)
        .def_readonly("maximum_moment_tail_error",&BupoFittingPlan::maximum_moment_tail_error)
        .def_readonly("partition_identity",&BupoFittingPlan::partition_identity);
    py::class_<BupoFittingReceipt>(m,"BupoFittingReceipt")
        .def_readonly("source_identity",&BupoFittingReceipt::source_identity)
        .def_readonly("input_identity",&BupoFittingReceipt::input_identity)
        .def_readonly("partition_identity",&BupoFittingReceipt::partition_identity)
        .def_readonly("direction",&BupoFittingReceipt::direction)
        .def_readonly("provider",&BupoFittingReceipt::provider)
        .def_readonly("algorithm",&BupoFittingReceipt::algorithm)
        .def_readonly("normalization",&BupoFittingReceipt::normalization)
        .def_readonly("near_shell_calls",&BupoFittingReceipt::near_shell_calls)
        .def_readonly("near_primitive_products",&BupoFittingReceipt::near_primitive_products)
        .def_readonly("far_blocks",&BupoFittingReceipt::far_blocks)
        .def_readonly("traversal_visits",&BupoFittingReceipt::traversal_visits)
        .def_readonly("pair_moment_panels",&BupoFittingReceipt::pair_moment_panels)
        .def_readonly("auxiliary_moment_panels",&BupoFittingReceipt::auxiliary_moment_panels)
        .def_readonly("moment_primitive_products",&BupoFittingReceipt::moment_primitive_products)
        .def_readonly("modeled_peak_bytes",&BupoFittingReceipt::modeled_peak_bytes)
        .def_readonly("input_l1",&BupoFittingReceipt::input_l1)
        .def_readonly("approximation_bound",&BupoFittingReceipt::approximation_bound)
        .def_readonly("compact_bound",&BupoFittingReceipt::compact_bound)
        .def_readonly("penetration_bound",&BupoFittingReceipt::penetration_bound)
        .def_readonly("moment_tail_bound",&BupoFittingReceipt::moment_tail_bound);
    py::class_<BupoFittingForward>(m,"BupoFittingForward")
        .def_property_readonly("values",[](const BupoFittingForward& r) { return Eigen::VectorXd(r.values); })
        .def_readonly("receipt",&BupoFittingForward::receipt);
    py::class_<BupoFittingAdjoint>(m,"BupoFittingAdjoint")
        .def_property_readonly("values",[](const BupoFittingAdjoint& r) { return Eigen::MatrixXd(r.values); })
        .def_readonly("receipt",&BupoFittingAdjoint::receipt);
    py::class_<BupoFittingCoupling>(m,"BupoFittingCoupling")
        .def(py::init<const vibeqc::BasisSet&,const vibeqc::BasisSet&,const BupoFittingOptions&,const ResourceBudget&>(),
             py::arg("orbital"),py::arg("auxiliary"),py::arg("options")=BupoFittingOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",[](const BupoFittingCoupling& b) { return BupoFittingPlan(b.plan()); })
        .def_property_readonly("source_identity",&BupoFittingCoupling::source_identity)
        .def("forward",[](const BupoFittingCoupling& b,const py::array& a) {
            check_array(a); const auto n=py::ssize_t(b.plan().n_basis);
            if(a.ndim()!=2 || a.shape(0)!=n || a.shape(1)!=n) throw py::value_error("MANTA BUPO fitting density shape");
            Eigen::MatrixXd input(n,n); const auto* data=static_cast<const double*>(a.data());
            if(a.flags() & py::array::f_style) input=Eigen::Map<const Eigen::MatrixXd>(data,n,n);
            else input=Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>(data,n,n);
            py::gil_scoped_release release; return b.forward(input);
        },py::arg("density"))
        .def("adjoint",[](const BupoFittingCoupling& b,const py::array& a) {
            check_array(a);
            if(a.ndim()!=1 || a.shape(0)!=py::ssize_t(b.plan().n_auxiliary)) throw py::value_error("MANTA BUPO fitting coefficient shape");
            const Eigen::VectorXd input=Eigen::Map<const Eigen::VectorXd>(static_cast<const double*>(a.data()),a.shape(0));
            py::gil_scoped_release release; return b.adjoint(input);
        },py::arg("coefficients"));
}
