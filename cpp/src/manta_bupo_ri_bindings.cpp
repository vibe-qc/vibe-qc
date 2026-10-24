#include "vibeqc/manta_bupo_ri.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace {
Eigen::MatrixXd density(const vibeqc::manta::BupoRIJBuilder& b,const py::array& a) {
    const auto n=py::ssize_t(b.plan().n_basis);
    if(!a.dtype().is(py::dtype::of<double>()) || a.ndim()!=2 || a.shape(0)!=n || a.shape(1)!=n
       || !(a.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA RI-BUPO density must be contiguous square float64 matching the basis");
    const auto* ptr=static_cast<const double*>(a.data());
    if(a.flags() & py::array::f_style) return Eigen::Map<const Eigen::MatrixXd>(ptr,n,n);
    return Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>(ptr,n,n);
}
}
void bind_manta_bupo_ri(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<BupoRIPlan>(m,"BupoRIPlan")
        .def_readonly("n_basis",&BupoRIPlan::n_basis)
        .def_readonly("n_auxiliary",&BupoRIPlan::n_auxiliary)
        .def_readonly("shell_calls",&BupoRIPlan::shell_calls)
        .def_readonly("primitive_products",&BupoRIPlan::primitive_products)
        .def_readonly("scalar_terms",&BupoRIPlan::scalar_terms)
        .def_readonly("kernel_terms",&BupoRIPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&BupoRIPlan::modeled_peak_bytes);
    py::class_<BupoRIReceipt>(m,"BupoRIReceipt")
        .def_readonly("source_identity",&BupoRIReceipt::source_identity)
        .def_readonly("exact_ri_source_identity",&BupoRIReceipt::exact_ri_source_identity)
        .def_readonly("density_identity",&BupoRIReceipt::density_identity)
        .def_readonly("algorithm",&BupoRIReceipt::algorithm)
        .def_readonly("provider",&BupoRIReceipt::provider)
        .def_readonly("kernel",&BupoRIReceipt::kernel)
        .def_readonly("retained_rank",&BupoRIReceipt::retained_rank)
        .def_readonly("modeled_peak_bytes",&BupoRIReceipt::modeled_peak_bytes)
        .def_readonly("coulomb_energy",&BupoRIReceipt::coulomb_energy)
        .def_readonly("metric_rhs_norm",&BupoRIReceipt::metric_rhs_norm)
        .def_readonly("metric_residual_norm",&BupoRIReceipt::metric_residual_norm)
        .def_readonly("metric_inverse_norm_bound",&BupoRIReceipt::metric_inverse_norm_bound)
        .def_readonly("forward_error_l2_bound",&BupoRIReceipt::forward_error_l2_bound)
        .def_readonly("approximation_energy_bound",&BupoRIReceipt::approximation_energy_bound)
        .def_readonly("approximation_j_bound",&BupoRIReceipt::approximation_j_bound)
        .def_readonly("adjoint_j_bound",&BupoRIReceipt::adjoint_j_bound)
        .def_readonly("forward_propagated_j_bound",&BupoRIReceipt::forward_propagated_j_bound)
        .def_readonly("forward",&BupoRIReceipt::forward)
        .def_readonly("adjoint",&BupoRIReceipt::adjoint);
    py::class_<BupoRIResult>(m,"BupoRIResult")
        .def_property_readonly("J",[](const BupoRIResult& r) { return Eigen::MatrixXd(r.J); })
        .def_property_readonly("rhs",[](const BupoRIResult& r) { return Eigen::VectorXd(r.rhs); })
        .def_property_readonly("coefficients",[](const BupoRIResult& r) { return Eigen::VectorXd(r.coefficients); })
        .def_readonly("receipt",&BupoRIResult::receipt);
    py::class_<BupoRIJBuilder,vibeqc::JKBuilder,std::shared_ptr<BupoRIJBuilder>>(m,"BupoRIJBuilder")
        .def(py::init<const vibeqc::BasisSet&,const vibeqc::BasisSet&,const BupoFittingOptions&,const RIOptions&,const ResourceBudget&>(),
            py::arg("orbital"),py::arg("auxiliary"),py::arg("coupling_options")=BupoFittingOptions{},
            py::arg("metric_options")=RIOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",&BupoRIJBuilder::plan,py::return_value_policy::copy)
        .def_property_readonly("coupling_plan",&BupoRIJBuilder::coupling_plan,py::return_value_policy::copy)
        .def_property_readonly("metric_info",&BupoRIJBuilder::metric_info,py::return_value_policy::copy)
        .def_property_readonly("metric",&BupoRIJBuilder::metric,py::return_value_policy::copy)
        .def_property_readonly("factor",&BupoRIJBuilder::factor,py::return_value_policy::copy)
        .def_property_readonly("source_identity",&BupoRIJBuilder::source_identity)
        .def_property_readonly("exact_ri_source_identity",&BupoRIJBuilder::exact_ri_source_identity)
        .def("build",[](const BupoRIJBuilder& b,const py::array& a) { const auto d=density(b,a); py::gil_scoped_release release; return b.build(d); })
        .def("build_J",[](const BupoRIJBuilder& b,const py::array& a) { const auto d=density(b,a); py::gil_scoped_release release; return b.build_J(d); })
        .def("build_exact",[](const BupoRIJBuilder& b,const py::array& a) { const auto d=density(b,a); py::gil_scoped_release release; return b.build_exact(d); });
}
