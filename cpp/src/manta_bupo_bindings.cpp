#include "vibeqc/manta_bupo.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
namespace py=pybind11;
namespace {
Eigen::MatrixXd density_copy(const vibeqc::manta::BupoJBuilder& builder,const py::array& input) {
    const auto n=static_cast<py::ssize_t>(builder.plan().n_basis);
    if(!input.dtype().is(py::dtype::of<double>()) || input.ndim()!=2 || input.shape(0)!=n || input.shape(1)!=n)
        throw py::value_error("MANTA BUPO density must be a square float64 array matching the basis");
    if(!(input.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA BUPO density must be C- or F-contiguous");
    Eigen::MatrixXd value(n,n); const auto* ptr=static_cast<const double*>(input.data());
    if(input.flags() & py::array::f_style) value=Eigen::Map<const Eigen::MatrixXd>(ptr,n,n);
    else value=Eigen::Map<const Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>>(ptr,n,n);
    return value;
}
}
void bind_manta_bupo(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<BupoOptions>(m,"BupoOptions")
        .def(py::init<>())
        .def_readwrite("order",&BupoOptions::order)
        .def_readwrite("collect_timings",&BupoOptions::collect_timings)
        .def_readwrite("opening",&BupoOptions::opening)
        .def_readwrite("integral_tolerance",&BupoOptions::integral_tolerance)
        .def_readwrite("near_backend",&BupoOptions::near_backend)
        .def_readwrite("extents",&BupoOptions::extents)
        .def_readwrite("bubbles",&BupoOptions::bubbles);
    py::class_<BupoPlan>(m,"BupoPlan")
        .def_readonly("timings_collected",&BupoPlan::timings_collected)
        .def_readonly("setup_wall_seconds",&BupoPlan::setup_wall_seconds)
        .def_readonly("provider_setup_wall_seconds",&BupoPlan::provider_setup_wall_seconds)
        .def_readonly("extent_setup_wall_seconds",&BupoPlan::extent_setup_wall_seconds)
        .def_readonly("tree_setup_wall_seconds",&BupoPlan::tree_setup_wall_seconds)
        .def_readonly("error_model_setup_wall_seconds",&BupoPlan::error_model_setup_wall_seconds)
        .def_readonly("partition_setup_wall_seconds",&BupoPlan::partition_setup_wall_seconds)
        .def_readonly("n_basis",&BupoPlan::n_basis)
        .def_readonly("shell_pairs",&BupoPlan::shell_pairs)
        .def_readonly("nodes",&BupoPlan::nodes)
        .def_readonly("moments",&BupoPlan::moments)
        .def_readonly("near_shell_calls",&BupoPlan::near_shell_calls)
        .def_readonly("near_primitive_quartets",&BupoPlan::near_primitive_quartets)
        .def_readonly("far_blocks",&BupoPlan::far_blocks)
        .def_readonly("moment_panels",&BupoPlan::moment_panels)
        .def_readonly("moment_primitive_pairs",&BupoPlan::moment_primitive_pairs)
        .def_readonly("traversal_visits",&BupoPlan::traversal_visits)
        .def_readonly("shell_calls",&BupoPlan::shell_calls)
        .def_readonly("primitive_products",&BupoPlan::primitive_products)
        .def_readonly("scalar_terms",&BupoPlan::scalar_terms)
        .def_readonly("kernel_terms",&BupoPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&BupoPlan::modeled_peak_bytes)
        .def_readonly("maximum_integral_error",&BupoPlan::maximum_integral_error)
        .def_readonly("maximum_compact_error",&BupoPlan::maximum_compact_error)
        .def_readonly("maximum_penetration_error",&BupoPlan::maximum_penetration_error)
        .def_readonly("maximum_moment_tail_error",&BupoPlan::maximum_moment_tail_error)
        .def_readonly("partition_identity",&BupoPlan::partition_identity);
    py::class_<BupoReceipt>(m,"BupoReceipt")
        .def_readonly("timings_collected",&BupoReceipt::timings_collected)
        .def_readonly("validation_wall_seconds",&BupoReceipt::validation_wall_seconds)
        .def_readonly("leaf_moments_wall_seconds",&BupoReceipt::leaf_moments_wall_seconds)
        .def_readonly("upward_wall_seconds",&BupoReceipt::upward_wall_seconds)
        .def_readonly("far_admission_wall_seconds",&BupoReceipt::far_admission_wall_seconds)
        .def_readonly("far_apply_wall_seconds",&BupoReceipt::far_apply_wall_seconds)
        .def_readonly("near_integrals_wall_seconds",&BupoReceipt::near_integrals_wall_seconds)
        .def_readonly("near_digest_wall_seconds",&BupoReceipt::near_digest_wall_seconds)
        .def_readonly("downward_wall_seconds",&BupoReceipt::downward_wall_seconds)
        .def_readonly("reverse_moments_wall_seconds",&BupoReceipt::reverse_moments_wall_seconds)
        .def_readonly("finalize_wall_seconds",&BupoReceipt::finalize_wall_seconds)
        .def_readonly("source_identity",&BupoReceipt::source_identity)
        .def_readonly("density_identity",&BupoReceipt::density_identity)
        .def_readonly("partition_identity",&BupoReceipt::partition_identity)
        .def_readonly("near_provider",&BupoReceipt::near_provider)
        .def_readonly("moment_provider",&BupoReceipt::moment_provider)
        .def_readonly("algorithm",&BupoReceipt::algorithm)
        .def_readonly("normalization",&BupoReceipt::normalization)
        .def_readonly("near_shell_calls",&BupoReceipt::near_shell_calls)
        .def_readonly("near_primitive_quartets",&BupoReceipt::near_primitive_quartets)
        .def_readonly("near_null_tiles",&BupoReceipt::near_null_tiles)
        .def_readonly("far_blocks",&BupoReceipt::far_blocks)
        .def_readonly("moment_panels",&BupoReceipt::moment_panels)
        .def_readonly("moment_primitive_pairs",&BupoReceipt::moment_primitive_pairs)
        .def_readonly("traversal_visits",&BupoReceipt::traversal_visits)
        .def_readonly("modeled_peak_bytes",&BupoReceipt::modeled_peak_bytes)
        .def_readonly("near_primitive_evaluations_measured",&BupoReceipt::near_primitive_evaluations_measured)
        .def_readonly("density_l1",&BupoReceipt::density_l1)
        .def_readonly("coulomb_energy",&BupoReceipt::coulomb_energy)
        .def_readonly("approximation_j_bound",&BupoReceipt::approximation_j_bound)
        .def_readonly("compact_j_bound",&BupoReceipt::compact_j_bound)
        .def_readonly("penetration_j_bound",&BupoReceipt::penetration_j_bound)
        .def_readonly("moment_tail_j_bound",&BupoReceipt::moment_tail_j_bound)
        .def_readonly("wall_seconds",&BupoReceipt::wall_seconds);
    py::class_<BupoResult>(m,"BupoResult")
        .def_property_readonly("J",[](const BupoResult& r) { return Eigen::MatrixXd(r.J); })
        .def_readonly("receipt",&BupoResult::receipt);
    py::class_<BupoJBuilder,vibeqc::JKBuilder,std::shared_ptr<BupoJBuilder>>(m,"BupoJBuilder")
        .def(py::init<const vibeqc::BasisSet&,const BupoOptions&,const ResourceBudget&>(),py::arg("basis"),py::arg("options")=BupoOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",[](const BupoJBuilder& b) { return BupoPlan(b.plan()); })
        .def_property_readonly("source_identity",&BupoJBuilder::source_identity)
        .def("build",[](const BupoJBuilder& b,const py::array& input) { const auto d=density_copy(b,input); py::gil_scoped_release release; return b.build(d); },py::arg("density"))
        .def("build_J",[](const BupoJBuilder& b,const py::array& input) { const auto d=density_copy(b,input); py::gil_scoped_release release; return b.build_J(d); },py::arg("density"));
}
