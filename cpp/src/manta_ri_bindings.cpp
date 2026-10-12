#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vibeqc/manta_ri_metric.hpp"
#include <cstring>

namespace py=pybind11;
namespace {
Eigen::MatrixXd density(const vibeqc::manta::RIJKBuilder& builder,const py::array& array) {
    const auto n=static_cast<py::ssize_t>(builder.plan().n_basis);
    if(!array.dtype().is(py::dtype::of<double>()) || array.ndim()!=2 || array.shape(0)!=n || array.shape(1)!=n)
        throw py::value_error("MANTA RI density must be a square float64 ndarray matching the basis");
    if(!(array.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA RI density must be C- or F-contiguous");
    Eigen::MatrixXd result(n,n);
    const auto* data=static_cast<const char*>(array.data());
    if(array.flags() & py::array::f_style) std::memcpy(result.data(),data,array.size()*sizeof(double));
    else for(py::ssize_t i=0;i<n;++i) for(py::ssize_t j=0;j<n;++j)
        std::memcpy(&result(i,j),data+(i*n+j)*sizeof(double),sizeof(double));
    return result;
}
py::ssize_t metric_dimension(const py::array& array) {
    if(!array.dtype().is(py::dtype::of<double>()) || array.ndim()!=2 || array.shape(0)<=0 || array.shape(1)!=array.shape(0))
        throw py::value_error("MANTA RI metric response requires a nonempty square float64 ndarray");
    if(!(array.flags() & (py::array::c_style | py::array::f_style)))
        throw py::value_error("MANTA RI metric response requires C- or F-contiguous arrays");
    return array.shape(0);
}
Eigen::MatrixXd metric_matrix(const py::array& array,py::ssize_t n) {
    if(metric_dimension(array)!=n) throw py::value_error("MANTA RI metric response matrix dimensions");
    Eigen::MatrixXd result(n,n);
    const auto* data=static_cast<const char*>(array.data());
    if(array.flags() & py::array::f_style) std::memcpy(result.data(),data,array.size()*sizeof(double));
    else for(py::ssize_t i=0;i<n;++i) for(py::ssize_t j=0;j<n;++j)
        std::memcpy(&result(i,j),data+(i*n+j)*sizeof(double),sizeof(double));
    return result;
}
} // namespace

void bind_manta_ri(py::module_& m) {
    using namespace vibeqc::manta;
    py::class_<RIOptions>(m,"RIOptions")
        .def(py::init<>())
        .def_readwrite("absolute_cutoff",&RIOptions::absolute_cutoff)
        .def_readwrite("relative_cutoff",&RIOptions::relative_cutoff)
        .def_readwrite("negative_relative_tolerance",&RIOptions::negative_relative_tolerance)
        .def_readwrite("mode_block",&RIOptions::mode_block);
    py::class_<RIMetricResponsePlan>(m,"RIMetricResponsePlan")
        .def_readonly("dimension",&RIMetricResponsePlan::dimension)
        .def_readonly("scalar_terms",&RIMetricResponsePlan::scalar_terms)
        .def_readonly("kernel_terms",&RIMetricResponsePlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&RIMetricResponsePlan::modeled_peak_bytes);
    py::class_<RIMetricResponseInfo>(m,"RIMetricResponseInfo")
        .def_readonly("rank",&RIMetricResponseInfo::rank)
        .def_readonly("cutoff",&RIMetricResponseInfo::cutoff)
        .def_readonly("cutoff_distance",&RIMetricResponseInfo::cutoff_distance)
        .def_readonly("spectral_guard",&RIMetricResponseInfo::spectral_guard)
        .def_readonly("rank_stability_radius",&RIMetricResponseInfo::rank_stability_radius)
        .def_readonly("retained_discarded_gap",&RIMetricResponseInfo::retained_discarded_gap)
        .def_readonly("has_split",&RIMetricResponseInfo::has_split)
        .def_readonly("source_identity",&RIMetricResponseInfo::source_identity)
        .def_readonly("factor_identity",&RIMetricResponseInfo::factor_identity);
    py::class_<RIMetricDerivative>(m,"RIMetricDerivative")
        .def_readonly("inverse",&RIMetricDerivative::inverse)
        .def_readonly("projector",&RIMetricDerivative::projector)
        .def_readonly("source_identity",&RIMetricDerivative::source_identity)
        .def_readonly("direction_identity",&RIMetricDerivative::direction_identity)
        .def_readonly("modeled_peak_bytes",&RIMetricDerivative::modeled_peak_bytes);
    m.def("plan_ri_metric_response",&plan_ri_metric_response,py::arg("dimension"),
          py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{});
    py::class_<RIMetricResponse>(m,"RIMetricResponse")
        .def(py::init([](const py::array& metric,const RIOptions& options,const ResourceBudget& budget) {
            const auto n=metric_dimension(metric);
            plan_ri_metric_response(n,options,budget); // Before payload copy/eigensolve.
            return RIMetricResponse(metric_matrix(metric,n),options,budget);
        }),py::arg("metric").noconvert(),py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",&RIMetricResponse::plan,py::return_value_policy::copy)
        .def_property_readonly("info",&RIMetricResponse::info,py::return_value_policy::copy)
        .def_property_readonly("inverse",&RIMetricResponse::inverse,py::return_value_policy::copy)
        .def_property_readonly("projector",&RIMetricResponse::projector,py::return_value_policy::copy)
        .def_property_readonly("factor",&RIMetricResponse::factor,py::return_value_policy::copy)
        .def("apply",[](const RIMetricResponse& r,const py::array& direction) {
            return r.apply(metric_matrix(direction,r.plan().dimension));
        },py::arg("direction").noconvert())
        .def("adjoint",[](const RIMetricResponse& r,const py::array& inverse_seed,const py::array& projector_seed) {
            const auto a=metric_matrix(inverse_seed,r.plan().dimension),b=metric_matrix(projector_seed,r.plan().dimension);
            return r.adjoint(a,b);
        },py::arg("inverse_seed").noconvert(),py::arg("projector_seed").noconvert());
    py::class_<RIPlan>(m,"RIPlan")
        .def_readonly("n_basis",&RIPlan::n_basis)
        .def_readonly("n_auxiliary",&RIPlan::n_auxiliary)
        .def_readonly("maximum_rank",&RIPlan::maximum_rank)
        .def_readonly("mode_block",&RIPlan::mode_block)
        .def_readonly("metric_shell_calls",&RIPlan::metric_shell_calls)
        .def_readonly("coupling_shell_calls_per_pass",&RIPlan::coupling_shell_calls_per_pass)
        .def_readonly("maximum_coupling_passes",&RIPlan::maximum_coupling_passes)
        .def_readonly("shell_calls",&RIPlan::shell_calls)
        .def_readonly("primitive_products",&RIPlan::primitive_products)
        .def_readonly("scalar_terms",&RIPlan::scalar_terms)
        .def_readonly("kernel_terms",&RIPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&RIPlan::modeled_peak_bytes);
    py::class_<RIMetricInfo>(m,"RIMetricInfo")
        .def_readonly("rank",&RIMetricInfo::rank)
        .def_readonly("shell_calls",&RIMetricInfo::shell_calls)
        .def_readonly("primitive_products",&RIMetricInfo::primitive_products)
        .def_readonly("cutoff",&RIMetricInfo::cutoff)
        .def_readonly("minimum_eigenvalue",&RIMetricInfo::minimum_eigenvalue)
        .def_readonly("maximum_eigenvalue",&RIMetricInfo::maximum_eigenvalue)
        .def_readonly("factor_identity",&RIMetricInfo::factor_identity);
    py::class_<RIReceipt>(m,"RIReceipt")
        .def_readonly("provider",&RIReceipt::provider)
        .def_readonly("provider_version",&RIReceipt::provider_version)
        .def_readonly("algorithm",&RIReceipt::algorithm)
        .def_readonly("normalization",&RIReceipt::normalization)
        .def_readonly("kernel",&RIReceipt::kernel)
        .def_readonly("omega",&RIReceipt::omega)
        .def_readonly("source_identity",&RIReceipt::source_identity)
        .def_readonly("density_identity",&RIReceipt::density_identity)
        .def_readonly("coulomb_energy",&RIReceipt::coulomb_energy)
        .def_readonly("metric_rhs_norm",&RIReceipt::metric_rhs_norm)
        .def_readonly("metric_residual_norm",&RIReceipt::metric_residual_norm)
        .def_readonly("metric_residual_evaluated",&RIReceipt::metric_residual_evaluated)
        .def_readonly("retained_rank",&RIReceipt::retained_rank)
        .def_readonly("coupling_passes",&RIReceipt::coupling_passes)
        .def_readonly("shell_calls",&RIReceipt::shell_calls)
        .def_readonly("primitive_products",&RIReceipt::primitive_products)
        .def_readonly("integral_elements_consumed",&RIReceipt::integral_elements_consumed)
        .def_readonly("modeled_peak_bytes",&RIReceipt::modeled_peak_bytes);
    py::class_<RIResult>(m,"RIResult")
        .def_readonly("J",&RIResult::J)
        .def_readonly("K",&RIResult::K)
        .def_readonly("receipt",&RIResult::receipt);
    py::class_<RIExchangePair>(m,"RIExchangePair")
        .def_readonly("K_alpha",&RIExchangePair::K_alpha)
        .def_readonly("K_beta",&RIExchangePair::K_beta)
        .def_readonly("density_identities",&RIExchangePair::density_identities)
        .def_readonly("receipt",&RIExchangePair::receipt);
    m.def("plan_ri",&plan_ri,py::arg("orbital"),py::arg("auxiliary"),py::arg("options")=RIOptions{},
          py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{});
    py::class_<RIJKBuilder,vibeqc::JKBuilder,std::shared_ptr<RIJKBuilder>>(m,"RIJKBuilder")
        .def(py::init<const vibeqc::BasisSet&,const vibeqc::BasisSet&,const RIOptions&,const KernelRequest&,const ResourceBudget&>(),
             py::arg("orbital"),py::arg("auxiliary"),py::arg("options")=RIOptions{},
             py::arg("request")=KernelRequest{},py::arg("budget")=ResourceBudget{})
        .def_property_readonly("plan",&RIJKBuilder::plan,py::return_value_policy::copy)
        .def_property_readonly("metric_info",&RIJKBuilder::metric_info,py::return_value_policy::copy)
        .def_property_readonly("source_identity",&RIJKBuilder::source_identity)
        .def_property_readonly("metric",&RIJKBuilder::metric,py::return_value_policy::copy)
        .def_property_readonly("factor",&RIJKBuilder::factor,py::return_value_policy::copy)
        .def("build",[](const RIJKBuilder& b,const py::array& d){return b.build(density(b,d));},py::arg("density").noconvert())
        .def("build_J",[](const RIJKBuilder& b,const py::array& d){return b.build_J(density(b,d));},py::arg("density").noconvert())
        .def("build_K",[](const RIJKBuilder& b,const py::array& d){return b.build_K(density(b,d));},py::arg("density").noconvert())
        .def("build_J_slot",[](const RIJKBuilder& b,const py::array& d,int slot){return b.build_J_slot(density(b,d),slot);},
             py::arg("density").noconvert(),py::arg("slot"))
        .def("build_K_slot",[](const RIJKBuilder& b,const py::array& d,int slot){return b.build_K_slot(density(b,d),slot);},
             py::arg("density").noconvert(),py::arg("slot"))
        .def("build_K_pair",[](const RIJKBuilder& b,const py::array& a,const py::array& beta){
            const auto da=density(b,a),db=density(b,beta);
            return b.build_K_pair(da,db);
        },py::arg("density_alpha").noconvert(),py::arg("density_beta").noconvert())
        .def("build_exchange_pair",[](const RIJKBuilder& b,const py::array& a,const py::array& beta){
            const auto da=density(b,a),db=density(b,beta);
            return b.build_exchange_pair(da,db);
        },py::arg("density_alpha").noconvert(),py::arg("density_beta").noconvert())
        .def("build_g_rhf",[](const RIJKBuilder& b,const py::array& d,double alpha){return b.build_g_rhf(density(b,d),alpha);},
             py::arg("density").noconvert(),py::arg("alpha_hf")=1.);
}
