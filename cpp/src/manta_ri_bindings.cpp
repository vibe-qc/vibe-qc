#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "vibeqc/manta_ri_metric.hpp"
#include "vibeqc/manta_admission.hpp"
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
    auto sp=py::class_<HermitianSpectrumPlan>(m,"HermitianSpectrumPlan");
#define SP(name) sp.def_readonly(#name,&HermitianSpectrumPlan::name)
    SP(dimension);SP(scalar_terms);SP(kernel_terms);SP(modeled_peak_bytes);
#undef SP
    auto sc=py::class_<HermitianSpectrumCertificate>(m,"HermitianSpectrumCertificate");
#define SC(name) sc.def_readonly(#name,&HermitianSpectrumCertificate::name)
    SC(spectrum_certified);SC(rank_certified);SC(rank);SC(residual_bound);SC(orthogonality_bound);
    SC(eigenvalue_error_bound);SC(cutoff_lower);SC(cutoff_upper);SC(rank_stability_radius);
#undef SC
    m.def("plan_hermitian_spectrum",&plan_hermitian_spectrum,py::arg("dimension"),
          py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{});
    m.def("certify_hermitian_spectrum",[](const py::array& metric,const py::array& values,
            const py::array& vectors,const RIOptions& options,const ResourceBudget& budget) {
        if(metric.ndim()!=2 || metric.shape(0)<=0 || metric.shape(1)!=metric.shape(0))
            throw py::value_error("MANTA spectral enclosure requires square metric");
        const auto n=metric.shape(0);
        for(const auto* a:{&metric,&vectors}) {
            if(!a->dtype().is(py::dtype::of<std::complex<double>>()) || a->ndim()!=2 || a->shape(0)!=n || a->shape(1)!=n ||
               !(a->flags() & (py::array::c_style | py::array::f_style)))
                throw py::value_error("MANTA spectral enclosure requires matching contiguous complex128 matrices");
        }
        if(!values.dtype().is(py::dtype::of<double>()) || values.ndim()!=1 || values.shape(0)!=n || !(values.flags() & py::array::c_style))
            throw py::value_error("MANTA spectral enclosure requires a contiguous float64 eigenvalue vector");
        plan_hermitian_spectrum(n,options,budget); // Before payload copies or inspection.
        auto copy=[n](const py::array& a) {
            Eigen::MatrixXcd result(n,n);const auto* data=static_cast<const char*>(a.data());
            for(py::ssize_t i=0;i<n;++i) for(py::ssize_t j=0;j<n;++j)
                std::memcpy(&result(i,j),data+i*a.strides(0)+j*a.strides(1),sizeof(std::complex<double>));
            return result;
        };
        Eigen::VectorXd eigenvalues(n);std::memcpy(eigenvalues.data(),values.data(),n*sizeof(double));
        const auto a=copy(metric),q=copy(vectors);
        return certify_hermitian_spectrum(a,eigenvalues,q,options,budget);
    },py::arg("metric").noconvert(),py::arg("eigenvalues").noconvert(),py::arg("eigenvectors").noconvert(),
      py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{});
    auto ic=py::class_<HermitianInverseCertificate>(m,"HermitianInverseCertificate");
#define IC(name) ic.def_readonly(#name,&HermitianInverseCertificate::name)
    IC(spectrum);IC(qualified);IC(retained_lower);IC(inverse_norm_bound);IC(divided_difference_bound);
    IC(residual_frobenius_bound);IC(residual_term);IC(orthogonality_term);IC(factor_product_bound);IC(inverse_error_bound);
#undef IC
    m.def("plan_hermitian_inverse",&plan_hermitian_inverse,py::arg("dimension"),
          py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{});
    m.def("certify_hermitian_inverse",[](const py::array& metric,const py::array& values,
            const py::array& vectors,const py::array& factor,const RIOptions& options,const ResourceBudget& budget) {
        if(metric.ndim()!=2 || metric.shape(0)<=0 || metric.shape(1)!=metric.shape(0))
            throw py::value_error("MANTA spectral enclosure requires square metric");
        const auto n=metric.shape(0);
        for(const auto* a:{&metric,&vectors}) {
            if(!a->dtype().is(py::dtype::of<std::complex<double>>()) || a->ndim()!=2 || a->shape(0)!=n || a->shape(1)!=n ||
               !(a->flags() & (py::array::c_style | py::array::f_style)))
                throw py::value_error("MANTA spectral enclosure requires matching contiguous complex128 matrices");
        }
        if(!values.dtype().is(py::dtype::of<double>()) || values.ndim()!=1 || values.shape(0)!=n || !(values.flags() & py::array::c_style))
            throw py::value_error("MANTA spectral enclosure requires a contiguous float64 eigenvalue vector");
        if(!factor.dtype().is(py::dtype::of<std::complex<double>>()) || factor.ndim()!=2 || factor.shape(0)!=n ||
           factor.shape(1)>n || !(factor.flags() & (py::array::c_style | py::array::f_style)))
            throw py::value_error("MANTA inverse enclosure requires a matching contiguous complex128 factor");
        plan_hermitian_inverse(n,options,budget); // Before payload copies or inspection.
        auto copy=[n](const py::array& a) {
            Eigen::MatrixXcd result(n,a.shape(1));const auto* data=static_cast<const char*>(a.data());
            for(py::ssize_t i=0;i<n;++i) for(py::ssize_t j=0;j<a.shape(1);++j)
                std::memcpy(&result(i,j),data+i*a.strides(0)+j*a.strides(1),sizeof(std::complex<double>));
            return result;
        };
        Eigen::VectorXd eigenvalues(n);std::memcpy(eigenvalues.data(),values.data(),n*sizeof(double));
        const auto a=copy(metric),q=copy(vectors);
        return certify_hermitian_inverse(a,eigenvalues,q,copy(factor),options,budget);
    },py::arg("metric").noconvert(),py::arg("eigenvalues").noconvert(),py::arg("eigenvectors").noconvert(),py::arg("factor").noconvert(),
      py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{});
    auto column=py::class_<HermitianInverseColumnBound>(m,"HermitianInverseColumnBound");
#define C(n) column.def_readonly(#n,&HermitianInverseColumnBound::n)
    C(qualified);C(coupling_norm);C(factor_score);C(residual_norm);C(discarded_norm);
#undef C
    auto family=py::class_<HermitianInverseFamilyBound>(m,"HermitianInverseFamilyBound");
#define F(n) family.def_readonly(#n,&HermitianInverseFamilyBound::n)
    F(qualified);F(j_error_bound);F(energy_error_bound);F(j_error_bounds);
#undef F
    m.def("plan_hermitian_inverse_action",&plan_hermitian_inverse_action,py::arg("dimension"),py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{});
    m.def("plan_hermitian_inverse_family",&plan_hermitian_inverse_family,py::arg("dimension"),py::arg("columns"),py::arg("budget")=ResourceBudget{});
    m.def("plan_hermitian_inverse_column",&plan_hermitian_inverse_column,py::arg("dimension"),py::arg("budget")=ResourceBudget{});
    auto action=py::class_<HermitianInverseAction,std::shared_ptr<HermitianInverseAction>>(m,"HermitianInverseAction");
    action.def(py::init([](const py::array& metric,const py::array& values,
            const py::array& vectors,const py::array& factor,const RIOptions& options,const ResourceBudget& budget) {
        if(metric.ndim()!=2 || metric.shape(0)<=0 || metric.shape(1)!=metric.shape(0))
            throw py::value_error("MANTA spectral enclosure requires square metric");
        const auto n=metric.shape(0);
        for(const auto* a:{&metric,&vectors}) {
            if(!a->dtype().is(py::dtype::of<std::complex<double>>()) || a->ndim()!=2 || a->shape(0)!=n || a->shape(1)!=n ||
               !(a->flags() & (py::array::c_style | py::array::f_style)))
                throw py::value_error("MANTA spectral enclosure requires matching contiguous complex128 matrices");
        }
        if(!values.dtype().is(py::dtype::of<double>()) || values.ndim()!=1 || values.shape(0)!=n || !(values.flags() & py::array::c_style))
            throw py::value_error("MANTA spectral enclosure requires a contiguous float64 eigenvalue vector");
        if(!factor.dtype().is(py::dtype::of<std::complex<double>>()) || factor.ndim()!=2 || factor.shape(0)!=n ||
           factor.shape(1)>n || !(factor.flags() & (py::array::c_style | py::array::f_style)))
            throw py::value_error("MANTA inverse enclosure requires a matching contiguous complex128 factor");
        plan_hermitian_inverse_action(n,options,budget); // Before payload copies or inspection.
        auto copy=[n](const py::array& a) {
            Eigen::MatrixXcd result(n,a.shape(1));const auto* data=static_cast<const char*>(a.data());
            for(py::ssize_t i=0;i<n;++i) for(py::ssize_t j=0;j<a.shape(1);++j)
                std::memcpy(&result(i,j),data+i*a.strides(0)+j*a.strides(1),sizeof(std::complex<double>));
            return result;
        };
        Eigen::VectorXd eigenvalues(n);std::memcpy(eigenvalues.data(),values.data(),n*sizeof(double));
        const auto a=copy(metric),q=copy(vectors);
        return std::make_shared<HermitianInverseAction>(a,eigenvalues,q,copy(factor),options,budget);
    }),py::arg("metric").noconvert(),py::arg("eigenvalues").noconvert(),py::arg("eigenvectors").noconvert(),py::arg("factor").noconvert(),
      py::arg("options")=RIOptions{},py::arg("budget")=ResourceBudget{});

    action.def_property_readonly("certificate",&HermitianInverseAction::certificate,py::return_value_policy::copy)
        .def("column",[](const HermitianInverseAction& a,const py::array& t,const ResourceBudget& b) {
            if(t.ndim()!=1 || t.shape(0)!=static_cast<py::ssize_t>(a.dimension()) || !t.dtype().is(py::dtype::of<std::complex<double>>()) || !(t.flags()&py::array::c_style))
                throw py::value_error("MANTA metric column requires contiguous matching complex128 vector");
            plan_hermitian_inverse_column(a.dimension(),b);
            Eigen::VectorXcd copy(t.size());std::memcpy(copy.data(),t.data(),t.size()*sizeof(std::complex<double>));return a.column(copy,b);
        },py::arg("coupling").noconvert(),py::arg("budget")=ResourceBudget{})
        .def("family",[](const HermitianInverseAction& a,const py::list& columns,double limit,const py::list& entries,const ResourceBudget& b) {
            using namespace admission;const Count count=py::len(columns);
            plan_hermitian_inverse_family(a.dimension(),count,b);
            if(py::len(entries) && py::len(entries)!=count)throw py::value_error("MANTA metric family entry shape mismatch");
            return a.family(columns.cast<std::vector<HermitianInverseColumnBound>>(),limit,entries.cast<std::vector<double>>(),b);
        },py::arg("columns"),py::arg("density_l1_limit"),py::arg("density_entry_limits")=py::list{},py::arg("budget")=ResourceBudget{});
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
    py::class_<RIFactorExportPlan>(m,"RIFactorExportPlan")
        .def_readonly("n_basis",&RIFactorExportPlan::n_basis)
        .def_readonly("retained_rank",&RIFactorExportPlan::retained_rank)
        .def_readonly("exported_values",&RIFactorExportPlan::exported_values)
        .def_readonly("shell_calls",&RIFactorExportPlan::shell_calls)
        .def_readonly("primitive_products",&RIFactorExportPlan::primitive_products)
        .def_readonly("integral_elements",&RIFactorExportPlan::integral_elements)
        .def_readonly("contraction_products",&RIFactorExportPlan::contraction_products)
        .def_readonly("accumulation_terms",&RIFactorExportPlan::accumulation_terms)
        .def_readonly("kernel_terms",&RIFactorExportPlan::kernel_terms)
        .def_readonly("modeled_peak_bytes",&RIFactorExportPlan::modeled_peak_bytes);
    py::class_<RIFactorExportReceipt>(m,"RIFactorExportReceipt")
        .def_readonly("provider",&RIFactorExportReceipt::provider)
        .def_readonly("provider_version",&RIFactorExportReceipt::provider_version)
        .def_readonly("algorithm",&RIFactorExportReceipt::algorithm)
        .def_readonly("normalization",&RIFactorExportReceipt::normalization)
        .def_readonly("kernel",&RIFactorExportReceipt::kernel)
        .def_readonly("omega",&RIFactorExportReceipt::omega)
        .def_readonly("source_identity",&RIFactorExportReceipt::source_identity)
        .def_readonly("coupling_identity",&RIFactorExportReceipt::coupling_identity)
        .def_readonly("metric_factor_identity",&RIFactorExportReceipt::metric_factor_identity)
        .def_readonly("export_identity",&RIFactorExportReceipt::export_identity)
        .def_readonly("n_basis",&RIFactorExportReceipt::n_basis)
        .def_readonly("retained_rank",&RIFactorExportReceipt::retained_rank)
        .def_readonly("shell_calls",&RIFactorExportReceipt::shell_calls)
        .def_readonly("primitive_products",&RIFactorExportReceipt::primitive_products)
        .def_readonly("integral_elements_consumed",&RIFactorExportReceipt::integral_elements_consumed)
        .def_readonly("contraction_products",&RIFactorExportReceipt::contraction_products)
        .def_readonly("accumulation_terms",&RIFactorExportReceipt::accumulation_terms)
        .def_readonly("factor_array_bytes",&RIFactorExportReceipt::factor_array_bytes)
        .def_readonly("raw_coupling_array_bytes",&RIFactorExportReceipt::raw_coupling_array_bytes)
        .def_readonly("modeled_peak_bytes",&RIFactorExportReceipt::modeled_peak_bytes);
    py::class_<RIFactorExport>(m,"RIFactorExport")
        .def_property_readonly("coefficients",&RIFactorExport::coefficients,py::return_value_policy::copy)
        .def_property_readonly("receipt",&RIFactorExport::receipt,py::return_value_policy::copy);
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
        .def("plan_factor_export",&RIJKBuilder::plan_factor_export,py::arg("budget"))
        .def("export_factors",&RIJKBuilder::export_factors,py::arg("budget"))
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
