#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/manta_admission.hpp"
#include "manta_assembly_interval.hpp"
#include "vibeqc/manta_periodic_ri_exchange_bounds.hpp"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstring>

namespace py=pybind11;
using namespace vibeqc::manta;
namespace {
using Complex=std::complex<double>;
template<class T> T item(const py::array& a,py::ssize_t offset) {
    T result;std::memcpy(&result,static_cast<const char*>(a.data())+offset,sizeof(T));return result;
}
bool contiguous(const py::array& a) {return a.flags()&(py::array::c_style|py::array::f_style);}
Count vectors(const py::array& a) {
    if(!a.dtype().is(py::dtype::of<double>()) || a.ndim()!=2 || a.shape(1)!=3 || !contiguous(a))
        throw py::value_error("MANTA RI Fock vectors require C- or F-contiguous float64 (n,3) arrays");
    return a.shape(0);
}
md::Matrix copy_vectors(const py::array& a) {
    md::Matrix result(a.shape(0),3);
    for(Eigen::Index i=0;i<result.rows();++i)for(int j=0;j<3;++j)result(i,j)=item<double>(a,i*a.strides(0)+j*a.strides(1));return result;
}
std::vector<Complex> matrix(const py::array& a,Count spins,Count nk,Count n,bool spin) {
    if(!a.dtype().is(py::dtype::of<Complex>()) || a.ndim()!=(spin?4:3) || !contiguous(a) ||
       (spin && a.shape(0)!=py::ssize_t(spins)) || a.shape(spin)!=py::ssize_t(nk) ||
       a.shape(spin+1)!=py::ssize_t(n) || a.shape(spin+2)!=py::ssize_t(n))
        throw py::value_error("MANTA RI Fock requires complex128 H[k,AO,AO] and D[2,k,AO,AO]");
    std::vector<Complex> result(a.size());Count out=0;
    for(Count s=0;s<(spin?spins:1);++s)for(Count k=0;k<nk;++k)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j)
        result[out++]=item<Complex>(a,(spin?s*a.strides(0):0)+k*a.strides(spin)+i*a.strides(spin+1)+j*a.strides(spin+2));
    return result;
}
py::array_t<Complex> owned(const std::vector<Complex>& values,std::vector<py::ssize_t> shape) {
    py::array_t<Complex> out(shape);if(!values.empty())std::memcpy(out.mutable_data(),values.data(),values.size()*sizeof(Complex));return out;
}
}
void bind_manta_periodic_ri_fock(py::module_& m) {
    auto assembly=py::class_<PeriodicAssemblyRoundoff>(m,"PeriodicAssemblyRoundoff");
#define A(n) assembly.def_readonly(#n,&PeriodicAssemblyRoundoff::n)
    A(exchange_correction_roundoff_included);A(exchange_correction_fock_bound);A(exchange_correction_energy_bound);
    A(direct_exchange_roundoff_included);A(direct_exchange_fock_bound);A(direct_exchange_energy_bound);
    A(fitted_exchange_roundoff_included);A(fitted_exchange_fock_bound);A(fitted_exchange_energy_bound);
    A(retained_coulomb_included);A(evaluated);A(paired_coulomb_included);A(fock_bound);A(energy_bound);A(combined_fock_bound);A(combined_energy_bound);
#undef A
    auto ap=py::class_<PeriodicAssemblyPlan>(m,"PeriodicAssemblyPlan");
#define A(n) ap.def_readonly(#n,&PeriodicAssemblyPlan::n)
    A(n_basis);A(momenta);A(spins);A(modeled_peak_bytes);A(scalar_terms);A(kernel_terms);
#undef A
    m.def("plan_periodic_assembly_roundoff",[](Count n,Count nk,Count ns,const ResourceBudget& b) {
        return assembly_interval::plan(n,nk,ns,b);
    },py::arg("n_basis"),py::arg("momenta"),py::arg("spins"),py::arg("budget")=ResourceBudget{});
    m.def("plan_periodic_correction_roundoff",&assembly_interval::correction_plan,
        py::arg("n_basis"),py::arg("momenta"),py::arg("spins"),py::arg("budget")=ResourceBudget{});
    m.def("certify_periodic_correction_roundoff",[](const py::array& s,const py::array& d,
        double madelung,double scale,const py::array& actual,const ResourceBudget& b) {
        if(s.ndim()!=3 || d.ndim()!=4)throw py::value_error("MANTA correction requires S[k,n,n] and D[spin,k,n,n]");
        const Count n=s.shape(1),nk=s.shape(0),ns=d.shape(0);
        assembly_interval::correction_plan(n,nk,ns,b);
        auto overlap=matrix(s,ns,nk,n,false),density=matrix(d,ns,nk,n,true),values=matrix(actual,ns,nk,n,true);
        py::array_t<double> out({py::ssize_t(ns),py::ssize_t(nk),py::ssize_t(n),py::ssize_t(n)});
        auto* errors=out.mutable_data();
        {py::gil_scoped_release release;
        for(Count spin=0;spin<ns;++spin)for(Count k=0;k<nk;++k) {
            const Count offset=k*n*n,block=(spin*nk+k)*n*n;
            assembly_interval::Correction audit(n,overlap.data()+offset,density.data()+block,madelung,scale);
            for(Count i=0;i<n;++i)for(Count j=0;j<n;++j)errors[block+i*n+j]=audit.error(i,j,values[block+i*n+j]);
        }}return out;
    },py::arg("overlap").noconvert(),py::arg("density").noconvert(),py::arg("madelung"),py::arg("exchange_scale"),
        py::arg("correction").noconvert(),py::arg("budget")=ResourceBudget{});
    m.def("certify_periodic_assembly_roundoff",[](const py::array& h,const py::array& d,const py::array& w,
        const py::array& j,const py::array& k,const py::array& xc,const py::array& correction,
        double scale,double ej,double exc,double enuc,const py::array& actual,double energy,const ResourceBudget& b,const py::object& input_errors,const py::object& correction_errors,bool fitted_exchange) {
        if(h.ndim()!=3 || d.ndim()!=4)throw py::value_error("MANTA assembly requires H[k,n,n] and D[spin,k,n,n]");
        const Count n=h.shape(1),nk=h.shape(0),ns=d.shape(0);
        assembly_interval::plan(n,nk,ns,b); // Preflight before copies or payload inspection.
        if(!w.dtype().is(py::dtype::of<double>()) || w.ndim()!=1 || w.shape(0)!=py::ssize_t(nk) || !contiguous(w))
            throw py::value_error("MANTA assembly requires contiguous float64 weights[k]");
        auto hh=matrix(h,ns,nk,n,false),dd=matrix(d,ns,nk,n,true),jj=matrix(j,ns,nk,n,false),
            kk=matrix(k,ns,nk,n,true),vv=matrix(xc,ns,nk,n,true),cc=matrix(correction,ns,nk,n,true),ff=matrix(actual,ns,nk,n,true);
        std::vector<double> weights(nk);for(Count ik=0;ik<nk;++ik)weights[ik]=item<double>(w,ik*w.strides(0));
        auto read_errors=[&](const py::object& input_errors) {
        std::vector<double> errors;
        if(!input_errors.is_none()) {
            if(!py::isinstance<py::array>(input_errors))throw py::value_error("MANTA assembly input errors require a float64 array");
            const auto a=py::cast<py::array>(input_errors);
            if(!a.dtype().is(py::dtype::of<double>()) || a.ndim()!=4 || !contiguous(a) ||
               a.shape(0)!=py::ssize_t(ns) || a.shape(1)!=py::ssize_t(nk) || a.shape(2)!=py::ssize_t(n) || a.shape(3)!=py::ssize_t(n))
                throw py::value_error("MANTA assembly input errors require float64 [spin,k,n,n]");
            errors.resize(dd.size());Count out=0;
            for(Count spin=0;spin<ns;++spin)for(Count ik=0;ik<nk;++ik)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j)
                errors[out++]=item<double>(a,spin*a.strides(0)+ik*a.strides(1)+i*a.strides(2)+j*a.strides(3));
        }
        return errors;};
        auto errors=read_errors(input_errors),correction_bounds=read_errors(correction_errors);
        py::gil_scoped_release release;
        assembly_interval::Audit audit(n,nk,ns,weights,hh,dd.data(),jj,vv.data(),ej,exc,enuc);
        audit.exchange(kk,scale,errors,fitted_exchange);for(Count q=0;q<cc.size();++q)
            audit.correction(q,cc[q],correction_bounds.empty()?0.:correction_bounds[q],!correction_bounds.empty());
        return audit.finish(ff.data(),ff.size(),energy);
    },py::arg("hcore").noconvert(),py::arg("density").noconvert(),py::arg("weights").noconvert(),
        py::arg("coulomb").noconvert(),py::arg("exchange").noconvert(),py::arg("xc").noconvert(),py::arg("correction").noconvert(),
        py::arg("exchange_scale"),py::arg("coulomb_energy"),py::arg("xc_energy"),py::arg("nuclear_energy"),
        py::arg("fock").noconvert(),py::arg("energy"),py::arg("budget")=ResourceBudget{},py::arg("exchange_errors")=py::none(),py::arg("correction_errors")=py::none(),py::arg("fitted_exchange")=false);
    auto p=py::class_<PeriodicRIFockPlan>(m,"PeriodicRIFockPlan");
#define P(name) p.def_readonly(#name,&PeriodicRIFockPlan::name)
    P(has_j);P(has_k);P(n_basis);P(n_auxiliary);P(momenta);P(channels);P(retained_source_bytes);P(driver_live_bytes);P(modeled_peak_bytes);
    P(metric_shell_calls);P(metric_primitive_products);P(build_shell_calls);P(build_primitive_products);
    P(shell_calls);P(primitive_products);P(scalar_terms);P(kernel_terms);P(gamma);P(pair);
#undef P
    py::class_<PeriodicRIFittingMetricTrial>(m,"PeriodicRIFittingMetricTrial")
        .def_readonly("cutoff",&PeriodicRIFittingMetricTrial::cutoff)
        .def_readonly("rank",&PeriodicRIFittingMetricTrial::rank)
        .def_readonly("bound",&PeriodicRIFittingMetricTrial::bound);
    auto domain=py::class_<PeriodicRIFittingDomainTrial>(m,"PeriodicRIFittingDomainTrial");
#define D(n) domain.def_readonly(#n,&PeriodicRIFittingDomainTrial::n)
    D(pair_cutoff_bohr);D(auxiliary_cutoff_bohr);D(metric_cutoff);D(reciprocal_cube_radius);D(vectors);D(rank);D(domain_attempt);D(metric_refinement);D(bound);
#undef D
    auto auxiliary=py::class_<PeriodicRIFittingAuxiliaryTrial>(m,"PeriodicRIFittingAuxiliaryTrial");
#define A(n) auxiliary.def_readonly(#n,&PeriodicRIFittingAuxiliaryTrial::n)
    A(candidate);A(functions);A(generation_identity);A(auxiliary_basis_identity);A(source);
#undef A
    py::class_<PeriodicRIFittingReferenceTrial>(m,"PeriodicRIFittingReferenceTrial")
        .def_readonly("preparation",&PeriodicRIFittingReferenceTrial::preparation)
        .def_readonly("bound",&PeriodicRIFittingReferenceTrial::bound)
        .def_readonly("refinement",&PeriodicRIFittingReferenceTrial::refinement)
        .def_readonly("ready",&PeriodicRIFittingReferenceTrial::ready);
    auto i=py::class_<PeriodicRIFockInfo>(m,"PeriodicRIFockInfo");
#define I(name) i.def_readonly(#name,&PeriodicRIFockInfo::name)
    I(accuracy);I(fitting_bound);I(fitting_trials);I(reference_trials);I(fitting_reference);I(reference_search);I(domain_trials);I(selected_fitting_plan);I(auxiliary_trials);I(selected_auxiliary_generation);I(exchange_kernel);I(physical_omega);I(has_j);I(has_k);I(source_identity);I(frame_identity);I(weight_identity);I(metric_shell_calls);I(metric_primitive_products);
    I(reverse_metric_residual);I(reverse_projector_residual);I(reverse_inverse_residual);I(metrics);
#undef I
    auto r=py::class_<PeriodicRIFockReceipt,PeriodicFockReceipt>(m,"PeriodicRIFockReceipt");
#define R(name) r.def_readonly(#name,&PeriodicRIFockReceipt::name)
    R(fitted_exchange_roundoff_evaluated);R(fitted_exchange_roundoff_bound);
    R(retained_j_roundoff_evaluated);R(retained_j_roundoff_bound);R(retained_energy_roundoff_bound);
    R(weighted_input_roundoff_evaluated);R(weighted_input_roundoff_bound);R(coupling_forward_input_error_included);R(paired_input_roundoff_included);
    R(forward_metric_roundoff_evaluated);R(forward_metric_coefficient_roundoff_bound);R(forward_metric_energy_roundoff_bound);R(paired_roundoff_evaluated);R(paired_j_roundoff_bound);R(paired_energy_roundoff_bound);
    R(coupling_roundoff_evaluated);R(coupling_forward_roundoff_bound);R(coupling_adjoint_roundoff_bound);R(metric_roundoff_evaluated);R(metric_j_roundoff_evaluated);R(metric_j_roundoff_bound);R(metric_projection_roundoff_bound);R(metric_coefficient_roundoff_bound);R(metric_energy_roundoff_bound);
    R(reference_coupling_identity);R(coupling_error_bound);R(approximation_j_bound);R(approximation_energy_bound);R(coupling_multipole_partition_builds);R(coupling_multipole_partition_reuses);R(coupling_multipole_partition_bypasses);R(coupling_multipole_trials);R(coupling_multipole_images);R(coupling_multipole_tree_nodes);R(coupling_multipole_far_blocks);R(coupling_multipole_translations);R(coupling_multipole_moment_panels);R(coupling_multipole_envelope_evaluations);R(coupling_multipole_reused_moment_panels);R(coupling_multipole_reused_extents);R(coupling_multipole_global_pairs);R(coupling_multipole_global_pair_reuses);R(coupling_multipole_global_auxiliary);R(coupling_multipole_global_projections);R(coupling_multipole_global_auxiliary_projections);R(coupling_multipole_pair_groups);R(coupling_multipole_compressed_pairs);R(coupling_multipole_compressed_nodes);R(coupling_multipole_action_products);R(coupling_multipole_action_translations);
    R(exchange_kernel);R(physical_omega);R(has_j);R(has_k);R(frame_identity);R(weight_identity);R(shell_calls);R(primitive_products);R(coupling_passes);
#undef R
    py::class_<PeriodicRIFockResult>(m,"PeriodicRIFockResult")
        .def_readonly("receipt",&PeriodicRIFockResult::receipt).def_readonly("components",&PeriodicRIFockResult::components)
        .def_property_readonly("J",[](const PeriodicRIFockResult& r){return owned(r.J,{py::ssize_t(r.receipt.has_j?r.momenta:0),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});})
        .def_property_readonly("K_roundoff_bounds",[](const PeriodicRIFockResult& r) {
            py::array_t<double> out({py::ssize_t(r.receipt.fitted_exchange_roundoff_evaluated?2:0),py::ssize_t(r.momenta),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});
            std::copy(r.K_roundoff_bounds.begin(),r.K_roundoff_bounds.end(),out.mutable_data());return out;})
        .def_property_readonly("K",[](const PeriodicRIFockResult& r){return owned(r.K,{py::ssize_t(r.receipt.has_k?2:0),py::ssize_t(r.momenta),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});})
        .def_property_readonly("values",[](const PeriodicRIFockResult& r){return owned(r.values,{2,py::ssize_t(r.momenta),py::ssize_t(r.n_basis),py::ssize_t(r.n_basis)});});
    m.def("plan_periodic_ri_fock",&plan_periodic_ri_fock,py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),
        py::arg("gamma_vectors"),py::arg("maximum_pair_vectors"),py::arg("momenta"),
        py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("ri_options")=RIOptions{},py::arg("budget")=ResourceBudget{},
        py::arg("make_j")=true,py::arg("make_k")=true,py::arg("exchange_kernel")=PeriodicGDFKernelOptions{},py::arg("accuracy")=PeriodicRIAccuracyOptions{});
    py::class_<PeriodicRIFockBuilder>(m,"PeriodicRIFockBuilder")
        .def(py::init([](const vibeqc::BasisSet& orbital,const vibeqc::BasisSet& auxiliary,const vibeqc::PeriodicSystem& system,
            const py::array& k,const py::array& w,const py::array& g,const py::array& q,const py::array& boundaries,
            const PeriodicGDFShortRangeOptions& o,const RIOptions& ri,const ResourceBudget& budget,bool make_j,bool make_k,const PeriodicGDFKernelOptions& physical,const PeriodicRIAccuracyOptions& accuracy) {
            using namespace admission;const Count nk=vectors(k),ng=vectors(g),nq=vectors(q),pairs=square(nk);
            if(!w.dtype().is(py::dtype::of<double>()) || w.ndim()!=1 || w.shape(0)!=py::ssize_t(nk) || !contiguous(w))
                throw py::value_error("MANTA RI Fock weights require contiguous float64 (nk,)");
            if(!boundaries.dtype().is(py::dtype::of<std::int64_t>()) || boundaries.ndim()!=1 ||
               boundaries.shape(0)!=py::ssize_t(add(pairs,1)) || !contiguous(boundaries))
                throw py::value_error("MANTA RI Fock offsets require contiguous int64 (nk*nk+1,)");
            plan_periodic_ri_fock(orbital,auxiliary,system,ng,make_k && nk>1?1:0,nk,o,ri,budget,make_j,make_k,physical,accuracy);
            Count maximum=0,previous=0;
            if(item<std::int64_t>(boundaries,0)!=0)throw py::value_error("MANTA RI Fock offsets must start at zero");
            for(Count p=0;p<pairs;++p) {
                const auto next=item<std::int64_t>(boundaries,(p+1)*boundaries.strides(0));
                if(next<0 || static_cast<Count>(next)<previous || static_cast<Count>(next)>nq)
                    throw py::value_error("MANTA RI Fock offsets must be monotone and bounded");
                const Count length=next-previous;
                if(!make_k && length)throw py::value_error("MANTA RI Fock J-only requires an empty pair domain");
                if(make_k && ((p/nk==p%nk && length) || (p/nk!=p%nk && !length)))
                    throw py::value_error("MANTA RI Fock K requires empty diagonal and nonempty off-diagonal segments");
                maximum=std::max(maximum,length);previous=next;
            }
            if(previous!=nq)throw py::value_error("MANTA RI Fock offsets must end at the vector count");
            plan_periodic_ri_fock(orbital,auxiliary,system,ng,maximum,nk,o,ri,budget,make_j,make_k,physical,accuracy);
            auto momenta=copy_vectors(k),gamma=copy_vectors(g),pair=copy_vectors(q);
            std::vector<double> weights(nk);std::memcpy(weights.data(),w.data(),nk*sizeof(double));
            std::vector<Count> offsets(pairs+1);for(Count p=0;p<=pairs;++p)offsets[p]=item<std::int64_t>(boundaries,p*boundaries.strides(0));
            return PeriodicRIFockBuilder(orbital,auxiliary,system,momenta,weights,gamma,pair,offsets,o,ri,budget,make_j,make_k,physical,accuracy);
        }),py::arg("orbital"),py::arg("auxiliary"),py::arg("system"),py::arg("momenta").noconvert(),py::arg("weights").noconvert(),
            py::arg("gamma_vectors").noconvert(),py::arg("pair_vectors").noconvert(),py::arg("transfer_offsets").noconvert(),
            py::arg("options")=PeriodicGDFShortRangeOptions{},py::arg("ri_options")=RIOptions{},py::arg("budget")=ResourceBudget{},
            py::arg("make_j")=true,py::arg("make_k")=true,py::arg("exchange_kernel")=PeriodicGDFKernelOptions{},py::arg("accuracy")=PeriodicRIAccuracyOptions{})
        .def_property_readonly("plan",&PeriodicRIFockBuilder::plan,py::return_value_policy::copy)
        .def_property_readonly("info",&PeriodicRIFockBuilder::info,py::return_value_policy::copy)
        .def("build_exchange",[](const PeriodicRIFockBuilder& b,const py::array& d,bool audit) {
            const auto& p=b.plan();auto density=matrix(d,2,p.momenta,p.n_basis,true);
            py::gil_scoped_release release;return b.build_exchange(density,audit);
        },py::arg("spin_densities").noconvert(),py::arg("audit_arithmetic")=false)
        .def("build",[](const PeriodicRIFockBuilder& b,const py::array& h,const py::array& d,double scale,bool audit) {
            const auto& p=b.plan();auto hcore=matrix(h,1,p.momenta,p.n_basis,false),density=matrix(d,2,p.momenta,p.n_basis,true);
            py::gil_scoped_release release;return b.build(hcore,density,scale,audit);
        },py::arg("hcore").noconvert(),py::arg("spin_densities").noconvert(),py::arg("exchange_scale")=1.,py::arg("audit_arithmetic")=false);
    auto eao=py::class_<PeriodicRIExchangeAccuracyOptions>(m,"PeriodicRIExchangeAccuracyOptions").def(py::init<>());
#define EA(n) eao.def_readwrite(#n,&PeriodicRIExchangeAccuracyOptions::n)
    EA(matrix_error);EA(energy_error);EA(metric_assembly_matrix_error);EA(metric_assembly_energy_error);EA(reference_ao_cube_radius);EA(reference_reciprocal_cube_radius);
    EA(ao_completion_cube_radius);EA(maximum_probes);EA(reciprocal_batch);EA(fourier_decay_bins);
#undef EA
    m.def("plan_periodic_ri_exchange_accuracy",&plan_periodic_ri_exchange_accuracy,py::arg("basis"),py::arg("ri_plan"),
        py::arg("options"),py::arg("retained_fitting_bytes")=0,py::arg("budget")=ResourceBudget{},py::arg("physical_kernel")=PeriodicGDFKernelOptions{});
    auto eo=py::class_<PeriodicRIExchangeFittingOptions>(m,"PeriodicRIExchangeFittingOptions");eo.def(py::init<>());
#define EO(n) eo.def_readwrite(#n,&PeriodicRIExchangeFittingOptions::n)
    EO(metric_assembly_matrix_error);EO(metric_assembly_energy_error);EO(density_l1_limit);EO(density_entry_limits);EO(matrix_error);EO(energy_error);EO(maximum_probes);EO(reciprocal_batch);EO(fourier_decay_bins);EO(ao_completion_cube_radius);EO(reciprocal_completion_cube_radius);
#undef EO
    auto ep=py::class_<PeriodicRIExchangeFittingPlan>(m,"PeriodicRIExchangeFittingPlan");
#define EP(n) ep.def_readonly(#n,&PeriodicRIExchangeFittingPlan::n)
    EP(n_basis);EP(momenta);EP(maximum_probes);EP(shell_calls);EP(primitive_products);EP(scalar_terms);EP(kernel_terms);EP(modeled_peak_bytes);
#undef EP
    auto er=py::class_<PeriodicRIExchangeFittingResult>(m,"PeriodicRIExchangeFittingResult");
#define ER(n) er.def_readonly(#n,&PeriodicRIExchangeFittingResult::n)
    ER(plan);ER(canonical_density_envelope_identity);ER(qualification);ER(ri_source_identity);ER(reference_source_identity);ER(finite_reference_source_identity);ER(pair_source_identity);ER(certificate_identity);
    ER(probes);ER(ri_shell_calls);ER(reference_shell_calls);ER(kernel);ER(physical_omega);ER(probe_roundoff_included);ER(source_construction_included);ER(targets_requested);ER(qualified);
    ER(metric_assembly_requested);ER(metric_assembly_qualified);ER(metric_assembly_matrix_bound);ER(metric_assembly_energy_bound);ER(metric_assembly_identity);ER(metric_assembly_error_bounds);ER(metric_assembly_channels);
    ER(density_l1_limit);ER(maximum_error);ER(energy_error_bound);ER(finite_difference_maximum);ER(reference_remainder_maximum);ER(finite_difference_energy_bound);ER(reference_remainder_energy_bound);
#undef ER
    auto ea=[](const PeriodicRIExchangeFittingResult& r,const std::vector<double>& v) {
        py::array_t<double> out({py::ssize_t(r.plan.momenta),py::ssize_t(r.plan.n_basis),py::ssize_t(r.plan.n_basis)});
        std::copy(v.begin(),v.end(),out.mutable_data());return out;
    };
    er.def_property_readonly("error_bounds",[ea](const PeriodicRIExchangeFittingResult& r){return ea(r,r.error_bounds);})
      .def_property_readonly("finite_difference_bounds",[ea](const PeriodicRIExchangeFittingResult& r){return ea(r,r.finite_difference_bounds);})
      .def_property_readonly("reference_remainder_bounds",[ea](const PeriodicRIExchangeFittingResult& r){return ea(r,r.reference_remainder_bounds);})
      .def_property_readonly("density_entry_limits",[](const PeriodicRIExchangeFittingResult& r){return r.density_entry_limits;})
      .def("validate_density",[](const PeriodicRIExchangeFittingResult& r,const py::array& d) {
          if(d.ndim()!=4 || (d.shape(0)!=1 && d.shape(0)!=2))throw py::value_error("MANTA exchange fitting requires one or two spin densities");
          const Count spins=d.shape(0);auto density=matrix(d,spins,r.plan.momenta,r.plan.n_basis,true);
          py::gil_scoped_release release;r.validate_density(density,spins);
      },py::arg("spin_densities").noconvert());
    m.def("plan_periodic_ri_exchange_fitting_capacity",&plan_periodic_ri_exchange_fitting_capacity,
        py::arg("basis"),py::arg("ri_plan"),py::arg("reference_cells"),py::arg("reference_vectors"),
        py::arg("reference_options"),py::arg("options"),py::arg("retained_fitting_bytes")=0,py::arg("budget")=ResourceBudget{});
    m.def("plan_periodic_ri_exchange_fitting",&plan_periodic_ri_exchange_fitting,py::arg("ri"),py::arg("reference_pairs"),py::arg("reference_vectors"),py::arg("options"),py::arg("budget")=ResourceBudget{});
    m.def("certify_periodic_ri_exchange_fitting",[](const PeriodicRIFockBuilder& ri,const PeriodicPairSource& pairs,
        const py::array& q,const py::array& offsets,const PeriodicRIExchangeFittingOptions& o,const ResourceBudget& b) {
        const Count count=vectors(q);plan_periodic_ri_exchange_fitting(ri,pairs,count,o,b);
        if(!offsets.dtype().is(py::dtype::of<std::int64_t>()) || offsets.ndim()!=1 || !contiguous(offsets) ||
           offsets.size()!=py::ssize_t(admission::add(admission::square(ri.plan().momenta),1)))throw py::value_error("MANTA exchange fitting requires complete int64 offsets");
        std::vector<Count> values(offsets.size());
        for(Count i=0;i<values.size();++i){auto x=item<std::int64_t>(offsets,i*offsets.strides(0));if(x<0)throw py::value_error("MANTA exchange fitting offsets must be nonnegative");values[i]=x;}
        auto wave=copy_vectors(q);py::gil_scoped_release release;return certify_periodic_ri_exchange_fitting(ri,pairs,wave,values,o,b);
    },py::arg("ri"),py::arg("reference_pairs"),py::arg("reference_vectors").noconvert(),py::arg("offsets").noconvert(),py::arg("options"),py::arg("budget")=ResourceBudget{});

}
