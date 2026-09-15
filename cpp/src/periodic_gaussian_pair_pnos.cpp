#include "vibeqc/periodic_gaussian_pair_pnos.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

namespace vibeqc {
namespace {
namespace real=periodic_correlation_real_local_detail;
namespace local=periodic_correlation_local_detail;
using I=std::uint64_t;
using real::add;
using real::mul;
using real::Digest;
using Options=PeriodicGaussianPairPNOOptions;
using Live=PeriodicGaussianPairPNOLiveInventory;
using Caps=PeriodicGaussianPairPNOCaps;
using Plan=PeriodicGaussianPairPNOPlan;
using Result=PeriodicGaussianPairPNOResult;
constexpr char policy[]="Nejad2025-Eq39;D=4SS^T+12AA^T;all-pairs;no-delta-denominator;initial-SC-only;original-Fvv-recanonicalization;no-energy-multiplicity";
static_assert(sizeof(double)==8 && FLT_EVAL_METHOD==0,"Gaussian pair PNOs require binary64 evaluation");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian pair PNOs do not support fast-math"
#endif
void limit(I value,I cap,const char* message) { if(!cap || value>cap) throw std::length_error(message); }
void jacobi(const HermitianJacobiOptions& o) {
    if(!o.max_sweeps || !std::isfinite(o.relative_offdiagonal_tolerance)
        || o.relative_offdiagonal_tolerance<=0 || o.relative_offdiagonal_tolerance>=1)
        throw std::invalid_argument("Gaussian pair PNO eigensolver controls must be explicit and valid");
}
void options_valid(const Options& o) {
    if(!std::isfinite(o.occupation_cutoff) || o.occupation_cutoff<0
        || !std::isfinite(o.denominator_floor) || o.denominator_floor<=0
        || !std::isfinite(o.maximum_initial_fvv_offdiagonal_norm) || o.maximum_initial_fvv_offdiagonal_norm<0
        || !std::isfinite(o.semicanonical_orthonormality_tolerance)
        || o.semicanonical_orthonormality_tolerance<=0 || o.semicanonical_orthonormality_tolerance>=1)
        throw std::invalid_argument("Gaussian pair PNO scientific controls must be explicit and finite");
    jacobi(o.pno_eigensolver); jacobi(o.semicanonical_eigensolver);
}
void objects(const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
             const PeriodicGaussianRealLocalProvider& wrapper) {
    local::validate_reference(ref);
    const auto& provider=wrapper.provider(); const auto context=wrapper.context_handle();
    if(!context || !wrapper.matched_finite_gaussian_hf_recipe() || !provider.matched_finite_gaussian_hf_recipe()
        || basis.state_handle().get()!=ref.state_handle().get() || provider.state_handle().get()!=ref.state_handle().get()
        || basis.allocation_identity()!=ref.dimensions().allocation_identity
        || provider.identity_sha256()!=wrapper.identity_sha256()
        || provider.source_context_identity_sha256()!=context->source_context_identity_sha256()
        || wrapper.source_context_identity_sha256()!=context->source_context_identity_sha256()
        || provider.hf_reference_source_identity_sha256()!=wrapper.hf_reference_source_identity_sha256()
        || provider.local_basis_identity_sha256()!=basis.local_basis_identity_sha256()
        || provider.basis_certificate_identity_sha256()!=basis.identity_sha256())
        throw std::invalid_argument("Gaussian pair PNO actual source/basis/state owners or receipts disagree");
    const auto callback=provider.integral_provider(basis);
    const auto& b=basis.memory(); const auto& p=provider.memory();
    if(!b.occupied_count || !b.virtual_count || b.orbital_count!=add(b.occupied_count,b.virtual_count)
        || b.occupied_count!=p.occupied_count || b.virtual_count!=p.virtual_count
        || b.orbital_count!=p.orbital_count || p.n_cells!=ref.state().n_kpoints()
        || p.n_auxiliary!=ref.dimensions().n_auxiliary || context->mesh().mesh()!=ref.state().mesh()
        || context->mesh().is_shift()!=ref.state().is_shift()
        || context->inventory().ao.function_count!=ref.state().n_basis()
        || context->inventory().auxiliary.function_count!=p.n_auxiliary
        || wrapper.memory().retained_row_bytes!=p.retained_row_bytes
        || wrapper.memory().occupied_count!=p.occupied_count || wrapper.memory().virtual_count!=p.virtual_count
        || callback.retained_numerical_bytes!=add(p.retained_row_bytes,mul(16,b.occupied_count))
        || callback.maximum_transient_numerical_bytes || p.scalar_work_units!=wrapper.memory().scalar_work_units)
        throw std::invalid_argument("Gaussian pair PNO actual provider shape or inventory differs");
}
void options_wire(Digest& h,const Options& o) {
    for(double value:{o.occupation_cutoff,o.denominator_floor,o.maximum_initial_fvv_offdiagonal_norm,
        o.semicanonical_orthonormality_tolerance}) h.real(value);
    for(const auto& e:{o.pno_eigensolver,o.semicanonical_eigensolver}) {
        h.u64(e.max_sweeps); h.real(e.relative_offdiagonal_tolerance);
    }
}
void plan_wire(Digest& h,const Plan& p) {
    for(auto value:{p.occupied_count,p.virtual_count,p.occupied_slot_i,p.occupied_slot_j,p.integral_calls,
        p.provider_work_units,p.numerical_work_units,p.work_units,p.integral_amplitude_phase_bytes,
        p.density_phase_bytes,p.semicanonical_phase_upper_bytes,p.peak_owned_numerical_bytes,
        p.retained_output_upper_bytes,p.borrowed_basis_bytes,p.borrowed_provider_row_bytes,p.state_resident_bytes,
        p.fixed_control_storage_bytes,p.replicas_per_node,p.reference_base_node_bytes,
        p.per_worker_inventoried_bytes,p.required_node_memory_bytes}) h.u64(value);
    options_wire(h,p.options);
    h.u64(p.live.other_live_bytes_per_worker); h.u64(p.live.fixed_backend_margin_bytes_per_worker);
    for(auto value:{p.caps.maximum_owned_numerical_bytes,p.caps.maximum_per_worker_inventoried_bytes,
        p.caps.maximum_node_inventoried_bytes,p.caps.maximum_integral_calls,p.caps.maximum_work_units}) h.u64(value);
}
std::string array_identity(const char* domain,const double* data,I count,I n) {
    Digest h(domain); h.u64(n); h.u64(count);
    for(I at=0;at<count;++at) h.real(data[at]);
    return h.finish();
}
double offdiagonal_norm(const double* f,I n) {
    double squared=0;
    for(I a=0;a<n;++a) for(I b=0;b<n;++b) {
        const double x=real::finite(f[a*n+b]);
        if(x!=f[b*n+a]) throw std::invalid_argument("Gaussian pair PNO Fock block must be exactly symmetric");
        if(a!=b) real::norm_lane(squared,std::abs(x));
    }
    return real::sqrt_up(squared);
}
void symmetric(const std::vector<double>& x,I n,double& maximum,const char* message) {
    for(I a=0;a<n;++a) for(I b=a+1;b<n;++b) {
        maximum=std::max(maximum,real::difference_up(x[a*n+b],x[b*n+a]));
        if(x[a*n+b]!=x[b*n+a]) throw std::invalid_argument(message);
    }
}
} // namespace

Plan plan_periodic_gaussian_pair_pnos(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    I i,I j,const Options& options,const Live& live,const Caps& caps) {
    options_valid(options); real::float_environment(); objects(ref,basis,provider);
    if(!live.fixed_backend_margin_bytes_per_worker)
        throw std::invalid_argument("Gaussian pair PNO backend margin must be explicit and positive");
    Plan p; p.options=options; p.live=live; p.caps=caps;
    p.occupied_count=basis.memory().occupied_count; p.virtual_count=basis.memory().virtual_count;
    p.occupied_slot_i=i; p.occupied_slot_j=j;
    if(i>=p.occupied_count || j>=p.occupied_count) throw std::out_of_range("Gaussian pair PNO occupied slot out of range");
    const auto n=p.virtual_count,nn=mul(n,n),o=p.occupied_count;
    p.integral_calls=nn; p.provider_work_units=mul(nn,provider.memory().scalar_work_units);
    const auto cube=mul(add(n,1),mul(add(n,1),add(n,1)));
    p.numerical_work_units=add(mul(4096,mul(add(add(options.pno_eigensolver.max_sweeps,
        options.semicanonical_eigensolver.max_sweeps),2),cube)),
        add(mul(1024,mul(add(add(o,n),1),add(add(o,n),1))),65536));
    p.work_units=add(p.provider_work_units,p.numerical_work_units);
    p.integral_amplitude_phase_bytes=add(mul(16,nn),mul(8,n));
    const auto density=plan_restricted_pair_pnos(n);
    p.density_phase_bytes=add(add(mul(8,nn),mul(8,n)),density.peak_owned_numerical_bytes);
    const auto rotation=plan_restricted_pair_semicanonicalization(n,n);
    p.semicanonical_phase_upper_bytes=add(add(mul(8,nn),mul(8,n)),rotation.peak_owned_numerical_bytes);
    p.peak_owned_numerical_bytes=std::max({p.integral_amplitude_phase_bytes,p.density_phase_bytes,p.semicanonical_phase_upper_bytes});
    if(p.peak_owned_numerical_bytes!=add(mul(48,nn),mul(16,n)))
        throw std::logic_error("Gaussian pair PNO shared-leaf lifetime formula changed");
    p.retained_output_upper_bytes=add(mul(8,nn),mul(16,n));
    p.borrowed_basis_bytes=basis.memory().retained_output_bytes;
    p.borrowed_provider_row_bytes=provider.memory().retained_row_bytes;
    p.state_resident_bytes=ref.state_resident_bytes();
    const auto expected_basis=add(mul(16,o),mul(8,add(add(mul(o,o),nn),mul(o,n))));
    if(p.borrowed_basis_bytes!=expected_basis || p.state_resident_bytes!=ref.state().resident_bytes()
        || ref.dimensions().external_bytes<p.state_resident_bytes)
        throw std::logic_error("Gaussian pair PNO borrowed payload inventory changed");
    p.fixed_control_storage_bytes=65536+sizeof(Plan)+sizeof(Result)+sizeof(Options)+sizeof(Live)+sizeof(Caps)
        +sizeof(PeriodicCorrelationAdmittedReference)+sizeof(PeriodicCorrelationRealLocalBasis)
        +sizeof(PeriodicGaussianRealLocalProvider)+sizeof(PeriodicGaussianSourceContext)
        +sizeof(RestrictedPairMP2Result)+sizeof(RestrictedPairPNOResult)+sizeof(RestrictedPairSemicanonicalResult);
    const auto& d=ref.dimensions(); const auto& b=ref.budget();
    p.replicas_per_node=mul(b.mpi_ranks,b.workers_per_rank);
    p.reference_base_node_bytes=add(add(d.external_bytes,d.shared_bytes),
        mul(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)));
    p.per_worker_inventoried_bytes=add(p.peak_owned_numerical_bytes,add(p.borrowed_basis_bytes,
        add(p.borrowed_provider_row_bytes,add(p.fixed_control_storage_bytes,
        add(live.other_live_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker)))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    real::extent(p.peak_owned_numerical_bytes); real::extent(p.borrowed_basis_bytes);
    if(nn>std::vector<double>().max_size()) throw std::length_error("Gaussian pair PNO vector extent exceeds cap");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian pair PNO owned memory cap exceeded");
    limit(p.per_worker_inventoried_bytes,caps.maximum_per_worker_inventoried_bytes,"Gaussian pair PNO worker memory cap exceeded");
    limit(p.required_node_memory_bytes,caps.maximum_node_inventoried_bytes,"Gaussian pair PNO node memory cap exceeded");
    limit(p.required_node_memory_bytes,b.memory_limit_bytes,"Gaussian pair PNO admitted-reference memory cap exceeded");
    limit(p.integral_calls,caps.maximum_integral_calls,"Gaussian pair PNO integral call cap exceeded");
    limit(p.work_units,caps.maximum_work_units,"Gaussian pair PNO work cap exceeded");
    Digest h("vibeqc.periodic.gaussian-pair-pnos.plan");
    for(const auto& s:{ref.state().state_identity_sha256(),ref.dimensions().allocation_identity,
        basis.identity_sha256(),provider.identity_sha256(),provider.hf_reference_source_identity_sha256()}) h.string(s);
    plan_wire(h,p); h.string(policy); const auto sha=h.finish();
    std::copy(sha.begin(),sha.end(),p.identity_ascii.begin()); return p;
}

const std::vector<double>& Result::coefficients() const {
    if(!state_ || !context_ || coefficients_.size()!=mul(memory_.virtual_count,diagnostics_.retained_dimension)
        || energies_.size()!=diagnostics_.retained_dimension || occupations_.size()!=memory_.virtual_count)
        throw std::logic_error("Gaussian pair PNO result is moved or malformed");
    return coefficients_;
}
const std::vector<double>& Result::energies() const { (void) coefficients(); return energies_; }
const std::vector<double>& Result::original_pno_occupations() const { (void) coefficients(); return occupations_; }

Result make_periodic_gaussian_pair_pnos(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& wrapper,
    I i,I j,const Options& options,const Live& live,const Caps& caps) {
    const auto p=plan_periodic_gaussian_pair_pnos(ref,basis,wrapper,i,j,options,live,caps);
    const auto& provider=wrapper.provider(); const auto n=p.virtual_count,o=p.occupied_count,nn=mul(n,n);
    Result result; result.memory_=p; result.state_=ref.state_handle(); result.context_=wrapper.context_handle();
    result.i_=basis.occupied(i); result.j_=basis.occupied(j); result.basis_=basis.identity_sha256();
    result.provider_=wrapper.identity_sha256(); result.hf_=wrapper.hf_reference_source_identity_sha256();
    auto& d=result.diagnostics_; const auto* fvv=basis.f_vv_data(); const auto* foo=basis.f_oo_data();
    d.initial_fvv_offdiagonal_norm_upper_bound=offdiagonal_norm(fvv,n);
    d.ignored_occupied_offdiagonal_norm_upper_bound=offdiagonal_norm(foo,o);
    if(d.initial_fvv_offdiagonal_norm_upper_bound>options.maximum_initial_fvv_offdiagonal_norm)
        throw std::invalid_argument("Gaussian pair PNO initial semicanonical Fvv projection exceeds explicit norm budget");
    d.integral_projection_error_bound=provider.diagnostics().maximum_eri_projection_error_bound;
    std::vector<double> g(static_cast<std::size_t>(nn)),eps(static_cast<std::size_t>(n));
    for(I a=0;a<n;++a) {
        eps[a]=fvv[a*n+a];
        for(I b=0;b<n;++b) {
            const auto integral=provider.integral(i,o+a,j,o+b);
            g[a*n+b]=real::finite(integral.value);
            d.maximum_scalar_integral_roundoff_error=std::max(d.maximum_scalar_integral_roundoff_error,integral.roundoff_error_bound);
            ++d.completed_integral_calls;
        }
    }
    if(i==j) symmetric(g,n,d.maximum_diagonal_integral_asymmetry,"Gaussian diagonal pair PNO integrals are not exactly symmetric");
    result.integrals_=array_identity("vibeqc.periodic.gaussian-pair-pnos.integrals",g.data(),nn,n);
    auto mp2=restricted_pair_semicanonical_mp2(g.data(),g.size(),eps.data(),eps.size(),n,
        foo[i*o+i],foo[j*o+j],options.denominator_floor,mul(8,nn));
    if(i==j) symmetric(mp2.amplitudes,n,d.maximum_diagonal_amplitude_asymmetry,
        "Gaussian diagonal pair PNO initial amplitudes are not exactly symmetric");
    result.amplitudes_=array_identity("vibeqc.periodic.gaussian-pair-pnos.amplitudes",mp2.amplitudes.data(),nn,n);
    d.minimum_denominator=mp2.minimum_denominator; d.maximum_denominator=mp2.maximum_denominator;
    d.maximum_absolute_initial_amplitude=mp2.maximum_absolute_amplitude; d.maximum_initial_residual=mp2.maximum_residual;
    d.initial_residual_frobenius_norm=mp2.residual_frobenius_norm;
    std::vector<double>().swap(g);
    // Deliberate periodic Eq.39 normalization. Never use legacy Diagonal,
    // and never rescale amplitudes or energy to compensate its molecular norm.
    auto pnos=restricted_pair_pnos(mp2.amplitudes.data(),mp2.amplitudes.size(),n,RestrictedPairKind::OffDiagonal,
        options.occupation_cutoff,plan_restricted_pair_pnos(n).peak_owned_numerical_bytes,options.pno_eigensolver);
    std::vector<double>().swap(mp2.amplitudes); std::vector<double>().swap(eps);
    result.density_=array_identity("vibeqc.periodic.gaussian-pair-pnos.density",pnos.density.data(),nn,n);
    d.retained_dimension=pnos.retained_dimension; d.density_trace=pnos.density_trace;
    d.discarded_occupation_sum=pnos.discarded_occupation_sum; d.minimum_occupation=pnos.minimum_occupation;
    d.negative_occupation_count=pnos.negative_occupation_count;
    d.amplitude_scaling_underflow_count=pnos.amplitude_scaling_underflow_count;
    d.density_underflow_entry_count=pnos.density_underflow_entry_count;
    d.density_eigensystem_relative_residual=pnos.eigensystem_relative_residual;
    d.density_eigenvector_orthogonality_error=pnos.eigenvector_orthogonality_error; d.pno_eigensolver=pnos.eigensolver;
    std::vector<double>().swap(pnos.density);
    const auto r=d.retained_dimension; const auto rotation=plan_restricted_pair_semicanonicalization(n,r);
    d.actual_semicanonical_phase_bytes=add(add(mul(8,mul(n,r)),mul(8,n)),rotation.peak_owned_numerical_bytes);
    if(d.actual_semicanonical_phase_bytes>p.semicanonical_phase_upper_bytes)
        throw std::logic_error("Gaussian pair PNO retained semicanonical phase exceeds admission");
    auto sc=restricted_pair_semicanonicalize(pnos.coefficients.data(),pnos.coefficients.size(),fvv,nn,n,r,
        options.semicanonical_orthonormality_tolerance,rotation.peak_owned_numerical_bytes,options.semicanonical_eigensolver);
    d.semicanonical_input_orthonormality_error=sc.input_orthonormality_error;
    d.semicanonical_output_orthonormality_error=sc.output_orthonormality_error;
    d.semicanonical_reduced_relative_residual=sc.reduced_eigensystem_relative_residual;
    d.semicanonical_projected_fock_relative_residual=sc.projected_fock_relative_residual;
    d.semicanonical_subspace_projector_error=sc.subspace_projector_frobenius_error;
    d.semicanonical_full_space_relative_residual=sc.full_space_relative_residual;
    d.semicanonical_fock_scaling_underflow_count=sc.fock_scaling_underflow_entries;
    d.semicanonical_eigensolver=sc.eigensolver;
    result.coefficients_=std::move(sc.coefficients); result.energies_=std::move(sc.energies);
    result.occupations_=std::move(pnos.occupations); std::vector<double>().swap(pnos.coefficients);
    d.retained_output_bytes=mul(8,add(add(mul(n,r),r),n));
    if(d.completed_integral_calls!=p.integral_calls || d.retained_output_bytes>p.retained_output_upper_bytes)
        throw std::logic_error("Gaussian pair PNO completed output differs from admission");
    Digest payload("vibeqc.periodic.gaussian-pair-pnos.payload"); payload.u64(n); payload.u64(r);
    for(const auto* array:{&result.occupations_,&result.coefficients_,&result.energies_})
        for(double value:*array) payload.real(value);
    result.payload_=payload.finish();
    Digest identity("vibeqc.periodic.gaussian-pair-pnos.identity");
    for(const auto& s:{ref.state().state_identity_sha256(),ref.dimensions().allocation_identity,
        result.context_->source_context_identity_sha256(),result.hf_,result.basis_,result.provider_,
        wrapper.consumed_sources_identity_sha256(),result.integrals_,result.amplitudes_,result.density_,result.payload_}) identity.string(s);
    for(auto value:{i,j,result.i_.occupied_index,result.i_.cell,result.j_.occupied_index,result.j_.cell}) identity.u64(value);
    options_wire(identity,options); identity.string(policy);
    for(double value:{d.initial_fvv_offdiagonal_norm_upper_bound,d.ignored_occupied_offdiagonal_norm_upper_bound,
        d.maximum_scalar_integral_roundoff_error,d.integral_projection_error_bound}) identity.real(value);
    result.identity_=identity.finish(); return result;
}

} // namespace vibeqc
