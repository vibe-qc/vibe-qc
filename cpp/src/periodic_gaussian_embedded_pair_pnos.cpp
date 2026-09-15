#include "vibeqc/periodic_gaussian_embedded_pair_pnos.hpp"
#include "vibeqc/periodic_gaussian_pair_pno_frame.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "embedded Gaussian pair PNOs forbid fast/finite-only math"
#endif
namespace vibeqc {
namespace {
namespace real=periodic_correlation_real_local_detail;
namespace local=periodic_correlation_local_detail;
using U=std::uint64_t;
using real::add;using real::mul;using real::Digest;
using Plan=PeriodicGaussianEmbeddedPairPNOPlan;
using Result=PeriodicGaussianEmbeddedPairPNOResult;
using Options=PeriodicGaussianEmbeddedPairPNOOptions;
using Live=PeriodicGaussianEmbeddedPairPNOLiveInventory;
using Caps=PeriodicGaussianEmbeddedPairPNOCaps;
using Embedding=PeriodicCorrelationRealPAOEmbedding;
static_assert(sizeof(double)==8 && FLT_EVAL_METHOD==0,"embedded PNOs require binary64 evaluation");
constexpr char policy[]="Nejad2025-Eq39;pair-frame-density;no-delta-prefactor;raw-ordered-projections;"
    "explicit-diagonal-G-and-F-transpose-projections;initial-diagonal-projected-F-only;"
    "original-projected-F-recanonicalization;retain-authentic-generation-D;common-export;no-PNO-padding;no-renormalization;"
    "no-energy-multiplicity;not-production-distinct-domain-scaling";
void limit(U value,U cap,const char* text) { if(value>cap) throw std::length_error(text); }
void finite_nonnegative(double x) {
    if(!std::isfinite(x) || x<0.0) throw std::invalid_argument("embedded PNO projection/cutoff budgets must be finite nonnegative");
}
void positive(double x) { finite_nonnegative(x);if(x==0.0) throw std::invalid_argument("embedded PNO numerical audit budget must be positive"); }
void options_valid(const Options& o,const Live& live) {
    finite_nonnegative(o.pno.occupation_cutoff);positive(o.pno.denominator_floor);
    finite_nonnegative(o.pno.maximum_initial_fvv_offdiagonal_norm);
    positive(o.pno.semicanonical_orthonormality_tolerance);
    if(o.pno.semicanonical_orthonormality_tolerance>=1.0 || !live.fixed_backend_margin_bytes_per_worker)
        throw std::invalid_argument("embedded PNO orthonormality/backend controls are invalid");
    finite_nonnegative(o.maximum_diagonal_exchange_projection_norm);finite_nonnegative(o.maximum_fock_symmetry_projection_norm);
    positive(o.maximum_exported_gram_error);positive(o.maximum_exported_fock_error);positive(o.maximum_exported_subspace_error);
    for(const auto& j:{o.pno.pno_eigensolver,o.pno.semicanonical_eigensolver})
        if(!j.max_sweeps || !std::isfinite(j.relative_offdiagonal_tolerance)
            || j.relative_offdiagonal_tolerance<=0.0 || j.relative_offdiagonal_tolerance>=1.0)
            throw std::invalid_argument("embedded PNO eigensolver controls must be explicit");
}
void sha(const std::string& s) {
    if(s.size()!=64) throw std::invalid_argument("embedded PNO source SHA extent differs");
    for(char c:s) if(!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("embedded PNO source SHA is malformed");
}
void objects(const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& b,
    const PeriodicGaussianRealLocalProvider& wrapper,const Embedding& e) {
    local::validate_reference(ref);const auto& p=wrapper.provider();const auto& ctx=wrapper.context_handle();
    if(!ctx || !wrapper.matched_finite_gaussian_hf_recipe() || !p.matched_finite_gaussian_hf_recipe()
        || b.state_handle()!=ref.state_handle() || p.state_handle()!=ref.state_handle() || e.state_handle()!=ref.state_handle()
        || b.allocation_identity()!=ref.dimensions().allocation_identity || e.allocation_identity()!=ref.dimensions().allocation_identity
        || p.identity_sha256()!=wrapper.identity_sha256() || p.local_basis_identity_sha256()!=b.local_basis_identity_sha256()
        || p.basis_certificate_identity_sha256()!=b.identity_sha256() || e.common_basis_identity_sha256()!=b.identity_sha256()
        || p.hf_reference_source_identity_sha256()!=wrapper.hf_reference_source_identity_sha256()
        || p.source_context_identity_sha256()!=wrapper.source_context_identity_sha256())
        throw std::invalid_argument("embedded PNO actual source/basis/embedding owners or receipts differ");
    for(const auto* s:std::array<const std::string*,12>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &b.identity_sha256(),&b.payload_sha256(),&wrapper.identity_sha256(),&wrapper.hf_reference_source_identity_sha256(),
        &wrapper.source_context_identity_sha256(),&p.payload_sha256(),&e.identity_sha256(),&e.payload_sha256(),
        &e.common_frame_identity_sha256(),&e.pair_frame_identity_sha256()}) sha(*s);
    if(!std::equal(ctx->source_context_identity_ascii().begin(),ctx->source_context_identity_ascii().end(),
            wrapper.source_context_identity_sha256().begin()))
        throw std::invalid_argument("embedded PNO physical context receipt differs");
    const auto& bm=b.memory();const auto& pm=p.memory();const auto& em=e.memory();
    const auto c=b.virtual_selection(),ec=e.common_selection();
    if(!bm.occupied_count || !bm.virtual_count || !em.pair_dimension || em.pair_dimension>bm.virtual_count
        || em.common_dimension!=bm.virtual_count || c.begin!=ec.begin || c.count!=ec.count || c.translation_cell!=ec.translation_cell
        || bm.occupied_count!=pm.occupied_count || bm.virtual_count!=pm.virtual_count
        || bm.orbital_count!=add(bm.occupied_count,bm.virtual_count) || bm.orbital_count!=pm.orbital_count
        || em.n_cells!=ref.state().n_kpoints() || em.n_basis!=ref.state().n_basis()
        || pm.n_cells!=ref.state().n_kpoints() || pm.n_auxiliary!=ref.dimensions().n_auxiliary
        || ctx->mesh().mesh()!=ref.state().mesh() || ctx->mesh().is_shift()!=ref.state().is_shift()
        || ctx->inventory().ao.function_count!=ref.state().n_basis() || ctx->inventory().auxiliary.function_count!=pm.n_auxiliary
        || wrapper.memory().retained_row_bytes!=pm.retained_row_bytes
        || wrapper.memory().occupied_count!=bm.occupied_count || wrapper.memory().virtual_count!=bm.virtual_count
        || wrapper.memory().orbital_count!=bm.orbital_count
        || wrapper.memory().scalar_work_units!=pm.scalar_work_units || !pm.scalar_work_units)
        throw std::invalid_argument("embedded PNO actual source/embedding dimensions differ");
    const auto callback=p.integral_provider(b);
    if(callback.retained_numerical_bytes!=add(pm.retained_row_bytes,mul(16,bm.occupied_count))
        || callback.maximum_transient_numerical_bytes)
        throw std::logic_error("embedded PNO native scalar provider inventory differs");
    (void)b.f_oo_data();(void)b.occupied_indices_data();(void)p.rows_data();
    (void)e.coefficients_data();(void)e.energies_data();
}
double product(double a,double b) {
    const double x=real::finite(a*b);
    if(x==0.0 && a!=0.0 && b!=0.0) throw std::overflow_error("embedded PNO contraction product underflows");
    return x;
}
void accumulate(double x,double& sum,double& correction) {
    real::finite(x);const double next=real::finite(sum+x);
    const double delta=std::abs(sum)>=std::abs(x) ? (sum-next)+x : (x-next)+sum;
    correction=real::finite(correction+delta);sum=next;
}
struct Sum {
    double sum=0,correction=0;
    void include(double x) { accumulate(x,sum,correction); }
    double value() const { return real::finite(sum+correction); }
};
double average(double a,double b) {
    if(a==b) return a;
    int exponent=0;std::frexp(std::max(std::abs(a),std::abs(b)),&exponent);
    const double x=std::scalbn(a,-exponent),y=std::scalbn(b,-exponent);
    if(std::scalbn(x,exponent)!=a || std::scalbn(y,exponent)!=b)
        throw std::overflow_error("embedded PNO transpose projection loses input range");
    return real::finite(std::scalbn(x+y,exponent-1));
}
void project_symmetric(std::vector<double>& matrix,U n,double cap,double& max_defect,double& correction) {
    double squared=0;
    // Audit the ENTIRE unmodified matrix before writing a projected entry.
    for(U a=0;a<n;++a) for(U b=a+1;b<n;++b) {
        const double x=matrix[a*n+b],y=matrix[b*n+a],z=average(x,y);
        max_defect=std::max(max_defect,real::difference_up(x,y));
        real::norm_lane(squared,real::difference_up(x,z));real::norm_lane(squared,real::difference_up(y,z));
    }
    correction=real::sqrt_up(squared);
    if(correction>cap) throw std::invalid_argument("embedded PNO raw transpose projection exceeds explicit Frobenius budget");
    for(U a=0;a<n;++a) for(U b=a+1;b<n;++b) matrix[a*n+b]=matrix[b*n+a]=average(matrix[a*n+b],matrix[b*n+a]);
}
double offdiagonal(const double* f,U n) {
    double norm=0;
    for(U a=0;a<n;++a) for(U b=0;b<n;++b) {
        const double x=real::finite(f[a*n+b]);
        if(x!=f[b*n+a]) throw std::invalid_argument("embedded PNO input Fock must be exactly symmetric");
        if(a!=b) real::norm_lane(norm,std::abs(x));
    }
    return real::sqrt_up(norm);
}
std::string array_hash(const char* domain,const double* x,U elements,U n) {
    Digest h(domain);h.u64(n);h.u64(elements);for(U a=0;a<elements;++a) h.real(x[a]);return h.finish();
}
std::string sources(const PeriodicCorrelationAdmittedReference& ref,const PeriodicCorrelationRealLocalBasis& basis,
    const PeriodicGaussianRealLocalProvider& p,const Embedding& e) {
    Digest h("vibeqc.periodic.gaussian-embedded-pair-pnos.sources");
    for(const auto* s:std::array<const std::string*,9>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &basis.identity_sha256(),&basis.payload_sha256(),&p.identity_sha256(),&p.provider().payload_sha256(),
        &p.hf_reference_source_identity_sha256(),&e.identity_sha256(),&e.payload_sha256()}) h.string(*s);
    const auto* labels=basis.occupied_indices_data();for(U a=0;a<mul(2,basis.memory().occupied_count);++a) h.u64(labels[a]);
    const auto* f=basis.f_oo_data();for(U a=0;a<basis.memory().retained_fock_bytes/8;++a) h.real(f[a]);
    const auto* rows=p.provider().rows_data();for(U a=0;a<p.memory().retained_row_bytes/8;++a) h.real(rows[a]);
    const U n=e.memory().common_dimension,m=e.memory().pair_dimension;
    Digest ep("vibeqc.periodic.correlation.real-pao-embedding.payload");ep.u64(n);ep.u64(m);
    for(U a=0;a<mul(n,m);++a) {const double x=e.coefficients_data()[a];h.real(x);ep.real(x);}
    for(U a=0;a<m;++a) {const double x=e.energies_data()[a];h.real(x);ep.real(x);}
    if(ep.finish()!=e.payload_sha256()) throw std::invalid_argument("embedded PNO actual embedding payload differs from its physical certificate");
    return h.finish();
}
void options_wire(Digest& h,const Options& o) {
    for(double x:{o.pno.occupation_cutoff,o.pno.denominator_floor,o.pno.maximum_initial_fvv_offdiagonal_norm,
        o.pno.semicanonical_orthonormality_tolerance,o.maximum_diagonal_exchange_projection_norm,
        o.maximum_fock_symmetry_projection_norm,o.maximum_exported_gram_error,o.maximum_exported_fock_error,
        o.maximum_exported_subspace_error}) h.real(x);
    for(const auto& j:{o.pno.pno_eigensolver,o.pno.semicanonical_eigensolver}) {h.u64(j.max_sweeps);h.real(j.relative_offdiagonal_tolerance);}
}
} // namespace

PeriodicGaussianPairPNOFrameValidationPlan plan_periodic_gaussian_pair_pno_frame_validation(
    const PeriodicGaussianPairPNOFrameView& frame) {
    if(frame.is_direct_gram()) {
        const auto source=plan_periodic_gaussian_gram_pair_pno_payload_validation(frame.direct_gram());
        return {source.numerical_lanes,source.work_units,source.control_storage_bytes};
    }
    PeriodicGaussianPairPNOFrameValidationPlan p;
    p.numerical_lanes=frame.retained_numerical_bytes()/8;
    p.work_units=mul(512,add(p.numerical_lanes,1024));
    p.control_storage_bytes=4096+sizeof(Digest)+sizeof(PeriodicGaussianPairPNOFrameView)
        +sizeof(PeriodicGaussianPairPNOFrameValidationPlan)+65;
    // SHA-256 has a uint64 bit-length. Include domain/shape overhead.
    if(p.numerical_lanes>(std::numeric_limits<U>::max()/8-1024)/8)
        throw std::length_error("PNO frame payload exceeds digest length domain");
    return p;
}
void validate_periodic_gaussian_pair_pno_frame_payload(const PeriodicGaussianPairPNOFrameView& frame) {
    if(frame.is_direct_gram()) {
        const auto p=plan_periodic_gaussian_pair_pno_frame_validation(frame);
        verify_periodic_gaussian_gram_pair_pno_payload(frame.direct_gram(),p.work_units);
        return;
    }
    (void)plan_periodic_gaussian_pair_pno_frame_validation(frame);real::float_environment();
    const bool embedded=frame.is_embedded();
    Digest payload(embedded ? "vibeqc.periodic.gaussian-embedded-pair-pnos.payload-v2" : "vibeqc.periodic.gaussian-pair-pnos.payload");
    payload.u64(frame.common_virtual_dimension());
    if(embedded) payload.u64(frame.generation_dimension());
    payload.u64(frame.retained_dimension());
    for(const auto* array:{&frame.original_pno_occupations(),&frame.coefficients()})
        for(double x:*array) payload.real(x);
    if(embedded) for(double x:frame.embedded().generation_coefficients()) payload.real(x);
    for(double x:frame.energies()) payload.real(x);
    if(payload.finish()!=frame.payload_sha256()) throw std::invalid_argument("PNO frame numerical payload differs from source receipt");
}

PeriodicGaussianEmbeddedPairPNOCountPlan plan_periodic_gaussian_embedded_pair_pno_counts(
    const PeriodicGaussianEmbeddedPairPNOCountInput& input,const Options& options,const Live& live) {
    real::float_environment();options_valid(options,live);
    const U n=input.common_virtual_dimension,m=input.generation_dimension,o=input.occupied_count;
    if(!n || !m || m>n || !o || !input.provider_scalar_work_units || input.provider_retained_row_bytes%8)
        throw std::invalid_argument("embedded PNO count-only source census is invalid");
    const U mm=mul(m,m),nm=mul(n,m),nn=mul(n,n);
    PeriodicGaussianEmbeddedPairPNOCountPlan p;
    p.integral_calls=nn;p.provider_work_units=mul(nn,input.provider_scalar_work_units);
    p.projection_phase_bytes=mul(24,mm);p.amplitude_phase_bytes=add(mul(24,mm),mul(8,m));
    p.density_phase_bytes=add(mul(16,mm),plan_restricted_pair_pnos(m).peak_owned_numerical_bytes);
    p.semicanonical_phase_upper_bytes=add(add(mul(16,mm),mul(8,m)),plan_restricted_pair_semicanonicalization(m,m).peak_owned_numerical_bytes);
    p.export_phase_upper_bytes=add(mul(8,nm),add(mul(8,mm),mul(16,m)));
    p.peak_owned_numerical_bytes=std::max({p.projection_phase_bytes,p.amplitude_phase_bytes,p.density_phase_bytes,
        p.semicanonical_phase_upper_bytes,p.export_phase_upper_bytes});
    p.retained_generation_coefficient_upper_bytes=mul(8,mm);
    p.retained_output_upper_bytes=p.export_phase_upper_bytes;
    if(p.density_phase_bytes!=add(mul(56,mm),mul(8,m)) || p.semicanonical_phase_upper_bytes!=add(mul(56,mm),mul(16,m)))
        throw std::logic_error("embedded PNO shared numerical phase contracts changed");
    p.borrowed_basis_bytes=add(mul(16,o),mul(8,add(add(mul(o,o),nn),mul(o,n))));
    p.borrowed_provider_row_bytes=input.provider_retained_row_bytes;
    p.borrowed_embedding_bytes=add(mul(8,nm),mul(8,m));
    p.complete_borrowed_numerical_bytes=add(p.borrowed_basis_bytes,add(p.borrowed_provider_row_bytes,p.borrowed_embedding_bytes));
    p.control_storage_reservation_bytes=131072U+3U*sizeof(Plan)+2U*sizeof(Result)+2U*sizeof(Options)+2U*sizeof(Live)+2U*sizeof(Caps)
        +sizeof(PeriodicCorrelationAdmittedReference)+sizeof(PeriodicCorrelationRealLocalBasis)+sizeof(PeriodicGaussianRealLocalProvider)
        +sizeof(PeriodicGaussianSourceContext)+sizeof(Embedding)+2U*sizeof(RestrictedPairMP2Result)
        +2U*sizeof(RestrictedPairPNOResult)+2U*sizeof(RestrictedPairSemicanonicalResult)+12U*sizeof(Digest)+64U*65U
        +2U*sizeof(PeriodicGaussianEmbeddedPairPNOCountInput)+2U*sizeof(PeriodicGaussianEmbeddedPairPNOCountPlan);
    p.control_storage_reservation_bytes=add(p.control_storage_reservation_bytes,live.other_live_control_bytes_per_worker);
    const U cube=mul(add(m,1),mul(add(m,1),add(m,1)));
    const U algebra=add(mul(nn,mm),add(mul(nn,mul(m,add(m,1))),add(mul(nm,m),add(mul(o,o),nn))));
    p.numerical_work_units=add(mul(4096,mul(add(add(options.pno.pno_eigensolver.max_sweeps,
        options.pno.semicanonical_eigensolver.max_sweeps),2),cube)),
        add(mul(1024,algebra),mul(512,add(add(p.complete_borrowed_numerical_bytes/8,mm),65536))));
    p.work_units=add(p.provider_work_units,p.numerical_work_units);
    for(U bytes:{p.peak_owned_numerical_bytes,p.complete_borrowed_numerical_bytes,p.control_storage_reservation_bytes,
        p.retained_output_upper_bytes}) real::extent(bytes);
    if(mm>std::vector<double>().max_size() || nm>std::vector<double>().max_size())
        throw std::length_error("embedded PNO native payload vector extent exceeded");
    return p;
}

Plan plan_periodic_gaussian_embedded_pair_pnos(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,const Embedding& embedding,
    U i,U j,const Options& options,const Live& live,const Caps& caps) {
    real::float_environment();options_valid(options,live);objects(ref,basis,provider,embedding);
    Plan p;p.occupied_count=basis.memory().occupied_count;p.common_virtual_dimension=basis.memory().virtual_count;
    p.generation_dimension=embedding.memory().pair_dimension;p.occupied_slot_i=i;p.occupied_slot_j=j;
    if(i>=p.occupied_count || j>=p.occupied_count) throw std::out_of_range("embedded PNO occupied slot exceeds actual basis");
    limit(p.common_virtual_dimension,caps.maximum_common_dimension,"embedded PNO common dimension cap");
    limit(p.generation_dimension,caps.maximum_generation_dimension,"embedded PNO generation dimension cap");
    if(embedding.diagnostics().embedding_gram_frobenius_upper_bound>options.pno.semicanonical_orthonormality_tolerance)
        throw std::invalid_argument("embedded PNO pair-frame Gram certificate exceeds requested generation tolerance");
    const auto count=plan_periodic_gaussian_embedded_pair_pno_counts({p.occupied_count,p.common_virtual_dimension,p.generation_dimension,
        provider.memory().scalar_work_units,provider.memory().retained_row_bytes},options,live);
    p.integral_calls=count.integral_calls;p.provider_work_units=count.provider_work_units;
    p.numerical_work_units=count.numerical_work_units;p.work_units=count.work_units;
    p.projection_phase_bytes=count.projection_phase_bytes;p.amplitude_phase_bytes=count.amplitude_phase_bytes;
    p.density_phase_bytes=count.density_phase_bytes;p.semicanonical_phase_upper_bytes=count.semicanonical_phase_upper_bytes;
    p.export_phase_upper_bytes=count.export_phase_upper_bytes;p.peak_owned_numerical_bytes=count.peak_owned_numerical_bytes;
    p.retained_output_upper_bytes=count.retained_output_upper_bytes;p.borrowed_basis_bytes=count.borrowed_basis_bytes;
    p.retained_generation_coefficient_upper_bytes=count.retained_generation_coefficient_upper_bytes;
    p.borrowed_provider_row_bytes=count.borrowed_provider_row_bytes;p.borrowed_embedding_bytes=count.borrowed_embedding_bytes;
    p.complete_borrowed_numerical_bytes=count.complete_borrowed_numerical_bytes;
    p.control_storage_reservation_bytes=count.control_storage_reservation_bytes;
    if(p.borrowed_basis_bytes!=basis.memory().retained_output_bytes || p.borrowed_embedding_bytes!=embedding.memory().output_numerical_bytes
        || ref.state_resident_bytes()!=ref.state().resident_bytes() || ref.dimensions().external_bytes<ref.state_resident_bytes())
        throw std::logic_error("embedded PNO complete live owner census differs");
    const auto& d=ref.dimensions();const auto& b=ref.budget();p.replicas_per_node=mul(b.mpi_ranks,b.workers_per_rank);
    if(!p.replicas_per_node) throw std::invalid_argument("embedded PNO requires positive native replicas");
    p.reference_base_node_bytes=add(add(d.external_bytes,d.shared_bytes),mul(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)));
    p.per_worker_inventoried_bytes=add(p.complete_borrowed_numerical_bytes,add(p.peak_owned_numerical_bytes,
        add(p.control_storage_reservation_bytes,add(live.other_live_numerical_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"embedded PNO owned cap");
    limit(p.control_storage_reservation_bytes,caps.maximum_control_storage_bytes_per_worker,"embedded PNO control cap");
    limit(p.per_worker_inventoried_bytes,caps.maximum_per_worker_inventoried_bytes,"embedded PNO worker cap");
    limit(p.required_node_memory_bytes,caps.maximum_node_inventoried_bytes,"embedded PNO node cap");
    limit(p.required_node_memory_bytes,b.memory_limit_bytes,"embedded PNO admitted-reference node cap");
    limit(p.integral_calls,caps.maximum_integral_calls,"embedded PNO integral count cap");
    limit(p.work_units,caps.maximum_work_units,"embedded PNO work cap");return p;
}

namespace {
void export_audit(const double* C,const double* X,const double* F,const double* eps,U n,U m,U t,const Options& o,
    PeriodicGaussianEmbeddedPairPNODiagnostics& d) {
    // These are measured binary64 residuals: outward norm arithmetic below
    // does not turn the preceding rounded compensated dots into intervals.
    double gram=0,fock=0,subspace=0;
    for(U a=0;a<t;++a) for(U b=0;b<t;++b) {
        Sum g,f;
        for(U l=0;l<n;++l) {
            g.include(product(C[l*t+a],C[l*t+b]));
            for(U q=0;q<n;++q) f.include(product(product(C[l*t+a],F[l*n+q]),C[q*t+b]));
        }
        real::norm_lane(gram,real::difference_up(g.value(),a==b ? 1.0 : 0.0));
        real::norm_lane(fock,real::difference_up(f.value(),a==b ? eps[a] : 0.0));
    }
    for(U l=0;l<n;++l) for(U a=0;a<t;++a) {
        Sum reconstructed;
        for(U b=0;b<m;++b) {
            Sum overlap;for(U q=0;q<n;++q) overlap.include(product(X[q*m+b],C[q*t+a]));
            reconstructed.include(product(X[l*m+b],overlap.value()));
        }
        real::norm_lane(subspace,real::difference_up(C[l*t+a],reconstructed.value()));
    }
    d.exported_gram_frobenius_upper_bound=real::sqrt_up(gram);
    d.exported_fock_frobenius_upper_bound=real::sqrt_up(fock);
    d.exported_subspace_frobenius_upper_bound=real::sqrt_up(subspace);
    if(d.exported_gram_frobenius_upper_bound>o.maximum_exported_gram_error
        || d.exported_fock_frobenius_upper_bound>o.maximum_exported_fock_error
        || d.exported_subspace_frobenius_upper_bound>o.maximum_exported_subspace_error)
        throw std::invalid_argument("embedded PNO final common Gram/Fock/subspace audit exceeded explicit budget");
}
} // namespace

void Result::require_live() const {
    if(!state_ || !context_ || !memory_.generation_dimension || !memory_.common_virtual_dimension
        || memory_.generation_dimension>memory_.common_virtual_dimension
        || diagnostics_.retained_dimension>memory_.generation_dimension
        || coefficients_.size()!=mul(memory_.common_virtual_dimension,diagnostics_.retained_dimension)
        || generation_coefficients_.size()!=mul(memory_.generation_dimension,diagnostics_.retained_dimension)
        || diagnostics_.retained_generation_coefficient_bytes!=mul(8,generation_coefficients_.size())
        || diagnostics_.retained_output_bytes!=mul(8,add(add(coefficients_.size(),generation_coefficients_.size()),
            add(energies_.size(),occupations_.size())))
        || energies_.size()!=diagnostics_.retained_dimension || occupations_.size()!=memory_.generation_dimension)
        throw std::logic_error("embedded PNO result is consumed or malformed");
}
const std::vector<double>& Result::coefficients() const {require_live();return coefficients_;}
const std::vector<double>& Result::generation_coefficients() const {require_live();return generation_coefficients_;}
const std::vector<double>& Result::energies() const {require_live();return energies_;}
const std::vector<double>& Result::original_pno_occupations() const {require_live();return occupations_;}
bool Result::complete_generation_pair_space() const {require_live();return diagnostics_.retained_dimension==memory_.generation_dimension;}
bool Result::complete_common_virtual_space() const {require_live();return diagnostics_.retained_dimension==memory_.common_virtual_dimension;}

Result make_periodic_gaussian_embedded_pair_pnos(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,const Embedding& embedding,
    U i,U j,const Options& supplied_options,const Live& supplied_live,const Caps& supplied_caps) {
    const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_embedded_pair_pnos(ref,basis,provider,embedding,i,j,options,live,caps);
    const U n=p.common_virtual_dimension,m=p.generation_dimension,o=p.occupied_count,mm=mul(m,m);
    const double* X=embedding.coefficients_data();const double* F=basis.f_vv_data();const double* Foo=basis.f_oo_data();
    Result result;result.memory_=p;result.options_=options;result.sources_=sources(ref,basis,provider,embedding);
    result.i_=basis.occupied(i);result.j_=basis.occupied(j);result.basis_=basis.identity_sha256();
    result.provider_=provider.identity_sha256();result.hf_=provider.hf_reference_source_identity_sha256();result.embedding_=embedding.identity_sha256();
    auto& d=result.diagnostics_;auto& gdiag=d.generation;
    (void)offdiagonal(F,n);gdiag.ignored_occupied_offdiagonal_norm_upper_bound=offdiagonal(Foo,o);
    gdiag.integral_projection_error_bound=provider.provider().diagnostics().maximum_eri_projection_error_bound;
    std::vector<double> G(static_cast<std::size_t>(mm)),Fp(static_cast<std::size_t>(mm));
    {
        std::vector<double> compensation(static_cast<std::size_t>(mm));
        Digest common_g("vibeqc.periodic.gaussian-pair-pnos.integrals");
        common_g.u64(n);common_g.u64(p.integral_calls);
        for(U a=0;a<n;++a) for(U b=0;b<n;++b) {
            if(gdiag.completed_integral_calls>=p.integral_calls) throw std::length_error("embedded PNO integral count exhausted");
            ++gdiag.completed_integral_calls; // BEFORE the actual native scalar call.
            const auto integral=provider.provider().integral(i,o+a,j,o+b);real::float_environment();
            const double value=real::finite(integral.value);
            common_g.real(value);
            gdiag.maximum_scalar_integral_roundoff_error=std::max(gdiag.maximum_scalar_integral_roundoff_error,integral.roundoff_error_bound);
            for(U u=0;u<m;++u) for(U v=0;v<m;++v)
                accumulate(product(product(X[a*m+u],value),X[b*m+v]),G[u*m+v],compensation[u*m+v]);
        }
        result.common_g_=common_g.finish();
        for(U u=0;u<mm;++u) {G[u]=real::finite(G[u]+compensation[u]);compensation[u]=0.0;}
        for(U a=0;a<n;++a) for(U b=0;b<n;++b) for(U u=0;u<m;++u) for(U v=0;v<m;++v)
            accumulate(product(product(X[a*m+u],F[a*n+b]),X[b*m+v]),Fp[u*m+v],compensation[u*m+v]);
        for(U u=0;u<mm;++u) Fp[u]=real::finite(Fp[u]+compensation[u]);
    }
    result.raw_g_=array_hash("vibeqc.periodic.gaussian-embedded-pair-pnos.raw-G",G.data(),mm,m);
    result.raw_f_=array_hash("vibeqc.periodic.gaussian-embedded-pair-pnos.raw-F",Fp.data(),mm,m);
    if(i==j) project_symmetric(G,m,options.maximum_diagonal_exchange_projection_norm,
        d.maximum_raw_exchange_transpose_defect,d.diagonal_exchange_projection_frobenius_upper_bound);
    project_symmetric(Fp,m,options.maximum_fock_symmetry_projection_norm,
        d.maximum_raw_fock_transpose_defect,d.fock_symmetry_projection_frobenius_upper_bound);
    gdiag.maximum_diagonal_integral_asymmetry=d.maximum_raw_exchange_transpose_defect;
    result.g_=array_hash("vibeqc.periodic.gaussian-embedded-pair-pnos.G",G.data(),mm,m);
    result.f_=array_hash("vibeqc.periodic.gaussian-embedded-pair-pnos.F",Fp.data(),mm,m);
    gdiag.initial_fvv_offdiagonal_norm_upper_bound=offdiagonal(Fp.data(),m);
    if(gdiag.initial_fvv_offdiagonal_norm_upper_bound>options.pno.maximum_initial_fvv_offdiagonal_norm)
        throw std::invalid_argument("embedded PNO initial pair Fock offdiagonal projection exceeds explicit norm budget");
    std::vector<double> eps(static_cast<std::size_t>(m));for(U a=0;a<m;++a) eps[a]=Fp[a*m+a];
    auto initial=restricted_pair_semicanonical_mp2(G.data(),G.size(),eps.data(),eps.size(),m,Foo[i*o+i],Foo[j*o+j],
        options.pno.denominator_floor,mul(8,mm));
    if(i==j) for(U a=0;a<m;++a) for(U b=a+1;b<m;++b) if(initial.amplitudes[a*m+b]!=initial.amplitudes[b*m+a])
        throw std::logic_error("embedded PNO diagonal SC-MP2 symmetry differs despite shared denominators");
    result.amplitudes_=array_hash("vibeqc.periodic.gaussian-embedded-pair-pnos.T",initial.amplitudes.data(),mm,m);
    gdiag.minimum_denominator=initial.minimum_denominator;gdiag.maximum_denominator=initial.maximum_denominator;
    gdiag.maximum_absolute_initial_amplitude=initial.maximum_absolute_amplitude;gdiag.maximum_initial_residual=initial.maximum_residual;
    gdiag.initial_residual_frobenius_norm=initial.residual_frobenius_norm;
    std::vector<double>().swap(G);std::vector<double>().swap(eps);
    auto pnos=restricted_pair_pnos(initial.amplitudes.data(),initial.amplitudes.size(),m,RestrictedPairKind::OffDiagonal,
        options.pno.occupation_cutoff,plan_restricted_pair_pnos(m).peak_owned_numerical_bytes,options.pno.pno_eigensolver);
    std::vector<double>().swap(initial.amplitudes);
    result.density_=array_hash("vibeqc.periodic.gaussian-embedded-pair-pnos.D",pnos.density.data(),mm,m);
    const U t=d.retained_dimension=gdiag.retained_dimension=pnos.retained_dimension;
    gdiag.density_trace=pnos.density_trace;gdiag.discarded_occupation_sum=pnos.discarded_occupation_sum;
    gdiag.minimum_occupation=pnos.minimum_occupation;gdiag.negative_occupation_count=pnos.negative_occupation_count;
    gdiag.amplitude_scaling_underflow_count=pnos.amplitude_scaling_underflow_count;gdiag.density_underflow_entry_count=pnos.density_underflow_entry_count;
    gdiag.density_eigensystem_relative_residual=pnos.eigensystem_relative_residual;
    gdiag.density_eigenvector_orthogonality_error=pnos.eigenvector_orthogonality_error;gdiag.pno_eigensolver=pnos.eigensolver;
    std::vector<double>().swap(pnos.density);
    const auto rotation=plan_restricted_pair_semicanonicalization(m,t);
    d.actual_semicanonical_phase_bytes=add(add(mul(8,mm),add(mul(8,mul(m,t)),mul(8,m))),rotation.peak_owned_numerical_bytes);
    limit(d.actual_semicanonical_phase_bytes,p.semicanonical_phase_upper_bytes,"embedded PNO actual local rotation phase exceeds admission");
    auto sc=restricted_pair_semicanonicalize(pnos.coefficients.data(),pnos.coefficients.size(),Fp.data(),Fp.size(),m,t,
        options.pno.semicanonical_orthonormality_tolerance,rotation.peak_owned_numerical_bytes,options.pno.semicanonical_eigensolver);
    gdiag.actual_semicanonical_phase_bytes=d.actual_semicanonical_phase_bytes;
    gdiag.semicanonical_input_orthonormality_error=sc.input_orthonormality_error;gdiag.semicanonical_output_orthonormality_error=sc.output_orthonormality_error;
    gdiag.semicanonical_reduced_relative_residual=sc.reduced_eigensystem_relative_residual;
    gdiag.semicanonical_projected_fock_relative_residual=sc.projected_fock_relative_residual;
    gdiag.semicanonical_subspace_projector_error=sc.subspace_projector_frobenius_error;
    gdiag.semicanonical_full_space_relative_residual=sc.full_space_relative_residual;
    gdiag.semicanonical_fock_scaling_underflow_count=sc.fock_scaling_underflow_entries;gdiag.semicanonical_eigensolver=sc.eigensolver;
    std::vector<double>().swap(Fp);std::vector<double>().swap(pnos.coefficients);
    d.actual_export_phase_bytes=mul(8,add(add(mul(n,t),mul(m,t)),add(t,m)));
    limit(d.actual_export_phase_bytes,p.export_phase_upper_bytes,"embedded PNO export phase exceeds admission");
    // Riplinger2013 doi:10.1063/1.4773581 Eq.25 and its recanonicalization
    // text: retain the original local expansion after the projected-Fock
    // rotation. C=X D is only a compatibility export; X^T C is not D's source.
    result.generation_coefficients_=std::move(sc.coefficients);
    result.coefficients_.resize(static_cast<std::size_t>(mul(n,t)));
    for(U a=0;a<n;++a) for(U b=0;b<t;++b) {
        Sum value;for(U u=0;u<m;++u) value.include(product(X[a*m+u],result.generation_coefficients_[u*t+b]));result.coefficients_[a*t+b]=value.value();
    }
    result.energies_=std::move(sc.energies);result.occupations_=std::move(pnos.occupations);
    export_audit(result.coefficients_.data(),X,F,result.energies_.data(),n,m,t,options,d);
    gdiag.retained_output_bytes=mul(8,add(add(mul(m,t),t),m));d.retained_output_bytes=d.actual_export_phase_bytes;
    d.retained_generation_coefficient_bytes=mul(8,mul(m,t));
    if(gdiag.completed_integral_calls!=p.integral_calls || d.retained_output_bytes>p.retained_output_upper_bytes)
        throw std::logic_error("embedded PNO completed source/output census differs");
    objects(ref,basis,provider,embedding);if(sources(ref,basis,provider,embedding)!=result.sources_)
        throw std::invalid_argument("embedded PNO actual source payload changed");
    real::float_environment();result.state_=ref.state_handle();result.context_=provider.context_handle();
    Digest payload("vibeqc.periodic.gaussian-embedded-pair-pnos.payload-v2");payload.u64(n);payload.u64(m);payload.u64(t);
    for(const auto* v:{&result.occupations_,&result.coefficients_,&result.generation_coefficients_,&result.energies_}) for(double x:*v) payload.real(x);
    result.payload_=payload.finish();Digest identity("vibeqc.periodic.gaussian-embedded-pair-pnos.identity");
    for(const auto* s:std::array<const std::string*,15>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &result.hf_,&result.basis_,&result.provider_,&result.embedding_,&result.sources_,&result.raw_g_,&result.g_,
        &result.raw_f_,&result.f_,&result.amplitudes_,&result.density_,&result.payload_,&result.common_g_}) identity.string(*s);
    for(U x:{i,j,result.i_.occupied_index,result.i_.cell,result.j_.occupied_index,result.j_.cell,n,m,t}) identity.u64(x);
    options_wire(identity,options);identity.string(policy);identity.string(real::kFloatPolicy);result.identity_=identity.finish();return result;
}
} // namespace vibeqc
