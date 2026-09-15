#include "vibeqc/periodic_correlation_real_pao_embedding.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"
#include "periodic_correlation_real_pao_embedding_internal.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "real PAO embeddings forbid fast/finite-only math"
#endif

namespace vibeqc {
namespace {
namespace local=periodic_correlation_local_detail;
namespace real=periodic_correlation_real_local_detail;
using U=std::uint64_t;
using Z=std::complex<double>;
using real::add;using real::mul;using real::Digest;using real::Sum;
using Plan=PeriodicCorrelationRealPAOEmbeddingMemoryPlan;
using Result=PeriodicCorrelationRealPAOEmbedding;
using Options=PeriodicCorrelationRealPAOEmbeddingOptions;
using Live=PeriodicCorrelationRealPAOEmbeddingLiveInventory;
using Caps=PeriodicCorrelationRealPAOEmbeddingCaps;
using Selection=PeriodicCorrelationVirtualBlockSelection;
static_assert(sizeof(Z)==16 && FLT_EVAL_METHOD==0,"real PAO embeddings require binary64 evaluation");
constexpr char policy[]="actual-retained-virtual-PAO-frames;full-BZ-1/Nk;native-cross-overlap;"
    "explicit-real-projection;no-repair;original-S-containment;original-F-projections;"
    "copied-pair-eps-only-after-original-F-gates;not-production-domain-scaling";

void limit(U value,U cap,const char* text) { if (value>cap) throw std::length_error(text); }
void positive(double value) {
    if (!std::isfinite(value) || value<=0.0)
        throw std::invalid_argument("real PAO embedding error budgets must be finite strictly positive");
}
void options_valid(const Options& o,const Live& l) {
    for (double x:{o.maximum_cross_overlap_imaginary_norm,o.maximum_embedding_gram_error,
        o.maximum_pair_metric_error,o.maximum_containment_norm,o.maximum_pair_fock_error,
        o.maximum_common_fock_error,o.maximum_embedded_fock_error}) positive(x);
    if (!l.fixed_backend_margin_bytes_per_worker)
        throw std::invalid_argument("real PAO embedding requires a positive backend margin");
}
void sha(const std::string& s) {
    if (s.size()!=64) throw std::invalid_argument("real PAO embedding requires complete native SHA receipts");
    for (char c:s) if (!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("real PAO embedding native SHA receipt is malformed");
}
bool same_selection(const Selection& a,const Selection& b) {
    return a.begin==b.begin && a.count==b.count && a.translation_cell==b.translation_cell;
}
void selection_wire(Digest& h,const Selection& s) {
    h.u64(s.begin);h.u64(s.count);h.u64(s.translation_cell);
}
struct Inputs {
    const PeriodicCorrelationAdmittedReference& ref;
    const PeriodicCorrelationRealLocalBasis& basis;
    const PeriodicCorrelationPAODomain& cd;
    const PeriodicCorrelationRealPAOSpace& cr;
    const Selection& c;
    const PeriodicCorrelationPAODomain& pd;
    const PeriodicCorrelationRealPAOSpace& pr;
    const Selection& p;
};
void metadata(const Inputs& in) {
    const auto& cs=in.cr.space();const auto& ps=in.pr.space();
    local::validate_pao(in.ref,in.cd,cs);local::validate_pao(in.ref,in.pd,ps);
    if (!in.c.count || !in.p.count || in.p.count>in.c.count)
        throw std::invalid_argument("real PAO embedding requires positive common/pair dimensions with pair <= common");
    for (const auto& entry:std::array<std::pair<const Selection*,const PeriodicCorrelationPAOSpace*>,2>{{{&in.c,&cs},{&in.p,&ps}}}) {
        const auto& s=*entry.first;const auto& space=*entry.second;
        if (s.begin>=space.retained_dimension() || s.count>space.retained_dimension()-s.begin
            || s.translation_cell>=in.ref.state().n_kpoints())
            throw std::out_of_range("real PAO embedding selection exceeds its actual native frame");
    }
    if (in.basis.state_handle()!=in.ref.state_handle() || in.basis.allocation_identity()!=in.ref.dimensions().allocation_identity
        || in.basis.memory().n_basis!=in.ref.state().n_basis() || in.basis.memory().n_cells!=in.ref.state().n_kpoints()
        || in.basis.memory().virtual_count!=in.c.count || !same_selection(in.basis.virtual_selection(),in.c))
        throw std::invalid_argument("real PAO embedding common basis state/allocation/selection differs");
    for (const auto* s:std::array<const std::string*,16>{&in.ref.state().state_identity_sha256(),
        &in.ref.state().numerical_payload_sha256(),&in.ref.dimensions().allocation_identity,
        &in.basis.identity_sha256(),&in.basis.payload_sha256(),&in.basis.local_basis_identity_sha256(),
        &in.cd.pao_domain_identity_sha256(),&in.cd.matrix_payload_sha256(),&cs.pao_space_identity_sha256(),&cs.payload_sha256(),
        &in.cr.identity_sha256(),&in.pd.pao_domain_identity_sha256(),&in.pd.matrix_payload_sha256(),
        &ps.pao_space_identity_sha256(),&ps.payload_sha256(),&in.pr.identity_sha256()}) sha(*s);
    if (!std::equal(in.basis.virtual_domain_identity_ascii().begin(),in.basis.virtual_domain_identity_ascii().end(),
            in.cd.pao_domain_identity_sha256().begin())
        || !std::equal(in.basis.virtual_space_identity_ascii().begin(),in.basis.virtual_space_identity_ascii().end(),
            cs.pao_space_identity_sha256().begin()))
        throw std::invalid_argument("real PAO embedding common basis does not certify these physical PAO frames");
    // These getters also reject consumed or replaced compact payloads.
    (void)in.basis.f_oo_data();(void)in.basis.f_vv_data();(void)in.basis.f_ov_data();
    (void)in.basis.occupied_indices_data();
    (void)cs.coefficients_data();(void)cs.energies_data();(void)cs.overlap_eigenvalues_data();
    (void)ps.coefficients_data();(void)ps.energies_data();(void)ps.overlap_eigenvalues_data();
    if (in.cr.memory().compact.output_numerical_bytes!=cs.memory().output_numerical_bytes
        || in.pr.memory().compact.output_numerical_bytes!=ps.memory().output_numerical_bytes)
        throw std::logic_error("real PAO embedding real-wrapper numeric inventory differs");
}
void domain_payload(Digest& h,const PeriodicCorrelationPAODomain& d) {
    h.string(d.pao_domain_identity_sha256());h.string(d.domain_index_sha256());
    h.string(d.matrix_payload_sha256());h.u64(d.domain_dimension());
    for (U a=0;a<d.domain_dimension();++a) {
        const auto column=d.column(a);h.u64(column.cell);h.u64(column.ao);
        for (U b=0;b<d.domain_dimension();++b) {h.complex(d.overlap(a,b));h.complex(d.fock(a,b));}
    }
}
void space_payload(Digest& h,const PeriodicCorrelationPAOSpace& s) {
    h.string(s.pao_space_identity_sha256());h.string(s.payload_sha256());
    h.u64(s.domain_dimension());h.u64(s.retained_dimension());
    for (U a=0;a<s.domain_dimension();++a) {
        h.real(s.overlap_eigenvalue(a));
        for (U b=0;b<s.retained_dimension();++b) h.complex(s.coefficient(a,b));
    }
    for (U b=0;b<s.retained_dimension();++b) h.real(s.energy(b));
}
std::string source_payload(const Inputs& in) {
    // Full finite source scans occur only AFTER the enclosing work/memory
    // gate. No whole state or domain/space payload is copied or cached.
    Digest h("vibeqc.periodic.correlation.real-pao-embedding.sources");
    const auto& s=in.ref.state();h.string(s.state_identity_sha256());h.string(s.numerical_payload_sha256());
    h.string(in.ref.dimensions().allocation_identity);
    for (int x:s.mesh()) h.u64(static_cast<U>(x));
    for (int x:s.is_shift()) h.u64(static_cast<U>(x));
    for (U a=0;a<3;++a) for (U b=0;b<3;++b) h.real(s.reciprocal_lattice()(a,b));
    h.u64(s.n_kpoints());h.u64(s.n_basis());h.u64(s.n_effective_orbitals());
    h.real(s.reference_energy_per_cell());h.real(s.uniform_weight());
    for (U k=0;k<s.n_kpoints();++k) {
        for (U a=0;a<3;++a) h.real(s.kpoint_cartesian(k)[a]);h.real(s.weight(k));
        const auto& overlap=s.overlap(k);const auto& fock=s.fock(k);const auto& coeff=s.coefficients(k);
        for (U a=0;a<s.n_basis();++a) {
            for (U b=0;b<s.n_basis();++b) {h.complex(overlap(a,b));h.complex(fock(a,b));}
            for (U b=0;b<s.n_effective_orbitals();++b) h.complex(coeff(a,b));
        }
        for (U a=0;a<s.n_effective_orbitals();++a) {
            h.real(s.orbital_energies(k)[a]);h.real(s.occupations(k)[a]);
            h.u64(s.frozen_core_mask(k)[a]);h.u64(s.correlated_occupied_mask(k)[a]);h.u64(s.virtual_mask(k)[a]);
        }
    }
    domain_payload(h,in.cd);if (&in.cd!=&in.pd) domain_payload(h,in.pd);
    space_payload(h,in.cr.space());if (&in.cr.space()!=&in.pr.space()) space_payload(h,in.pr.space());
    h.string(in.cr.identity_sha256());h.string(in.pr.identity_sha256());
    h.string(in.basis.identity_sha256());h.string(in.basis.payload_sha256());
    const auto& b=in.basis.memory();const auto* indices=in.basis.occupied_indices_data();
    for (U a=0;a<mul(2,b.occupied_count);++a) h.u64(indices[a]);
    const auto* F=in.basis.f_oo_data();for (U a=0;a<b.retained_fock_bytes/8;++a) h.real(F[a]);
    selection_wire(h,in.c);selection_wire(h,in.p);return h.finish();
}
std::string frame(const Inputs& in,bool common) {
    if (!common) return periodic_correlation_real_pao_embedding_detail::pair_frame_identity(in.ref,in.pd,in.pr,in.p);
    Digest h("vibeqc.periodic.correlation.real-pao-embedding.frame");
    const auto& d=common ? in.cd : in.pd;const auto& r=common ? in.cr : in.pr;
    h.string(in.ref.state().state_identity_sha256());h.string(in.ref.dimensions().allocation_identity);
    h.string(d.pao_domain_identity_sha256());h.string(r.space().pao_space_identity_sha256());h.string(r.identity_sha256());
    selection_wire(h,common ? in.c : in.p);
    // The physical pair frame is independent of the common frame; only the
    // common receipt additionally binds the actual selected integral basis.
    h.u64(common);if (common) {h.string(in.basis.identity_sha256());h.string(in.basis.payload_sha256());}
    return h.finish();
}
void gate(double value,double cap,const char* text) { if (value>cap) throw std::invalid_argument(text); }
} // namespace

Plan plan_periodic_correlation_real_pao_embedding(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicCorrelationPAODomain& cd,
    const PeriodicCorrelationRealPAOSpace& cr,const Selection& c,const PeriodicCorrelationPAODomain& pd,
    const PeriodicCorrelationRealPAOSpace& pr,const Selection& p,const Options& options,const Live& live,const Caps& caps) {
    real::float_environment();options_valid(options,live);
    const Inputs in{ref,basis,cd,cr,c,pd,pr,p};metadata(in);
    Plan m;m.n_cells=ref.state().n_kpoints();m.n_basis=ref.state().n_basis();
    m.common_dimension=c.count;m.pair_dimension=p.count;
    limit(c.count,caps.maximum_common_dimension,"real PAO embedding common dimension cap");
    limit(p.count,caps.maximum_pair_dimension,"real PAO embedding pair dimension cap");
    m.overlap=plan_periodic_correlation_pao_space_overlap(ref,cd,cr.space(),c,pd,pr.space(),p);
    m.unique_domain_owners=m.overlap.unique_domain_owners;m.unique_space_owners=m.overlap.unique_space_owners;
    m.unique_real_wrapper_owners=&cr==&pr ? 1 : 2;
    m.live_domain_bytes=m.overlap.live_domain_bytes;m.live_space_bytes=m.overlap.live_space_bytes;
    const auto& bm=basis.memory();const U o=bm.occupied_count,n=c.count,r=p.count,N=m.n_basis,K=m.n_cells;
    const U expected_fock=mul(8,add(add(mul(o,o),mul(n,n)),mul(o,n)));
    m.live_basis_bytes=add(mul(16,o),expected_fock);
    if (bm.retained_fock_bytes!=expected_fock || bm.retained_index_bytes!=mul(16,o)
        || bm.retained_output_bytes!=m.live_basis_bytes)
        throw std::logic_error("real PAO embedding full common-basis inventory differs");
    m.complete_borrowed_numerical_bytes=add(m.live_basis_bytes,add(m.live_domain_bytes,m.live_space_bytes));
    const U nr=mul(n,r);m.output_numerical_bytes=mul(8,add(nr,r));
    m.overlap_phase_bytes=m.overlap.peak_owned_numerical_bytes;
    m.conversion_phase_bytes=add(m.overlap.output_bytes,m.output_numerical_bytes);
    m.physical_audit_phase_bytes=add(m.output_numerical_bytes,mul(144,N));
    m.peak_owned_numerical_bytes=std::max({m.overlap_phase_bytes,m.conversion_phase_bytes,m.physical_audit_phase_bytes});
    // Compact objects reside inline in their real wrappers. Count wrapper
    // objects once and their separate compact/string payload allocations by
    // actual object address, never by an equal content receipt.
    m.control_storage_reservation_bytes=131072U+3U*sizeof(Plan)+2U*sizeof(Result)+2U*sizeof(Options)
        +2U*sizeof(Live)+2U*sizeof(Caps)+sizeof(PeriodicCorrelationAdmittedReference)
        +sizeof(PeriodicCorrelationRealLocalBasis)+4U*65U+4U*sizeof(Digest)+12U*sizeof(std::vector<Z>)
        +2U*sizeof(PeriodicCorrelationPAOSpaceOverlap)+20U*65U;
    m.control_storage_reservation_bytes=add(m.control_storage_reservation_bytes,
        add(mul(m.unique_domain_owners,sizeof(PeriodicCorrelationPAODomain)+4U*65U),
            add(mul(m.unique_real_wrapper_owners,sizeof(PeriodicCorrelationRealPAOSpace)+65U),
                add(mul(m.unique_space_owners,6U*65U),live.other_live_control_bytes_per_worker))));
    const auto& dims=ref.dimensions();const auto& budget=ref.budget();
    m.replicas_per_node=mul(budget.mpi_ranks,budget.workers_per_rank);
    if (!m.replicas_per_node) throw std::invalid_argument("real PAO embedding requires positive native replicas");
    m.reference_base_node_bytes=add(add(dims.external_bytes,dims.shared_bytes),
        mul(budget.mpi_ranks,add(dims.per_rank_bytes,dims.localization_window_bytes_per_rank)));
    m.per_worker_inventoried_bytes=add(m.complete_borrowed_numerical_bytes,
        add(m.peak_owned_numerical_bytes,add(m.control_storage_reservation_bytes,
            add(live.other_live_numerical_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))));
    m.required_node_memory_bytes=add(m.reference_base_node_bytes,mul(m.replicas_per_node,m.per_worker_inventoried_bytes));
    const U neff=ref.state().n_effective_orbitals();
    const U state_lanes=add(64,mul(K,add(add(mul(4,mul(N,N)),mul(2,mul(N,neff))),add(mul(5,neff),4))));
    const U source_lanes=add(state_lanes,add(m.complete_borrowed_numerical_bytes/8,4096));
    // Two complete payload-validation/hash passes plus all fixed receipt
    // parsing. SHA and compensated/outward scalar costs are conservative
    // abstract work units, not FLOPs or a performance claim.
    m.source_payload_validation_work_units=mul(512,source_lanes);
    const U common_expansion=add(mul(N,cd.domain_dimension()),add(mul(N,N),add(mul(2,mul(N,neff)),mul(4,N))));
    const U pair_expansion=add(mul(N,pd.domain_dimension()),add(mul(N,N),add(mul(2,mul(N,neff)),mul(4,N))));
    const U expansions=add(mul(n,common_expansion),mul(2,pair_expansion));
    const U contractions=add(mul(16,mul(N,N)),add(mul(16,mul(N,n)),mul(64,N)));
    m.physical_audit_work_units=mul(256,add(mul(mul(K,mul(r,r)),add(expansions,contractions)),
        add(mul(mul(n,n),mul(r,r)),add(mul(n,mul(r,r)),nr))));
    m.work_units=add(mul(128,m.overlap.work_units),add(m.source_payload_validation_work_units,
        add(m.physical_audit_work_units,mul(256,add(m.output_numerical_bytes/8,4096)))));
    for (U bytes:{m.peak_owned_numerical_bytes,m.complete_borrowed_numerical_bytes,m.control_storage_reservation_bytes,
        mul(8,source_lanes),mul(144,N)}) real::extent(bytes);
    if (nr>std::vector<double>().max_size() || r>std::vector<double>().max_size() || mul(9,N)>std::vector<Z>().max_size())
        throw std::length_error("real PAO embedding native vector extent exceeded");
    limit(m.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"real PAO embedding owned numerical cap");
    limit(m.control_storage_reservation_bytes,caps.maximum_control_storage_bytes_per_worker,"real PAO embedding control cap");
    limit(m.per_worker_inventoried_bytes,caps.maximum_per_worker_inventoried_bytes,"real PAO embedding worker cap");
    limit(m.required_node_memory_bytes,caps.maximum_node_inventoried_bytes,"real PAO embedding node cap");
    limit(m.required_node_memory_bytes,budget.memory_limit_bytes,"real PAO embedding admitted reference node cap");
    limit(m.work_units,caps.maximum_work_units,"real PAO embedding work cap");
    return m;
}

namespace {
Z quadratic(const PeriodicMeanFieldComplexMatrix& M,const Z* left,const Z* right,U N,Z* scratch) {
    for (U a=0;a<N;++a) {
        Sum s;for (U b=0;b<N;++b) s.include(M(a,b)*right[b]);scratch[a]=s.value();
    }
    Sum result;for (U a=0;a<N;++a) result.include(std::conj(left[a])*scratch[a]);return result.value();
}
void physical_audits(const Inputs& in,const Plan& m,const Options& o,const double* X,const double* eps,
    PeriodicCorrelationRealPAOEmbeddingDiagnostics& d) {
    const U N=m.n_basis,n=m.common_dimension,r=m.pair_dimension,K=m.n_cells;
    std::vector<Z> storage(static_cast<std::size_t>(mul(9,N)));
    Z* pa=storage.data();Z* pb=pa+N;Z* ea=pb+N;Z* eb=ea+N;
    Z* ac=eb+N;Z* bc=ac+N;Z* col=bc+N;Z* scratch=col+N;
    const double w=real::finite(1.0/static_cast<double>(K));
    const double w_upper=real::up(w);
    double gram_error=0,pair_fock_error=0,common_fock_error=0,embedded_fock_error=0,containment_absolute=0;
    Sum containment;
    const auto& state=in.ref.state();const double* Fcommon=in.basis.f_vv_data();
    for (U a=0;a<r;++a) for (U b=0;b<r;++b) {
        Sum pair_gram,pair_fock,embedded_fock;
        for (U k=0;k<K;++k) {
            local::fill_virtual_columns(state,in.pd,in.pr.space(),in.p.begin+a,1,in.p.translation_cell,k,pa,scratch);
            local::fill_virtual_columns(state,in.pd,in.pr.space(),in.p.begin+b,1,in.p.translation_cell,k,pb,scratch);
            std::fill(ea,ea+N,Z{});std::fill(eb,eb+N,Z{});std::fill(ac,ac+N,Z{});std::fill(bc,bc+N,Z{});
            for (U l=0;l<n;++l) {
                local::fill_virtual_columns(state,in.cd,in.cr.space(),in.c.begin+l,1,in.c.translation_cell,k,col,scratch);
                for (U mu=0;mu<N;++mu) {
                    real::accumulate(col[mu]*X[l*r+a],ea[mu],ac[mu]);
                    real::accumulate(col[mu]*X[l*r+b],eb[mu],bc[mu]);
                }
            }
            for (U mu=0;mu<N;++mu) {ea[mu]=real::finite(ea[mu]+ac[mu]);eb[mu]=real::finite(eb[mu]+bc[mu]);}
            pair_gram.include(quadratic(state.overlap(k),pa,pb,N,scratch));
            pair_fock.include(quadratic(state.fock(k),pa,pb,N,scratch));
            embedded_fock.include(quadratic(state.fock(k),ea,eb,N,scratch));
            if (a==b) {
                for (U mu=0;mu<N;++mu) col[mu]=real::finite(pa[mu]-ea[mu]);
                containment.include(quadratic(state.overlap(k),col,col,N,scratch));
                double absolute=0;
                for (U mu=0;mu<N;++mu) for (U nu=0;nu<N;++nu)
                    absolute=real::add_up(absolute,real::mul_up(real::norm_up(col[mu]),
                        real::mul_up(real::norm_up(state.overlap(k)(mu,nu)),real::norm_up(col[nu]))));
                containment_absolute=real::add_up(containment_absolute,real::mul_up(w_upper,absolute));
            }
        }
        const Z S=real::finite(pair_gram.value()*w),F=real::finite(pair_fock.value()*w),E=real::finite(embedded_fock.value()*w);
        Sum projected;
        for (U l=0;l<n;++l) for (U q=0;q<n;++q)
            projected.include(real::finite(real::finite(X[l*r+a]*Fcommon[l*n+q])*X[q*r+b]));
        const Z XF=projected.value();
        real::norm_difference(gram_error,S,Z(a==b ? 1.0 : 0.0,0));
        real::norm_difference(pair_fock_error,F,Z(a==b ? eps[a] : 0.0,0));
        real::norm_difference(common_fock_error,XF,F);
        real::norm_difference(embedded_fock_error,E,XF);
    }
    const Z signed_metric=real::finite(containment.value()*w);
    d.containment_metric_quadratic_real=signed_metric.real();d.containment_metric_quadratic_imaginary=signed_metric.imag();
    d.containment_s_absolute_upper_bound=real::sqrt_up(containment_absolute);
    d.pair_metric_frobenius_upper_bound=real::sqrt_up(gram_error);
    d.pair_original_fock_frobenius_upper_bound=real::sqrt_up(pair_fock_error);
    d.common_projected_fock_frobenius_upper_bound=real::sqrt_up(common_fock_error);
    d.embedded_original_fock_frobenius_upper_bound=real::sqrt_up(embedded_fock_error);
    gate(d.pair_metric_frobenius_upper_bound,o.maximum_pair_metric_error,"real PAO embedding pair original-S metric gate");
    gate(d.containment_s_absolute_upper_bound,o.maximum_containment_norm,"real PAO embedding pair is not contained in common physical space");
    gate(d.pair_original_fock_frobenius_upper_bound,o.maximum_pair_fock_error,"real PAO embedding pair eps do not match original projected Fock");
    gate(d.common_projected_fock_frobenius_upper_bound,o.maximum_common_fock_error,"real PAO embedding common-to-pair original Fock consistency gate");
    gate(d.embedded_original_fock_frobenius_upper_bound,o.maximum_embedded_fock_error,"real PAO embedding original embedded/common Fock gate");
}
} // namespace

void Result::require_live() const {
    if (!state_ || !memory_.common_dimension || !memory_.pair_dimension
        || coefficients_.size()!=mul(memory_.common_dimension,memory_.pair_dimension)
        || energies_.size()!=memory_.pair_dimension || identity_.size()!=64)
        throw std::logic_error("real PAO embedding result is consumed or incomplete");
}
const double* Result::coefficients_data() const { require_live();return coefficients_.data(); }
const double* Result::energies_data() const { require_live();return energies_.data(); }
double Result::coefficient(std::size_t c,std::size_t p) const {
    require_live();if (c>=memory_.common_dimension || p>=memory_.pair_dimension)
        throw std::out_of_range("real PAO embedding coefficient out of range");
    return coefficients_[c*memory_.pair_dimension+p];
}
double Result::energy(std::size_t p) const {
    require_live();if (p>=memory_.pair_dimension) throw std::out_of_range("real PAO embedding energy out of range");return energies_[p];
}

Result make_periodic_correlation_real_pao_embedding(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicCorrelationPAODomain& cd,
    const PeriodicCorrelationRealPAOSpace& cr,const Selection& c,const PeriodicCorrelationPAODomain& pd,
    const PeriodicCorrelationRealPAOSpace& pr,const Selection& p,const Options& supplied_options,
    const Live& supplied_live,const Caps& supplied_caps) {
    const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto m=plan_periodic_correlation_real_pao_embedding(ref,basis,cd,cr,c,pd,pr,p,options,live,caps);
    const Inputs in{ref,basis,cd,cr,c,pd,pr,p};Result result;
    result.memory_=m;result.options_=options;result.common_=c;result.pair_=p;
    result.sources_=source_payload(in);result.basis_=basis.identity_sha256();
    result.common_frame_=frame(in,true);result.pair_frame_=frame(in,false);
    const U n=m.common_dimension,r=m.pair_dimension;
    {
        auto overlap=make_periodic_correlation_pao_space_overlap(ref,cd,cr.space(),c,pd,pr.space(),p,
            {m.overlap.peak_owned_numerical_bytes,m.overlap.work_units});
        result.overlap_=overlap.identity_sha256();
        result.coefficients_.resize(static_cast<std::size_t>(mul(n,r)));
        result.energies_.resize(static_cast<std::size_t>(r));
        double imaginary=0;
        for (U a=0;a<mul(n,r);++a) {
            const auto z=real::finite(overlap.data()[a]);real::norm_lane(imaginary,std::abs(z.imag()));result.coefficients_[a]=z.real();
        }
        result.diagnostics_.cross_overlap_imaginary_frobenius_upper_bound=real::sqrt_up(imaginary);
        gate(result.diagnostics_.cross_overlap_imaginary_frobenius_upper_bound,options.maximum_cross_overlap_imaginary_norm,
            "real PAO embedding imaginary cross overlap exceeds explicit projection budget");
        for (U a=0;a<r;++a) result.energies_[a]=real::finite(pr.space().energy(p.begin+a));
    }
    double gram=0;
    for (U a=0;a<r;++a) for (U b=0;b<r;++b) {
        Sum sum;for (U l=0;l<n;++l) sum.include(real::finite(result.coefficients_[l*r+a]*result.coefficients_[l*r+b]));
        real::norm_difference(gram,sum.value(),Z(a==b ? 1.0 : 0.0,0));
    }
    result.diagnostics_.embedding_gram_frobenius_upper_bound=real::sqrt_up(gram);
    gate(result.diagnostics_.embedding_gram_frobenius_upper_bound,options.maximum_embedding_gram_error,
        "real PAO embedding X^T X gate rejects incomplete or nonorthonormal pair embedding");
    physical_audits(in,m,options,result.coefficients_.data(),result.energies_.data(),result.diagnostics_);
    metadata(in);if (source_payload(in)!=result.sources_)
        throw std::invalid_argument("real PAO embedding native source changed during construction");
    real::float_environment();result.state_=ref.state_handle();result.allocation_=ref.dimensions().allocation_identity;
    Digest payload("vibeqc.periodic.correlation.real-pao-embedding.payload");payload.u64(n);payload.u64(r);
    for (double x:result.coefficients_) payload.real(x);for (double x:result.energies_) payload.real(x);
    result.payload_=payload.finish();
    Digest identity("vibeqc.periodic.correlation.real-pao-embedding.identity");
    for (const auto* s:std::array<const std::string*,8>{&ref.state().state_identity_sha256(),&result.allocation_,&result.basis_,
        &result.common_frame_,&result.pair_frame_,&result.overlap_,&result.sources_,&result.payload_}) identity.string(*s);
    for (double x:{options.maximum_cross_overlap_imaginary_norm,options.maximum_embedding_gram_error,
        options.maximum_pair_metric_error,options.maximum_containment_norm,options.maximum_pair_fock_error,
        options.maximum_common_fock_error,options.maximum_embedded_fock_error}) identity.real(x);
    identity.string(policy);identity.string(real::kFloatPolicy);result.identity_=identity.finish();return result;
}

} // namespace vibeqc
