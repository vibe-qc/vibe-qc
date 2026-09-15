#include "vibeqc/periodic_gaussian_gram_pair_pnos.hpp"

#include <cfloat>
#include <optional>
#include "periodic_correlation_real_local_internal.hpp"
#include "periodic_correlation_real_pao_embedding_internal.hpp"

namespace vibeqc {
namespace {
namespace real=periodic_correlation_real_local_detail;
namespace local=periodic_correlation_local_detail;
using U=std::uint64_t;
using C=std::complex<double>;
using real::add;
using real::mul;
using real::Digest;
using Ref=PeriodicCorrelationAdmittedReference;
using Basis=PeriodicCorrelationRealLocalBasis;
using Geometry=PeriodicGaussianPairDomainGeometry;
using Result=PeriodicGaussianGramPairPNOResult;
using Plan=PeriodicGaussianGramPairPNOPlan;
using Config=PeriodicGaussianGramPairPNOConfig;
using Options=PeriodicGaussianGramPairPNOOptions;
using Live=PeriodicGaussianGramPairPNOLiveInventory;
using Caps=PeriodicGaussianGramPairPNOCaps;
using Gram=PeriodicGaussianDensityGramBlock;
constexpr char policy[]="DirectPAOGram;actual-unique-occupied-plus-pair-PAO-basis;"
    "Nejad2025-Eqs37-40-no-diagonal-density-division;local-original-F;"
    "generation-D-retained;original-S-export-C=XD;GPNO=DtGD;"
    "no-common-provider-rows-or-common-G-replay;no-energy-multiplicity;not-production-DLPNO";
static_assert(sizeof(C)==16 && sizeof(U)==8 && FLT_EVAL_METHOD==0,
    "Gram pair PNOs require binary64 and uint64 storage");
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gram pair PNOs forbid fast/finite-only math"
#endif
void positive(U x) { if(!x) throw std::invalid_argument("Gram pair PNO resources must be positive explicit controls"); }
void limit(U x,U cap,const char* message) { positive(cap);if(x>cap) throw std::length_error(message); }
U subtract(U a,U b) { if(a<b) throw std::logic_error("Gram pair PNO inventory subtraction underflows");return a-b; }
void nonnegative(double x) {
    if(!std::isfinite(x) || x<0) throw std::invalid_argument("Gram pair PNO correction budgets must be finite nonnegative");
}
void positive_real(double x) {
    if(!std::isfinite(x) || x<=0) throw std::invalid_argument("Gram pair PNO tolerances must be finite positive");
}
void sha(const std::string& s) {
    if(s.size()!=64) throw std::invalid_argument("Gram pair PNO source receipt has invalid extent");
    for(char c:s) if(!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("Gram pair PNO source receipt is malformed");
}
bool same_label(PeriodicCorrelationPlacedOccupied a,PeriodicCorrelationPlacedOccupied b) {
    return a.occupied_index==b.occupied_index && a.cell==b.cell;
}
void metric_caps(const PeriodicGaussianMetricCaps& c) {
    for(U x:{c.maximum_owned_numeric_bytes,c.maximum_per_replica_inventoried_bytes,
        c.maximum_node_inventoried_bytes,c.maximum_candidate_evaluations,c.maximum_work_units}) positive(x);
}
void controls(const Config& c,const Options& o,const Live& live,const Caps& caps) {
    positive(c.gram.auxiliary_block);positive(c.gram.panel.ao_pair_block);
    positive(live.fixed_backend_margin_bytes_per_worker);
    real::tolerance_pair(o.local_basis.coefficient_tr_absolute_tolerance,o.local_basis.coefficient_tr_relative_tolerance);
    real::tolerance_pair(o.local_basis.orthonormality_absolute_tolerance,o.local_basis.orthonormality_relative_tolerance);
    real::tolerance_pair(o.local_basis.fock_absolute_tolerance,o.local_basis.fock_relative_tolerance);
    positive_real(o.local_basis.maximum_fock_projection_error);
    real::tolerance_pair(o.gram.reversal_absolute_tolerance,o.gram.reversal_relative_tolerance);
    real::tolerance_pair(o.gram.conjugacy_absolute_tolerance,o.gram.conjugacy_relative_tolerance);
    real::tolerance_pair(o.gram.self_q_absolute_tolerance,o.gram.self_q_relative_tolerance);
    positive_real(o.gram.maximum_eri_projection_error);positive_real(o.gram.maximum_scalar_roundoff_error);
    nonnegative(o.pno.pno.occupation_cutoff);positive_real(o.pno.pno.denominator_floor);
    nonnegative(o.pno.pno.maximum_initial_fvv_offdiagonal_norm);
    positive_real(o.pno.pno.semicanonical_orthonormality_tolerance);
    if(o.pno.pno.semicanonical_orthonormality_tolerance>=1)
        throw std::invalid_argument("Gram pair PNO orthonormality tolerance must be below one");
    nonnegative(o.pno.maximum_diagonal_exchange_projection_norm);
    nonnegative(o.pno.maximum_fock_symmetry_projection_norm);
    nonnegative(o.maximum_retained_diagonal_projection_norm);
    positive_real(o.maximum_occupied_fock_difference);
    positive_real(o.pno.maximum_exported_gram_error);positive_real(o.pno.maximum_exported_fock_error);
    positive_real(o.pno.maximum_exported_subspace_error);
    for(const auto& j:{o.pno.pno.pno_eigensolver,o.pno.pno.semicanonical_eigensolver}) {
        positive(j.max_sweeps);positive_real(j.relative_offdiagonal_tolerance);
        if(j.relative_offdiagonal_tolerance>=1) throw std::invalid_argument("Gram pair PNO eigensolver tolerance must be below one");
    }
    for(U x:{caps.maximum_common_dimension,caps.maximum_generation_dimension,caps.maximum_owned_numerical_bytes,
        caps.maximum_control_storage_bytes,caps.maximum_worker_bytes,caps.maximum_node_bytes,caps.maximum_work_units,
        caps.maximum_gram_control_storage_bytes,caps.local_basis.maximum_owned_numerical_bytes,caps.local_basis.maximum_work_units,
        caps.gram.maximum_left_density_count,caps.gram.maximum_right_density_count,caps.gram.maximum_output_elements,
        caps.gram.maximum_factor_panels,caps.gram.maximum_tile_calls,caps.gram.maximum_image_candidate_evaluations,
        caps.gram.maximum_leaf_control_bytes,caps.gram.panel.maximum_tile_calls,caps.gram.panel.maximum_image_candidate_evaluations}) positive(x);
    metric_caps(caps.gram.resources);metric_caps(caps.gram.metric);metric_caps(caps.gram.panel.resources);
    metric_caps(caps.gram.panel.tile.resources);
}
void metadata(const PeriodicGaussianRHFResult& hf,const Ref& ref,const PeriodicCorrelationWannier& w,
    const Basis& b,const Geometry& geometry) {
    real::float_environment();local::validate_wannier(ref,w);
    const auto& d=geometry.domain();const auto& s=geometry.real_space().space();const auto& e=geometry.embedding();
    local::validate_pao(ref,d,s);
    const auto& ctx=hf.context_handle();const auto& bm=b.memory();const auto& em=e.memory();
    if(!hf.converged() || !hf.matched_finite_gaussian_hf_source() || !ctx
        || hf.state_handle()!=ref.state_handle() || geometry.state_handle()!=ref.state_handle()
        || geometry.context_handle()!=ctx || b.state_handle()!=ref.state_handle()
        || b.allocation_identity()!=ref.dimensions().allocation_identity
        || geometry.allocation_identity()!=ref.dimensions().allocation_identity
        || b.contract_version()!=kPeriodicCorrelationRealLocalBasisContractVersion
        || geometry.basis_identity_sha256()!=b.identity_sha256()
        || e.common_basis_identity_sha256()!=b.identity_sha256()
        || geometry.hf_reference_source_identity_sha256()!=hf.reference_source_identity_sha256())
        throw std::invalid_argument("Gram pair PNO requires exact actual HF/common basis/pair geometry owners");
    if(!bm.occupied_count || !bm.virtual_count || bm.orbital_count!=add(bm.occupied_count,bm.virtual_count)
        || bm.n_cells!=ref.state().n_kpoints() || bm.n_basis!=ref.state().n_basis()
        || geometry.occupied_slot_i()>=bm.occupied_count || geometry.occupied_slot_j()>=bm.occupied_count
        || em.common_dimension!=bm.virtual_count || !em.pair_dimension || em.pair_dimension>bm.virtual_count
        || e.state_handle()!=ref.state_handle() || e.allocation_identity()!=ref.dimensions().allocation_identity
        || ctx->mesh().mesh()!=ref.state().mesh() || ctx->mesh().is_shift()!=ref.state().is_shift()
        || ctx->inventory().ao.function_count!=bm.n_basis
        || ctx->inventory().auxiliary.function_count!=ref.dimensions().n_auxiliary)
        throw std::invalid_argument("Gram pair PNO source dimensions or pair occupied slots differ");
    const auto a=b.virtual_selection(),c=e.common_selection(),p=e.pair_selection();
    if(a.begin!=c.begin || a.count!=c.count || a.translation_cell!=c.translation_cell
        || p.count!=em.pair_dimension || p.begin>=s.retained_dimension()
        || p.count>s.retained_dimension()-p.begin || p.translation_cell>=bm.n_cells)
        throw std::invalid_argument("Gram pair PNO common/pair virtual selection differs from embedding");
    (void)b.occupied_indices_data();(void)b.f_oo_data();(void)b.f_vv_data();(void)b.f_ov_data();
    (void)e.coefficients_data();(void)e.energies_data();
    (void)geometry.retained_numerical_bytes();(void)geometry.retained_control_storage_bytes();
    for(const auto* receipt:std::array<const std::string*,10>{&b.identity_sha256(),&b.payload_sha256(),
        &b.local_basis_identity_sha256(),&geometry.identity_sha256(),&geometry.builder_identity_sha256(),
        &e.identity_sha256(),&e.payload_sha256(),&e.common_frame_identity_sha256(),&e.pair_frame_identity_sha256(),
        &ref.dimensions().allocation_identity}) sha(*receipt);
}
U fixed_controls() {
    // Fixed logical allowance includes caller local-index array[4], every
    // scalar/codec/control object and allocator bookkeeping, not exact RSS.
    return 131072+3*sizeof(Plan)+2*sizeof(Result)+2*sizeof(Options)+2*sizeof(Caps)+2*sizeof(Live)+2*sizeof(Config)
        +sizeof(Basis)+sizeof(Ref)+sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicGaussianSourceContext)
        +sizeof(PeriodicCorrelationWannier)+2*sizeof(Gram)+3*sizeof(PeriodicGaussianDensityGramPlan)
        +2*sizeof(RestrictedPairMP2Result)+2*sizeof(RestrictedPairPNOResult)+2*sizeof(RestrictedPairSemicanonicalResult)
        +12*sizeof(Digest)+10*sizeof(std::vector<double>)+64*65;
}
double product(double a,double b) {
    const double x=real::finite(a*b);
    if(x==0 && a!=0 && b!=0) throw std::overflow_error("Gram pair PNO contraction product underflows");
    return x;
}
void accumulate(double x,double& sum,double& correction) {
    real::finite(x);const double next=real::finite(sum+x);
    correction=real::finite(correction+(std::abs(sum)>=std::abs(x)?(sum-next)+x:(x-next)+sum));sum=next;
}
struct Sum {
    double sum=0,correction=0;
    void include(double x) {accumulate(x,sum,correction);}
    double value() const {return real::finite(sum+correction);}
};
double average(double a,double b) {
    if(a==b) return a;
    int exponent=0;std::frexp(std::max(std::abs(a),std::abs(b)),&exponent);
    const double x=std::scalbn(a,-exponent),y=std::scalbn(b,-exponent);
    if(std::scalbn(x,exponent)!=a || std::scalbn(y,exponent)!=b)
        throw std::overflow_error("Gram pair PNO transpose projection loses input range");
    return real::finite(std::scalbn(x+y,exponent-1));
}
void symmetric_projection(std::vector<double>& G,U n,double cap,double& maximum,double& norm) {
    double squared=0;
    for(U a=0;a<n;++a) for(U b=a+1;b<n;++b) {
        const double x=G[a*n+b],y=G[b*n+a],z=average(x,y);
        maximum=std::max(maximum,real::difference_up(x,y));
        real::norm_lane(squared,real::difference_up(x,z));real::norm_lane(squared,real::difference_up(y,z));
    }
    norm=real::sqrt_up(squared);
    if(norm>cap) throw std::invalid_argument("Gram pair PNO diagonal projection exceeds explicit Frobenius budget");
    for(U a=0;a<n;++a) for(U b=a+1;b<n;++b) G[a*n+b]=G[b*n+a]=average(G[a*n+b],G[b*n+a]);
}
double offdiagonal(const double* F,U n) {
    double norm=0;
    for(U a=0;a<n;++a) for(U b=0;b<n;++b) {
        const double x=real::finite(F[a*n+b]);
        if(x!=F[b*n+a]) throw std::invalid_argument("Gram pair PNO original real Fock is not exactly symmetric");
        if(a!=b) real::norm_lane(norm,std::abs(x));
    }
    return real::sqrt_up(norm);
}
std::string array_hash(const char* name,const double* data,U elements,U n) {
    Digest h(name);h.u64(n);h.u64(elements);for(U at=0;at<elements;++at) h.real(data[at]);return h.finish();
}
PeriodicGaussianSourceCaps exact_basis_caps(const PeriodicGaussianSourceContext& c) {
    const auto& v=c.inventory();const auto& a=v.ao;const auto& b=v.auxiliary;
    return {sizeof(PeriodicGaussianSourceContext),c.mesh().size(),add(a.shell_count,b.shell_count),
        add(a.contraction_count,b.contraction_count),add(add(a.exponent_count,a.coefficient_count),add(b.exponent_count,b.coefficient_count)),
        add(a.content_wire_bytes,b.content_wire_bytes),v.combined_borrowed_active_numeric_bytes,v.work_units_upper_bound};
}
std::string input_payload(const PeriodicGaussianRHFResult& hf,const Ref& ref,const BasisSet& ao,const BasisSet& aux,
    const PeriodicCorrelationWannier& w,const C* gauges,std::size_t gauge_count,const Basis& basis,const Geometry& geometry) {
    metadata(hf,ref,w,basis,geometry);
    hf.context_handle()->verify_bases(ao,aux,exact_basis_caps(*hf.context_handle()));
    const auto gauge=local::validate_gauges(w,gauges,gauge_count);
    // Reproduce the SAME native common-basis selection/lineage wire, using
    // its sealed component hashes rather than borrowing common geometry.
    // Foo agreement alone cannot distinguish sign changes or rotations in
    // degenerate occupied spaces. No caller-labelled lineage is accepted.
    const auto selected=basis.virtual_selection();
    Digest selection("vibeqc.periodic.correlation.local-orbital-factors.selection");
    selection.u64(basis.memory().occupied_count);
    for(U a=0;a<mul(2,basis.memory().occupied_count);++a) selection.u64(basis.occupied_indices_data()[a]);
    selection.u64(selected.begin);selection.u64(selected.count);selection.u64(selected.translation_cell);
    Digest lineage("vibeqc.periodic.correlation.local-orbital-factors.basis");
    lineage.string(ref.state().state_identity_sha256());lineage.string(ref.state().calculation_identity());
    lineage.string(ref.dimensions().allocation_identity);lineage.string(selection.finish());
    lineage.string(w.wannier_identity_sha256());lineage.string(gauge);
    lineage.string(std::string(basis.virtual_domain_identity_ascii().begin(),basis.virtual_domain_identity_ascii().end()));
    lineage.string(std::string(basis.virtual_space_identity_ascii().begin(),basis.virtual_space_identity_ascii().end()));
    if(lineage.finish()!=basis.local_basis_identity_sha256())
        throw std::invalid_argument("Gram pair PNO supplied Wannier/gauge does not match the original common basis localization");
    const auto& d=geometry.domain();const auto& s=geometry.real_space().space();const auto& e=geometry.embedding();
    if(periodic_correlation_real_pao_embedding_detail::pair_frame_identity(ref,d,geometry.real_space(),e.pair_selection())
        !=e.pair_frame_identity_sha256()) throw std::invalid_argument("Gram pair PNO geometry pair-frame receipt differs");
    const U D=d.domain_dimension(),r=s.retained_dimension(),n=e.memory().common_dimension,m=e.memory().pair_dimension;
    Digest h("vibeqc.periodic.gaussian-gram-pair-pnos.sources");
    h.string(hf.reference_source_identity_sha256());h.string(hf.context_handle()->source_context_identity_sha256());
    for(const auto* x:std::array<const std::string*,10>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &w.wannier_identity_sha256(),&gauge,&basis.identity_sha256(),&basis.payload_sha256(),&geometry.identity_sha256(),
        &e.identity_sha256(),&e.common_frame_identity_sha256(),&e.pair_frame_identity_sha256()}) h.string(*x);
    static_assert(kPeriodicCorrelationWannierContractVersion==1,"Gram seed Wannier coefficient codec version changed");
    Digest wp("vibeqc.periodic.correlation.wannier.coefficients");
    for(int component:ref.state().mesh()) wp.u32(component);
    wp.u64(w.n_basis());wp.u64(w.n_home_occupied());
    for(U cell=0;cell<w.n_cells();++cell) {
        const auto* values=w.cell_coefficients(cell);
        for(U x=0;x<mul(w.n_basis(),w.n_home_occupied());++x) {h.complex(values[x]);wp.complex(values[x]);}
    }
    if(wp.finish()!=w.coefficient_payload_sha256())
        throw std::invalid_argument("Gram pair PNO Wannier coefficient payload differs from its native receipt");
    real::indices(basis.occupied_indices_data(),mul(2,basis.memory().occupied_count),basis.memory().occupied_count,ref.state());
    Digest bp("vibeqc.periodic.correlation.real-local-basis.payload");
    bp.u64(basis.memory().occupied_count);bp.u64(basis.memory().virtual_count);
    for(U a=0;a<mul(2,basis.memory().occupied_count);++a) {const U x=basis.occupied_indices_data()[a];bp.u64(x);h.u64(x);}
    for(U a=0;a<basis.memory().retained_fock_bytes/8;++a) {const double x=basis.f_oo_data()[a];bp.real(x);h.real(x);}
    if(bp.finish()!=basis.payload_sha256()) throw std::invalid_argument("Gram pair PNO common Fock payload differs");
    Digest di("vibeqc.periodic.correlation.pao.domain");for(int x:ref.state().mesh()) di.u32(x);
    di.u64(ref.state().n_basis());di.u64(D);
    for(U a=0;a<D;++a) {const auto c=d.column(a);di.u64(c.cell);di.u64(c.ao);h.u64(c.cell);h.u64(c.ao);}
    Digest dm("vibeqc.periodic.correlation.pao.matrices");dm.u64(D);
    for(const auto* matrix:{d.overlap_data(),d.fock_data()}) for(U a=0;a<mul(D,D);++a) {dm.complex(matrix[a]);h.complex(matrix[a]);}
    Digest sp("vibeqc.periodic.correlation.pao.space.payload");sp.u64(D);sp.u64(r);
    for(U a=0;a<mul(D,r);++a) {sp.complex(s.coefficients_data()[a]);h.complex(s.coefficients_data()[a]);}
    for(U a=0;a<r;++a) {sp.real(s.energies_data()[a]);h.real(s.energies_data()[a]);}
    for(U a=0;a<D;++a) {sp.real(s.overlap_eigenvalues_data()[a]);h.real(s.overlap_eigenvalues_data()[a]);}
    Digest ep("vibeqc.periodic.correlation.real-pao-embedding.payload");ep.u64(n);ep.u64(m);
    for(U a=0;a<mul(n,m);++a) {ep.real(e.coefficients_data()[a]);h.real(e.coefficients_data()[a]);}
    for(U a=0;a<m;++a) {ep.real(e.energies_data()[a]);h.real(e.energies_data()[a]);}
    if(di.finish()!=d.domain_index_sha256() || dm.finish()!=d.matrix_payload_sha256()
        || sp.finish()!=s.payload_sha256() || ep.finish()!=e.payload_sha256())
        throw std::invalid_argument("Gram pair PNO generation geometry payload differs from native receipts");
    return h.finish();
}
void option_wire(Digest& h,const Options& o) {
    for(double x:{o.local_basis.coefficient_tr_absolute_tolerance,o.local_basis.coefficient_tr_relative_tolerance,
        o.local_basis.orthonormality_absolute_tolerance,o.local_basis.orthonormality_relative_tolerance,
        o.local_basis.fock_absolute_tolerance,o.local_basis.fock_relative_tolerance,o.local_basis.maximum_fock_projection_error,
        o.gram.reversal_absolute_tolerance,o.gram.reversal_relative_tolerance,o.gram.conjugacy_absolute_tolerance,
        o.gram.conjugacy_relative_tolerance,o.gram.self_q_absolute_tolerance,o.gram.self_q_relative_tolerance,
        o.gram.maximum_eri_projection_error,o.gram.maximum_scalar_roundoff_error,
        o.pno.pno.occupation_cutoff,o.pno.pno.denominator_floor,o.pno.pno.maximum_initial_fvv_offdiagonal_norm,
        o.pno.pno.semicanonical_orthonormality_tolerance,o.pno.maximum_diagonal_exchange_projection_norm,
        o.pno.maximum_fock_symmetry_projection_norm,o.pno.maximum_exported_gram_error,o.pno.maximum_exported_fock_error,
        o.pno.maximum_exported_subspace_error,o.maximum_retained_diagonal_projection_norm,o.maximum_occupied_fock_difference}) h.real(x);
    for(const auto& j:{o.pno.pno.pno_eigensolver,o.pno.pno.semicanonical_eigensolver}) {h.u64(j.max_sweeps);h.real(j.relative_offdiagonal_tolerance);}
}
void export_audit(const double* Cc,const double* X,const double* F,const double* eps,U n,U m,U r,
    const Options& options,PeriodicGaussianGramPairPNODiagnostics& d) {
    // Measured residuals of compensated, rounded contractions, not interval
    // proofs of exact represented-array dot products or eigenspaces.
    double gram=0,fock=0,subspace=0;
    for(U a=0;a<r;++a) for(U b=0;b<r;++b) {
        Sum g,f;
        for(U l=0;l<n;++l) {
            g.include(product(Cc[l*r+a],Cc[l*r+b]));
            for(U q=0;q<n;++q) f.include(product(product(Cc[l*r+a],F[l*n+q]),Cc[q*r+b]));
        }
        real::norm_lane(gram,real::difference_up(g.value(),a==b?1.0:0.0));
        real::norm_lane(fock,real::difference_up(f.value(),a==b?eps[a]:0.0));
    }
    for(U l=0;l<n;++l) for(U a=0;a<r;++a) {
        Sum reconstructed;
        for(U b=0;b<m;++b) {
            Sum overlap;for(U q=0;q<n;++q) overlap.include(product(X[q*m+b],Cc[q*r+a]));
            reconstructed.include(product(X[l*m+b],overlap.value()));
        }
        real::norm_lane(subspace,real::difference_up(Cc[l*r+a],reconstructed.value()));
    }
    d.exported_gram_frobenius_upper_bound=real::sqrt_up(gram);
    d.exported_fock_frobenius_upper_bound=real::sqrt_up(fock);
    d.exported_subspace_frobenius_upper_bound=real::sqrt_up(subspace);
    if(d.exported_gram_frobenius_upper_bound>options.pno.maximum_exported_gram_error
        || d.exported_fock_frobenius_upper_bound>options.pno.maximum_exported_fock_error
        || d.exported_subspace_frobenius_upper_bound>options.pno.maximum_exported_subspace_error)
        throw std::invalid_argument("Gram pair PNO common export Gram/Fock/subspace residual exceeds explicit budget");
}
std::string output_payload(const Result& result) {
    Digest h("vibeqc.periodic.gaussian-gram-pair-pnos.payload");
    h.u64(result.memory().common_virtual_dimension);h.u64(result.memory().generation_dimension);h.u64(result.diagnostics().retained_dimension);
    for(const auto* array:{&result.original_pno_occupations(),&result.coefficients(),&result.generation_coefficients(),
        &result.energies(),&result.exchange_integrals()}) for(double x:*array) h.real(x);
    return h.finish();
}
} // namespace

Plan plan_periodic_gaussian_gram_pair_pnos(const PeriodicGaussianRHFResult& hf,const Ref& ref,
    const PeriodicCorrelationWannier& w,const Basis& basis,const Geometry& geometry,
    const Config& config,const Options& options,const Live& live,const Caps& caps) {
    controls(config,options,live,caps);metadata(hf,ref,w,basis,geometry);
    Plan p;p.n_cells=ref.state().n_kpoints();p.n_basis=ref.state().n_basis();
    p.occupied_count=basis.memory().occupied_count;p.common_virtual_dimension=basis.memory().virtual_count;
    p.generation_dimension=geometry.embedding().memory().pair_dimension;
    p.occupied_slot_i=geometry.occupied_slot_i();p.occupied_slot_j=geometry.occupied_slot_j();
    p.local_occupied_count=same_label(basis.occupied(p.occupied_slot_i),basis.occupied(p.occupied_slot_j))?1:2;
    const U n=p.common_virtual_dimension,m=p.generation_dimension,q=p.local_occupied_count,mm=mul(m,m),nm=mul(n,m);
    limit(n,caps.maximum_common_dimension,"Gram pair PNO common dimension cap exceeded");
    limit(m,caps.maximum_generation_dimension,"Gram pair PNO generation dimension cap exceeded");
    limit(m,caps.gram.maximum_left_density_count,"Gram pair PNO left density cap exceeded");
    limit(m,caps.gram.maximum_right_density_count,"Gram pair PNO right density cap exceeded");
    limit(mm,caps.gram.maximum_output_elements,"Gram pair PNO Gram output cap exceeded");
    p.local_basis=plan_periodic_correlation_real_local_basis(ref,w,geometry.domain(),geometry.real_space().space(),q,
        geometry.embedding().pair_selection());
    limit(p.local_basis.peak_owned_numerical_bytes,caps.local_basis.maximum_owned_numerical_bytes,"Gram pair PNO local basis owned cap exceeded");
    limit(p.local_basis.work_units,caps.local_basis.maximum_work_units,"Gram pair PNO local basis work cap exceeded");
    p.gram_selection.left={q,m};
    // Second occupied label is position1 only for distinct physical labels.
    p.gram_selection.right={q==1?q:add(add(q,m)-1,q),m};
    p.gram_phase_upper_bytes=add(p.local_basis.retained_output_bytes,caps.gram.resources.maximum_owned_numeric_bytes);
    p.copy_phase_bytes=mul(16,mm);p.amplitude_phase_bytes=add(mul(16,mm),mul(8,m));
    p.density_phase_bytes=add(mul(16,mm),plan_restricted_pair_pnos(m).peak_owned_numerical_bytes);
    p.semicanonical_phase_upper_bytes=add(add(mul(16,mm),mul(8,m)),plan_restricted_pair_semicanonicalization(m,m).peak_owned_numerical_bytes);
    p.retained_integral_phase_upper_bytes=add(mul(32,mm),mul(40,m));
    p.export_phase_upper_bytes=mul(8,add(nm,add(mul(2,mm),mul(2,m))));
    p.retained_output_upper_bytes=p.export_phase_upper_bytes;
    const U numerical=std::max({p.copy_phase_bytes,p.amplitude_phase_bytes,p.density_phase_bytes,
        p.semicanonical_phase_upper_bytes,p.retained_integral_phase_upper_bytes,p.export_phase_upper_bytes});
    p.peak_owned_numerical_bytes=std::max({p.local_basis.peak_owned_numerical_bytes,p.gram_phase_upper_bytes,
        add(p.local_basis.retained_output_bytes,numerical)});
    p.borrowed_common_basis_bytes=basis.memory().retained_output_bytes;
    p.borrowed_geometry_bytes=geometry.retained_numerical_bytes();
    p.borrowed_wannier_bytes=w.memory().retained_coefficient_bytes;p.borrowed_gauge_bytes=w.memory().caller_gauge_bytes;
    p.borrowed_gaussian_bytes=hf.context_handle()->inventory().combined_borrowed_active_numeric_bytes;
    p.complete_borrowed_numerical_bytes=add(add(p.borrowed_common_basis_bytes,p.borrowed_geometry_bytes),
        add(add(p.borrowed_wannier_bytes,p.borrowed_gauge_bytes),p.borrowed_gaussian_bytes));
    p.fixed_control_storage_bytes=fixed_controls();
    p.control_storage_reservation_bytes=add(add(p.fixed_control_storage_bytes,geometry.retained_control_storage_bytes()),
        add(caps.maximum_gram_control_storage_bytes,live.other_live_control_bytes_per_worker));
    const U cube=mul(add(m,1),mul(add(m,1),add(m,1))),nn=mul(n,n);
    p.input_validation_work_units=add(mul(2,hf.context_handle()->inventory().work_units_upper_bound),
        mul(2048,add(65536,add(p.complete_borrowed_numerical_bytes/8,mul(p.occupied_count,p.occupied_count)))));
    const U algebra=add(mul(nn,mm),add(mul(nm,m),add(mul(mm,m),mul(p.occupied_count,p.occupied_count))));
    p.numerical_work_units=add(mul(4096,mul(add(add(options.pno.pno.pno_eigensolver.max_sweeps,
        options.pno.pno.semicanonical_eigensolver.max_sweeps),2),cube)),
        add(mul(1024,algebra),mul(512,add(p.retained_output_upper_bytes/8,65536))));
    // Includes the three bounded nested Gram metadata plans without assuming
    // their resource caps are numerical work that executes multiple times.
    p.input_validation_work_units=add(p.input_validation_work_units,
        mul(3*1048576,add(1024,add(m,geometry.domain().domain_dimension()))));
    p.gram_work_units_upper_bound=caps.gram.resources.maximum_work_units;
    p.work_units=add(add(p.local_basis.work_units,p.gram_work_units_upper_bound),
        add(p.input_validation_work_units,p.numerical_work_units));
    const auto& d=ref.dimensions();const auto& b=ref.budget();
    if(ref.state_resident_bytes()!=ref.state().resident_bytes() || d.external_bytes<ref.state_resident_bytes())
        throw std::invalid_argument("Gram pair PNO reference baseline inventory differs");
    p.replicas_per_node=mul(b.mpi_ranks,b.workers_per_rank);positive(p.replicas_per_node);
    p.reference_base_node_bytes=add(add(d.external_bytes,d.shared_bytes),mul(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)));
    p.worker_bytes=add(p.peak_owned_numerical_bytes,add(p.complete_borrowed_numerical_bytes,
        add(p.control_storage_reservation_bytes,add(live.other_live_numerical_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.worker_bytes));
    for(U bytes:{p.peak_owned_numerical_bytes,p.complete_borrowed_numerical_bytes,p.control_storage_reservation_bytes,
        p.retained_output_upper_bytes,mul(8,mm),mul(8,nm)}) real::extent(bytes);
    if(mm>std::vector<double>().max_size() || nm>std::vector<double>().max_size())
        throw std::length_error("Gram pair PNO vector extent exceeds native capacity");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gram pair PNO owned cap exceeded");
    limit(p.control_storage_reservation_bytes,caps.maximum_control_storage_bytes,"Gram pair PNO control cap exceeded");
    limit(p.worker_bytes,caps.maximum_worker_bytes,"Gram pair PNO worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.maximum_node_bytes,"Gram pair PNO node cap exceeded");
    limit(p.required_node_memory_bytes,b.memory_limit_bytes,"Gram pair PNO admitted-reference node cap exceeded");
    limit(p.work_units,caps.maximum_work_units,"Gram pair PNO work cap exceeded");
    return p;
}

void Result::require_live() const {
    const U n=memory_.common_virtual_dimension,m=memory_.generation_dimension,r=diagnostics_.retained_dimension;
    if(!state_ || !context_ || !n || !m || m>n || r>m || !memory_.occupied_count
        || memory_.occupied_slot_i>=memory_.occupied_count || memory_.occupied_slot_j>=memory_.occupied_count
        || coefficients_.size()!=mul(n,r) || generation_coefficients_.size()!=mul(m,r)
        || energies_.size()!=r || occupations_.size()!=m || exchange_.size()!=mul(r,r)
        || diagnostics_.retained_frame_bytes!=mul(8,add(add(coefficients_.size(),generation_coefficients_.size()),add(r,m)))
        || diagnostics_.retained_output_bytes!=add(diagnostics_.retained_frame_bytes,mul(8,exchange_.size()))
        || diagnostics_.retained_generation_coefficient_bytes!=mul(8,generation_coefficients_.size()))
        throw std::logic_error("Gram pair PNO owner is consumed or malformed");
}
const std::vector<double>& Result::coefficients() const {require_live();return coefficients_;}
const std::vector<double>& Result::generation_coefficients() const {require_live();return generation_coefficients_;}
const std::vector<double>& Result::energies() const {require_live();return energies_;}
const std::vector<double>& Result::original_pno_occupations() const {require_live();return occupations_;}
const std::vector<double>& Result::exchange_integrals() const {require_live();return exchange_;}
U Result::retained_numerical_bytes() const {require_live();return diagnostics_.retained_output_bytes;}
bool Result::complete_generation_pair_space() const {require_live();return diagnostics_.retained_dimension==memory_.generation_dimension;}
bool Result::complete_common_virtual_space() const {require_live();return diagnostics_.retained_dimension==memory_.common_virtual_dimension;}

PeriodicGaussianGramPairPNOPayloadValidationPlan plan_periodic_gaussian_gram_pair_pno_payload_validation(const Result& result) {
    PeriodicGaussianGramPairPNOPayloadValidationPlan p;p.numerical_lanes=result.retained_numerical_bytes()/8;
    real::extent(mul(8,p.numerical_lanes));
    p.work_units=mul(512,add(p.numerical_lanes,1024));
    p.control_storage_bytes=4096+sizeof(Digest)+sizeof(p)+65;return p;
}
void verify_periodic_gaussian_gram_pair_pno_payload(const Result& result,U maximum_work_units) {
    const auto p=plan_periodic_gaussian_gram_pair_pno_payload_validation(result);
    limit(p.work_units,maximum_work_units,"Gram pair PNO payload verification work cap exceeded");
    real::float_environment();
    if(output_payload(result)!=result.payload_sha256()) throw std::invalid_argument("Gram pair PNO payload receipt differs");
}

Result make_periodic_gaussian_gram_pair_pnos(const PeriodicGaussianRHFResult& hf,const Ref& ref,
    const BasisSet& ao,const BasisSet& aux,const PeriodicCorrelationWannier& w,const C* gauges,std::size_t gauge_count,
    const Basis& basis,const Geometry& geometry,const Config& supplied_config,const Options& supplied_options,
    const Live& supplied_live,const Caps& supplied_caps) {
    const auto config=supplied_config;const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_gram_pair_pnos(hf,ref,w,basis,geometry,config,options,live,caps);
    const U n=p.common_virtual_dimension,m=p.generation_dimension,q=p.local_occupied_count,mm=mul(m,m),o=p.occupied_count;
    Result result;result.memory_=p;result.options_=options;
    result.sources_=input_payload(hf,ref,ao,aux,w,gauges,gauge_count,basis,geometry);
    result.i_=basis.occupied(p.occupied_slot_i);result.j_=basis.occupied(p.occupied_slot_j);
    result.basis_=basis.identity_sha256();result.hf_=hf.reference_source_identity_sha256();
    result.context_sha_=hf.context_handle()->source_context_identity_sha256();
    result.geometry_=geometry.identity_sha256();result.embedding_=geometry.embedding().identity_sha256();
    const std::array<U,4> labels{result.i_.occupied_index,result.i_.cell,result.j_.occupied_index,result.j_.cell};
    auto local_basis=make_periodic_correlation_real_local_basis(ref,w,gauges,gauge_count,geometry.domain(),
        geometry.real_space().space(),labels.data(),mul(2,q),q,geometry.embedding().pair_selection(),options.local_basis,caps.local_basis);
    if(local_basis.memory().retained_output_bytes!=p.local_basis.retained_output_bytes)
        throw std::logic_error("Gram pair PNO local basis retained census changed");
    result.local_basis_=local_basis.identity_sha256();auto& d=result.diagnostics_;auto& gd=d.generation;
    d.local_basis=local_basis.diagnostics();
    if(d.local_basis.fock_projection_frobenius_upper_bound>options.pno.maximum_fock_symmetry_projection_norm)
        throw std::invalid_argument("Gram pair PNO raw-complex local Fock projection exceeds explicit budget");
    const double* F=local_basis.f_vv_data();const double* Foo=local_basis.f_oo_data();
    double occupied_difference=0;
    for(U a=0;a<q;++a) for(U b=0;b<q;++b) {
        const U ia=a?p.occupied_slot_j:p.occupied_slot_i,ib=b?p.occupied_slot_j:p.occupied_slot_i;
        real::norm_lane(occupied_difference,real::difference_up(Foo[a*q+b],basis.f_oo_data()[ia*o+ib]));
    }
    d.occupied_fock_frobenius_upper_bound=real::sqrt_up(occupied_difference);
    if(d.occupied_fock_frobenius_upper_bound>options.maximum_occupied_fock_difference)
        throw std::invalid_argument("Gram pair PNO local/common occupied Fock agreement exceeds explicit budget");
    gd.ignored_occupied_offdiagonal_norm_upper_bound=offdiagonal(basis.f_oo_data(),o);
    gd.initial_fvv_offdiagonal_norm_upper_bound=offdiagonal(F,m);
    if(gd.initial_fvv_offdiagonal_norm_upper_bound>options.pno.pno.maximum_initial_fvv_offdiagonal_norm)
        throw std::invalid_argument("Gram pair PNO initial local Fvv offdiagonal norm exceeds explicit budget");
    result.f_=array_hash("vibeqc.periodic.gaussian-gram-pair-pnos.local-F",F,mm,m);
    const U domain_space_bytes=add(p.local_basis.live_domain_bytes,p.local_basis.live_space_bytes);
    PeriodicGaussianDensityGramLiveInventory gram_live;
    gram_live.fixed_backend_margin_bytes_per_worker=live.fixed_backend_margin_bytes_per_worker;
    auto gram_caps=caps.gram;
    gram_caps.resources.maximum_per_replica_inventoried_bytes=std::min(gram_caps.resources.maximum_per_replica_inventoried_bytes,p.worker_bytes);
    gram_caps.resources.maximum_node_inventoried_bytes=std::min(gram_caps.resources.maximum_node_inventoried_bytes,p.required_node_memory_bytes);
    const auto bare=plan_periodic_gaussian_density_gram(hf,ref,w,geometry.domain(),geometry.real_space().space(),local_basis,
        p.gram_selection,config.gram,options.gram,gram_live,gram_caps);
    limit(bare.control_storage_reservation_bytes,caps.maximum_gram_control_storage_bytes,"Gram pair PNO nested Gram control cap exceeded");
    gram_live.other_retained_bytes_per_worker=add(live.other_live_numerical_bytes_per_worker,
        add(p.borrowed_common_basis_bytes,add(subtract(p.borrowed_geometry_bytes,domain_space_bytes),
        subtract(p.control_storage_reservation_bytes,bare.control_storage_reservation_bytes))));
    const auto exact=plan_periodic_gaussian_density_gram(hf,ref,w,geometry.domain(),geometry.real_space().space(),local_basis,
        p.gram_selection,config.gram,options.gram,gram_live,gram_caps);
    if(exact.per_worker_inventoried_bytes>p.worker_bytes || exact.required_node_memory_bytes>p.required_node_memory_bytes
        || exact.work_units>p.gram_work_units_upper_bound || exact.peak_owned_numerical_bytes>caps.gram.resources.maximum_owned_numeric_bytes)
        throw std::logic_error("Gram pair PNO exact Gram child exceeds enclosing admission");
    std::vector<double> G;
    {
        auto gram=build_periodic_gaussian_density_gram(hf,ref,ao,aux,w,gauges,gauge_count,geometry.domain(),
            geometry.real_space().space(),local_basis,p.gram_selection,config.gram,options.gram,gram_live,gram_caps);
        if(gram.state_handle()!=ref.state_handle() || gram.context_handle()!=hf.context_handle()
            || gram.basis_identity_sha256()!=local_basis.identity_sha256()
            || gram.hf_reference_source_identity_sha256()!=result.hf_ || gram.memory().output_elements!=mm)
            throw std::invalid_argument("Gram pair PNO consumed Gram source differs");
        d.gram_memory=gram.memory();d.gram=gram.diagnostics();
        result.gram_=gram.identity_sha256();result.gram_payload_=gram.payload_sha256();result.gram_sources_=gram.consumed_sources_identity_sha256();
        G.assign(gram.data(),gram.data()+mm);
    } // The raw Gram owner is released before T/density workspace exists.
    result.raw_g_=array_hash("vibeqc.periodic.gaussian-gram-pair-pnos.raw-G",G.data(),mm,m);
    if(result.diagonal_pair()) symmetric_projection(G,m,options.pno.maximum_diagonal_exchange_projection_norm,
        d.maximum_raw_exchange_transpose_defect,d.diagonal_exchange_projection_frobenius_upper_bound);
    result.g_=array_hash("vibeqc.periodic.gaussian-gram-pair-pnos.G",G.data(),mm,m);
    gd.maximum_diagonal_integral_asymmetry=d.maximum_raw_exchange_transpose_defect;
    gd.integral_projection_error_bound=d.gram.real_projection.maximum_eri_projection_error_bound;
    gd.maximum_scalar_integral_roundoff_error=d.gram.maximum_integral_roundoff_error;
    // There are no old scalar-provider calls. The actual Gram source/panel
    // counts live in diagnostics.gram; zero here is not a missing replay.
    gd.completed_integral_calls=0;
    RestrictedPairMP2Result initial;
    {
        std::vector<double> eps(static_cast<std::size_t>(m));for(U a=0;a<m;++a) eps[a]=F[a*m+a];
        initial=restricted_pair_semicanonical_mp2(G.data(),G.size(),eps.data(),eps.size(),m,Foo[0],Foo[q*q-1],
            options.pno.pno.denominator_floor,mul(8,mm));
    }
    result.amplitudes_=array_hash("vibeqc.periodic.gaussian-gram-pair-pnos.T",initial.amplitudes.data(),mm,m);
    gd.minimum_denominator=initial.minimum_denominator;gd.maximum_denominator=initial.maximum_denominator;
    gd.maximum_absolute_initial_amplitude=initial.maximum_absolute_amplitude;gd.maximum_initial_residual=initial.maximum_residual;
    gd.initial_residual_frobenius_norm=initial.residual_frobenius_norm;
    // Nejad Eq.39 has no molecular (1+delta_ij) normalization. Physical
    // diagonal symmetry has already been gated in G; no density rescaling.
    auto pnos=restricted_pair_pnos(initial.amplitudes.data(),initial.amplitudes.size(),m,RestrictedPairKind::OffDiagonal,
        options.pno.pno.occupation_cutoff,plan_restricted_pair_pnos(m).peak_owned_numerical_bytes,options.pno.pno.pno_eigensolver);
    std::vector<double>().swap(initial.amplitudes);
    result.density_=array_hash("vibeqc.periodic.gaussian-gram-pair-pnos.D",pnos.density.data(),mm,m);
    const U r=d.retained_dimension=gd.retained_dimension=pnos.retained_dimension;
    gd.density_trace=pnos.density_trace;gd.discarded_occupation_sum=pnos.discarded_occupation_sum;
    gd.minimum_occupation=pnos.minimum_occupation;gd.negative_occupation_count=pnos.negative_occupation_count;
    gd.amplitude_scaling_underflow_count=pnos.amplitude_scaling_underflow_count;gd.density_underflow_entry_count=pnos.density_underflow_entry_count;
    gd.density_eigensystem_relative_residual=pnos.eigensystem_relative_residual;
    gd.density_eigenvector_orthogonality_error=pnos.eigenvector_orthogonality_error;gd.pno_eigensolver=pnos.eigensolver;
    std::vector<double>().swap(pnos.density);
    const auto rotation=plan_restricted_pair_semicanonicalization(m,r);
    d.actual_semicanonical_phase_bytes=add(add(mul(8,mm),add(mul(8,mul(m,r)),mul(8,m))),rotation.peak_owned_numerical_bytes);
    limit(d.actual_semicanonical_phase_bytes,p.semicanonical_phase_upper_bytes,"Gram pair PNO actual rotation phase exceeds admission");
    auto sc=restricted_pair_semicanonicalize(pnos.coefficients.data(),pnos.coefficients.size(),F,mm,m,r,
        options.pno.pno.semicanonical_orthonormality_tolerance,rotation.peak_owned_numerical_bytes,options.pno.pno.semicanonical_eigensolver);
    gd.actual_semicanonical_phase_bytes=d.actual_semicanonical_phase_bytes;
    gd.semicanonical_input_orthonormality_error=sc.input_orthonormality_error;gd.semicanonical_output_orthonormality_error=sc.output_orthonormality_error;
    gd.semicanonical_reduced_relative_residual=sc.reduced_eigensystem_relative_residual;
    gd.semicanonical_projected_fock_relative_residual=sc.projected_fock_relative_residual;
    gd.semicanonical_subspace_projector_error=sc.subspace_projector_frobenius_error;
    gd.semicanonical_full_space_relative_residual=sc.full_space_relative_residual;
    gd.semicanonical_fock_scaling_underflow_count=sc.fock_scaling_underflow_entries;gd.semicanonical_eigensolver=sc.eigensolver;
    std::vector<double>().swap(pnos.coefficients);
    result.generation_coefficients_=std::move(sc.coefficients);result.energies_=std::move(sc.energies);result.occupations_=std::move(pnos.occupations);
    const U rr=mul(r,r),mr=mul(m,r);
    d.actual_retained_integral_phase_bytes=add(mul(8,add(mm,add(mr,add(r,m)))),add(mul(16,rr),mul(16,r)));
    limit(d.actual_retained_integral_phase_bytes,p.retained_integral_phase_upper_bytes,"Gram pair PNO actual retained-G phase exceeds admission");
    result.exchange_.resize(static_cast<std::size_t>(rr));
    {
        std::vector<double> correction(static_cast<std::size_t>(rr)),row(static_cast<std::size_t>(r)),row_correction(static_cast<std::size_t>(r));
        const auto* D=result.generation_coefficients_.data();
        for(U a=0;a<m;++a) {
            std::fill(row.begin(),row.end(),0.0);std::fill(row_correction.begin(),row_correction.end(),0.0);
            for(U b=0;b<m;++b) for(U beta=0;beta<r;++beta)
                accumulate(product(G[a*m+b],D[b*r+beta]),row[beta],row_correction[beta]);
            for(U beta=0;beta<r;++beta) row[beta]=real::finite(row[beta]+row_correction[beta]);
            for(U alpha=0;alpha<r;++alpha) for(U beta=0;beta<r;++beta)
                accumulate(product(D[a*r+alpha],row[beta]),result.exchange_[alpha*r+beta],correction[alpha*r+beta]);
        }
        for(U a=0;a<rr;++a) result.exchange_[a]=real::finite(result.exchange_[a]+correction[a]);
    }
    std::vector<double>().swap(G);
    result.raw_retained_g_=array_hash("vibeqc.periodic.gaussian-gram-pair-pnos.raw-retained-G",result.exchange_.data(),rr,r);
    if(result.diagonal_pair()) symmetric_projection(result.exchange_,r,options.maximum_retained_diagonal_projection_norm,
        d.maximum_raw_retained_transpose_defect,d.retained_diagonal_projection_frobenius_upper_bound);
    result.retained_g_=array_hash("vibeqc.periodic.gaussian-gram-pair-pnos.retained-G",result.exchange_.data(),rr,r);
    d.actual_export_phase_bytes=mul(8,add(mul(n,r),add(mr,add(add(r,m),rr))));
    limit(d.actual_export_phase_bytes,p.export_phase_upper_bytes,"Gram pair PNO actual export phase exceeds admission");
    const auto* X=geometry.embedding().coefficients_data();result.coefficients_.resize(static_cast<std::size_t>(mul(n,r)));
    for(U a=0;a<n;++a) for(U b=0;b<r;++b) {
        Sum x;for(U u=0;u<m;++u) x.include(product(X[a*m+u],result.generation_coefficients_[u*r+b]));
        result.coefficients_[a*r+b]=x.value();
    }
    export_audit(result.coefficients_.data(),X,basis.f_vv_data(),result.energies_.data(),n,m,r,options,d);
    d.retained_generation_coefficient_bytes=mul(8,mr);
    d.retained_frame_bytes=mul(8,add(mul(n,r),add(mr,add(r,m))));d.retained_output_bytes=d.actual_export_phase_bytes;
    gd.retained_output_bytes=mul(8,add(mr,add(r,m)));
    if(d.retained_output_bytes>p.retained_output_upper_bytes)
        throw std::logic_error("Gram pair PNO output census exceeds admission");
    if(input_payload(hf,ref,ao,aux,w,gauges,gauge_count,basis,geometry)!=result.sources_)
        throw std::invalid_argument("Gram pair PNO original source payload changed");
    real::float_environment();result.state_=ref.state_handle();result.context_=hf.context_handle();
    result.payload_=output_payload(result);
    Digest identity("vibeqc.periodic.gaussian-gram-pair-pnos.identity");
    for(const auto* x:std::array<const std::string*,19>{&ref.state().state_identity_sha256(),&ref.dimensions().allocation_identity,
        &result.basis_,&result.local_basis_,&result.hf_,&result.context_sha_,&result.geometry_,&result.embedding_,
        &result.gram_,&result.gram_payload_,&result.gram_sources_,&result.raw_g_,&result.g_,&result.f_,
        &result.raw_retained_g_,&result.retained_g_,&result.amplitudes_,&result.density_,&result.payload_}) identity.string(*x);
    identity.string(result.sources_);identity.string(policy);identity.string(real::kFloatPolicy);
    for(U x:{p.occupied_count,n,m,r,p.occupied_slot_i,p.occupied_slot_j,result.i_.occupied_index,result.i_.cell,result.j_.occupied_index,result.j_.cell,
        config.gram.auxiliary_block,config.gram.panel.ao_pair_block}) identity.u64(x);
    option_wire(identity,options);result.identity_=identity.finish();return result;
}
} // namespace vibeqc
