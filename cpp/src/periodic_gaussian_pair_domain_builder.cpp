#include "vibeqc/periodic_gaussian_pair_domain_builder.hpp"

#include <cfloat>
#include <type_traits>
#include "vibeqc/periodic_gaussian_mixed_pair_factors.hpp"
#include "periodic_correlation_real_local_internal.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0) || FLT_EVAL_METHOD != 0
#error "Gaussian pair-domain builder requires strict binary64 evaluation"
#endif

namespace vibeqc {
namespace {
namespace real=periodic_correlation_real_local_detail;
namespace local=periodic_correlation_local_detail;
using U=std::uint64_t;
using real::add;using real::mul;using real::Digest;
using Builder=PeriodicGaussianPairDomainBuilder;
using Geometry=PeriodicGaussianPairDomainGeometry;
using Live=PeriodicGaussianPairDomainBuilderLive;
using PreparationPlan=PeriodicGaussianPairDomainBuilderPlan;
using PreparationCaps=PeriodicGaussianPairDomainBuilderCaps;
using PairPlan=PeriodicGaussianPairDomainEmbeddingPlan;
using PairCaps=PeriodicGaussianPairDomainCaps;
using Domain=PeriodicCorrelationPAODomain;
using RealSpace=PeriodicCorrelationRealPAOSpace;
using Basis=PeriodicCorrelationRealLocalBasis;
using Ref=PeriodicCorrelationAdmittedReference;
static_assert(std::is_nothrow_move_constructible<Domain>::value && std::is_nothrow_move_constructible<RealSpace>::value,
    "Gaussian pair-domain final geometry transfers must not throw");
static_assert(std::is_nothrow_move_constructible<PeriodicCorrelationRealPAOEmbedding>::value
    && std::is_nothrow_move_constructible<Geometry>::value,"Gaussian pair-domain bundle transfers must not throw");

void limit(U value,U cap,const char* message) {if(!cap || value>cap) throw std::length_error(message);}
U subtract(U all,U part) {
    if(part>all) throw std::logic_error("Gaussian pair-domain live subset exceeds its complete owner");
    return all-part;
}
void sha(const std::string& value) {
    if(value.size()!=64) throw std::invalid_argument("Gaussian pair-domain source SHA extent differs");
    for(char c:value) if(!((c>='0' && c<='9') || (c>='a' && c<='f')))
        throw std::invalid_argument("Gaussian pair-domain source SHA is malformed");
}
void live_valid(const Live& live) {
    real::float_environment();
    if(!live.backend_margin_bytes_per_worker) throw std::invalid_argument("Gaussian pair-domain builder requires positive backend allowance");
}
U common_bytes(const Domain& d,const RealSpace& s) {
    return add(add(d.memory().retained_domain_index_bytes,d.memory().retained_matrix_bytes),s.space().memory().output_numerical_bytes);
}
U home_controls(U homes) {
    // Native actual-domain wrapper owns its selector inline plus nine SHAs.
    return mul(homes,sizeof(PeriodicGaussianOccupiedPAODomain)+9U*65U);
}
U retained_controls(U homes) {
    // Inline topology/domain/space objects already occur in sizeof(Builder).
    return add(sizeof(Builder),add(home_controls(homes),(6U+4U+4U+7U)*65U));
}
U geometry_controls() {
    // Children occur inline in sizeof(Geometry): domain four strings,
    // compact+real space seven, embedding eight, bundle five.
    return sizeof(Geometry)+24U*65U;
}
U fixed_controls() {
    return 131072U+4U*sizeof(PreparationPlan)+3U*sizeof(PairPlan)+2U*sizeof(PreparationCaps)+2U*sizeof(PairCaps)
        +3U*sizeof(Live)+3U*sizeof(PeriodicGaussianPairDomainOptions)+3U*sizeof(PeriodicCorrelationOccupiedPAODomainOptions)
        +sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicGaussianLocalizationResult)+sizeof(PeriodicGaussianSourceContext)
        +sizeof(Ref)+sizeof(Basis)+3U*sizeof(BasisSet)+sizeof(PeriodicSystem)+16U*sizeof(Digest)+64U*65U
        +2U*geometry_controls();
}
U union_control_upper() {
    return 65536U+sizeof(PeriodicCorrelationTranslationPairTopology)+4U*64U
        +2U*(sizeof(PeriodicCorrelationOccupiedPAODomain)+7U*64U);
}
U embedding_controls() {
    // Exact current embedding control formula for distinct common/pair
    // domain and real-space owners, with zero caller controls. Check its
    // published actual plan again before use, never substitute a cap.
    using P=PeriodicCorrelationRealPAOEmbeddingMemoryPlan;using R=PeriodicCorrelationRealPAOEmbedding;
    using O=PeriodicCorrelationRealPAOEmbeddingOptions;using L=PeriodicCorrelationRealPAOEmbeddingLiveInventory;
    using C=PeriodicCorrelationRealPAOEmbeddingCaps;using Z=std::complex<double>;
    return 131072U+3U*sizeof(P)+2U*sizeof(R)+2U*sizeof(O)+2U*sizeof(L)+2U*sizeof(C)+sizeof(Ref)+sizeof(Basis)
        +4U*65U+4U*sizeof(Digest)+12U*sizeof(std::vector<Z>)+2U*sizeof(PeriodicCorrelationPAOSpaceOverlap)+20U*65U
        +2U*(sizeof(Domain)+4U*65U)+2U*(sizeof(RealSpace)+65U)+12U*65U;
}
U occupied_controls(const PeriodicGaussianOccupiedPAODomainMemoryPlan& plan) {
    return add(plan.additional_control_storage_bytes,add(plan.selector.fixed_control_storage_bytes,plan.selector.borrowed_owner_control_bytes));
}
void reference_node(const Ref& ref,U worker,U& replicas,U& base,U& node) {
    const auto& b=ref.budget();const auto& d=ref.dimensions();replicas=mul(b.mpi_ranks,b.workers_per_rank);
    if(!replicas || ref.state_resident_bytes()!=ref.state().resident_bytes() || d.external_bytes<ref.state_resident_bytes())
        throw std::invalid_argument("Gaussian pair-domain reference baseline or replica census differs");
    base=add(add(d.external_bytes,d.shared_bytes),mul(b.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)));
    node=add(base,mul(replicas,worker));
    limit(node,b.memory_limit_bytes,"Gaussian pair-domain enclosing admitted node cap exceeded");
}
void common_metadata(const Ref& ref,const Basis& b,const Domain& d,const RealSpace& r) {
    local::validate_reference(ref);const auto& s=r.space();local::validate_pao(ref,d,s);
    const auto v=b.virtual_selection();
    if(b.state_handle()!=ref.state_handle() || b.allocation_identity()!=ref.dimensions().allocation_identity
        || d.state_handle()!=ref.state_handle() || s.state_handle()!=ref.state_handle()
        || !b.memory().occupied_count || !v.count || v.count!=b.memory().virtual_count
        || v.begin>=s.retained_dimension() || v.count>s.retained_dimension()-v.begin || v.translation_cell>=ref.state().n_kpoints()
        || b.memory().n_cells!=ref.state().n_kpoints() || b.memory().n_basis!=ref.state().n_basis()
        || r.memory().compact.output_numerical_bytes!=s.memory().output_numerical_bytes)
        throw std::invalid_argument("Gaussian pair-domain common frame owner/selection differs");
    for(const auto* x:std::array<const std::string*,9>{&b.identity_sha256(),&b.payload_sha256(),&b.local_basis_identity_sha256(),
        &d.pao_domain_identity_sha256(),&d.matrix_payload_sha256(),&s.pao_space_identity_sha256(),&s.payload_sha256(),
        &r.identity_sha256(),&ref.dimensions().allocation_identity}) sha(*x);
    if(!std::equal(b.virtual_domain_identity_ascii().begin(),b.virtual_domain_identity_ascii().end(),d.pao_domain_identity_sha256().begin())
        || !std::equal(b.virtual_space_identity_ascii().begin(),b.virtual_space_identity_ascii().end(),s.pao_space_identity_sha256().begin()))
        throw std::invalid_argument("Gaussian pair-domain common basis does not certify supplied PAO geometry");
    (void)b.occupied_indices_data();(void)b.f_oo_data();(void)s.coefficients_data();(void)s.energies_data();
}
void preparation_metadata(const PeriodicGaussianRHFResult& hf,const Ref& ref,
    const PeriodicGaussianLocalizationResult& loc,const Basis& b,const Domain& d,const RealSpace& s) {
    common_metadata(ref,b,d,s);
    if(!hf.converged() || hf.state_handle()!=ref.state_handle() || !hf.context_handle() || !loc.converged()
        || loc.optimizer().state_handle()!=ref.state_handle() || loc.wannier().state_handle()!=ref.state_handle()
        || loc.optimizer().allocation_identity()!=ref.dimensions().allocation_identity
        || loc.wannier().allocation_identity()!=ref.dimensions().allocation_identity
        || loc.wannier().localization_identity_sha256()!=loc.optimizer().optimizer_identity_sha256()
        || hf.plan().n_basis!=ref.state().n_basis() || hf.plan().n_kpoints!=ref.state().n_kpoints()
        || loc.memory().n_basis!=ref.state().n_basis() || loc.memory().n_points!=ref.state().n_kpoints()
        || loc.memory().n_active!=ref.state().n_correlated_occupied()
        || loc.source_image_cutoff_bohr()!=hf.context_handle()->options().ao_pair_image_cutoff_bohr)
        throw std::invalid_argument("Gaussian pair-domain requires exact HF/localization source owners and image recipe");
}
void preparation_sources(const PeriodicGaussianRHFResult& hf,const Ref& ref,const BasisSet& ao,const BasisSet& aux,
    const BasisSet& minimal,const PeriodicSystem& system,const PeriodicGaussianLocalizationResult& loc,
    const Basis& b,const Domain& d,const RealSpace& s) {
    preparation_metadata(hf,ref,loc,b,d,s);hf.verify_physical_inputs(ao,aux,system);loc.verify_original_inputs(ref,ao,minimal,system);
    const auto& om=loc.optimizer().memory();
    const auto gauge=local::validate_gauges(loc.wannier(),loc.optimizer().gauges_data(),mul(om.n_points,mul(om.n_active,om.n_active)));
    if(real::local_basis_identity(ref,loc.wannier(),gauge,d,s.space(),b.occupied_indices_data(),b.memory().occupied_count,
            b.virtual_selection())!=b.local_basis_identity_sha256())
        throw std::invalid_argument("Gaussian pair-domain occupied basis is not from this exact localization gauge");
}
std::string common_payload(const Domain& d,const RealSpace& r) {
    Digest h("vibeqc.periodic.gaussian-pair-domain-builder.common");const auto& s=r.space();
    h.string(d.pao_domain_identity_sha256());h.string(s.pao_space_identity_sha256());h.string(r.identity_sha256());
    h.u64(d.domain_dimension());h.u64(s.retained_dimension());
    for(U i=0;i<d.domain_dimension();++i) {const auto c=d.column(i);h.u64(c.cell);h.u64(c.ao);}
    for(U i=0;i<d.memory().matrix_element_count;++i) {h.complex(d.overlap_data()[i]);h.complex(d.fock_data()[i]);}
    for(U i=0;i<mul(s.domain_dimension(),s.retained_dimension());++i) h.complex(s.coefficients_data()[i]);
    for(U i=0;i<s.retained_dimension();++i) h.real(s.energies_data()[i]);
    for(U i=0;i<s.domain_dimension();++i) h.real(s.overlap_eigenvalues_data()[i]);
    return h.finish();
}
std::string payload(const std::vector<PeriodicGaussianOccupiedPAODomain>& domains,const std::vector<U>& mapping,
    const PeriodicCorrelationTranslationPairTopology& topology,const Domain& d,const RealSpace& r) {
    Digest h("vibeqc.periodic.gaussian-pair-domain-builder.payload");h.string(common_payload(d,r));
    h.u64(mapping.size());for(U x:mapping) h.u64(x);
    h.string(topology.topology_identity_sha256());h.u64(topology.row_count());
    for(U q=0;q<topology.row_count();++q) {const auto& x=topology.row(q);h.u64(x.home_orbital);h.u64(x.partner_orbital);
        h.u64(x.translation_linear_index);h.u64(x.placed_multiplicity);}
    h.u64(domains.size());
    for(const auto& wrapper:domains) {
        h.string(wrapper.identity_sha256());const auto& domain=wrapper.domain();const auto& m=domain.memory();const auto& g=domain.diagnostics();
        h.string(domain.payload_identity_sha256());h.u64(m.atom_cell_count);
        for(U at=0;at<m.atom_cell_count;++at) h.real(domain.populations_data()[at]);
        h.u64(g.seed_count);for(U at=0;at<g.seed_count;++at) {const auto x=domain.seed(at);h.u64(x.cell);h.u64(x.atom);}
        h.u64(g.expanded_count);for(U at=0;at<g.expanded_count;++at) {const auto x=domain.expanded(at);h.u64(x.cell);h.u64(x.atom);}
    }
    return h.finish();
}
PeriodicCorrelationOccupiedPAODomainInventory occupied_live(const Live& live,U other,U control) {
    PeriodicCorrelationOccupiedPAODomainInventory v;v.other_live_numerical_bytes=add(live.other_live_numerical_bytes_per_worker,other);
    v.other_live_control_bytes=control;v.backend_allowance_bytes=live.backend_margin_bytes_per_worker;return v;
}
} // namespace

PreparationPlan plan_periodic_gaussian_pair_domain_builder(const PeriodicGaussianRHFResult& hf,const Ref& ref,
    const PeriodicGaussianLocalizationResult& loc,const Basis& basis,const Domain& domain,const RealSpace& space,
    const PeriodicCorrelationOccupiedPAODomainOptions& options,const Live& live,const PreparationCaps& caps) {
    live_valid(live);preparation_metadata(hf,ref,loc,basis,domain,space);
    PreparationPlan p;p.n_cells=ref.state().n_kpoints();p.n_basis=ref.state().n_basis();p.n_atoms=hf.plan().atom_count;
    p.n_home_occupied=ref.state().n_correlated_occupied();p.common_virtual_dimension=basis.memory().virtual_count;
    if(!p.n_home_occupied || !p.n_atoms) throw std::invalid_argument("Gaussian pair-domain builder requires occupied/atomic input");
    const auto counts=estimate_periodic_correlation_translation_pair_counts(ref.state().mesh(),p.n_home_occupied);
    p.topology_rows=counts.candidate_count;
    if(p.topology_rows!=ref.dimensions().expected_pair_candidate_count) throw std::invalid_argument("Gaussian pair-domain topology census differs");
    limit(p.n_home_occupied,caps.maximum_home_domains,"Gaussian pair-domain home owner cap exceeded");
    limit(p.topology_rows,caps.maximum_topology_rows,"Gaussian pair-domain topology row cap exceeded");
    p.atom_mapping_bytes=mul(8,p.n_basis);p.topology_row_bytes=mul(32,p.topology_rows);
    p.common_geometry_numerical_bytes=common_bytes(domain,space);p.borrowed_basis_bytes=basis.memory().retained_output_bytes;
    p.borrowed_localization_bytes=loc.memory().output_numerical_bytes;
    p.borrowed_gaussian_bytes=add(loc.memory().borrowed_gaussian_numerical_bytes,hf.context_handle()->inventory().auxiliary.borrowed_active_numeric_bytes);
    p.retained_control_upper_bytes=retained_controls(p.n_home_occupied);
    const U own_control=add(fixed_controls(),p.retained_control_upper_bytes);
    const auto bare_leaf=plan_periodic_gaussian_occupied_pao_domain(hf,ref,loc,0,options,
        occupied_live({0,0,live.backend_margin_bytes_per_worker},0,0),caps.occupied);
    p.control_storage_reservation_bytes=add(add(own_control,occupied_controls(bare_leaf)),live.other_live_control_bytes_per_worker);
    limit(p.control_storage_reservation_bytes,caps.maximum_control_storage_bytes,"Gaussian pair-domain preparation control cap exceeded");
    const U atom_cells=mul(p.n_cells,p.n_atoms);
    const U one_output=add(mul(8,atom_cells),mul(16,add(std::min(atom_cells,caps.occupied.selector.maximum_seed_atom_cells),
        std::min(atom_cells,caps.occupied.selector.maximum_expanded_atom_cells))));
    p.home_domain_output_upper_bytes=mul(p.n_home_occupied,one_output);
    const U persistent=add(p.common_geometry_numerical_bytes,add(p.atom_mapping_bytes,p.topology_row_bytes));
    p.retained_numerical_upper_bytes=add(persistent,p.home_domain_output_upper_bytes);
    p.peak_owned_numerical_bytes=add(persistent,std::max(p.home_domain_output_upper_bytes,
        add(mul(p.n_home_occupied-1,one_output),caps.occupied.maximum_owned_numerical_bytes)));
    p.input_validation_work_units=add(mul(2,add(hf.plan().input_check_work_units,loc.memory().input_verification_work_units)),
        mul(4096,add(32768,add(p.retained_numerical_upper_bytes/8,add(p.borrowed_basis_bytes/8,p.borrowed_localization_bytes/8)))));
    p.domain_work_units=mul(p.n_home_occupied,caps.occupied.maximum_work_units);p.work_units=add(p.input_validation_work_units,p.domain_work_units);
    p.worker_bytes=add(p.peak_owned_numerical_bytes,add(p.borrowed_basis_bytes,add(p.borrowed_localization_bytes,
        add(p.borrowed_gaussian_bytes,add(p.control_storage_reservation_bytes,add(live.other_live_numerical_bytes_per_worker,live.backend_margin_bytes_per_worker))))));
    reference_node(ref,p.worker_bytes,p.replicas_per_node,p.reference_base_node_bytes,p.required_node_memory_bytes);
    for(U x:{p.retained_numerical_upper_bytes,p.peak_owned_numerical_bytes,p.control_storage_reservation_bytes,p.atom_mapping_bytes}) real::extent(x);
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian pair-domain preparation owned cap exceeded");
    limit(p.worker_bytes,caps.maximum_worker_bytes,"Gaussian pair-domain preparation worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.maximum_node_bytes,"Gaussian pair-domain preparation node cap exceeded");
    limit(p.work_units,caps.maximum_work_units,"Gaussian pair-domain preparation work cap exceeded");
    if(p.n_home_occupied>std::vector<PeriodicGaussianOccupiedPAODomain>().max_size() || p.n_basis>std::vector<U>().max_size())
        throw std::length_error("Gaussian pair-domain preparation vector extents exceeded");
    // Subordinate exact occupied leaf gate, only after the whole macro gate.
    const auto child=occupied_live(live,add(p.borrowed_basis_bytes,add(persistent,mul(p.n_home_occupied-1,one_output))),
        subtract(p.control_storage_reservation_bytes,occupied_controls(bare_leaf)));
    const auto leaf=plan_periodic_gaussian_occupied_pao_domain(hf,ref,loc,0,options,child,caps.occupied);
    if(add(leaf.selector.retained_population_bytes,leaf.selector.retained_domain_bytes_upper)!=one_output)
        throw std::logic_error("Gaussian pair-domain selector output upper changed");
    return p;
}

void Builder::require_live() const {
    if(!state_ || !context_ || !topology_ || !common_domain_ || !common_space_
        || domains_.size()!=memory_.n_home_occupied || atom_mapping_.size()!=memory_.n_basis
        || topology_->state_handle()!=state_ || topology_->row_count()!=memory_.topology_rows
        || common_domain_->state_handle()!=state_ || common_space_->space().state_handle()!=state_)
        throw std::logic_error("Gaussian pair-domain builder is consumed or malformed");
}
U Builder::retained_numerical_bytes() const {require_live();return diagnostics_.retained_numerical_bytes;}
U Builder::retained_control_storage_bytes() const {require_live();return diagnostics_.retained_control_storage_bytes;}

Builder make_periodic_gaussian_pair_domain_builder(const PeriodicGaussianRHFResult& hf,const Ref& ref,
    const BasisSet& ao,const BasisSet& aux,const BasisSet& minimal,const PeriodicSystem& system,
    const PeriodicGaussianLocalizationResult& loc,const Basis& basis,Domain&& domain,RealSpace&& space,
    const PeriodicCorrelationOccupiedPAODomainOptions& supplied_options,const Live& supplied_live,const PreparationCaps& supplied_caps) {
    const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_pair_domain_builder(hf,ref,loc,basis,domain,space,options,live,caps);
    preparation_sources(hf,ref,ao,aux,minimal,system,loc,basis,domain,space);
    const auto original_common=common_payload(domain,space);
    Builder result;result.memory_=p;result.state_=ref.state_handle();result.context_=hf.context_handle();
    result.atom_mapping_.resize(static_cast<std::size_t>(p.n_basis));U offset=0;
    for(U shell=0;shell<ao.nshells();++shell) {
        const int atom=ao.shell_atom_index(shell);if(atom<0 || static_cast<U>(atom)>=p.n_atoms)
            throw std::invalid_argument("Gaussian pair-domain actual shell atom label exceeds source");
        const U count=ao.libint()[static_cast<std::size_t>(shell)].size();
        if(count>p.n_basis-offset) throw std::invalid_argument("Gaussian pair-domain actual shell census exceeds source");
        for(U q=0;q<count;++q) result.atom_mapping_[offset++]=static_cast<U>(atom);
    }
    if(offset!=p.n_basis) throw std::invalid_argument("Gaussian pair-domain actual AO-map census differs");
    result.topology_.emplace(make_periodic_correlation_translation_pair_topology(ref));
    result.domains_.reserve(static_cast<std::size_t>(p.n_home_occupied));
    const U persistent=add(p.common_geometry_numerical_bytes,add(p.atom_mapping_bytes,p.topology_row_bytes));
    const auto bare_leaf=plan_periodic_gaussian_occupied_pao_domain(hf,ref,loc,0,options,
        occupied_live({0,0,live.backend_margin_bytes_per_worker},0,0),caps.occupied);
    U held=0;
    for(U home=0;home<p.n_home_occupied;++home) {
        const auto child=occupied_live(live,add(p.borrowed_basis_bytes,add(persistent,held)),
            subtract(p.control_storage_reservation_bytes,occupied_controls(bare_leaf)));
        auto selected=select_periodic_gaussian_occupied_pao_domain(hf,ref,ao,aux,minimal,system,loc,home,options,child,caps.occupied);
        held=add(held,selected.domain().diagnostics().retained_numerical_bytes);
        limit(held,p.home_domain_output_upper_bytes,"Gaussian pair-domain actual retained home payload exceeds admission");
        result.domains_.push_back(std::move(selected));
    }
    Digest map_identity("vibeqc.periodic.correlation.occupied-pao-domain.ao-atom-map");
    map_identity.u64(p.n_basis);map_identity.u64(p.n_atoms);for(U value:result.atom_mapping_) map_identity.u64(value);
    const auto mapping_receipt=map_identity.finish();
    for(const auto& endpoint:result.domains_) if(endpoint.domain().mapping_identity_sha256()!=mapping_receipt)
        throw std::invalid_argument("Gaussian pair-domain native map differs from selected physical endpoint map");
    preparation_sources(hf,ref,ao,aux,minimal,system,loc,basis,domain,space);
    if(common_payload(domain,space)!=original_common) throw std::invalid_argument("Gaussian pair-domain common payload changed during preparation");
    result.basis_=basis.identity_sha256();result.hf_=hf.reference_source_identity_sha256();result.localization_=loc.localization_identity_sha256();
    result.allocation_=ref.dimensions().allocation_identity;
    result.payload_=payload(result.domains_,result.atom_mapping_,*result.topology_,domain,space);
    Digest h("vibeqc.periodic.gaussian-pair-domain-builder.identity");
    for(const auto* x:std::array<const std::string*,6>{&result.basis_,&result.hf_,&result.localization_,&result.allocation_,&result.payload_,&ref.state().state_identity_sha256()}) h.string(*x);
    h.string("actual-Gaussian-HF-and-localization;native-atom-map;home-domains;canonical-union-stream;no-full-pair-embedding-cache");
    result.identity_=h.finish();result.diagnostics_.constructed_home_domains=p.n_home_occupied;
    result.diagnostics_.retained_home_domain_bytes=held;result.diagnostics_.retained_numerical_bytes=add(persistent,held);
    result.diagnostics_.retained_control_storage_bytes=p.retained_control_upper_bytes;
    // ALL fallible work has finished. Move geometry only now; no source
    // borrowed pointer remains in the returned preparation owner.
    result.common_domain_.emplace(std::move(domain));result.common_space_.emplace(std::move(space));return result;
}

PairPlan plan_periodic_gaussian_pair_domain_embedding(const Ref& ref,const Basis& basis,const Builder& builder,
    U i,U j,const PeriodicGaussianPairDomainOptions& options,const Live& live,const PairCaps& caps) {
    live_valid(live);builder.require_live();common_metadata(ref,basis,*builder.common_domain_,*builder.common_space_);
    if(builder.state_!=ref.state_handle() || builder.basis_!=basis.identity_sha256() || builder.allocation_!=ref.dimensions().allocation_identity)
        throw std::invalid_argument("Gaussian pair-domain embedding requires original builder reference/basis owners");
    if(i>=basis.memory().occupied_count || j>=basis.memory().occupied_count) throw std::out_of_range("Gaussian pair-domain occupied slot exceeds basis");
    if(!options.domain.require_time_reversal || !options.domain.require_real_matrices)
        throw std::invalid_argument("Gaussian pair-domain generation requires explicit real/TR PAO geometry");
    for(U cap:{caps.pair_union.maximum_owned_numerical_bytes,caps.pair_union.maximum_control_storage_bytes,
        caps.pair_union.maximum_work_units,caps.maximum_pao_domain_owned_numerical_bytes,caps.real_space.maximum_owned_numerical_bytes,
        caps.real_space.maximum_work_units,caps.embedding.maximum_owned_numerical_bytes,caps.embedding.maximum_control_storage_bytes_per_worker,
        caps.embedding.maximum_work_units}) if(!cap) throw std::invalid_argument("Gaussian pair-domain child resource ceilings must be positive");
    PairPlan p;p.n_cells=builder.memory_.n_cells;p.n_basis=builder.memory_.n_basis;p.common_virtual_dimension=basis.memory().virtual_count;
    p.maximum_generation_dimension=p.common_virtual_dimension;p.occupied_slot_i=i;p.occupied_slot_j=j;
    p.borrowed_builder_numerical_bytes=builder.retained_numerical_bytes();p.borrowed_basis_numerical_bytes=basis.memory().retained_output_bytes;
    p.retained_embedding_upper_bytes=add(mul(8,mul(p.common_virtual_dimension,p.common_virtual_dimension)),mul(8,p.common_virtual_dimension));
    p.peak_owned_numerical_bytes=add(add(caps.pair_union.maximum_owned_numerical_bytes,caps.maximum_pao_domain_owned_numerical_bytes),
        add(caps.real_space.maximum_owned_numerical_bytes,caps.embedding.maximum_owned_numerical_bytes));
    p.control_storage_reservation_bytes=add(add(fixed_controls(),builder.retained_control_storage_bytes()),
        add(add(union_control_upper(),embedding_controls()),live.other_live_control_bytes_per_worker));
    const U D=std::min(mul(p.n_cells,p.n_basis),caps.pair_union.maximum_ao_columns);
    if(!D) throw std::invalid_argument("Gaussian pair-domain AO domain cap must be positive");
    p.maximum_domain_dimension=D;
    const U m=std::min(D,p.maximum_generation_dimension);
    p.retained_pair_geometry_upper_bytes=add(add(mul(16,D),mul(32,mul(D,D))),
        add(add(mul(16,mul(D,m)),mul(8,add(D,m))),add(mul(8,mul(p.common_virtual_dimension,m)),mul(8,m))));
    p.retained_geometry_control_upper_bytes=geometry_controls();
    p.pao_domain_work_upper=mul(4096,mul(add(p.n_cells,1),mul(mul(add(p.n_basis,1),mul(add(p.n_basis,1),add(p.n_basis,1))),mul(add(D,1),add(D,1)))));
    p.metadata_validation_work_units=mul(4096,add(32768,add(p.borrowed_builder_numerical_bytes/8,
        add(p.borrowed_basis_numerical_bytes/8,mul(builder.memory_.n_home_occupied,1024)))));
    p.child_work_upper=add(caps.pair_union.maximum_work_units,add(caps.real_space.maximum_work_units,caps.embedding.maximum_work_units));
    p.work_units=add(p.metadata_validation_work_units,add(p.pao_domain_work_upper,p.child_work_upper));
    p.worker_bytes=add(add(p.borrowed_builder_numerical_bytes,p.borrowed_basis_numerical_bytes),add(p.peak_owned_numerical_bytes,
        add(p.control_storage_reservation_bytes,add(live.other_live_numerical_bytes_per_worker,live.backend_margin_bytes_per_worker))));
    reference_node(ref,p.worker_bytes,p.replicas_per_node,p.reference_base_node_bytes,p.required_node_memory_bytes);
    for(U x:{p.peak_owned_numerical_bytes,p.retained_embedding_upper_bytes,p.retained_pair_geometry_upper_bytes,
        p.retained_geometry_control_upper_bytes,p.control_storage_reservation_bytes}) real::extent(x);
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian pair-domain embedding macro owned cap exceeded");
    limit(p.control_storage_reservation_bytes,caps.maximum_control_storage_bytes,"Gaussian pair-domain embedding macro control cap exceeded");
    limit(p.worker_bytes,caps.maximum_worker_bytes,"Gaussian pair-domain embedding macro worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.maximum_node_bytes,"Gaussian pair-domain embedding macro node cap exceeded");
    limit(p.work_units,caps.maximum_work_units,"Gaussian pair-domain embedding macro work cap exceeded");return p;
}

Geometry make_periodic_gaussian_pair_domain_geometry(const Ref& ref,const Basis& basis,
    const Builder& builder,U i,U j,const PeriodicGaussianPairDomainOptions& supplied_options,const Live& supplied_live,const PairCaps& supplied_caps) {
    const auto options=supplied_options;const auto live=supplied_live;const auto caps=supplied_caps;
    const auto p=plan_periodic_gaussian_pair_domain_embedding(ref,basis,builder,i,j,options,live,caps);
    if(payload(builder.domains_,builder.atom_mapping_,*builder.topology_,*builder.common_domain_,*builder.common_space_)!=builder.payload_)
        throw std::invalid_argument("Gaussian pair-domain builder payload differs from its sealed source");
    const auto left=basis.occupied(i),right=basis.occupied(j);
    const auto resolution=resolve_periodic_correlation_placed_pair(*builder.topology_,left.occupied_index,left.cell,right.occupied_index,right.cell);
    const auto row=builder.topology_->row(resolution.row_index);
    const auto& home=builder.domains_.at(row.home_orbital);const auto& partner=builder.domains_.at(row.partner_orbital);
    for(const auto* endpoint:{&home,&partner}) if(endpoint->context_handle()!=builder.context_
        || endpoint->hf_reference_source_identity_sha256()!=builder.hf_ || endpoint->localization_identity_sha256()!=builder.localization_)
        throw std::invalid_argument("Gaussian pair-domain endpoint physical source differs from builder");
    U endpoints=home.domain().diagnostics().retained_numerical_bytes;
    if(&home!=&partner) endpoints=add(endpoints,partner.domain().diagnostics().retained_numerical_bytes);
    const U union_borrowed=add(endpoints,add(builder.memory_.atom_mapping_bytes,builder.memory_.topology_row_bytes));
    PeriodicCorrelationPairPAODomainInventory union_live;
    union_live.other_live_numerical_bytes=add(live.other_live_numerical_bytes_per_worker,
        add(p.borrowed_basis_numerical_bytes,subtract(p.borrowed_builder_numerical_bytes,union_borrowed)));
    union_live.backend_allowance_bytes=live.backend_margin_bytes_per_worker;
    const auto union_bare=plan_periodic_correlation_pair_pao_domain(ref,*builder.topology_,resolution.row_index,
        home.domain(),partner.domain(),builder.memory_.n_atoms,union_live,caps.pair_union);
    const U union_controls=add(union_bare.fixed_control_storage_bytes,union_bare.borrowed_owner_control_bytes);
    if(union_controls>union_control_upper()) throw std::logic_error("Gaussian pair-domain union control formula changed");
    union_live.other_live_control_bytes=subtract(p.control_storage_reservation_bytes,union_controls);
    auto pair=make_periodic_correlation_pair_pao_domain(ref,*builder.topology_,resolution.row_index,home.domain(),partner.domain(),
        builder.atom_mapping_.data(),builder.atom_mapping_.size(),builder.memory_.n_atoms,union_live,caps.pair_union);
    const U D=pair.diagnostics().ao_column_count;if(!D) throw std::invalid_argument("Gaussian pair-domain selected union is empty");
    auto domain=make_periodic_correlation_pao_domain(ref,pair.cell_ao_indices_data(),mul(2,D),D,
        caps.maximum_pao_domain_owned_numerical_bytes,options.domain);
    auto space=make_periodic_correlation_real_pao_space(ref,domain,options.real_space,caps.real_space);
    const U m=space.space().retained_dimension();
    if(!m || m>p.maximum_generation_dimension) throw std::invalid_argument("Gaussian pair-domain rank is empty or exceeds common frame");
    PeriodicCorrelationVirtualBlockSelection selected;selected.begin=0;selected.count=m;selected.translation_cell=resolution.common_translation_cell;
    PeriodicCorrelationRealPAOEmbeddingLiveInventory embedding_live;
    embedding_live.other_live_numerical_bytes_per_worker=add(live.other_live_numerical_bytes_per_worker,
        add(subtract(p.borrowed_builder_numerical_bytes,builder.memory_.common_geometry_numerical_bytes),pair.diagnostics().retained_numerical_bytes));
    embedding_live.fixed_backend_margin_bytes_per_worker=live.backend_margin_bytes_per_worker;
    const auto embedding_bare=plan_periodic_correlation_real_pao_embedding(ref,basis,*builder.common_domain_,*builder.common_space_,basis.virtual_selection(),
        domain,space,selected,options.embedding,embedding_live,caps.embedding);
    if(embedding_bare.control_storage_reservation_bytes!=embedding_controls())
        throw std::logic_error("Gaussian pair-domain embedding control formula changed");
    embedding_live.other_live_control_bytes_per_worker=subtract(p.control_storage_reservation_bytes,embedding_bare.control_storage_reservation_bytes);
    auto result=make_periodic_correlation_real_pao_embedding(ref,basis,*builder.common_domain_,*builder.common_space_,basis.virtual_selection(),
        domain,space,selected,options.embedding,embedding_live,caps.embedding);
    if(result.memory().output_numerical_bytes>p.retained_embedding_upper_bytes)
        throw std::logic_error("Gaussian pair-domain embedding output exceeds its macro admission");
    builder.require_live();common_metadata(ref,basis,*builder.common_domain_,*builder.common_space_);
    if(builder.basis_!=basis.identity_sha256() || payload(builder.domains_,builder.atom_mapping_,*builder.topology_,
        *builder.common_domain_,*builder.common_space_)!=builder.payload_)
        throw std::invalid_argument("Gaussian pair-domain source changed during embedding construction");
    Geometry output;output.state_=ref.state_handle();output.context_=builder.context_;
    output.i_=i;output.j_=j;output.builder_=builder.identity_;output.basis_=basis.identity_sha256();
    output.hf_=builder.hf_;output.allocation_=ref.dimensions().allocation_identity;
    output.numerical_bytes_=add(common_bytes(domain,space),result.memory().output_numerical_bytes);
    if(D>p.maximum_domain_dimension || output.numerical_bytes_>p.retained_pair_geometry_upper_bytes)
        throw std::logic_error("Gaussian pair-domain retained geometry exceeds macro admission");
    Digest seal("vibeqc.periodic.gaussian-pair-domain-geometry.identity-v1");
    for(const auto* value:std::array<const std::string*,11>{&output.builder_,&output.basis_,&output.hf_,&output.allocation_,
        &ref.state().state_identity_sha256(),&domain.pao_domain_identity_sha256(),&domain.matrix_payload_sha256(),
        &space.identity_sha256(),&space.space().payload_sha256(),&result.identity_sha256(),&result.payload_sha256()}) {
        sha(*value);seal.string(*value);
    }
    seal.u64(i);seal.u64(j);seal.u64(D);seal.u64(m);seal.u64(output.numerical_bytes_);
    output.identity_=seal.finish();
    // No fallible work remains, and no source view is retained across moves.
    output.domain_.emplace(std::move(domain));output.space_.emplace(std::move(space));
    output.embedding_.emplace(std::move(result));return output;
}

PeriodicCorrelationRealPAOEmbedding make_periodic_gaussian_pair_domain_embedding(const Ref& ref,const Basis& basis,
    const Builder& builder,U i,U j,const PeriodicGaussianPairDomainOptions& options,const Live& live,const PairCaps& caps) {
    auto geometry=make_periodic_gaussian_pair_domain_geometry(ref,basis,builder,i,j,options,live,caps);
    return std::move(*geometry.embedding_);
}

void Geometry::require_live() const {
    if(!state_ || !context_ || !domain_ || !space_ || !embedding_)
        throw std::invalid_argument("Gaussian pair-domain geometry is empty or consumed");
    const auto& d=*domain_;const auto& s=space_->space();const auto& e=*embedding_;
    if(d.state_handle()!=state_ || s.state_handle()!=state_ || e.state_handle()!=state_
        || s.pao_domain_identity_sha256()!=d.pao_domain_identity_sha256()
        || d.domain_dimension()!=s.domain_dimension() || !s.retained_dimension()
        || e.pair_selection().begin!=0 || e.pair_selection().count!=s.retained_dimension()
        || e.memory().pair_dimension!=s.retained_dimension() || e.common_basis_identity_sha256()!=basis_
        || e.allocation_identity()!=allocation_ || s.allocation_identity()!=allocation_
        || add(common_bytes(d,*space_),e.memory().output_numerical_bytes)!=numerical_bytes_)
        throw std::invalid_argument("Gaussian pair-domain geometry source or retained census differs");
    (void)d.overlap_data();(void)d.fock_data();(void)s.coefficients_data();(void)s.energies_data();
    (void)e.coefficients_data();(void)e.energies_data();
}
const Domain& Geometry::domain() const & {require_live();return *domain_;}
const RealSpace& Geometry::real_space() const & {require_live();return *space_;}
const PeriodicCorrelationRealPAOEmbedding& Geometry::embedding() const & {require_live();return *embedding_;}
PeriodicGaussianPairPNOGeometryView Geometry::geometry_view() const & {
    require_live();return {*domain_,*space_,*embedding_};
}
U Geometry::retained_numerical_bytes() const {require_live();return numerical_bytes_;}
U Geometry::retained_control_storage_bytes() const {require_live();return geometry_controls();}
const Domain& Builder::common_domain() const & {require_live();return *common_domain_;}
const RealSpace& Builder::common_real_space() const & {require_live();return *common_space_;}
} // namespace vibeqc
