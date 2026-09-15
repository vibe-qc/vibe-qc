#include "vibeqc/periodic_gaussian_triple_spaces.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian triple spaces forbid fast/finite-only math"
#endif

namespace vibeqc {
namespace {
namespace real = periodic_correlation_real_local_detail;
namespace local = periodic_correlation_local_detail;
using U = std::uint64_t;
using real::add;
using real::mul;
using real::Digest;
using Plan = PeriodicGaussianTripleSpacesPlan;
using Result = PeriodicGaussianTripleSpaces;
using Live = PeriodicGaussianTripleSpacesLiveInventory;
using Caps = PeriodicGaussianTripleSpacesCaps;
using Options = BoundedRestrictedTNOOptions;
static_assert(sizeof(double) == 8 && FLT_EVAL_METHOD == 0,
    "Gaussian triple spaces require binary64 evaluation");
constexpr char policy[] = "actual-finite-Gaussian;converged-pair-CCSD-connected-T2;MP2-pair-frames-only;"
    "Riplinger2013-Eq10-13-molecular-density-normalization;sorted-occupied-multisets;"
    "all-unordered-triples;no-weak-screening;zero-cutoff-completes-union-not-common;"
    "original-Fvv-recanonicalization;no-triples-amplitudes-or-energy;not-production-DLPNO";
U subtract(U a, U b) {
    if (b > a) throw std::logic_error("Gaussian triple spaces ownership subtraction is inconsistent");
    return a-b;
}
U frame_validation_upper(U n) { return mul(512,add(add(mul(2,mul(n,n)),mul(2,n)),1024)); }
void verify_frame_payload(const PeriodicGaussianPairPNOFrameView& frame,U n) {
    const auto p=plan_periodic_gaussian_pair_pno_frame_validation(frame);
    if (p.work_units>frame_validation_upper(n) || p.control_storage_bytes>65536)
        throw std::logic_error("Gaussian triple spaces frame validation exceeds driver admission");
    frame.verify_payload();
}
U pairs(U n) { return n%2 ? mul(n,add(n,1)/2) : mul(n/2,add(n,1)); }
U triples(U n) {
    std::array<U,3> factors{n,add(n,1),add(n,2)};
    for (U divisor : {2U,3U}) for (U& factor : factors)
        if (factor%divisor == 0) { factor/=divisor; break; }
    return mul(mul(factors[0],factors[1]),factors[2]);
}
void limit(U value, U cap, const char* message) {
    if (value > cap) throw std::length_error(message);
}
void sha(const std::string& s) {
    if (s.size()!=64) throw std::invalid_argument("Gaussian triple spaces require native SHA-256 receipts");
    for (unsigned char ch:s)
        if (!(ch>='0' && ch<='9') && !(ch>='a' && ch<='f'))
            throw std::invalid_argument("Gaussian triple spaces receipt is not lowercase SHA-256");
}
bool placed_same(PeriodicCorrelationPlacedOccupied a, PeriodicCorrelationPlacedOccupied b) {
    return a.occupied_index==b.occupied_index && a.cell==b.cell;
}
bool same_ascii(const std::string& value,const std::array<char,64>& ascii) {
    return value.size()==ascii.size() && std::equal(value.begin(),value.end(),ascii.begin());
}
struct Sources {
    const PeriodicCorrelationAdmittedReference& ref;
    const PeriodicCorrelationRealLocalBasis& basis;
    const PeriodicGaussianRealLocalProvider& provider;
    const PeriodicGaussianPairMP2Result& mp2;
    const PeriodicGaussianPairCCSDResult& ccsd;
};
void basic_sources(const Sources& s, U o, U n, U P) {
    local::validate_reference(s.ref);
    const auto& provider=s.provider.provider();
    const auto& context=s.provider.context_handle();
    if (!context || !s.provider.matched_finite_gaussian_hf_recipe()
        || !provider.matched_finite_gaussian_hf_recipe()
        || s.basis.state_handle().get()!=s.ref.state_handle().get()
        || provider.state_handle().get()!=s.ref.state_handle().get()
        || s.basis.allocation_identity()!=s.ref.dimensions().allocation_identity
        || provider.identity_sha256()!=s.provider.identity_sha256()
        || !same_ascii(provider.source_context_identity_sha256(),context->source_context_identity_ascii())
        || !same_ascii(s.provider.source_context_identity_sha256(),context->source_context_identity_ascii())
        || provider.hf_reference_source_identity_sha256()!=s.provider.hf_reference_source_identity_sha256()
        || provider.local_basis_identity_sha256()!=s.basis.local_basis_identity_sha256()
        || provider.basis_certificate_identity_sha256()!=s.basis.identity_sha256())
        throw std::invalid_argument("Gaussian triple spaces require matching exact reference/basis/provider owners");
    const auto integral=provider.integral_provider(s.basis); // validates moved native owners, no ERI call
    const auto& bp=s.basis.memory();const auto& pp=provider.memory();
    if (!o || !n || bp.orbital_count!=add(o,n) || pp.occupied_count!=o || pp.virtual_count!=n
        || pp.orbital_count!=bp.orbital_count || pp.n_cells!=s.ref.state().n_kpoints()
        || pp.n_auxiliary!=s.ref.dimensions().n_auxiliary
        || context->mesh().mesh()!=s.ref.state().mesh() || context->mesh().is_shift()!=s.ref.state().is_shift()
        || context->inventory().ao.function_count!=s.ref.state().n_basis()
        || context->inventory().auxiliary.function_count!=pp.n_auxiliary
        || s.provider.memory().retained_row_bytes!=pp.retained_row_bytes
        || s.provider.memory().occupied_count!=o || s.provider.memory().virtual_count!=n
        || integral.retained_numerical_bytes!=add(pp.retained_row_bytes,mul(16,o))
        || integral.maximum_transient_numerical_bytes)
        throw std::invalid_argument("Gaussian triple spaces actual provider dimensions or inventory differ");
    if (!s.mp2.converged() || !s.ccsd.converged()
        || !s.mp2.matched_finite_gaussian_hf_recipe() || !s.ccsd.matched_finite_gaussian_hf_recipe()
        || (s.ccsd.split_bare_particle_hole() && !s.ccsd.particle_hole_physical_source_certified())
        || s.ccsd.state_handle().get()!=s.ref.state_handle().get()
        || s.ccsd.context_handle().get()!=s.provider.context_handle().get()
        || s.mp2.state_handle().get()!=s.ref.state_handle().get()
        || s.mp2.context_handle().get()!=s.provider.context_handle().get()
        || s.ccsd.memory().domain_generated!=s.mp2.domain_generated()
        || s.mp2.memory().occupied_count!=o || s.mp2.memory().common_virtual_dimension!=n
        || s.mp2.memory().pair_count!=P || s.mp2.diagnostics().completed_pairs!=P
        || s.ccsd.memory().occupied_count!=o || s.ccsd.memory().common_virtual_dimension!=n
        || s.ccsd.memory().pair_count!=P || s.ccsd.diagnostics().completed_singles!=o
        || s.ccsd.warmstart_identity_sha256()!=s.mp2.identity_sha256()
        || s.ccsd.pair_spaces_identity_sha256()!=s.mp2.pair_spaces_identity_sha256()
        || s.ccsd.basis_identity_sha256()!=s.basis.identity_sha256()
        || s.mp2.provider_identity_sha256()!=s.provider.identity_sha256()
        || s.ccsd.provider_identity_sha256()!=s.provider.identity_sha256()
        || s.mp2.hf_reference_source_identity_sha256()!=s.provider.hf_reference_source_identity_sha256()
        || s.ccsd.hf_reference_source_identity_sha256()!=s.provider.hf_reference_source_identity_sha256())
        throw std::invalid_argument("Gaussian triple spaces require matching converged actual CCSD and original MP2 frames");
    for (const auto* solver : {&s.ccsd.solver().memory()})
        if (solver->n_occupied!=o || solver->common_virtual_dimension!=n || solver->pair_count!=P)
            throw std::invalid_argument("Gaussian triple spaces CCSD solver dimensions differ");
    const auto& mp=s.mp2.solver().memory();
    if (mp.n_occupied!=o || mp.common_virtual_dimension!=n || mp.pair_count!=P)
        throw std::invalid_argument("Gaussian triple spaces MP2 solver dimensions differ");
    for (const auto* value : std::array<const std::string*,16>{&s.ref.state().state_identity_sha256(),
        &s.ref.dimensions().allocation_identity,&s.basis.identity_sha256(),&s.basis.payload_sha256(),
        &s.provider.identity_sha256(),&s.provider.hf_reference_source_identity_sha256(),
        &s.provider.consumed_sources_identity_sha256(),&s.mp2.identity_sha256(),
        &s.mp2.pair_spaces_identity_sha256(),&s.mp2.solver().input_identity_sha256(),&s.mp2.solver().payload_sha256(),
        &s.ccsd.identity_sha256(),&s.ccsd.singles_spaces_identity_sha256(),
        &s.ccsd.solver().input_identity_sha256(),&s.ccsd.solver().payload_sha256(),&s.provider.source_context_identity_sha256()}) sha(*value);
}
void pno_source(const Sources& s, const PeriodicGaussianPairPNOFrameView& pn, U i,U j,U n) {
    if (pn.state_handle().get()!=s.ref.state_handle().get()
        || pn.context_handle().get()!=s.provider.context_handle().get()
        || pn.basis_identity_sha256()!=s.basis.identity_sha256()
        || pn.provider_identity_sha256()!=s.provider.identity_sha256()
        || pn.hf_reference_source_identity_sha256()!=s.provider.hf_reference_source_identity_sha256()
        || pn.occupied_count()!=s.basis.memory().occupied_count
        || pn.occupied_slot_i()!=i || pn.occupied_slot_j()!=j || pn.common_virtual_dimension()!=n
        || pn.is_embedded()!=s.mp2.domain_generated()
        || !placed_same(pn.occupied_i(),s.basis.occupied(i)) || !placed_same(pn.occupied_j(),s.basis.occupied(j)))
        throw std::invalid_argument("Gaussian triple spaces PNO state/context/frame/occupied identity differs");
    for (const auto* value : {&pn.identity_sha256(),&pn.payload_sha256(),&pn.basis_identity_sha256(),
        &pn.provider_identity_sha256(),&pn.hf_reference_source_identity_sha256(),&pn.common_exchange_integral_identity_sha256(),
        &pn.initial_amplitude_identity_sha256(),&pn.density_identity_sha256()}) sha(*value);
    if (pn.is_embedded()) {
        const auto& e=pn.embedded();
        for (const auto* value:{&e.embedding_identity_sha256(),&e.raw_exchange_identity_sha256(),
            &e.projected_exchange_identity_sha256(),&e.raw_fock_identity_sha256(),&e.projected_fock_identity_sha256(),
            &e.source_payload_receipt_sha256()}) sha(*value);
        if (i==j) {
            const auto& source=s.mp2.diagonal_generation_embedding(i);
            if (source.state_handle().get()!=s.ref.state_handle().get()
                || source.allocation_identity()!=s.ref.dimensions().allocation_identity
                || source.common_basis_identity_sha256()!=s.basis.identity_sha256()
                || source.memory().common_dimension!=n || source.memory().pair_dimension!=pn.generation_dimension()
                || source.identity_sha256()!=e.embedding_identity_sha256())
                throw std::invalid_argument("Gaussian triple spaces original diagonal generation frame differs");
            (void)source.coefficients_data();(void)source.energies_data();
        }
    }
}
void validate_all_owners(const Sources& s,const Plan& p) {
    const U o=p.occupied_count,n=p.common_virtual_dimension,P=p.pair_count;
    basic_sources(s,o,n,P);
    U pair_bytes=0,pair_elements=0,mp2_borrowed=0,singles_bytes=0,singles_elements=0;
    U pair_generation=0,singles_generation=0,embedding_bytes=0,pair_receipt_bytes=0;
    U pair_generation_coefficients=0,singles_generation_coefficients=0;
    U geometry_bytes=0,geometry_controls=0;
    for (U i=0;i<o;++i) {
        const auto pn=s.ccsd.singles_frame(i);pno_source(s,pn,i,i,n);
        const U r=pn.retained_dimension(),m=pn.generation_dimension();
        const auto t=s.ccsd.solver().stored_singles_view(i);
        if (r>n || s.ccsd.solver().singles_rank(i)!=r || pn.coefficients().size()!=mul(n,r)
            || pn.energies().size()!=r || pn.original_pno_occupations().size()!=m || t.element_count!=r)
            throw std::invalid_argument("Gaussian triple spaces independent CCSD singles payload differs");
        const U generation_bytes=pn.is_embedded()?mul(8,mul(m,r)):0;
        const U bytes=add(generation_bytes,mul(8,add(add(mul(n,r),r),m)));
        singles_generation_coefficients=add(singles_generation_coefficients,generation_bytes);
        if (bytes!=pn.retained_numerical_bytes())
            throw std::logic_error("Gaussian triple spaces singles retained inventory differs");
        singles_bytes=add(singles_bytes,bytes);singles_elements=add(singles_elements,r);
        singles_generation=add(singles_generation,m);
        if (s.mp2.domain_generated())
            embedding_bytes=add(embedding_bytes,s.mp2.diagonal_generation_embedding(i).memory().output_numerical_bytes);
        for (U j=i;j<o;++j) {
            if(s.mp2.domain_generated()) {
                const auto& g=s.mp2.pair_generation_geometry(i,j);
                geometry_bytes=add(geometry_bytes,g.retained_numerical_bytes());
                geometry_controls=add(geometry_controls,g.retained_control_storage_bytes());
                for(const auto* receipt:{&g.identity_sha256(),&g.builder_identity_sha256(),&g.basis_identity_sha256(),
                    &g.hf_reference_source_identity_sha256(),&g.allocation_identity()}) sha(*receipt);
            }
            const auto& pair=s.mp2.pair(i,j);const auto pair_pno=pair.frame();
            pno_source(s,pair_pno,i,j,n);const U rank=pair.memory().retained_dimension;
            const auto cc=s.ccsd.solver().stored_pair_view(i,j);
            const auto mp=s.mp2.solver().stored_amplitudes_view(i,j);
            if (rank>n || pair.memory().virtual_count!=n
                || pair.memory().generation_dimension!=pair_pno.generation_dimension() || pair.memory().occupied_slot_i!=i
                || pair.memory().occupied_slot_j!=j || pair_pno.retained_dimension()!=rank
                || pair.allocation_identity()!=s.ref.dimensions().allocation_identity
                || pair.local_basis_identity_sha256()!=s.basis.local_basis_identity_sha256()
                || pair.consumed_sources_identity_sha256()!=s.provider.consumed_sources_identity_sha256()
                || pair.coefficients().size()!=mul(n,rank) || pair.energies().size()!=rank
                || pair.exchange_integrals().size()!=mul(rank,rank)
                || pair_pno.original_pno_occupations().size()!=pair_pno.generation_dimension()
                || s.ccsd.solver().pair_rank(i,j)!=rank || s.mp2.solver().pair_rank(i,j)!=rank
                || cc.element_count!=mul(rank,rank) || mp.element_count!=mul(rank,rank))
                throw std::invalid_argument("Gaussian triple spaces pair frame/CCSD amplitude/source identity differs");
            for (const auto* value : {&pair.identity_sha256(),&pair.payload_sha256(),
                &pair.raw_exchange_integral_identity_sha256(),&pair.exchange_integral_identity_sha256(),
                &pair.allocation_identity(),&pair.local_basis_identity_sha256(),&pair.consumed_sources_identity_sha256()}) sha(*value);
            const U borrowed=mul(8,add(add(mul(n,rank),rank),mul(rank,rank)));
            const U local_bytes=pair_pno.is_embedded()?mul(8,mul(pair_pno.generation_dimension(),rank)):0;
            const U bytes=add(local_bytes,add(borrowed,mul(8,pair_pno.generation_dimension())));
            pair_generation_coefficients=add(pair_generation_coefficients,local_bytes);
            if (bytes!=pair.memory().retained_output_bytes)
                throw std::logic_error("Gaussian triple spaces pair retained inventory differs");
            pair_bytes=add(pair_bytes,bytes);mp2_borrowed=add(mp2_borrowed,borrowed);
            pair_generation=add(pair_generation,pair_pno.generation_dimension());
            pair_receipt_bytes=add(pair_receipt_bytes,add(7U*65U,pair_pno.retained_receipt_payload_bytes()));
            pair_elements=add(pair_elements,mul(rank,rank));
        }
    }
    const auto& cc=s.ccsd.solver().memory();const auto& mp=s.mp2.solver().memory();
    if (pair_bytes!=s.mp2.diagnostics().retained_pair_bytes || singles_bytes!=s.ccsd.diagnostics().retained_singles_bytes
        || mp.total_amplitude_elements!=pair_elements || mp.borrowed_pair_numeric_bytes!=mp2_borrowed
        || mp.amplitude_snapshot_bytes!=mul(8,pair_elements) || mp.retained_pair_record_bytes!=mul(16,P)
        || mp.output_numerical_bytes!=add(mul(8,pair_elements),mul(16,P))
        || cc.total_singles_elements!=singles_elements || cc.total_doubles_elements!=pair_elements
        || cc.amplitude_snapshot_bytes!=mul(8,add(singles_elements,pair_elements))
        || cc.retained_record_bytes!=mul(16,add(o,P))
        || cc.output_numerical_bytes!=add(cc.amplitude_snapshot_bytes,cc.retained_record_bytes)
        || add(geometry_bytes,add(pair_bytes,mp.output_numerical_bytes))!=p.borrowed_mp2_numerical_bytes
        || add(singles_bytes,cc.output_numerical_bytes)!=p.borrowed_ccsd_numerical_bytes
        || pair_generation!=s.mp2.diagnostics().generation_dimension_sum
        || pair_generation_coefficients!=s.mp2.diagnostics().retained_generation_coefficient_bytes
        || singles_generation_coefficients!=s.ccsd.diagnostics().retained_singles_generation_coefficient_bytes
        || singles_generation!=s.ccsd.diagnostics().singles_generation_dimension_sum
        || singles_generation!=s.ccsd.memory().singles_generation_dimension_sum
        || embedding_bytes!=s.mp2.diagnostics().retained_generation_embedding_bytes
        || geometry_bytes!=s.mp2.diagnostics().retained_pair_geometry_bytes
        || geometry_controls!=s.mp2.diagnostics().retained_pair_geometry_control_bytes
        || pair_receipt_bytes!=s.mp2.memory().retained_pair_seal_bytes
        || (s.mp2.domain_generated() && singles_generation!=s.mp2.diagnostics().diagonal_generation_dimension_sum)
        || (!s.mp2.domain_generated() && (pair_generation!=mul(P,n) || singles_generation!=mul(o,n))))
        throw std::logic_error("Gaussian triple spaces complete retained owner census differs");
}
BoundedRestrictedTNOInput geometry_input(const Sources& s,U i,U j,U k) {
    BoundedRestrictedTNOInput in;
    in.n_occupied=s.basis.memory().occupied_count;in.common_dimension=s.basis.memory().virtual_count;
    in.occupied={i,j,k};in.virtual_fock={s.basis.f_vv_data(),static_cast<std::size_t>(mul(in.common_dimension,in.common_dimension))};
    const std::array<U,3> left{i,i,j},right{j,k,k};
    for (U e=0;e<3;++e) {
        const auto& C=s.mp2.pair(left[e],right[e]).coefficients();
        const auto T=s.ccsd.solver().stored_pair_view(left[e],right[e]);
        in.edges[e]={left[e],right[e],s.ccsd.solver().pair_rank(left[e],right[e]),
            {C.data(),C.size()},{T.data,T.element_count}};
    }
    return in;
}
U full_source_numeric(const Plan& p) {
    return add(add(p.borrowed_basis_bytes,p.borrowed_provider_row_bytes),
        add(p.borrowed_mp2_numerical_bytes,p.borrowed_ccsd_numerical_bytes));
}
BoundedRestrictedTNOInventory leaf_inventory(const Plan& p,const Live& live,U retained,
    const BoundedRestrictedTNOMemoryPlan& leaf) {
    BoundedRestrictedTNOInventory inv;
    inv.numerical_replicas=p.replicas_per_node;inv.external_node_bytes=p.reference_base_node_bytes;
    inv.other_live_numerical_bytes_per_replica=add(live.other_live_numerical_bytes_per_worker,
        add(retained,subtract(full_source_numeric(p),leaf.borrowed_numerical_bytes)));
    inv.other_live_control_bytes_per_replica=subtract(p.control_storage_reservation_bytes,leaf.fixed_inventoried_object_bytes);
    inv.backend_margin_bytes_per_replica=live.fixed_backend_margin_bytes_per_worker;
    return inv;
}
void geometry_caps(const BoundedRestrictedTNOMemoryPlan& p,const BoundedRestrictedTNOCaps& c) {
    limit(p.common_dimension,c.maximum_common_dimension,"Gaussian triple spaces leaf common cap");
    limit(p.union_columns,c.maximum_union_columns,"Gaussian triple spaces leaf union cap");
    limit(p.peak_owned_numerical_bytes,c.maximum_owned_numerical_bytes,"Gaussian triple spaces leaf owned cap");
    limit(p.total_node_bytes,c.maximum_node_bytes,"Gaussian triple spaces leaf node cap");
    limit(p.control_storage_bytes_per_replica,c.maximum_control_storage_bytes_per_replica,"Gaussian triple spaces leaf control cap");
    limit(p.work_units_upper_bound,c.maximum_work_units,"Gaussian triple spaces leaf work cap");
}
BoundedRestrictedTNOMemoryPlan geometry_plan(const BoundedRestrictedTNOInput& in,const Options& options,
    const Plan& p,const Live& live,const Caps& caps,U retained,BoundedRestrictedTNOInventory* inventory=nullptr) {
    BoundedRestrictedTNOInventory minimal;
    minimal.backend_margin_bytes_per_replica=live.fixed_backend_margin_bytes_per_worker;
    const auto local_plan=plan_bounded_restricted_triple_natural_orbitals(in,options,minimal);
    if (local_plan.fixed_inventoried_object_bytes>p.leaf_fixed_control_reservation_bytes)
        throw std::logic_error("Gaussian triple spaces leaf control reservation changed");
    const auto inv=leaf_inventory(p,live,retained,local_plan);
    const auto plan=plan_bounded_restricted_triple_natural_orbitals(in,options,inv);
    geometry_caps(plan,caps.geometry);
    limit(plan.total_node_bytes,p.required_node_memory_bytes,"Gaussian triple spaces leaf exceeds enclosing node inventory");
    if (inventory) *inventory=inv;
    return plan;
}
struct Events {
    const Sources& sources;const Plan& plan;
    PeriodicGaussianTripleSpacesCallback callback=nullptr;void* context=nullptr;
    PeriodicGaussianTripleSpacesProgress event;
    std::array<char,64> ccsd{},mp2{},basis{},provider{};
    void pin() {
        std::copy(sources.ccsd.identity_sha256().begin(),sources.ccsd.identity_sha256().end(),ccsd.begin());
        std::copy(sources.mp2.identity_sha256().begin(),sources.mp2.identity_sha256().end(),mp2.begin());
        std::copy(sources.basis.identity_sha256().begin(),sources.basis.identity_sha256().end(),basis.begin());
        std::copy(sources.provider.identity_sha256().begin(),sources.provider.identity_sha256().end(),provider.begin());
    }
    void emit(PeriodicGaussianTripleSpacesStage stage) {
        if (event.callback_count>=plan.progress_callback_upper_bound)
            throw std::length_error("Gaussian triple spaces progress count exceeds admission");
        event.stage=stage;++event.callback_count;
        if (callback) callback(event,context);
        real::float_environment();validate_all_owners(sources,plan);
        if (!std::equal(ccsd.begin(),ccsd.end(),sources.ccsd.identity_sha256().begin())
            || !std::equal(mp2.begin(),mp2.end(),sources.mp2.identity_sha256().begin())
            || !std::equal(basis.begin(),basis.end(),sources.basis.identity_sha256().begin())
            || !std::equal(provider.begin(),provider.end(),sources.provider.identity_sha256().begin()))
            throw std::invalid_argument("Gaussian triple spaces input owner changed across callback");
    }
};
}  // namespace

Plan plan_periodic_gaussian_triple_spaces(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& mp2,const PeriodicGaussianPairCCSDResult& ccsd,
    const Options& options,const Live& live,const Caps& caps) {
    real::float_environment();
    if (!live.fixed_backend_margin_bytes_per_worker)
        throw std::invalid_argument("Gaussian triple spaces backend margin must be explicit and positive");
    const Sources sources{ref,basis,provider,mp2,ccsd};Plan p;
    const U o=p.occupied_count=basis.memory().occupied_count,n=p.common_virtual_dimension=basis.memory().virtual_count;
    if (!o || !n) throw std::invalid_argument("Gaussian triple spaces dimensions must be positive");
    const U P=p.pair_count=pairs(o),Q=p.triple_count=triples(o),nn=mul(n,n);
    limit(P,caps.maximum_pair_count,"Gaussian triple spaces pair count cap");
    limit(Q,caps.maximum_triple_count,"Gaussian triple spaces triple count cap");
    // Fixed-size metadata only until the complete enclosing admission below.
    // No coefficient/amplitude traversal, payload scan or callback yet.
    basic_sources(sources,o,n,P);
    p.borrowed_basis_bytes=basis.memory().retained_output_bytes;
    p.borrowed_provider_row_bytes=provider.memory().retained_row_bytes;
    p.borrowed_mp2_numerical_bytes=add(mp2.diagnostics().retained_pair_geometry_bytes,
        add(mp2.diagnostics().retained_pair_bytes,mp2.solver().memory().output_numerical_bytes));
    p.borrowed_ccsd_numerical_bytes=add(ccsd.diagnostics().retained_singles_bytes,ccsd.solver().memory().output_numerical_bytes);
    const U expected_basis=add(mul(16,o),mul(8,add(add(mul(o,o),nn),mul(o,n))));
    if (p.borrowed_basis_bytes!=expected_basis || ref.state_resident_bytes()!=ref.state().resident_bytes()
        || ref.dimensions().external_bytes<ref.state_resident_bytes())
        throw std::logic_error("Gaussian triple spaces basis/reference inventory differs");
    p.borrowed_mp2_control_bytes=add(sizeof(PeriodicGaussianPairMP2Result),add(mul(P,sizeof(PeriodicGaussianPairSpace)),add(mp2.memory().retained_pair_seal_bytes,6U*65U)));
    p.borrowed_mp2_control_bytes=add(p.borrowed_mp2_control_bytes,mp2.diagnostics().retained_pair_geometry_control_bytes);
    p.borrowed_ccsd_control_bytes=add(sizeof(PeriodicGaussianPairCCSDResult),add(mul(o,sizeof(PeriodicGaussianPairPNOFrame)),
        add(mul(o,(mp2.domain_generated()?14U:8U)*65U),9U*65U)));
    // Optional split-execution/source receipts are additional owned payloads,
    // not replacements for the original input/amplitude codec strings.
    for (const auto* receipt : {&ccsd.solver().split_execution_identity_sha256(),
        &ccsd.solver().consumed_particle_hole_identity_sha256(),
        &ccsd.physical_particle_hole_source_identity_sha256()}) {
        if (!receipt->empty()) {
            sha(*receipt);
            p.borrowed_ccsd_control_bytes=add(p.borrowed_ccsd_control_bytes,add(receipt->size(),1));
        }
    }
    const U output=add(mul(8,nn),mul(16,n));
    p.retained_triple_output_upper_bytes=mul(Q,output);
    p.serialized_leaf_owned_ceiling=caps.geometry.maximum_owned_numerical_bytes;
    p.peak_owned_numerical_bytes=std::max(p.retained_triple_output_upper_bytes,
        add(mul(Q-1,output),p.serialized_leaf_owned_ceiling));
    p.retained_triple_object_bytes=mul(Q,sizeof(BoundedRestrictedTNOResult));
    p.retained_triple_seal_bytes=mul(Q,2U*65U);
    p.leaf_fixed_control_reservation_bytes=65536U+4U*sizeof(BoundedRestrictedTNOResult)
        +4U*sizeof(BoundedRestrictedTNOInput)+4U*sizeof(BoundedRestrictedTNOMemoryPlan)
        +4U*sizeof(RestrictedPairSemicanonicalResult);
    U control=65536U+sizeof(Result)+sizeof(Plan)+sizeof(Events)+sizeof(Options)+sizeof(Live)+sizeof(Caps)
        +sizeof(PeriodicCorrelationAdmittedReference)+sizeof(PeriodicCorrelationRealLocalBasis)
        +sizeof(PeriodicGaussianRealLocalProvider)+sizeof(PeriodicGaussianSourceContext)+10U*65U;
    for (U bytes:{p.borrowed_mp2_control_bytes,p.borrowed_ccsd_control_bytes,p.retained_triple_object_bytes,
        p.retained_triple_seal_bytes,p.leaf_fixed_control_reservation_bytes,live.other_live_control_bytes_per_worker}) control=add(control,bytes);
    p.control_storage_reservation_bytes=control;
    const auto& d=ref.dimensions();const auto& budget=ref.budget();
    p.replicas_per_node=mul(budget.mpi_ranks,budget.workers_per_rank);
    p.reference_base_node_bytes=add(add(d.external_bytes,d.shared_bytes),mul(budget.mpi_ranks,add(d.per_rank_bytes,d.localization_window_bytes_per_rank)));
    p.per_worker_inventoried_bytes=add(full_source_numeric(p),add(p.peak_owned_numerical_bytes,
        add(control,add(live.other_live_numerical_bytes_per_worker,live.fixed_backend_margin_bytes_per_worker))));
    p.required_node_memory_bytes=add(p.reference_base_node_bytes,mul(p.replicas_per_node,p.per_worker_inventoried_bytes));
    p.progress_callback_upper_bound=add(Q,2);
    // Receipt/shape checks traverse all P pair and o singles owners at every
    // callback. Domain-generated source frame payloads are verified once in
    // the factory, separately charged here; the planner remains metadata-only.
    p.driver_work_units=mul(32768,mul(add(add(add(P,o),Q),1),add(Q,4)));
    if (mp2.domain_generated()) p.driver_work_units=add(p.driver_work_units,mul(add(P,o),frame_validation_upper(n)));
    p.work_units_upper_bound=add(p.driver_work_units,mul(Q,caps.geometry.maximum_work_units));
    real::extent(p.peak_owned_numerical_bytes);real::extent(p.retained_triple_object_bytes);
    if (Q>std::vector<BoundedRestrictedTNOResult>().max_size())
        throw std::length_error("Gaussian triple spaces result table exceeds native vector extent");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian triple spaces owned cap");
    limit(control,caps.maximum_control_storage_bytes_per_worker,"Gaussian triple spaces control cap");
    limit(p.per_worker_inventoried_bytes,caps.maximum_per_worker_inventoried_bytes,"Gaussian triple spaces worker cap");
    limit(p.required_node_memory_bytes,caps.maximum_node_inventoried_bytes,"Gaussian triple spaces node cap");
    limit(p.required_node_memory_bytes,budget.memory_limit_bytes,"Gaussian triple spaces admitted reference node cap");
    limit(p.progress_callback_upper_bound,caps.maximum_progress_callbacks,"Gaussian triple spaces callback cap");
    limit(p.work_units_upper_bound,caps.maximum_work_units,"Gaussian triple spaces work cap");
    validate_all_owners(sources,p);
    U retained=0;
    for (U i=0;i<o;++i) for (U j=i;j<o;++j) for (U k=j;k<o;++k) {
        const auto input=geometry_input(sources,i,j,k);
        const auto leaf=geometry_plan(input,options,p,live,caps,retained);
        p.exact_rank_plan_peak_owned_bytes=std::max(p.exact_rank_plan_peak_owned_bytes,add(retained,leaf.peak_owned_numerical_bytes));
        retained=add(retained,leaf.output_numerical_bytes_upper_bound);
        p.exact_rank_plan_geometry_work_units=add(p.exact_rank_plan_geometry_work_units,leaf.work_units_upper_bound);
    }
    p.exact_rank_plan_output_upper_bytes=retained;
    p.exact_rank_plan_peak_owned_bytes=std::max(p.exact_rank_plan_peak_owned_bytes,retained);
    if (retained>p.retained_triple_output_upper_bytes || p.exact_rank_plan_peak_owned_bytes>p.peak_owned_numerical_bytes
        || p.exact_rank_plan_geometry_work_units>mul(Q,caps.geometry.maximum_work_units))
        throw std::logic_error("Gaussian triple spaces exact metadata plan exceeds initial enclosing admission");
    return p;
}

const BoundedRestrictedTNOResult& Result::triple(U i,U j,U k) const {
    if (!state_ || !context_ || triples_.size()!=memory_.triple_count
        || diagnostics_.completed_triples!=memory_.triple_count)
        throw std::logic_error("Gaussian triple spaces owner is moved or incomplete");
    const U o=memory_.occupied_count;
    if (i>j || j>k || k>=o) throw std::out_of_range("Gaussian triple spaces require canonical i<=j<=k");
    U at=subtract(triples(o),triples(o-i));
    // Sum_{b=i}^{j-1}(o-b), without a traversal or unchecked products.
    at=add(at,subtract(pairs(o-i),pairs(o-j)));at=add(at,k-j);
    const auto& value=triples_.at(static_cast<std::size_t>(at));
    if (value.occupied!=std::array<U,3>{i,j,k} || value.union_rank>memory_.common_virtual_dimension
        || value.retained_rank>value.union_rank || value.occupations.size()!=value.union_rank
        || value.semicanonical.coefficients.size()!=mul(memory_.common_virtual_dimension,value.retained_rank)
        || value.semicanonical.energies.size()!=value.retained_rank)
        throw std::logic_error("Gaussian triple spaces retained geometry is moved or malformed");
    return value;
}

void validate_periodic_gaussian_triple_spaces(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& mp2,const PeriodicGaussianPairCCSDResult& ccsd,
    const PeriodicGaussianTripleSpaces& spaces) {
    real::float_environment();
    const auto& p=spaces.memory();const auto& diag=spaces.diagnostics();
    const U o=basis.memory().occupied_count,n=basis.memory().virtual_count;
    if (!o || !n || p.occupied_count!=o || p.common_virtual_dimension!=n
        || p.pair_count!=pairs(o) || p.triple_count!=triples(o)
        || diag.completed_triples!=p.triple_count
        || p.progress_callback_upper_bound!=add(p.triple_count,2)
        || diag.completed_progress_callbacks!=p.progress_callback_upper_bound)
        throw std::invalid_argument("Gaussian triple spaces downstream dimension/completion census differs");
    const Sources sources{ref,basis,provider,mp2,ccsd};
    validate_all_owners(sources,p);
    if (spaces.state_handle().get()!=ref.state_handle().get()
        || spaces.context_handle().get()!=provider.context_handle().get()
        || spaces.allocation_identity()!=ref.dimensions().allocation_identity
        || spaces.ccsd_identity_sha256()!=ccsd.identity_sha256()
        || spaces.ccsd_amplitude_payload_sha256()!=ccsd.solver().payload_sha256()
        || spaces.warmstart_identity_sha256()!=mp2.identity_sha256()
        || spaces.pair_spaces_identity_sha256()!=mp2.pair_spaces_identity_sha256()
        || spaces.basis_identity_sha256()!=basis.identity_sha256()
        || spaces.provider_identity_sha256()!=provider.identity_sha256()
        || spaces.hf_reference_source_identity_sha256()!=provider.hf_reference_source_identity_sha256()
        || !spaces.matched_finite_gaussian_hf_recipe())
        throw std::invalid_argument("Gaussian triple spaces downstream exact owner/source/frame receipt differs");
    for (const auto* value : {&spaces.identity_sha256(),&spaces.spaces_identity_sha256(),
        &spaces.ccsd_identity_sha256(),&spaces.ccsd_amplitude_payload_sha256(),&spaces.warmstart_identity_sha256(),
        &spaces.pair_spaces_identity_sha256(),&spaces.basis_identity_sha256(),&spaces.provider_identity_sha256(),
        &spaces.hf_reference_source_identity_sha256(),&spaces.allocation_identity()}) sha(*value);
    const auto& dimensions=ref.dimensions();const auto& budget=ref.budget();
    const U base=add(add(dimensions.external_bytes,dimensions.shared_bytes),
        mul(budget.mpi_ranks,add(dimensions.per_rank_bytes,dimensions.localization_window_bytes_per_rank)));
    if (p.replicas_per_node!=mul(budget.mpi_ranks,budget.workers_per_rank)
        || p.reference_base_node_bytes!=base || p.borrowed_basis_bytes!=basis.memory().retained_output_bytes
        || p.borrowed_provider_row_bytes!=provider.memory().retained_row_bytes)
        throw std::invalid_argument("Gaussian triple spaces downstream reference/resource inventory differs");
    U retained=0,work=0,empty_union=0,empty_retained=0,all_equal=0,completed=0;
    U total_rank=0,total_union=0,maximum_rank=0;
    bool full_union=true,full_retained=true;
    for (U i=0;i<o;++i) for (U j=i;j<o;++j) for (U k=j;k<o;++k) {
        const auto& geometry=spaces.triple(i,j,k);
        const auto input=geometry_input(sources,i,j,k);
        U columns=0,unique=0,maximum_edge_rank=0;
        for (U e=0;e<3;++e) {
            bool repeated=false;
            for (U previous=0;previous<e;++previous)
                repeated=repeated || (input.edges[e].occupied_i==input.edges[previous].occupied_i
                    && input.edges[e].occupied_j==input.edges[previous].occupied_j);
            if (!repeated) {
                columns=add(columns,input.edges[e].rank);++unique;
                maximum_edge_rank=std::max(maximum_edge_rank,input.edges[e].rank);
            }
        }
        const U u=geometry.union_rank,r=geometry.retained_rank;
        const U bytes=mul(8,add(add(mul(n,r),r),u));
        const auto& gm=geometry.memory;
        if (u>std::min(n,columns) || (columns==0)!=(u==0) || geometry.usable!=(r!=0)
            || gm.common_dimension!=n || gm.union_columns!=columns || gm.unique_edge_count!=unique
            || gm.maximum_edge_rank!=maximum_edge_rank || gm.union_rank_upper_bound!=std::min(n,columns)
            || geometry.semicanonical.domain_dimension!=n || geometry.semicanonical.selected_dimension!=r
            || geometry.semicanonical.memory.domain_dimension!=n || geometry.semicanonical.memory.selected_dimension!=r
            || geometry.semicanonical.eigensolver_performed!=(r!=0)
            || geometry.output_numerical_bytes!=bytes || bytes>gm.output_numerical_bytes_upper_bound
            || gm.peak_owned_numerical_bytes>p.serialized_leaf_owned_ceiling)
            throw std::invalid_argument("Gaussian triple spaces downstream triple rank/extent/inventory differs");
        const auto& a=geometry.options;const auto& b=spaces.options();
        if (a.occupation_cutoff!=b.occupation_cutoff
            || a.union_absolute_rank_cutoff!=b.union_absolute_rank_cutoff
            || a.union_relative_rank_cutoff!=b.union_relative_rank_cutoff
            || a.maximum_union_column_reconstruction_error!=b.maximum_union_column_reconstruction_error
            || a.input_orthonormality_tolerance!=b.input_orthonormality_tolerance
            || a.eigensystem_relative_reconstruction_tolerance!=b.eigensystem_relative_reconstruction_tolerance
            || a.eigenvector_orthogonality_tolerance!=b.eigenvector_orthogonality_tolerance
            || a.union_negative_absolute_tolerance!=b.union_negative_absolute_tolerance
            || a.union_negative_relative_tolerance!=b.union_negative_relative_tolerance
            || a.density_negative_absolute_tolerance!=b.density_negative_absolute_tolerance
            || a.density_negative_relative_tolerance!=b.density_negative_relative_tolerance
            || a.occupation_ambiguity_absolute_guard!=b.occupation_ambiguity_absolute_guard
            || a.occupation_ambiguity_relative_guard!=b.occupation_ambiguity_relative_guard
            || a.union_eigensolver.max_sweeps!=b.union_eigensolver.max_sweeps
            || a.union_eigensolver.relative_offdiagonal_tolerance!=b.union_eigensolver.relative_offdiagonal_tolerance
            || a.density_eigensolver.max_sweeps!=b.density_eigensolver.max_sweeps
            || a.density_eigensolver.relative_offdiagonal_tolerance!=b.density_eigensolver.relative_offdiagonal_tolerance
            || a.fock_eigensolver.max_sweeps!=b.fock_eigensolver.max_sweeps
            || a.fock_eigensolver.relative_offdiagonal_tolerance!=b.fock_eigensolver.relative_offdiagonal_tolerance
            || (a.occupation_cutoff==0.0 && r!=u))
            throw std::invalid_argument("Gaussian triple spaces downstream triple controls differ");
        sha(geometry.input_identity_sha256);sha(geometry.result_identity_sha256);
        retained=add(retained,bytes);work=add(work,gm.work_units_upper_bound);++completed;
        total_rank=add(total_rank,r);total_union=add(total_union,u);maximum_rank=std::max(maximum_rank,r);
        if (!u) ++empty_union;
        if (!r) ++empty_retained;
        if (i==k) ++all_equal;
        full_union=full_union && u==n;full_retained=full_retained && r==n;
    }
    const bool complete=o==mul(ref.state().n_kpoints(),ref.state().n_correlated_occupied())
        && n==mul(ref.state().n_kpoints(),ref.state().n_virtual());
    if (completed!=p.triple_count || retained!=diag.retained_numerical_bytes || work!=diag.geometry_work_units
        || retained>p.exact_rank_plan_output_upper_bytes || work!=p.exact_rank_plan_geometry_work_units
        || diag.empty_union_count!=empty_union || diag.empty_retained_space_count!=empty_retained
        || diag.all_equal_tuple_count!=all_equal || diag.all_unions_full_common_rank!=full_union
        || diag.total_retained_rank!=total_rank || diag.total_union_rank!=total_union
        || diag.maximum_retained_rank!=maximum_rank
        || diag.all_retained_full_common_rank!=full_retained || diag.complete_common_finite_torus_basis!=complete
        || mp2.diagnostics().complete_common_finite_torus_basis!=complete
        || ccsd.diagnostics().complete_common_finite_torus_basis!=complete)
        throw std::invalid_argument("Gaussian triple spaces downstream aggregate census/flags differ");
}

Result make_periodic_gaussian_triple_spaces(const PeriodicCorrelationAdmittedReference& ref,
    const PeriodicCorrelationRealLocalBasis& basis,const PeriodicGaussianRealLocalProvider& provider,
    const PeriodicGaussianPairMP2Result& mp2,const PeriodicGaussianPairCCSDResult& ccsd,
    const Options& caller_options,const Live& caller_live,const Caps& caller_caps,
    PeriodicGaussianTripleSpacesCallback callback,void* callback_context) {
    const auto options=caller_options;const auto live=caller_live;const auto caps=caller_caps;
    const auto p=plan_periodic_gaussian_triple_spaces(ref,basis,provider,mp2,ccsd,options,live,caps);
    const Sources sources{ref,basis,provider,mp2,ccsd};
    if (mp2.domain_generated()) for (U i=0;i<p.occupied_count;++i) {
        verify_frame_payload(ccsd.singles_frame(i),p.common_virtual_dimension);
        for (U j=i;j<p.occupied_count;++j) verify_frame_payload(mp2.pair(i,j).frame(),p.common_virtual_dimension);
    }
    Result result;result.memory_=p;result.options_=options;result.state_=ref.state_handle();result.context_=provider.context_handle();
    result.ccsd_=ccsd.identity_sha256();result.amplitudes_=ccsd.solver().payload_sha256();
    result.warmstart_=mp2.identity_sha256();result.pair_spaces_=mp2.pair_spaces_identity_sha256();
    result.basis_=basis.identity_sha256();result.provider_=provider.identity_sha256();
    result.hf_=provider.hf_reference_source_identity_sha256();result.allocation_=ref.dimensions().allocation_identity;
    auto& diag=result.diagnostics_;
    diag.complete_common_finite_torus_basis=ccsd.diagnostics().complete_common_finite_torus_basis;
    diag.all_unions_full_common_rank=true;diag.all_retained_full_common_rank=true;
    Events events{sources,p,callback,callback_context,{},{},{},{},{}};events.pin();
    events.emit(PeriodicGaussianTripleSpacesStage::Begin);
    result.triples_.reserve(static_cast<std::size_t>(p.triple_count));
    if (result.triples_.capacity()!=p.triple_count)
        throw std::length_error("Gaussian triple spaces allocator over-reserved its admitted object table");
    Digest spaces("vibeqc.periodic.gaussian-triple-spaces.geometry");spaces.u64(p.triple_count);
    const U o=p.occupied_count,n=p.common_virtual_dimension;
    for (U i=0;i<o;++i) for (U j=i;j<o;++j) for (U k=j;k<o;++k) {
        const auto input=geometry_input(sources,i,j,k);
        BoundedRestrictedTNOInventory inventory;
        const auto leaf=geometry_plan(input,options,p,live,caps,diag.retained_numerical_bytes,&inventory);
        auto geometry=bounded_restricted_triple_natural_orbitals(input,options,inventory,caps.geometry);
        if (geometry.output_numerical_bytes>leaf.output_numerical_bytes_upper_bound
            || geometry.memory.peak_owned_numerical_bytes!=leaf.peak_owned_numerical_bytes)
            throw std::logic_error("Gaussian triple spaces realized leaf exceeds its admitted plan");
        diag.retained_numerical_bytes=add(diag.retained_numerical_bytes,geometry.output_numerical_bytes);
        diag.geometry_work_units=add(diag.geometry_work_units,leaf.work_units_upper_bound);
        diag.total_retained_rank=add(diag.total_retained_rank,geometry.retained_rank);
        diag.total_union_rank=add(diag.total_union_rank,geometry.union_rank);
        diag.maximum_retained_rank=std::max(diag.maximum_retained_rank,geometry.retained_rank);
        if (diag.retained_numerical_bytes>p.exact_rank_plan_output_upper_bytes
            || diag.geometry_work_units>p.exact_rank_plan_geometry_work_units)
            throw std::logic_error("Gaussian triple spaces retained/work census exceeds its batch plan");
        if (!geometry.union_rank) ++diag.empty_union_count;
        if (!geometry.retained_rank) ++diag.empty_retained_space_count;
        if (i==k) ++diag.all_equal_tuple_count;
        diag.all_unions_full_common_rank=diag.all_unions_full_common_rank && geometry.union_rank==n;
        diag.all_retained_full_common_rank=diag.all_retained_full_common_rank && geometry.retained_rank==n;
        spaces.u64(i);spaces.u64(j);spaces.u64(k);spaces.string(geometry.input_identity_sha256);
        spaces.string(geometry.result_identity_sha256);
        events.event.occupied={i,j,k};events.event.union_rank=geometry.union_rank;
        events.event.retained_rank=geometry.retained_rank;
        result.triples_.push_back(std::move(geometry));++diag.completed_triples;
        events.event.completed_triples=diag.completed_triples;
        events.event.retained_numerical_bytes=diag.retained_numerical_bytes;
        events.event.geometry_work_units=diag.geometry_work_units;
        events.emit(PeriodicGaussianTripleSpacesStage::TripleComplete);
    }
    if (diag.completed_triples!=p.triple_count)
        throw std::logic_error("Gaussian triple spaces completed triple census differs");
    result.spaces_=spaces.finish();
    Digest identity("vibeqc.periodic.gaussian-triple-spaces.identity");
    for (const auto* s : std::array<const std::string*,12>{&ref.state().state_identity_sha256(),&result.allocation_,
        &result.ccsd_,&result.amplitudes_,&result.warmstart_,&result.pair_spaces_,&result.basis_,&result.provider_,
        &result.hf_,&result.spaces_,&ccsd.singles_spaces_identity_sha256(),&provider.source_context_identity_sha256()}) identity.string(*s);
    identity.u64(diag.complete_common_finite_torus_basis);identity.u64(diag.all_unions_full_common_rank);
    identity.u64(diag.all_retained_full_common_rank);identity.string(policy);result.identity_=identity.finish();
    events.emit(PeriodicGaussianTripleSpacesStage::Finished);
    diag.completed_progress_callbacks=events.event.callback_count;
    return result;
}
}  // namespace vibeqc
