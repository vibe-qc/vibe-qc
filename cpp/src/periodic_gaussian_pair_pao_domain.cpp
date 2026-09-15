#include "vibeqc/periodic_gaussian_pair_pao_domain.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian pair PAO domains forbid fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "Gaussian pair PAO domains require binary64 evaluation"
#endif

namespace vibeqc {
namespace {
namespace arithmetic=periodic_correlation_real_local_detail;
using arithmetic::add;
using arithmetic::mul;
using I=std::uint64_t;
using Endpoint=PeriodicGaussianOccupiedPAODomain;
using Inventory=PeriodicCorrelationPairPAODomainInventory;
using Caps=PeriodicGaussianPairPAODomainCaps;
using Plan=PeriodicGaussianPairPAODomainMemoryPlan;
using Result=PeriodicGaussianPairPAODomain;
using Topology=PeriodicCorrelationTranslationPairTopology;
static_assert(sizeof(Endpoint)>=sizeof(PeriodicCorrelationOccupiedPAODomain),
    "Gaussian occupied domain inline child inventory changed");
constexpr I extra_fixed_controls=65536+sizeof(Plan)+sizeof(Result)+sizeof(Caps)+sizeof(Inventory)
    +sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicGaussianSourceContext)
    +2*sizeof(BasisSet)+sizeof(PeriodicSystem)+16*65;
void limit(I value,I cap,const char* message) {
    if (!cap || value>cap) throw std::length_error(message);
}
void owners(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const Endpoint& home,const Endpoint& partner) {
    if (!ref.state_handle() || !hf.converged() || !hf.context_handle()
        || hf.state_handle().get()!=ref.state_handle().get())
        throw std::invalid_argument("Gaussian pair PAO domain requires exact captured HF state owner");
    for (const auto* endpoint : {&home,&partner}) {
        if (endpoint->context_handle().get()!=hf.context_handle().get()
            || endpoint->hf_reference_source_identity_sha256()!=hf.reference_source_identity_sha256()
            || endpoint->domain().state_handle().get()!=ref.state_handle().get()
            || endpoint->domain().allocation_identity()!=ref.dimensions().allocation_identity
            || endpoint->localization_identity_sha256().size()!=64 || endpoint->identity_sha256().size()!=64)
            throw std::invalid_argument("Gaussian pair PAO domain endpoint HF source/state/allocation differs");
    }
    if (home.localization_identity_sha256()!=partner.localization_identity_sha256())
        throw std::invalid_argument("Gaussian pair PAO domain endpoint global localization differs");
}
I gaussian_bytes(const PeriodicGaussianRHFResult& hf) {
    // Basis roles are conservatively counted separately. Shell atom labels
    // are not part of the HF integral-basis census, but are read to build the
    // new map. System lattice/coordinates/charges are physical input roles.
    const auto& census=hf.context_handle()->inventory();
    return add(census.combined_borrowed_active_numeric_bytes,
        add(72,add(mul(28,hf.plan().atom_count),mul(4,census.ao.shell_count))));
}
I additional_controls(const Endpoint& home,const Endpoint& partner) {
    const I distinct=&home==&partner ? 1 : 2;
    // Child union already accounts each generic inline endpoint and its
    // numerical payload. Count only the actual-HF wrapper remainder here.
    return add(extra_fixed_controls,mul(distinct,
        sizeof(Endpoint)-sizeof(PeriodicCorrelationOccupiedPAODomain)+3*65));
}
Inventory child_inventory(const PeriodicGaussianRHFResult& hf,const Endpoint& home,
    const Endpoint& partner,const Inventory& input) {
    auto child=input;
    child.other_live_numerical_bytes=add(child.other_live_numerical_bytes,gaussian_bytes(hf));
    child.other_live_control_bytes=add(child.other_live_control_bytes,additional_controls(home,partner));
    return child;
}
} // namespace

Plan plan_periodic_gaussian_pair_pao_domain(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const Topology& topology,I row,
    const Endpoint& home,const Endpoint& partner,const Inventory& inventory,const Caps& caps) {
    arithmetic::float_environment(); owners(hf,ref,home,partner);
    Plan p;
    p.atom_mapping_bytes=mul(8,ref.state().n_basis());
    p.borrowed_gaussian_numerical_bytes=gaussian_bytes(hf);
    p.additional_control_storage_bytes=additional_controls(home,partner);
    p.physical_input_validation_work_units=mul(2,hf.plan().input_check_work_units);
    const auto& ao=hf.context_handle()->inventory().ao;
    p.mapping_work_units=mul(128,add(1,add(ao.function_count,add(ao.shell_count,ao.contraction_count))));
    limit(add(p.physical_input_validation_work_units,p.mapping_work_units),caps.maximum_work_units,
        "Gaussian pair PAO domain physical input work cap exceeded");
    p.pair_union=plan_periodic_correlation_pair_pao_domain(ref,topology,row,home.domain(),partner.domain(),
        hf.plan().atom_count,child_inventory(hf,home,partner,inventory),caps.pair_union);
    if (p.pair_union.caller_mapping_bytes!=p.atom_mapping_bytes)
        throw std::logic_error("Gaussian pair PAO domain child mapping inventory differs");
    p.peak_owned_numerical_bytes=add(p.atom_mapping_bytes,p.pair_union.peak_owned_numerical_bytes);
    p.worker_bytes=p.pair_union.worker_bytes; p.required_node_memory_bytes=p.pair_union.required_node_memory_bytes;
    p.work_units=add(p.pair_union.planned_work_units,add(p.physical_input_validation_work_units,p.mapping_work_units));
    arithmetic::extent(p.atom_mapping_bytes); arithmetic::extent(p.peak_owned_numerical_bytes);
    limit(p.atom_mapping_bytes/8,std::vector<I>().max_size(),"Gaussian pair PAO atom-map vector extent exceeded");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,"Gaussian pair PAO domain owned cap exceeded");
    limit(p.worker_bytes,caps.maximum_worker_bytes,"Gaussian pair PAO domain worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.maximum_node_bytes,"Gaussian pair PAO domain node cap exceeded");
    limit(p.work_units,caps.maximum_work_units,"Gaussian pair PAO domain work cap exceeded");
    return p;
}

const PeriodicCorrelationPairPAODomain& Result::domain() const {
    if (!context_ || !domain_ || !domain_->state_handle())
        throw std::logic_error("Gaussian pair PAO domain result is consumed");
    (void)domain_->cell_ao_indices_data(); return *domain_;
}

Result make_periodic_gaussian_pair_pao_domain(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const BasisSet& ao,const BasisSet& auxiliary,
    const PeriodicSystem& system,const Topology& topology,I row,const Endpoint& home,const Endpoint& partner,
    const Inventory& inventory,const Caps& caps) {
    const auto p=plan_periodic_gaussian_pair_pao_domain(hf,ref,topology,row,home,partner,inventory,caps);
    hf.verify_physical_inputs(ao,auxiliary,system);
    std::vector<I> mapping(static_cast<std::size_t>(ref.state().n_basis()));
    I offset=0;
    for (I s=0;s<ao.nshells();++s) {
        const int atom=ao.shell_atom_index(s);
        if (atom<0 || static_cast<I>(atom)>=hf.plan().atom_count)
            throw std::invalid_argument("Gaussian pair PAO domain actual shell atom label is out of range");
        const I functions=ao.libint()[static_cast<std::size_t>(s)].size();
        if (functions>mapping.size()-offset)
            throw std::logic_error("Gaussian pair PAO domain shell expansion exceeds admitted AO count");
        for (I mu=0;mu<functions;++mu) mapping[offset++]=static_cast<I>(atom);
    }
    if (offset!=mapping.size())
        throw std::logic_error("Gaussian pair PAO domain shell expansion disagrees with AO count");
    Result result; result.memory_=p;
    result.domain_.emplace(make_periodic_correlation_pair_pao_domain(ref,topology,row,home.domain(),partner.domain(),
        mapping.data(),mapping.size(),hf.plan().atom_count,child_inventory(hf,home,partner,inventory),caps.pair_union));
    owners(hf,ref,home,partner); hf.verify_physical_inputs(ao,auxiliary,system);
    result.context_=hf.context_handle(); result.hf_=hf.reference_source_identity_sha256();
    result.localization_=home.localization_identity_sha256();
    arithmetic::Digest h("vibeqc.periodic.gaussian-pair-pao-domain.identity.v1");
    h.string(result.hf_); h.string(result.localization_); h.string(result.context_->source_context_identity_sha256());
    h.string(home.identity_sha256()); h.string(partner.identity_sha256());
    h.string(result.domain_->pair_pao_domain_identity_sha256());
    h.string("actual-HF;native-shell-atom-map;translated-initial-pair-union;no-extended-domain");
    result.identity_=h.finish(); return result;
}
} // namespace vibeqc
