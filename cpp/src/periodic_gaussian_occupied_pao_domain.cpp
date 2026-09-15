#include "vibeqc/periodic_gaussian_occupied_pao_domain.hpp"

#include <cfloat>
#include "periodic_correlation_real_local_internal.hpp"

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__ != 0)
#error "Gaussian occupied PAO domains forbid fast/finite-only math"
#endif
#if FLT_EVAL_METHOD != 0
#error "Gaussian occupied PAO domains require binary64 evaluation"
#endif

namespace vibeqc {
namespace {
namespace arithmetic=periodic_correlation_real_local_detail;
using I=std::uint64_t;
using arithmetic::add;
using arithmetic::mul;
using Options=PeriodicCorrelationOccupiedPAODomainOptions;
using Inventory=PeriodicCorrelationOccupiedPAODomainInventory;
using Caps=PeriodicGaussianOccupiedPAODomainCaps;
using Plan=PeriodicGaussianOccupiedPAODomainMemoryPlan;
using Result=PeriodicGaussianOccupiedPAODomain;
// The optimizer and Wannier are inline members of localization, and are
// already in the child's borrowed-owner controls. Count only the remainder
// of that wrapper, not a second copy of those inline objects.
static_assert(sizeof(PeriodicGaussianLocalizationResult)>=sizeof(PeriodicCorrelationIAOOptimizerResult)
    +sizeof(PeriodicCorrelationWannier), "Gaussian localization inline control inventory changed");
constexpr I extra_controls=65536+sizeof(Plan)+sizeof(Result)+sizeof(Caps)+sizeof(Inventory)
    +sizeof(Options)+sizeof(PeriodicGaussianRHFResult)+sizeof(PeriodicGaussianSourceContext)
    +3*sizeof(BasisSet)+sizeof(PeriodicSystem)+sizeof(PeriodicGaussianLocalizationResult)
    -sizeof(PeriodicCorrelationIAOOptimizerResult)-sizeof(PeriodicCorrelationWannier)+16*65;
void limit(I value,I cap,const char* message) {
    if (!cap || value>cap) throw std::length_error(message);
}
void owners(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
            const PeriodicGaussianLocalizationResult& localization,I occupied) {
    if (!ref.state_handle() || !hf.converged() || !hf.context_handle()
        || hf.state_handle().get()!=ref.state_handle().get())
        throw std::invalid_argument("Gaussian occupied PAO domain requires the exact captured HF state owner");
    if (!localization.converged())
        throw std::invalid_argument("Gaussian occupied PAO domain requires converged localization");
    const auto& opt=localization.optimizer(); const auto& w=localization.wannier();
    if (opt.state_handle().get()!=ref.state_handle().get() || w.state_handle().get()!=ref.state_handle().get()
        || opt.allocation_identity()!=ref.dimensions().allocation_identity
        || w.allocation_identity()!=ref.dimensions().allocation_identity
        || w.localization_identity_sha256()!=opt.optimizer_identity_sha256())
        throw std::invalid_argument("Gaussian occupied PAO domain localization state/allocation lineage differs");
    if (localization.source_image_cutoff_bohr()!=hf.context_handle()->options().ao_pair_image_cutoff_bohr)
        throw std::invalid_argument("Gaussian occupied PAO domain localization AO-image policy differs from HF");
    if (occupied>=ref.state().n_correlated_occupied())
        throw std::out_of_range("Gaussian occupied PAO domain active occupied index is out of range");
    if (hf.plan().n_basis!=ref.state().n_basis() || hf.plan().n_kpoints!=ref.state().n_kpoints()
        || localization.memory().n_basis!=ref.state().n_basis()
        || localization.memory().n_points!=ref.state().n_kpoints())
        throw std::invalid_argument("Gaussian occupied PAO domain native owner dimensions differ");
}
Inventory child_inventory(const PeriodicGaussianRHFResult& hf,
    const PeriodicGaussianLocalizationResult& localization,const Inventory& inventory) {
    Inventory child=inventory;
    child.other_live_numerical_bytes=add(child.other_live_numerical_bytes,
        add(localization.memory().borrowed_gaussian_numerical_bytes,
            hf.context_handle()->inventory().auxiliary.borrowed_active_numeric_bytes));
    child.other_live_control_bytes=add(child.other_live_control_bytes,extra_controls);
    return child;
}
void inputs(const PeriodicGaussianRHFResult& hf,const PeriodicCorrelationAdmittedReference& ref,
    const BasisSet& ao,const BasisSet& auxiliary,const BasisSet& minimal,const PeriodicSystem& system,
    const PeriodicGaussianLocalizationResult& localization,I occupied) {
    owners(hf,ref,localization,occupied);
    hf.verify_physical_inputs(ao,auxiliary,system);
    localization.verify_original_inputs(ref,ao,minimal,system);
}
} // namespace

Plan plan_periodic_gaussian_occupied_pao_domain(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const PeriodicGaussianLocalizationResult& localization,
    I occupied,const Options& options,const Inventory& inventory,const Caps& caps) {
    arithmetic::float_environment(); owners(hf,ref,localization,occupied);
    Plan p;
    p.atom_mapping_bytes=mul(8,ref.state().n_basis());
    p.borrowed_gaussian_numerical_bytes=add(localization.memory().borrowed_gaussian_numerical_bytes,
        hf.context_handle()->inventory().auxiliary.borrowed_active_numeric_bytes);
    p.additional_control_storage_bytes=extra_controls;
    // Both original sources are checked before and after selection. The
    // source census bounds the one shell/contraction/AO-map expansion.
    p.physical_input_validation_work_units=mul(2,add(hf.plan().input_check_work_units,
        localization.memory().input_verification_work_units));
    const auto& ao=hf.context_handle()->inventory().ao;
    p.mapping_work_units=mul(128,add(1,add(ao.function_count,add(ao.shell_count,ao.contraction_count))));
    limit(add(p.physical_input_validation_work_units,p.mapping_work_units),caps.maximum_work_units,
          "Gaussian occupied PAO domain physical input work cap exceeded");
    const auto child=child_inventory(hf,localization,inventory);
    p.selector=plan_periodic_correlation_occupied_pao_domain(ref,localization.optimizer(),localization.wannier(),
        hf.plan().atom_count,options,child,caps.selector);
    if (p.selector.caller_mapping_bytes!=p.atom_mapping_bytes)
        throw std::logic_error("Gaussian occupied PAO domain child mapping inventory differs");
    p.peak_owned_numerical_bytes=add(p.atom_mapping_bytes,p.selector.peak_owned_numerical_bytes);
    p.worker_bytes=p.selector.worker_bytes; p.required_node_memory_bytes=p.selector.required_node_memory_bytes;
    p.work_units=add(p.selector.planned_work_units,add(p.physical_input_validation_work_units,p.mapping_work_units));
    arithmetic::extent(p.atom_mapping_bytes); arithmetic::extent(p.peak_owned_numerical_bytes);
    limit(p.atom_mapping_bytes/8,std::vector<I>().max_size(),"Gaussian occupied PAO atom-map vector extent exceeded");
    limit(p.peak_owned_numerical_bytes,caps.maximum_owned_numerical_bytes,
          "Gaussian occupied PAO domain owned numerical cap exceeded");
    limit(p.worker_bytes,caps.maximum_worker_bytes,"Gaussian occupied PAO domain worker cap exceeded");
    limit(p.required_node_memory_bytes,caps.maximum_node_bytes,"Gaussian occupied PAO domain node cap exceeded");
    limit(p.work_units,caps.maximum_work_units,"Gaussian occupied PAO domain work cap exceeded");
    return p;
}

const PeriodicCorrelationOccupiedPAODomain& Result::domain() const {
    if (!context_ || !domain_ || !domain_->state_handle())
        throw std::logic_error("Gaussian occupied PAO domain result is consumed");
    (void)domain_->populations_data(); return *domain_;
}

Result select_periodic_gaussian_occupied_pao_domain(const PeriodicGaussianRHFResult& hf,
    const PeriodicCorrelationAdmittedReference& ref,const BasisSet& ao,const BasisSet& auxiliary,
    const BasisSet& minimal,const PeriodicSystem& system,const PeriodicGaussianLocalizationResult& localization,
    I occupied,const Options& options,const Inventory& inventory,const Caps& caps) {
    const auto p=plan_periodic_gaussian_occupied_pao_domain(hf,ref,localization,occupied,options,inventory,caps);
    inputs(hf,ref,ao,auxiliary,minimal,system,localization,occupied);
    std::vector<I> mapping(static_cast<std::size_t>(ref.state().n_basis()));
    I offset=0;
    for (I s=0;s<ao.nshells();++s) {
        const int atom=ao.shell_atom_index(s);
        if (atom<0 || static_cast<I>(atom)>=hf.plan().atom_count)
            throw std::invalid_argument("Gaussian occupied PAO domain actual shell atom label is out of range");
        const I functions=ao.libint()[static_cast<std::size_t>(s)].size();
        if (functions>mapping.size()-offset)
            throw std::logic_error("Gaussian occupied PAO domain shell expansion exceeds admitted AO count");
        for (I mu=0;mu<functions;++mu) mapping[offset++]=static_cast<I>(atom);
    }
    if (offset!=mapping.size())
        throw std::logic_error("Gaussian occupied PAO domain shell expansion disagrees with AO count");
    Result result; result.memory_=p;
    result.domain_.emplace(select_periodic_correlation_occupied_pao_domain(ref,localization.optimizer(),
        localization.wannier(),mapping.data(),mapping.size(),hf.plan().atom_count,occupied,options,
        child_inventory(hf,localization,inventory),caps.selector));
    inputs(hf,ref,ao,auxiliary,minimal,system,localization,occupied);
    result.context_=hf.context_handle(); result.hf_=hf.reference_source_identity_sha256();
    result.localization_=localization.localization_identity_sha256();
    arithmetic::Digest h("vibeqc.periodic.gaussian-occupied-pao-domain.identity.v1");
    h.string(result.hf_); h.string(result.localization_);
    h.string(localization.gaussian_input_identity_sha256());
    h.string(result.context_->source_context_identity_sha256());
    h.string(result.domain_->occupied_pao_domain_identity_sha256());
    h.string("actual-HF;original-AO-minimal-aux-cell;native-shell-atom-map;one-home-occupied;no-extended-domain");
    result.identity_=h.finish(); return result;
}
} // namespace vibeqc
