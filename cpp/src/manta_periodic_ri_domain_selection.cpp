#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/manta_admission.hpp"

namespace vibeqc::manta {
void validate_periodic_ri_domain_options(const PeriodicRIFittingDomainOptions& p) {
    admission::margin(p.budget);
    if(p.maximum_refinements>31 || p.maximum_reciprocal_cube_radius>128 ||
       p.initial_reciprocal_cube_radius>p.maximum_reciprocal_cube_radius ||
       !std::isfinite(p.growth_factor) || p.growth_factor<=1. ||
       !std::isfinite(p.maximum_pair_cutoff_bohr) || p.maximum_pair_cutoff_bohr<=0. ||
       !std::isfinite(p.maximum_auxiliary_cutoff_bohr) || p.maximum_auxiliary_cutoff_bohr<=0. ||
       (p.initial_pair_cutoff_bohr && (!std::isfinite(*p.initial_pair_cutoff_bohr) || *p.initial_pair_cutoff_bohr<0. || *p.initial_pair_cutoff_bohr>p.maximum_pair_cutoff_bohr)) ||
       (p.initial_auxiliary_cutoff_bohr && (!std::isfinite(*p.initial_auxiliary_cutoff_bohr) || *p.initial_auxiliary_cutoff_bohr<0. || *p.initial_auxiliary_cutoff_bohr>p.maximum_auxiliary_cutoff_bohr)) ||
       p.budget.other_live_bytes || p.budget.memory_bytes<=p.budget.backend_margin_bytes ||
       !p.budget.shell_calls || !p.budget.primitive_quartets || !p.budget.scalar_terms || !p.budget.kernel_terms)
        throw std::invalid_argument("MANTA fitting domain selection requires finite nested controls, bounded attempts and positive search caps");
}
void validate_periodic_ri_auxiliary_options(const PeriodicRIFittingAuxiliaryOptions& p) {
    admission::margin(p.budget);
    if(!std::isfinite(p.initial.beta) || p.initial.beta<=1. || p.initial.refinement>8 ||
       p.maximum_refinements>8-p.initial.refinement || p.initial.maximum_angular_momentum<0 ||
       p.initial.maximum_angular_momentum>32 || p.budget.other_live_bytes ||
       p.budget.memory_bytes<=p.budget.backend_margin_bytes || !p.budget.shell_calls ||
       !p.budget.primitive_quartets || !p.budget.scalar_terms || !p.budget.kernel_terms)
        throw std::invalid_argument("MANTA auxiliary selection requires bounded generation controls and positive search caps");
    if(p.prepared) {
        const auto& initial=p.prepared->initial();const auto& plan=p.prepared->plan();
        if(initial.beta!=p.initial.beta || initial.refinement!=p.initial.refinement ||
           initial.maximum_angular_momentum!=p.initial.maximum_angular_momentum ||
           p.prepared->maximum_refinements()!=p.maximum_refinements)
            throw std::invalid_argument("MANTA auxiliary selection prepared controls mismatch");
        admission::cap(plan.modeled_peak_bytes,p.generation_limits.memory_bytes,"MANTA auxiliary generation memory budget exceeded");
        admission::cap(plan.work_units,p.generation_limits.work_units,"MANTA auxiliary generation work budget exceeded");
        admission::cap(plan.maximum_shells,p.generation_limits.maximum_shells,"MANTA auxiliary generation shell budget exceeded");
        admission::cap(plan.maximum_functions,p.generation_limits.maximum_functions,"MANTA auxiliary generation function budget exceeded");
    }
}
PeriodicRIFittingAuxiliaryOptions prepare_periodic_ri_auxiliary_selection(const BasisSet& orbital,const BasisSet& seed,
    const PeriodicSystem& system,const PeriodicRIFittingAuxiliaryOptions& requested,const ResourceBudget& budget) {
    validate_periodic_ri_auxiliary_options(requested);auto out=requested;
    admission::cap(budget.other_live_bytes,budget.memory_bytes,"MANTA auxiliary metadata memory budget exceeded");
    auto limits=requested.generation_limits;
    limits.memory_bytes=std::min(limits.memory_bytes,budget.memory_bytes-budget.other_live_bytes);
    limits.work_units=std::min({limits.work_units,budget.scalar_terms,budget.kernel_terms/32});
    if(out.prepared) {
        admission::cap(out.prepared->plan().modeled_peak_bytes,limits.memory_bytes,"MANTA auxiliary metadata memory budget exceeded");
        admission::cap(out.prepared->plan().work_units,limits.work_units,"MANTA auxiliary metadata work budget exceeded");
        if(!out.prepared->matches_sources(orbital,seed))
            throw std::invalid_argument("MANTA auxiliary sequence orbital or seed source mismatch");
        return out;
    }
    // Admit the temporary cell molecule before copying its atom inventory.
    // The shared sequence plan includes this owner and its construction work.
    admission::cap(admission::add(65536,admission::mul(256,system.unit_cell.size())),limits.memory_bytes,
        "MANTA auxiliary cell metadata memory budget exceeded");
    admission::cap(admission::mul(4096,system.unit_cell.size()),limits.work_units,
        "MANTA auxiliary cell metadata work budget exceeded");
    const auto molecule=system.unit_cell_molecule();
    out.prepared=std::make_shared<ProductAuxiliarySequence>(molecule,orbital,seed,requested.initial,requested.maximum_refinements,limits);
    return out;
}
ResourceBudget periodic_ri_fitting_search_budget(const PeriodicRIFittingAccuracyOptions& o) {
    if(!o.reference_selection && !o.domain_selection && !o.auxiliary_selection)
        throw std::invalid_argument("MANTA fitting search requires a selection policy");
    if(o.reference_selection)validate_periodic_ri_reference_options(*o.reference_selection);
    if(o.domain_selection)validate_periodic_ri_domain_options(*o.domain_selection);
    if(o.auxiliary_selection)validate_periodic_ri_auxiliary_options(*o.auxiliary_selection);
    auto b=o.reference_selection?o.reference_selection->budget:o.domain_selection?o.domain_selection->budget:o.auxiliary_selection->budget;
    auto merge=[&](const ResourceBudget& d) {
        b.memory_bytes=std::min(b.memory_bytes,d.memory_bytes);b.shell_calls=std::min(b.shell_calls,d.shell_calls);
        b.primitive_quartets=std::min(b.primitive_quartets,d.primitive_quartets);
        b.scalar_terms=std::min(b.scalar_terms,d.scalar_terms);b.kernel_terms=std::min(b.kernel_terms,d.kernel_terms);
        b.backend_margin_bytes=std::max(b.backend_margin_bytes,d.backend_margin_bytes);
    };
    if(o.domain_selection)merge(o.domain_selection->budget);
    if(o.auxiliary_selection)merge(o.auxiliary_selection->budget);
    return b;
}
namespace {
PeriodicRIFockPlan domain_capacity(const BasisSet& basis,const BasisSet& aux,const PeriodicSystem& system,
    Count ng,Count nq,Count nk,const PeriodicGDFShortRangeOptions& options,const RIOptions& ri,const ResourceBudget& budget,
    bool j,bool k,const PeriodicGDFKernelOptions& physical,const PeriodicRIAccuracyOptions& component,const PeriodicRIFittingAccuracyOptions& accuracy) {
    if(!accuracy.domain_selection)return plan_periodic_ri_fock(basis,aux,system,ng,nq,nk,options,ri,budget,j,k,physical,component);
    const auto& p=*accuracy.domain_selection;validate_periodic_ri_domain_options(p);
    if(!j || k || component.enabled() || physical.kernel!="coulomb")
        throw std::invalid_argument("MANTA fitting domain selection requires J-only fitting without a separate component accuracy policy");
    auto selected=options;
    selected.pair_cutoff_bohr=p.initial_pair_cutoff_bohr.value_or(options.pair_cutoff_bohr);
    selected.auxiliary_cutoff_bohr=p.initial_auxiliary_cutoff_bohr.value_or(options.auxiliary_cutoff_bohr);
    if(selected.pair_cutoff_bohr<options.pair_cutoff_bohr || selected.pair_cutoff_bohr>p.maximum_pair_cutoff_bohr ||
       selected.auxiliary_cutoff_bohr<options.auxiliary_cutoff_bohr || selected.auxiliary_cutoff_bohr>p.maximum_auxiliary_cutoff_bohr)
        throw std::invalid_argument("MANTA fitting domain initial radii must nest the original source and fit within limits");
    for(Count i=0;i<p.maximum_refinements;++i) {
        selected.pair_cutoff_bohr=std::min(p.maximum_pair_cutoff_bohr,std::max(1.,selected.pair_cutoff_bohr*p.growth_factor));
        selected.auxiliary_cutoff_bohr=std::min(p.maximum_auxiliary_cutoff_bohr,std::max(1.,selected.auxiliary_cutoff_bohr*p.growth_factor));
    }
    const Count cube=std::min(p.maximum_reciprocal_cube_radius,admission::add(p.initial_reciprocal_cube_radius,p.maximum_refinements));
    return plan_periodic_ri_fock(basis,aux,system,periodic_gdf_domain_vector_capacity(ng,cube),0,nk,selected,ri,budget,true,false,physical,component);
}
} // namespace
PeriodicRIFockPlan plan_periodic_ri_fock_capacity(const BasisSet& basis,const BasisSet& aux,const PeriodicSystem& system,
    Count ng,Count nq,Count nk,const PeriodicGDFShortRangeOptions& options,const RIOptions& ri,const ResourceBudget& budget,
    bool j,bool k,const PeriodicGDFKernelOptions& physical,const PeriodicRIAccuracyOptions& component,const PeriodicRIFittingAccuracyOptions& accuracy) {
    auto plan=domain_capacity(basis,aux,system,ng,nq,nk,options,ri,budget,j,k,physical,component,accuracy);
    if(!accuracy.auxiliary_selection)return plan;
    const auto& policy=*accuracy.auxiliary_selection;validate_periodic_ri_auxiliary_options(policy);
    if(!j || k || component.enabled() || physical.kernel!="coulomb")
        throw std::invalid_argument("MANTA auxiliary selection requires J-only fitting without a separate component accuracy policy");
    if(!policy.prepared || !policy.prepared->matches_sources(basis,aux))
        throw std::invalid_argument("MANTA auxiliary selection requires a prepared sequence matching orbital and seed sources");
    // Componentwise capacity; no monotone cost or accuracy assumption about a
    // candidate ordering. Count planners consume immutable prepared metadata.
    for(Count i=0;i<policy.prepared->plan().candidates;++i) {
        const auto next=domain_capacity(basis,policy.prepared->candidate(i).basis(),system,ng,nq,nk,options,ri,budget,true,false,physical,component,accuracy);
#define M(n) plan.n=std::max(plan.n,next.n)
        M(n_auxiliary);M(retained_source_bytes);M(driver_live_bytes);M(modeled_peak_bytes);M(metric_shell_calls);M(metric_primitive_products);
        M(build_shell_calls);M(build_primitive_products);M(shell_calls);M(primitive_products);M(scalar_terms);M(kernel_terms);
#undef M
#define Q(n) plan.gamma.n=std::max(plan.gamma.n,next.gamma.n)
        Q(n_auxiliary);Q(maximum_rank);Q(mode_block);Q(maximum_coupling_passes);Q(shell_calls);Q(primitive_products);
        Q(scalar_terms);Q(kernel_terms);Q(modeled_peak_bytes);Q(accuracy_selection_passes);
#undef Q
    }
    return plan;
}
} // namespace vibeqc::manta
