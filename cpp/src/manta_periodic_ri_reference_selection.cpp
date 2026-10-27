#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_periodic_error.hpp"
#include <algorithm>
#include <tuple>

namespace vibeqc::manta {
namespace {
using namespace admission;
ResourceBudget unlimited(ResourceBudget b) {
    b.other_live_bytes=0;
    b.memory_bytes=b.shell_calls=b.primitive_quartets=b.scalar_terms=b.kernel_terms=std::numeric_limits<Count>::max();
    return b;
}
struct Ledger {
    ResourceBudget available;
    PeriodicRIFittingBoundPlan charged;
    void record(Count peak,Count shells,Count primitives,Count scalar,Count kernel) {
        cap(peak,available.memory_bytes,"MANTA reference search memory budget exceeded");
        cap(shells,available.shell_calls,"MANTA reference search shell budget exceeded");
        cap(primitives,available.primitive_quartets,"MANTA reference search primitive budget exceeded");
        cap(scalar,available.scalar_terms,"MANTA reference search scalar budget exceeded");
        cap(kernel,available.kernel_terms,"MANTA reference search kernel budget exceeded");
        charged.modeled_peak_bytes=std::max(charged.modeled_peak_bytes,peak);
        charged.shell_calls=add(charged.shell_calls,shells);charged.primitive_products=add(charged.primitive_products,primitives);
        charged.scalar_terms=add(charged.scalar_terms,scalar);charged.kernel_terms=add(charged.kernel_terms,kernel);
        available.shell_calls-=shells;available.primitive_quartets-=primitives;
        available.scalar_terms-=scalar;available.kernel_terms-=kernel;
    }
};
bool fitting_targets_met(const PeriodicRIFittingBoundReceipt& c,const PeriodicRIFittingAccuracyOptions& o) {
    return c.j_error_bound<=o.j_error && c.energy_error_bound<=o.energy_error;
}
std::string grow(PeriodicBupoSourceOptions& s,const PeriodicRIFittingReferenceOptions& p,
    const PeriodicRIFittingBoundReceipt& c,const PeriodicRIFittingAccuracyOptions& o) {
    const double pair=std::min(p.maximum_pair_cutoff_bohr,std::max(1.,*s.pairs.pair_cutoff_bohr*p.growth_factor));
    const bool pair_available=s.pair_cube_radius<p.maximum_cube_radius || pair>*s.pairs.pair_cutoff_bohr;
    const double tolerance=s.bupo.integral_tolerance*.1;
    const std::array<bool,4> available{tolerance<s.bupo.integral_tolerance,s.real_cube_radius<p.maximum_cube_radius,
        s.reciprocal_cube_radius<p.maximum_cube_radius,pair_available};
    const auto advance=[&](Count component) {
        if(component==0)s.bupo.integral_tolerance=tolerance;
        else if(component==1)++s.real_cube_radius;
        else if(component==2)++s.reciprocal_cube_radius;
        else {
            if(s.pair_cube_radius<p.maximum_cube_radius)++s.pair_cube_radius;
            s.pairs.pair_cutoff_bohr=pair;
        }
    };
    if(!p.selective_refinement) {
        bool changed=false;for(Count component=0;component<4;++component)if(available[component]){advance(component);changed=true;}
        return changed?"joint":"";
    }
    // Selection guidance from independent fixed-family contribution bounds.
    // This is a bounded greedy search, not a convergence/optimality proof.
    // Readiness and final acceptance still use the unchanged combined map.
    const std::array<double,4> j{c.reference_approximation_j_bound,c.reference_real_domain_j_bound,
        c.reference_reciprocal_domain_j_bound,c.reference_pair_image_j_bound};
    const std::array<double,4> energy{c.reference_approximation_energy_bound,c.reference_real_domain_energy_bound,
        c.reference_reciprocal_domain_energy_bound,c.reference_pair_image_energy_bound};
    const std::array<const char*,4> names{"approximation","real-domain","reciprocal-domain","pair-images"};
    long double largest=0;Count selected=4;
    for(Count component=0;component<4;++component)if(available[component]) {
        const long double score=std::max(static_cast<long double>(j[component])/p.reference_fraction/o.j_error,
            static_cast<long double>(energy[component])/p.reference_fraction/o.energy_error);
        if(score>largest){largest=score;selected=component;}
    }
    if(selected==4)return "";
    advance(selected);return names[selected];
}
} // namespace

void validate_periodic_ri_reference_options(const PeriodicRIFittingReferenceOptions& p) {
    const auto& s=p.initial;const auto& b=p.budget;
    margin(b);
    if(p.maximum_refinements>31 || p.maximum_cube_radius>128 ||
       std::max({s.pair_cube_radius,s.real_cube_radius,s.reciprocal_cube_radius})>p.maximum_cube_radius ||
       s.near_cube_radius>s.real_cube_radius)
        throw std::invalid_argument("MANTA reference selection refinement or cube limit is invalid");
    if(!std::isfinite(p.growth_factor) || p.growth_factor<=1. ||
       !std::isfinite(p.maximum_pair_cutoff_bohr) || p.maximum_pair_cutoff_bohr<=0. ||
       !std::isfinite(p.reference_fraction) || p.reference_fraction<=0. || p.reference_fraction>1. ||
       !std::isfinite(s.eta) || s.eta<=0. || !std::isfinite(s.bupo.integral_tolerance) || s.bupo.integral_tolerance<0. ||
       (s.pairs.pair_cutoff_bohr && (!std::isfinite(*s.pairs.pair_cutoff_bohr) || *s.pairs.pair_cutoff_bohr<0. ||
                                    *s.pairs.pair_cutoff_bohr>p.maximum_pair_cutoff_bohr)) ||
       !s.bupo.bound_infinite_domain || !s.bupo.bound_pair_images)
        throw std::invalid_argument("MANTA reference selection requires finite controls and full Coulomb bounds");
    if(b.other_live_bytes || b.memory_bytes<=b.backend_margin_bytes ||
       !b.shell_calls || !b.primitive_quartets || !b.scalar_terms || !b.kernel_terms)
        throw std::invalid_argument("MANTA reference selection budget requires positive caps and zero external live bytes");
}

void PeriodicRIFockBuilder::certify_fitting(const PeriodicRIFittingAccuracyOptions& o,
    const BoundedPeriodicDensityEnvelope& density,const ResourceBudget& budget,
    const PeriodicSystem& system,const RegularKMesh& mesh) {
    if(!o.reference_selection && !o.domain_selection && !o.auxiliary_selection) {certify_fitting(o,density,budget);return;}
    if(fitting_bound_ || !info_.has_j)
        throw std::invalid_argument("MANTA fitting search requires an uncertified fitted J owner");
    const auto reservation=plan_periodic_ri_fitting_accuracy(gamma_->plan(),o,budget);
    if((o.maximum_metric_refinements || o.domain_selection || o.auxiliary_selection) && (info_.has_k || info_.accuracy.enabled))
        throw std::invalid_argument("MANTA full fitting selection requires J-only fitting without a separate component accuracy policy");
    Ledger ledger;ledger.available=periodic_ri_fitting_search_budget(o);
    ledger.available.backend_margin_bytes=std::max(budget.backend_margin_bytes,ledger.available.backend_margin_bytes);
    const Count margin_extra=ledger.available.backend_margin_bytes-budget.backend_margin_bytes;
    ledger.charged.n_basis=reservation.n_basis;ledger.charged.momenta=reservation.momenta;
    ledger.charged.spins=reservation.spins;ledger.charged.maximum_probes=reservation.maximum_probes;
    ledger.charged.maximum_trials=reservation.maximum_trials;
    const auto* sequence=o.auxiliary_selection?o.auxiliary_selection->prepared.get():nullptr;
    const Count entries=mul(plan_.momenta,square(plan_.n_basis));
    const Count history=add(sequence?sequence->plan().modeled_peak_bytes:0,add(65536,add(mul(16384,add(reservation.maximum_trials,1)),mul(128,entries))));
    const Count setup=add(sequence?sequence->plan().work_units:0,add(65536,mul(4096,add(reservation.maximum_trials,entries))));
    ledger.record(add(margin_extra,add(history,gamma_->plan().modeled_peak_bytes)),0,0,setup,add(mul(16,setup),sequence?mul(16,sequence->plan().work_units):0));
    const auto family=periodic_ri_fitting_family(o,density);
    if(o.reference_selection)info_.reference_trials.reserve(o.reference_selection->maximum_refinements+1);
    if(o.maximum_metric_refinements)info_.fitting_trials.reserve(reservation.maximum_trials);
    if(o.domain_selection)info_.domain_trials.reserve(reservation.maximum_trials);
    if(sequence)info_.auxiliary_trials.reserve(reservation.maximum_trials);
    // Always enlarge from the original shared source: preserve its numerical
    // vectors and order, avoid recursively increasing union reservations, and
    // reset the requested metric controls for each new physical domain.
    const auto base=(o.domain_selection || sequence)?gamma_:std::shared_ptr<PeriodicRIQBuilder>{};
    const Count base_live=base?base->plan().modeled_peak_bytes:0;
    std::shared_ptr<const PreparedPeriodicBupoSource> reference=o.reference;
    std::unique_ptr<PeriodicRIFittingBoundSource> bound;
    PeriodicRIFockPlan capacity;
    double pair=0.,auxiliary=0.;Count cube=0;
    const auto leaf_budget=unlimited(budget);
    if(base) {
        const auto& source=base->coupling_source();
        capacity=plan_periodic_ri_fock_capacity(source.orbital(),source.auxiliary(),system,source.plan().vectors,0,
            plan_.momenta,source.options(),base->options(),leaf_budget,true,false,{}, {},o);
        pair=source.options().pair_cutoff_bohr;auxiliary=source.options().auxiliary_cutoff_bohr;
        if(o.domain_selection) {
            const auto& d=*o.domain_selection;
            pair=d.initial_pair_cutoff_bohr.value_or(pair);auxiliary=d.initial_auxiliary_cutoff_bohr.value_or(auxiliary);
            cube=d.initial_reciprocal_cube_radius;
        }
    }
    auto check_capacity=[&](const PeriodicRIFockPlan& p) {
#define C(n) cap(p.n,capacity.n,"MANTA selected fitting source exceeds admitted SCF capacity")
        C(retained_source_bytes);C(driver_live_bytes);C(modeled_peak_bytes);C(metric_shell_calls);C(metric_primitive_products);
        C(build_shell_calls);C(build_primitive_products);C(shell_calls);C(primitive_products);C(scalar_terms);C(kernel_terms);
#undef C
    };
    const double initial_pair=pair,initial_auxiliary=auxiliary;const Count initial_cube=cube;
    const ProductAuxiliaryBasis* candidate=nullptr;
    auto replace_domain=[&] {
        const auto& source=base->coupling_source();
        const auto& basis=candidate?candidate->basis():source.auxiliary();
        const auto generation=source.plan_auxiliary_domains(basis,pair,auxiliary,cube,leaf_budget);
        auto options=source.options();options.pair_cutoff_bohr=pair;options.auxiliary_cutoff_bohr=auxiliary;
        const auto upper_plan=plan_periodic_ri_fock(source.orbital(),basis,system,
            generation.maximum_vectors,0,plan_.momenta,options,base->options(),leaf_budget,true,false);
        check_capacity(upper_plan);
        const Count ref_live=reference?reference->plan().source_live_bytes:0;
        const Count peak=add(margin_extra,add(history,add(ref_live,add(base_live,add(gamma_->plan().modeled_peak_bytes,
            add(generation.generation_bytes,upper_plan.modeled_peak_bytes))))));
        // Admit old, original and replacement owners together, plus generation
        // and Fock conjugacy work. Rejected candidate work is never refunded.
        ledger.record(peak,upper_plan.shell_calls,upper_plan.primitive_products,
            add(upper_plan.scalar_terms,generation.generation_scalar_terms),
            add(upper_plan.kernel_terms,generation.generation_kernel_terms));
        auto enlarged=source.with_auxiliary_domains(basis,pair,auxiliary,cube,leaf_budget);
        const auto actual=plan_periodic_ri_fock(source.orbital(),basis,system,
            enlarged->plan().vectors,0,plan_.momenta,options,base->options(),leaf_budget,true,false);
        check_capacity(actual);
        auto next=std::make_shared<PeriodicRIQBuilder>(enlarged,2,base->options(),leaf_budget);
        if(candidate)replace_gamma_auxiliary(std::move(next),actual,*candidate);
        else replace_gamma_domains(std::move(next),actual);
    };
    if(base && (pair!=base->coupling_source().options().pair_cutoff_bohr ||
       auxiliary!=base->coupling_source().options().auxiliary_cutoff_bohr || cube))replace_domain();
    auto probe=[&] {
        auto leaf=ledger.available;leaf.other_live_bytes=add(history,add(base_live,reference->plan().source_live_bytes));
        const auto plan=PeriodicRIFittingBoundSource::plan_source(*gamma_,*reference->source(),family,leaf);
        ledger.record(plan.modeled_peak_bytes,plan.shell_calls,plan.primitive_products,plan.scalar_terms,plan.kernel_terms);
        return std::make_unique<PeriodicRIFittingBoundSource>(gamma_,reference->source(),family,leaf,false);
    };
    if(o.reference_selection) {
        const auto& policy=*o.reference_selection;auto source=policy.initial;
        if(!source.pairs.pair_cutoff_bohr)
            source.pairs.pair_cutoff_bohr=gamma_->coupling_source().source().short_range.pair_cutoff_bohr;
        if(*source.pairs.pair_cutoff_bohr>policy.maximum_pair_cutoff_bohr)
            throw std::invalid_argument("MANTA initial inherited reference pair cutoff exceeds its limit");
        bool ready=false;std::string refinement="initial";
        for(Count attempt=0;attempt<=policy.maximum_refinements;++attempt) {
            auto leaf=ledger.available;leaf.other_live_bytes=add(history,add(base_live,gamma_->plan().modeled_peak_bytes));
            reference=std::make_shared<PreparedPeriodicBupoSource>(gamma_->coupling_source().orbital(),system,mesh,source,64,leaf);
            const auto& p=reference->plan();
            ledger.record(p.modeled_peak_bytes,p.shell_calls,p.primitive_pairs,p.scalar_terms,p.kernel_terms);
            if(weights_!=reference->source()->pair_source().weights())
                throw std::invalid_argument("MANTA automatic fitting reference weights mismatch");
            bound=probe();const auto& c=bound->receipt();
            // Reference fraction is selection guidance, not an extra additive
            // certificate or an assumption that fitting errors are monotone.
            const auto reference_energy=c.reference_remainder_energy_bound;
            ready=fitting_targets_met(c,o) || (c.reference_remainder_j_bound<=policy.reference_fraction*o.j_error &&
                                  reference_energy<=policy.reference_fraction*o.energy_error);
            info_.reference_trials.push_back({reference->receipt(),c,ready,refinement});
            if(ready)break;
            if(attempt==policy.maximum_refinements)
                throw std::runtime_error("MANTA automatic fitting reference targets unmet at the refinement/domain limit");
            refinement=grow(source,policy,c,o);bound.reset();reference.reset();
            if(refinement.empty())throw std::runtime_error("MANTA automatic fitting reference targets unmet at the refinement/domain limit");
        }
        if(!ready)throw std::logic_error("MANTA reference selection has no admitted winner");
    } else {
        const auto& p=reference->plan();
        ledger.record(add(margin_extra,add(history,add(base_live,add(gamma_->plan().modeled_peak_bytes,p.modeled_peak_bytes)))),
            p.shell_calls,p.primitive_pairs,p.scalar_terms,p.kernel_terms);
        if(weights_!=reference->source()->pair_source().weights())
            throw std::invalid_argument("MANTA complete fitting reference weights mismatch");
    }
    info_.fitting_reference=reference->receipt();
    const Count domains=o.domain_selection?o.domain_selection->maximum_refinements:0;
    const Count candidates=sequence?sequence->plan().candidates:0;
    for(Count basis_attempt=0;basis_attempt<=candidates;++basis_attempt) {
        if(basis_attempt) {
            bound.reset();candidate=&sequence->candidate(basis_attempt-1);
            pair=initial_pair;auxiliary=initial_auxiliary;cube=initial_cube;
            replace_domain(); // Original metric controls reset for every basis.
        }
        for(Count domain=0;domain<=domains;++domain) {
            for(Count attempt=0;attempt<=o.maximum_metric_refinements;++attempt) {
                if(!bound)bound=probe();
                const auto& c=bound->receipt();const auto& metric=gamma_->metric_info();
                PeriodicRIFittingDomainTrial trial{pair,auxiliary,metric.cutoff,cube,gamma_->coupling_source().plan().vectors,metric.rank,domain,attempt,c};
                if(o.maximum_metric_refinements)info_.fitting_trials.push_back({metric.cutoff,metric.rank,c});
                if(o.domain_selection)info_.domain_trials.push_back(trial);
                if(sequence)info_.auxiliary_trials.push_back({basis_attempt,gamma_->plan().n_auxiliary,
                    candidate?candidate->receipt().identity:"seed",gamma_->coupling_source().source().short_range.auxiliary_basis_identity,trial});
                if(fitting_targets_met(c,o) && c.metric_targets_met()) {
                    info_.fitting_bound=c;info_.reference_search=ledger.charged;
                    if(base)info_.selected_fitting_plan=plan_;
                    if(candidate)info_.selected_auxiliary_generation=candidate->receipt();
                    fitting_bound_=std::move(bound);return;
                }
                const double next=std::max(o.minimum_metric_cutoff,metric.cutoff*o.metric_cutoff_reduction);
                if(attempt==o.maximum_metric_refinements || next>=metric.cutoff)break;
                bound.reset();
                const auto& q=gamma_->plan();const Count m=q.n_auxiliary;
                const Count refresh=mul(512,add(mul(m,square(m)),add(square(m),entries)));
                const Count peak=add(margin_extra,add(history,add(base_live,add(reference->plan().source_live_bytes,mul(2,q.modeled_peak_bytes)))));
                ledger.record(peak,q.shell_calls,q.primitive_products,add(q.scalar_terms,refresh),add(q.kernel_terms,mul(16,refresh)));
                replace_gamma(gamma_->refine_metric(next,leaf_budget));
            }
            if(!o.domain_selection || domain==domains)break;
            const auto& p=*o.domain_selection;
            const auto next_pair=std::min(p.maximum_pair_cutoff_bohr,std::max(1.,pair*p.growth_factor));
            const auto next_aux=std::min(p.maximum_auxiliary_cutoff_bohr,std::max(1.,auxiliary*p.growth_factor));
            const auto next_cube=std::min(p.maximum_reciprocal_cube_radius,cube+1);
            if(pair==next_pair && auxiliary==next_aux && cube==next_cube)break;
            bound.reset();pair=next_pair;auxiliary=next_aux;cube=next_cube;replace_domain();
        }
    }
    if(bound && !bound->receipt().rank_target_met())
        throw std::runtime_error("MANTA fitting metric rank target unmet at the auxiliary/domain/metric limit");
    if(bound && !bound->receipt().action_target_met())throw std::runtime_error("MANTA metric action targets unmet at the auxiliary/domain/metric limit");
    if(sequence)throw std::runtime_error("MANTA fitting error targets unmet at the auxiliary/domain/metric limit");
    if(o.domain_selection)throw std::runtime_error("MANTA fitting error targets unmet at the physical fitting domain/metric limit");
    throw std::runtime_error("MANTA fitting error targets unmet after automatic reference and bounded metric selection");
}
} // namespace vibeqc::manta
