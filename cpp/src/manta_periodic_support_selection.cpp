#include "vibeqc/manta_periodic_support_selection.hpp"
#include "manta_periodic_hf_source.hpp"
#include "manta_pair_image_internal.hpp"

namespace vibeqc::manta {
namespace {
void check_policy(const PeriodicSupportRequest& o,const PeriodicSupportSelectionOptions& p) {
    if(!p.maximum_attempts || p.maximum_attempts>1024 || !p.maximum_completion_cube_radius || p.maximum_completion_cube_radius>2147483647)
        throw std::invalid_argument("MANTA support selection requires 1..1024 attempts and a positive supported cube limit");
    if(!std::isfinite(p.growth_factor)||p.growth_factor<=1||p.growth_factor>4)
        throw std::invalid_argument("MANTA support selection growth factor must be in (1,4]");
    for(double x:{p.maximum_pair_cutoff_bohr,p.maximum_nuclear_cutoff_bohr,p.maximum_reciprocal_energy_cutoff})
        if(!std::isfinite(x)||x<=0)throw std::invalid_argument("MANTA support selection limits must be finite positive");
    if(!o.pair_cutoff_bohr || *o.pair_cutoff_bohr>p.maximum_pair_cutoff_bohr || o.nuclear_cutoff_bohr>p.maximum_nuclear_cutoff_bohr || o.reciprocal_energy_cutoff>p.maximum_reciprocal_energy_cutoff)
        throw std::invalid_argument("MANTA support selection initial physical support exceeds its limit");
    for(Count x:{o.one_electron_accuracy.completion_cube_radius,o.nuclear_accuracy.ao_completion_cube_radius,
        o.nuclear_accuracy.real_completion_cube_radius,o.nuclear_accuracy.reciprocal_completion_cube_radius,
        o.exchange_accuracy.ao_completion_cube_radius,o.exchange_accuracy.reciprocal_completion_cube_radius})
        if(x>p.maximum_completion_cube_radius)throw std::invalid_argument("MANTA support selection initial completion cube exceeds its limit");
}
void caps(const PeriodicSupportProbePlan& p,const ResourceBudget& b) {
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA support probe memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA support probe shell budget exceeded");
    cap(p.primitive_pairs,b.primitive_quartets,"MANTA support probe primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA support probe scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA support probe kernel budget exceeded");
}
Count reach(const Eigen::Matrix3d& inverse,double radius) {
    double maximum=0;for(int a=0;a<3;++a)maximum=std::max(maximum,inverse.row(a).norm()*radius);
    if(!std::isfinite(maximum)||maximum>2147483644.)throw std::invalid_argument("MANTA support proof extent exceeds supported labels");
    return static_cast<Count>(std::ceil(maximum))+2;
}
bool grow_proofs(PeriodicSupportRequest& o,const PeriodicSupportSelectionOptions& policy,const BasisSet& basis,const PeriodicSystem& s) {
    const Eigen::Matrix3d inverse=s.lattice.inverse(),reciprocal_inverse=s.reciprocal_lattice().inverse();
    const Count ao=reach(inverse,*o.pair_cutoff_bohr+max_shell_offset(basis.libint(),basis.libint()));
    const Count real=reach(inverse,o.nuclear_cutoff_bohr),reciprocal=reach(reciprocal_inverse,std::sqrt(2*o.reciprocal_energy_cutoff));
    if(std::max({ao,real,reciprocal})>policy.maximum_completion_cube_radius)return false;
    o.one_electron_accuracy.completion_cube_radius=std::max(o.one_electron_accuracy.completion_cube_radius,ao);
    o.nuclear_accuracy.ao_completion_cube_radius=std::max(o.nuclear_accuracy.ao_completion_cube_radius,ao);
    o.nuclear_accuracy.real_completion_cube_radius=std::max(o.nuclear_accuracy.real_completion_cube_radius,real);
    o.nuclear_accuracy.reciprocal_completion_cube_radius=std::max(o.nuclear_accuracy.reciprocal_completion_cube_radius,reciprocal);
    if(o.exchange_accuracy.enabled()) {
        o.exchange_accuracy.ao_completion_cube_radius=std::max(o.exchange_accuracy.ao_completion_cube_radius,ao);
        o.exchange_accuracy.reciprocal_completion_cube_radius=std::max(o.exchange_accuracy.reciprocal_completion_cube_radius,reciprocal);
    }
    return true;
}
double maximum(const md::Matrix& x) { return x.size()?x.maxCoeff():0; }
double maximum(const std::vector<double>& x) { return x.empty()?0:*std::max_element(x.begin(),x.end()); }
PeriodicSupportTrial probe(const BasisSet& basis,const PeriodicSystem& s,const RegularKMesh& mesh,
    const PeriodicSupportRequest& o,const PeriodicSupportDomains& d,const ResourceBudget& budget) {
    const auto& p=d.plan.probe;const Count nk=mesh.size(),n=basis.nbasis(),e=mul(nk,square(n));
    md::Matrix k(nk,3);for(Count ik=0;ik<nk;++ik)k.row(ik)=(s.reciprocal_lattice()*mesh.fractional_at(ik)).transpose();
    auto leaf=budget;leaf.other_live_bytes=p.modeled_peak_bytes-p.overlap.modeled_peak_bytes;
    const auto overlap=compute_bloch_one_electron(basis,"overlap",d.translations,k,o.pair_cutoff_bohr,leaf);
    BoundedPeriodicRHFCaps dc;dc.maximum_owned_numerical_bytes=p.density.peak_owned_numerical_bytes;
    dc.maximum_total_numerical_bytes=p.modeled_peak_bytes;dc.maximum_work_units=p.density.maximum_work_units;
    dc.other_live_numerical_bytes=p.modeled_peak_bytes-p.density.peak_owned_numerical_bytes-p.density.borrowed_input_bytes;
    const auto density=bound_bounded_periodic_density_envelope(nk,n,p.n_alpha,p.n_beta,overlap.values.data(),e,o.scf,dc);
    leaf.other_live_bytes=p.modeled_peak_bytes-p.one_electron.modeled_peak_bytes;
    const auto st=bound_periodic_one_electron_accuracy(basis,s.lattice,d.translations,k,o.pair_cutoff_bohr,density,o.one_electron_accuracy,leaf,false,overlap.values,o.scf);
    leaf.other_live_bytes=p.modeled_peak_bytes-p.nuclear.modeled_peak_bytes;
    const auto nuclear=bound_periodic_nuclear_accuracy(basis,s,d.reciprocal_vectors,d.translations,k,o.alpha,o.nuclear_cutoff_bohr,
        o.pair_cutoff_bohr,density,o.nuclear_accuracy,leaf,false);
    PeriodicSupportTrial t;t.reciprocal_energy_cutoff=o.reciprocal_energy_cutoff;t.pair_cutoff_bohr=*o.pair_cutoff_bohr;t.nuclear_cutoff_bohr=o.nuclear_cutoff_bohr;
    t.ao_cube_radius=o.nuclear_accuracy.ao_completion_cube_radius;t.real_cube_radius=o.nuclear_accuracy.real_completion_cube_radius;
    t.reciprocal_cube_radius=o.nuclear_accuracy.reciprocal_completion_cube_radius;
    t.overlap_error=st.images.receipt.maximum_overlap_error;t.kinetic_error=st.images.receipt.maximum_kinetic_error;t.kinetic_energy_error=st.kinetic_energy_error_bound;
    t.nuclear_matrix_error=nuclear.nuclear.receipt.maximum_potential_error;t.attraction_energy_error=nuclear.attraction_energy_error_bound;
    t.nuclear_energy_error=nuclear.nuclear.receipt.nuclear_energy_error_bound;
    t.overlap_identity=overlap.receipt.source_identity;t.nuclear_identity=nuclear.nuclear.receipt.nuclear_source_identity;
    t.one_electron_certificate=st.certificate_identity;t.nuclear_certificate=nuclear.certificate_identity;t.density_envelope_identity=st.density_envelope_identity;
    if(t.density_envelope_identity!=nuclear.density_envelope_identity)throw std::logic_error("MANTA support trial lost its shared canonical envelope");
    t.overlap_image_rank_requested=o.one_electron_accuracy.overlap_image_rank;
    if(st.overlap_rank) {
        t.overlap_image_rank_qualified=st.overlap_rank->qualified;
        t.overlap_image_perturbation_bound=st.overlap_rank->maximum_perturbation_bound;
        t.overlap_image_rank_radius=st.overlap_rank->minimum_stability_radius;
    }
    t.qualified=st.qualified&&nuclear.qualified;t.grow_pair=!st.qualified;
    if(!nuclear.qualified) {
        const auto& nr=nuclear.nuclear.receipt;
        if(t.nuclear_matrix_error>o.nuclear_accuracy.matrix_error||t.attraction_energy_error>o.nuclear_accuracy.attraction_energy_error) {
            std::vector<std::complex<double>> h(e);for(Count i=0;i<e;++i)h[i]=density.weighted_entry_bounds[i];
            auto score=[&](const md::Matrix& b) {return std::max(maximum(b)/o.nuclear_accuracy.matrix_error,
                pair_image_detail::density_trace_bound(b,h,std::vector<double>(nk,1.))/o.nuclear_accuracy.attraction_energy_error);};
            const double a=score(nuclear.nuclear.real_error_bound),b=score(nuclear.nuclear.reciprocal_error_bound),c=score(nuclear.nuclear.pair_image_error_bound);
            if(a>=std::max(b,c))t.grow_nuclear=true;else if(b>=c)t.grow_reciprocal=true;else t.grow_pair=true;
        }
        if(t.nuclear_energy_error>o.nuclear_accuracy.nuclear_energy_error) {
            if(nr.nuclear_real_error_bound>=nr.nuclear_reciprocal_error_bound)t.grow_nuclear=true;else t.grow_reciprocal=true;
        }
    }
    if(p.exchange) {
        PeriodicRHFReceipt correction;compute_madelung(s,mesh,o,p.madelung,correction);
        leaf.other_live_bytes=p.modeled_peak_bytes-p.exchange->modeled_peak_bytes;
        const auto exchange=bound_periodic_exchange_accuracy(basis,s.lattice,d.translations,k,d.exchange_vectors,d.transfer_offsets,
            o.pair_cutoff_bohr,overlap.values,density,o.exchange_q0=="BvK-ewald",correction.madelung,o.exchange_scale,o.exchange_accuracy,leaf,false);
        t.exchange_matrix_error=exchange.maximum_error;t.exchange_energy_error=exchange.energy_error_bound;
        t.exchange_certificate=exchange.certificate_identity;t.qualified=t.qualified&&exchange.qualified;
        if(!exchange.qualified) {
            const double image=o.exchange_scale*maximum(exchange.exchange.pair_image_error_bound)+exchange.maximum_correction_error;
            if(image>=o.exchange_scale*maximum(exchange.exchange.reciprocal_error_bound))t.grow_pair=true;else t.grow_reciprocal=true;
        }
    }
    t.modeled_peak_bytes=d.plan.modeled_peak_bytes;t.shell_calls=d.plan.shell_calls;t.primitive_pairs=d.plan.primitive_pairs;
    t.scalar_terms=d.plan.scalar_terms;t.kernel_terms=d.plan.kernel_terms;return t;
}
} // namespace

PeriodicSupportProbePlan plan_periodic_support_probe(const BasisSet& basis,const PeriodicSystem& s,const RegularKMesh& mesh,
    Count ng,Count nc,Count nq,const PeriodicSupportRequest& o,const ResourceBudget& b) {
    margin(b);PeriodicSupportProbePlan p;
    if(s.dim!=3||s.charge!=0||s.multiplicity<1||s.unit_cell.empty())throw std::invalid_argument("MANTA support selection requires a neutral all-electron 3D source");
    Count electrons=0;for(const auto& atom:s.unit_cell) {if(atom.Z<=0)throw std::invalid_argument("MANTA support selection requires positive nuclear charges");electrons=add(electrons,atom.Z);}
    if(Count(s.multiplicity-1)>electrons || (electrons-Count(s.multiplicity-1))%2)throw std::invalid_argument("MANTA support selection electron populations are inconsistent");
    p.n_beta=(electrons-Count(s.multiplicity-1))/2;p.n_alpha=electrons-p.n_beta;
    if(!o.one_electron_accuracy.enabled()||!o.nuclear_accuracy.enabled())throw std::invalid_argument("MANTA support selection requires one-electron and nuclear targets");
    if(!std::isfinite(o.exchange_scale)||o.exchange_scale<0||(o.exchange_accuracy.enabled()&&o.exchange_scale<=0))
        throw std::invalid_argument("MANTA support selection requires a positive active exact-exchange coefficient");
    if(o.exchange_q0!="strict-zero-mode"&&o.exchange_q0!="BvK-ewald")throw std::invalid_argument("MANTA support selection exchange convention is unsupported");
    if(!std::isfinite(o.madelung_tolerance)||o.madelung_tolerance<=0||o.madelung_tolerance>=1||
        (o.madelung_alpha&&(!std::isfinite(*o.madelung_alpha)||*o.madelung_alpha<=0)))throw std::invalid_argument("MANTA support selection Madelung controls are invalid");
    const Count nk=mesh.size(),e=mul(nk,square(basis.nbasis()));auto leaf=unlimited(b);
    p.overlap=plan_bloch_one_electron(basis,"overlap",nc,nk,o.pair_cutoff_bohr,leaf);
    p.density=plan_bounded_periodic_density_envelope(nk,basis.nbasis(),p.n_alpha,p.n_beta,o.scf);
    p.one_electron=plan_periodic_one_electron_accuracy(basis,nc,nk,o.pair_cutoff_bohr,o.one_electron_accuracy,leaf,o.scf);
    p.nuclear=plan_periodic_nuclear_accuracy(basis,s,ng,nc,nk,o.alpha,o.nuclear_cutoff_bohr,o.pair_cutoff_bohr,o.nuclear_accuracy,leaf);
    if(o.exchange_accuracy.enabled()) {
        p.exchange=plan_periodic_exchange_accuracy(basis,nc,nk,nq,o.pair_cutoff_bohr,o.exchange_q0=="BvK-ewald",o.exchange_accuracy,leaf);
        p.madelung=plan_madelung(s,mesh,o);
    }
    p.modeled_peak_bytes=add(b.other_live_bytes,add(65536,add(mul(128,e),add(p.overlap.modeled_peak_bytes,p.density.peak_owned_numerical_bytes))));
    p.shell_calls=p.overlap.shell_calls;p.primitive_pairs=p.overlap.primitive_pairs;
    p.scalar_terms=add(p.overlap.scalar_terms,add(p.density.maximum_work_units,add(mul(4096,e),p.madelung.kernel_terms)));
    p.kernel_terms=add(p.overlap.kernel_terms,mul(128,p.scalar_terms));
    auto charge=[&](const auto& q) {p.modeled_peak_bytes=add(p.modeled_peak_bytes,q.modeled_peak_bytes);
        p.shell_calls=add(p.shell_calls,q.shell_calls);p.primitive_pairs=add(p.primitive_pairs,q.primitive_pairs);
        p.scalar_terms=add(p.scalar_terms,q.scalar_terms);p.kernel_terms=add(p.kernel_terms,q.kernel_terms);};
    charge(p.one_electron);charge(p.nuclear);if(p.exchange)charge(*p.exchange);
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,p.madelung.modeled_peak_bytes);caps(p,b);return p;
}

PeriodicSupportSelectionResult select_periodic_supports(const BasisSet& basis,const PeriodicSystem& s,const RegularKMesh& mesh,
    const PeriodicSupportRequest& request,const PeriodicSupportSelectionOptions& policy,const ResourceBudget& budget) {
    check_policy(request,policy);margin(budget);PeriodicSupportSelectionResult out;out.policy=policy;out.selected=request;
    // Trial metadata, native/Python copies and serialized output remain live
    // beside both the search and the final method. Work is charged by admitted
    // trial upper bounds, including discarded trials, never only the winner.
    out.retained_bytes=add(65536,mul(16384,policy.maximum_attempts));
    auto b=budget;b.other_live_bytes=add(b.other_live_bytes,out.retained_bytes);
    cap(b.other_live_bytes,b.memory_bytes,"MANTA support selection history memory budget exceeded");
    // Admit the first trial before basis-center scans or history allocation.
    const auto initial=plan_periodic_support_trial(basis,s,mesh,out.selected,b);
    // Conservatively charge the first count-only preparation by its entire
    // admitted scalar/kernel upper bound. It executes no shell kernels.
    out.preparation_scalar_terms=out.scalar_terms=initial.scalar_terms;
    out.preparation_kernel_terms=out.kernel_terms=initial.kernel_terms;
    out.modeled_peak_bytes=initial.modeled_peak_bytes;
    b.scalar_terms-=initial.scalar_terms;b.kernel_terms-=initial.kernel_terms;
    out.trials.reserve(policy.maximum_attempts);
    for(Count attempt=0;attempt<policy.maximum_attempts;++attempt) {
        if(!grow_proofs(out.selected,policy,basis,s)) {out.termination="completion-cube-limit";break;}
        {
            const auto domains=build_periodic_support_domains(basis,s,mesh,out.selected,b);
            const auto trial=probe(basis,s,mesh,out.selected,domains,b);out.trials.push_back(trial);
            out.modeled_peak_bytes=std::max(out.modeled_peak_bytes,trial.modeled_peak_bytes);
            out.shell_calls=add(out.shell_calls,trial.shell_calls);out.primitive_pairs=add(out.primitive_pairs,trial.primitive_pairs);
            out.scalar_terms=add(out.scalar_terms,trial.scalar_terms);out.kernel_terms=add(out.kernel_terms,trial.kernel_terms);
            b.shell_calls-=trial.shell_calls;b.primitive_quartets-=trial.primitive_pairs;b.scalar_terms-=trial.scalar_terms;b.kernel_terms-=trial.kernel_terms;
            if(trial.qualified) {out.qualified=true;out.termination="targets-passed";break;}
        }
        if(attempt+1==policy.maximum_attempts) {out.termination="attempt-limit";break;}
        const auto& t=out.trials.back();bool enlarged=false;
        auto grow=[&](double& x,double ceiling,double factor) {const double next=std::min(ceiling,x*factor);if(next>x){x=next;enlarged=true;}};
        if(t.grow_pair)grow(*out.selected.pair_cutoff_bohr,policy.maximum_pair_cutoff_bohr,policy.growth_factor);
        if(t.grow_nuclear)grow(out.selected.nuclear_cutoff_bohr,policy.maximum_nuclear_cutoff_bohr,policy.growth_factor);
        if(t.grow_reciprocal)grow(out.selected.reciprocal_energy_cutoff,policy.maximum_reciprocal_energy_cutoff,policy.growth_factor*policy.growth_factor);
        if(!enlarged) {out.termination="physical-support-limit";break;}
    }
    Digest id;id.text("manta.periodic-component-support-selection.v1;fixed-source-before-scf;no-relaxation");
    id.word(policy.maximum_attempts);id.word(policy.maximum_completion_cube_radius);
    for(double x:{policy.growth_factor,policy.maximum_pair_cutoff_bohr,policy.maximum_nuclear_cutoff_bohr,policy.maximum_reciprocal_energy_cutoff})id.real(x);
    for(const auto& t:out.trials) {id.real(t.pair_cutoff_bohr);id.real(t.reciprocal_energy_cutoff);id.real(t.nuclear_cutoff_bohr);
        id.text(t.one_electron_certificate);id.text(t.nuclear_certificate);id.text(t.exchange_certificate);}
    id.text(out.termination);out.certificate_identity=id.hash.finish_hex();return out;
}
} // namespace vibeqc::manta
