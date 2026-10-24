#include "vibeqc/manta_periodic_ri_fitting_bounds.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_periodic_error.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "manta_ri_interval.hpp"
#include <algorithm>
#include <cstring>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
using periodic_error::upper;
using periodic_error::positive_add;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) { std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=v>>(56-8*i);hash.update(b.data(),8); }
    void text(const std::string& s) { word(s.size());hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double v) { if(v==0)v=0.;Count bits;std::memcpy(&bits,&v,8);word(bits); }
};
double magnitude(Complex z) { return upper(std::hypot(static_cast<long double>(z.real()),static_cast<long double>(z.imag()))); }
template<class T> const T& required(const std::shared_ptr<const T>& p) {
    if(!p)throw std::invalid_argument("MANTA fitting bound requires typed RI and BUPO sources");return *p;
}
void validate_rank_perturbation(const std::optional<double>& value) {
    if(value && (!std::isfinite(*value) || *value<0.))
        throw std::invalid_argument("MANTA metric rank perturbation must be finite nonnegative");
}
void compatible(const PeriodicRIQBuilder& ri,const PeriodicBupoSource& reference) {
    ri.validate_request(true,false);
    const auto& fit=ri.coupling_source();const auto& pair=reference.pair_source();
    if(!reference.bounds_full_coulomb())throw std::invalid_argument("MANTA fitting bound requires full Ewald and pair-image reference bounds");
    if(fit.source().kernel!="coulomb" || !fit.transfer().isZero(0.))
        throw std::invalid_argument("MANTA fitting bound requires a literal Gamma Coulomb source");
    if(fit.source().short_range.orbital_basis_identity!=pair.receipt().basis_identity ||
       fit.momenta().rows()!=pair.momenta().rows() ||
       !(fit.lattice().array()==pair.lattice().array()).all() ||
       !(fit.momenta().array()==pair.momenta().array()).all())
        throw std::invalid_argument("MANTA fitting bound source basis, cell or momentum frame mismatch");
}
void validate_action_targets(double j,double e) {
    if(!std::isfinite(j) || !std::isfinite(e) || j<0. || e<0. || (j>0.)!=(e>0.))
        throw std::invalid_argument("MANTA metric action targets require both positive finite J/energy values or both zero");
}
}
PeriodicRIFittingBoundPlan PeriodicRIFittingBoundSource::plan_source(const PeriodicRIQBuilder& ri,
    const PeriodicBupoSource& reference,const PeriodicRIFittingBoundOptions& o,const ResourceBudget& b) {
    compatible(ri,reference);return plan_counts(ri.plan(),reference.plan(),o,b);
}
PeriodicRIFittingBoundPlan PeriodicRIFittingBoundSource::plan_counts(const PeriodicRIQPlan& a,
    const PeriodicBupoPlan& c,const PeriodicRIFittingBoundOptions& o,const ResourceBudget& b) {
    using namespace admission; margin(b);validate_rank_perturbation(o.metric_rank_perturbation);validate_action_targets(o.metric_action_j_error,o.metric_action_energy_error);
    if(a.n_basis!=c.n_basis || a.momenta!=c.momenta || !a.spins)
        throw std::invalid_argument("MANTA fitting plan source extent mismatch");
    if(!std::isfinite(o.density_l1_limit) || o.density_l1_limit<=0.)
        throw std::invalid_argument("MANTA fitting bound requires a positive finite density L1 limit");
    for(double value:{o.j_error,o.energy_error})if(!std::isfinite(value) || value<0.)
        throw std::invalid_argument("MANTA fitting targets must be finite nonnegative");
    if((o.j_error>0.)!=(o.energy_error>0.))throw std::invalid_argument("MANTA fitting targets require both J and energy");
    PeriodicRIFittingBoundPlan p;
    p.n_basis=a.n_basis;p.momenta=a.momenta;p.spins=a.spins;
    const Count mesh=mul(a.momenta,square(a.n_basis));p.maximum_probes=mesh;
    if(!o.probe_batch_size || o.probe_batch_size>32)throw std::invalid_argument("MANTA fitting probe batch requires 1 to 32 densities");
    if(!o.maximum_probes || mesh>o.maximum_probes)throw std::invalid_argument("MANTA fitting probe limit exceeded");
    if(!o.density_entry_limits.empty() && o.density_entry_limits.size()!=mesh)
        throw std::invalid_argument("MANTA fitting density entry-limit shape mismatch");
    for(double value:o.density_entry_limits)if(!std::isfinite(value) || value<0.)
        throw std::invalid_argument("MANTA fitting density entry limits must be finite nonnegative");
    // Reuse the already prepared immutable producers. Their complete peak
    // plans conservatively reserve both retained sources plus each build.
    // One RI build is reserved after all probes for the returned consumer.
    const Count calls=add(mesh,1),extra=add(mul(2816,mul(mesh,add(mesh,1))),mul(8448,mesh));
    p.shell_calls=add(mul(calls,a.shell_calls),mul(mesh,c.shell_calls));
    p.primitive_products=add(mul(calls,a.primitive_products),mul(mesh,c.primitive_products));
    p.scalar_terms=add(64,add(extra,add(mul(calls,a.scalar_terms),mul(mesh,c.scalar_terms))));
    p.kernel_terms=add(512,add(mul(8,extra),add(mul(calls,a.kernel_terms),mul(mesh,c.kernel_terms))));
    p.modeled_peak_bytes=add(add(a.modeled_peak_bytes,c.modeled_peak_bytes),
        add(add(b.other_live_bytes,b.backend_margin_bytes),add(4096,add(mul(add(512,mul(16,a.spins)),mesh),mul(32,a.momenta)))));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,mul(std::min(mesh,o.probe_batch_size),c.batch_workspace_per_density));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(16640,mul(768,mesh)));
    if(o.metric_action_j_error>0.) {
        if(!a.coupling_norm_supported)throw std::invalid_argument("MANTA metric action targets require an explicit tile or compressed non-pair-tree coupling source");
        auto unlimited=b;unlimited.other_live_bytes=0;
        unlimited.memory_bytes=unlimited.scalar_terms=unlimited.kernel_terms=std::numeric_limits<Count>::max();
        const auto audit=plan_hermitian_inverse_action(a.n_auxiliary,RIOptions{},unlimited);
        const auto column=plan_hermitian_inverse_column(a.n_auxiliary,unlimited);
        const auto family=plan_hermitian_inverse_family(a.n_auxiliary,mesh,unlimited);
        if(!a.explicit_coupling_action) {
            const auto coupling=plan_periodic_gdf_coupling_norm_bounds(a.coupling_plan,unlimited);
            p.modeled_peak_bytes=add(p.modeled_peak_bytes,coupling.modeled_peak_bytes);
            p.shell_calls=add(p.shell_calls,coupling.shell_calls);p.primitive_products=add(p.primitive_products,coupling.primitive_products);
            p.scalar_terms=add(p.scalar_terms,coupling.scalar_terms);p.kernel_terms=add(p.kernel_terms,coupling.kernel_terms);
        }
        // Cache a scalar adjoint norm during the admitted column audit;
        // reserve its scan and outward propagation on the returned build.
        const Count propagation=add(256,mul(128,mesh));
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,64);
        p.scalar_terms=add(p.scalar_terms,propagation);p.kernel_terms=add(p.kernel_terms,mul(8,propagation));
        // One auxiliary-by-AO-shell-pair panel, scalar column envelopes and
        // the retained action audit. No complete auxiliary/AO-pair tensor.
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(audit.modeled_peak_bytes,
            add(mul(16,mul(a.n_auxiliary,a.maximum_coupling_tile_elements)),mul(512,mesh))));
        p.shell_calls=add(p.shell_calls,a.shell_calls);p.primitive_products=add(p.primitive_products,a.primitive_products);
        p.scalar_terms=add(p.scalar_terms,add(a.scalar_terms,add(audit.scalar_terms,add(mul(mesh,column.scalar_terms),family.scalar_terms))));
        p.kernel_terms=add(p.kernel_terms,add(a.kernel_terms,add(audit.kernel_terms,add(mul(mesh,column.kernel_terms),family.kernel_terms))));
    }
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA fitting-bound memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA fitting-bound shell-call budget exceeded");
    cap(p.primitive_products,b.primitive_quartets,"MANTA fitting-bound primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA fitting-bound scalar-work budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA fitting-bound kernel-work budget exceeded");return p;
}
PeriodicRIFittingBoundSource::PeriodicRIFittingBoundSource(std::shared_ptr<const PeriodicRIQBuilder> ri,
    std::shared_ptr<const PeriodicBupoSource> reference,const PeriodicRIFittingBoundOptions& o,const ResourceBudget& budget,bool enforce_targets)
    :plan_(plan_source(required(ri),required(reference),o,budget)),ri_(std::move(ri)),reference_(std::move(reference)),options_(o) {
    const Count n=plan_.n_basis,n2=n*n,nk=plan_.momenta,mesh=nk*n2;
    const auto& w=reference_->pair_source().weights();const auto& accuracy=ri_->metric_info().accuracy;
    const auto& fit_entries=ri_->accuracy_entry_limits();
    auto& r=receipt_;r.density_l1_limit=o.density_l1_limit;r.component_density_envelope=!o.density_entry_limits.empty();
    r.targets_requested=o.j_error>0.;r.ri_source_identity=ri_->metric_info().source_identity;
    r.metric_rank_perturbation=o.metric_rank_perturbation;r.metric_spectrum=ri_->metric_info().spectrum;
    r.metric_rank_qualified=o.metric_rank_perturbation && r.metric_spectrum.rank_certified &&
        *o.metric_rank_perturbation<r.metric_spectrum.rank_stability_radius;
    if(enforce_targets && !r.rank_target_met())
        throw std::runtime_error("MANTA fitting metric rank target unmet for the selected stored Gamma metric");
    r.metric_action_requested=o.metric_action_j_error>0.;r.metric_action_j_target=o.metric_action_j_error;r.metric_action_energy_target=o.metric_action_energy_error;
    if(r.metric_action_requested)certify_metric_action(budget);
    if(enforce_targets && !r.action_target_met())throw std::runtime_error("MANTA metric action targets unmet for the selected stored Gamma metric");
    r.reference_source_identity=reference_->source_identity();r.pair_source_identity=reference_->pair_source().receipt().source_identity;
    std::vector<long double> finite_sum(mesh),reference_sum(mesh);
    std::vector<double> finite_max(mesh),reference_max(mesh);
    std::vector<Complex> spin(plan_.spins*mesh);
    struct Probe {Count k,i,j;double scale,h;bool imaginary;};
    std::vector<Probe> probes;std::vector<std::vector<Complex>> totals;
    probes.reserve(o.probe_batch_size);totals.reserve(o.probe_batch_size);
    using Bounds=std::pair<std::vector<double>,std::vector<double>>;
    Bounds pending;
    std::array<std::vector<long double>,4> component_sum;
    std::array<std::vector<double>,4> component_max,pending_components;
    for(Count c=0;c<4;++c){component_sum[c].resize(mesh);component_max[c].resize(mesh);}

    // Normalized Hermitian probes: Eii and (Eij+Eji)/2, i(Eij-Eji)/2.
    // Batch boundaries may split a real/imaginary pair. Keep just its pending
    // response envelope, then apply the original hypot and summation order.
    auto flush=[&]() {
        if(probes.empty())return;
        PeriodicBupoBatchResult reference;
        if(o.probe_batch_size==1) {
            reference.results.push_back(reference_->build(totals.front()));
            const auto& receipt=reference.results.front().receipt;auto& work=reference.receipt;
            work.source_passes=1;work.real_shell_calls=receipt.direct.real_shell_calls;
            work.fourier_shell_calls=receipt.direct.fourier_shell_calls;work.overlap_shell_calls=receipt.direct.overlap_shell_calls;
            if(receipt.domain)work.domain_majorant_pairs_evaluated=receipt.domain->majorant_pairs_evaluated;
        } else reference=reference_->build_batch(totals,budget);
        ++r.reference_batches;r.reference_source_passes+=reference.receipt.source_passes;
        r.reference_shell_calls+=reference.receipt.real_shell_calls+reference.receipt.fourier_shell_calls+reference.receipt.overlap_shell_calls;
        r.reference_majorant_pairs+=reference.receipt.domain_majorant_pairs_evaluated;
        for(Count index=0;index<probes.size();++index) {
            const auto& probe=probes[index];const auto& exact=reference.results[index];
            std::fill(spin.begin(),spin.end(),Complex{});std::copy(totals[index].begin(),totals[index].end(),spin.begin());
            const auto fitted=ri_->build(spin,w,true,false);++r.probes;
            if(!exact.receipt.domain || !exact.receipt.pair_images)throw std::logic_error("MANTA fitting reference lost full Coulomb bounds");
            Bounds a{std::vector<double>(mesh),std::vector<double>(mesh)};
            if(!exact.domain_real_j_error_bound || !exact.domain_reciprocal_j_error_bound || !exact.pair_image_j_error_bound)
                throw std::logic_error("MANTA fitting reference lost component bounds");
            const std::array<const md::Matrix*,4> maps{&exact.finite_j_error_bound,&*exact.domain_real_j_error_bound,
                &*exact.domain_reciprocal_j_error_bound,&*exact.pair_image_j_error_bound};
            std::array<std::vector<double>,4> components;
            for(auto& values:components)values.resize(mesh);
            for(Count x=0;x<mesh;++x) {
                a.first[x]=upper(static_cast<long double>(magnitude(fitted.J[x]-exact.values[x]))/probe.scale);
                a.second[x]=upper(static_cast<long double>(exact.j_error_bound((x%n2)/n,x%n))/probe.scale);
                for(Count c=0;c<4;++c)components[c][x]=upper(static_cast<long double>((*maps[c])((x%n2)/n,x%n))/probe.scale);
            }
            if(probe.i!=probe.j && !probe.imaginary){pending=std::move(a);pending_components=std::move(components);continue;}
            if(probe.imaginary)for(Count x=0;x<mesh;++x) {
                a.first[x]=upper(std::hypot(static_cast<long double>(pending.first[x]),static_cast<long double>(a.first[x])));
                a.second[x]=upper(std::hypot(static_cast<long double>(pending.second[x]),static_cast<long double>(a.second[x])));
                for(Count c=0;c<4;++c)components[c][x]=upper(std::hypot(static_cast<long double>(pending_components[c][x]),static_cast<long double>(components[c][x])));
            }
            for(Count x=0;x<mesh;++x) {
                finite_max[x]=std::max(finite_max[x],a.first[x]);reference_max[x]=std::max(reference_max[x],a.second[x]);
                for(Count c=0;c<4;++c) {
                    component_max[c][x]=std::max(component_max[c][x],components[c][x]);
                    if(r.component_density_envelope)positive_add(static_cast<long double>(probe.h)*(probe.i==probe.j?1:2)*components[c][x],component_sum[c][x]);
                }
                if(r.component_density_envelope) {
                    positive_add(static_cast<long double>(probe.h)*(probe.i==probe.j?1:2)*a.first[x],finite_sum[x]);
                    positive_add(static_cast<long double>(probe.h)*(probe.i==probe.j?1:2)*a.second[x],reference_sum[x]);
                }
            }
        }
        probes.clear();totals.clear();
    };
    auto enqueue=[&](Count k,Count i,Count j,double scale,double h,bool imaginary) {
        probes.push_back({k,i,j,scale,h,imaginary});totals.emplace_back(mesh);
        const Complex v=(imaginary?Complex(0.,scale):Complex(scale,0.))/(w[k]*(i==j?1.:2.));
        if(!std::isfinite(v.real()) || !std::isfinite(v.imag()))throw std::overflow_error("MANTA fitting probe scaling overflow");
        totals.back()[k*n2+i*n+j]=v;totals.back()[k*n2+j*n+i]=std::conj(v);
        if(probes.size()==o.probe_batch_size)flush();
    };
    for(Count k=0;k<nk;++k)if(w[k]>0.)for(Count i=0;i<n;++i)for(Count j=i;j<n;++j) {
        const Count ij=k*n2+i*n+j,ji=k*n2+j*n+i;
        const double h=r.component_density_envelope?std::min(o.density_entry_limits[ij],o.density_entry_limits[ji]):o.density_l1_limit;
        if(h==0.)continue;
        double scale=std::min({1.,o.density_l1_limit/16.,h/16.});
        if(accuracy.enabled)scale=std::min(scale,accuracy.density_l1_limit/16.);
        if(!fit_entries.empty())scale=std::min({scale,fit_entries[ij]/16.,fit_entries[ji]/16.});
        if(!std::isfinite(scale) || scale<=0.)throw std::invalid_argument("MANTA fitting family cannot be probed within RI accuracy limits");
        enqueue(k,i,j,scale,h,false);if(i!=j)enqueue(k,i,j,scale,h,true);
    }
    flush();
    j_error_bounds_.resize(mesh);long double entry_energy=0.,finite_energy=0.,reference_energy=0.;
    std::array<double,4> component_j{};std::array<long double,4> component_energy{};
    for(Count x=0;x<mesh;++x) {
        double a=upper(static_cast<long double>(o.density_l1_limit)*finite_max[x]);
        double b=upper(static_cast<long double>(o.density_l1_limit)*reference_max[x]);
        if(r.component_density_envelope){a=std::min(a,upper(finite_sum[x]));b=std::min(b,upper(reference_sum[x]));}
        std::array<double,4> values{};
        for(Count c=0;c<4;++c) {
            values[c]=upper(static_cast<long double>(o.density_l1_limit)*component_max[c][x]);
            if(r.component_density_envelope)values[c]=std::min(values[c],upper(component_sum[c][x]));
            component_j[c]=std::max(component_j[c],values[c]);
        }
        j_error_bounds_[x]=upper(static_cast<long double>(a)+b);
        r.finite_difference_j_bound=std::max(r.finite_difference_j_bound,a);
        r.reference_remainder_j_bound=std::max(r.reference_remainder_j_bound,b);
        r.j_error_bound=std::max(r.j_error_bound,j_error_bounds_[x]);
        if(r.component_density_envelope && w[x/n2]>0.) {
            const Count transpose=(x/n2)*n2+(x%n)*n+(x%n2)/n;
            const long double coefficient=.5L*std::min(o.density_entry_limits[x],o.density_entry_limits[transpose]);
            positive_add(coefficient*j_error_bounds_[x],entry_energy);
            positive_add(coefficient*a,finite_energy);positive_add(coefficient*b,reference_energy);
            for(Count c=0;c<4;++c)positive_add(coefficient*values[c],component_energy[c]);
        }
    }
    r.energy_error_bound=upper(.5L*o.density_l1_limit*r.j_error_bound);
    if(r.component_density_envelope)r.energy_error_bound=std::min(r.energy_error_bound,upper(entry_energy));
    // Same fixed-family trace inequality as the combined energy bound.
    // Keep each map's entrywise envelope instead of replacing it by max(J).
    // These are separate upper bounds; their sum need not equal the separately contracted
    // combined bound, so they are not an additive error decomposition.
    r.finite_difference_energy_bound=upper(.5L*o.density_l1_limit*r.finite_difference_j_bound);
    r.reference_remainder_energy_bound=upper(.5L*o.density_l1_limit*r.reference_remainder_j_bound);
    if(r.component_density_envelope) {
        r.finite_difference_energy_bound=std::min(r.finite_difference_energy_bound,upper(finite_energy));
        r.reference_remainder_energy_bound=std::min(r.reference_remainder_energy_bound,upper(reference_energy));
    }
    // Independently valid contribution enclosures; no additivity or ordering
    // relative to the separately minimized combined certificate is assumed.
    const std::array<double*,4> component_j_fields{&r.reference_approximation_j_bound,&r.reference_real_domain_j_bound,&r.reference_reciprocal_domain_j_bound,&r.reference_pair_image_j_bound};
    const std::array<double*,4> component_e_fields{&r.reference_approximation_energy_bound,&r.reference_real_domain_energy_bound,&r.reference_reciprocal_domain_energy_bound,&r.reference_pair_image_energy_bound};
    for(Count c=0;c<4;++c) {
        *component_j_fields[c]=component_j[c];
        double energy=upper(.5L*o.density_l1_limit*component_j[c]);
        if(r.component_density_envelope)energy=std::min(energy,upper(component_energy[c]));
        *component_e_fields[c]=energy;
    }
    if(enforce_targets && r.targets_requested && (r.j_error_bound>o.j_error || r.energy_error_bound>o.energy_error))
        throw std::runtime_error("MANTA fitting error targets unmet for the declared fixed density family");
    Digest id;id.text("manta.periodic-ri-full-coulomb-bound.v1;hermitian-basis;no-roundoff-proof");
    id.text(r.ri_source_identity);id.text(r.reference_source_identity);id.real(o.density_l1_limit);id.real(o.j_error);id.real(o.energy_error);
    id.word(o.density_entry_limits.size());for(double h:o.density_entry_limits)id.real(h);
    for(double value:j_error_bounds_)id.real(value);id.real(r.energy_error_bound);r.certificate_identity=id.hash.finish_hex();
}
void PeriodicRIFittingBoundSource::validate_density(const std::vector<Complex>& d,const std::vector<double>& w) const {
    if(receipt_.targets_requested && (receipt_.j_error_bound>options_.j_error || receipt_.energy_error_bound>options_.energy_error))
        throw std::runtime_error("MANTA unqualified fitting trial cannot be consumed");
    if(!receipt_.action_target_met())throw std::runtime_error("MANTA unqualified metric action trial cannot be consumed");
    if(!receipt_.rank_target_met())throw std::runtime_error("MANTA unqualified metric rank trial cannot be consumed");
    const Count n=plan_.n_basis,n2=n*n,mesh=plan_.momenta*n2;
    if(d.size()!=plan_.spins*mesh || w!=reference_->pair_source().weights())
        throw std::invalid_argument("MANTA fitting certificate density shape or fixed weights mismatch");
    long double l1=0.;
    for(Count x=0;x<mesh;++x) {
        Complex total{};for(Count s=0;s<plan_.spins;++s)total+=d[s*mesh+x];
        const double value=upper(static_cast<long double>(w[x/n2])*magnitude(total));positive_add(value,l1);
        if(!options_.density_entry_limits.empty() && value>options_.density_entry_limits[x])
            throw std::invalid_argument("MANTA fitting density exceeds fixed entry envelope");
    }
    if(upper(l1)>options_.density_l1_limit)throw std::invalid_argument("MANTA fitting density exceeds fixed L1 envelope");
}
PeriodicRIQResult PeriodicRIFittingBoundSource::build(const std::vector<Complex>& d,const std::vector<double>& w) const {
    validate_density(d,w);auto result=ri_->build(d,w,true,false,receipt_.metric_action_requested);
    propagate_metric_roundoff(result.receipt);return result;
}
void PeriodicRIFittingBoundSource::propagate_metric_roundoff(PeriodicRIQReceipt& r) const {
    if(!receipt_.metric_action_qualified || !r.metric_roundoff_evaluated)return;
    if(r.source_identity!=ri_->metric_info().source_identity || !r.has_j)
        throw std::logic_error("MANTA metric arithmetic propagation source mismatch");
    spectral_interval::require_arithmetic();
    // Same stored T as Section 5.35: |T^T delta_c|_max <= max_p ||T_p||_1
    // * |delta_c|_max. This is the metric substep contribution through the
    // exact adjoint, excluding arithmetic in the executed adjoint itself.
    // Sun2017 Eqs.16-17, paired convention; scientific contract Section 5.36.
    r.metric_j_roundoff_bound=spectral_interval::times_up(receipt_.metric_action_adjoint_norm_bound,
        r.metric_coefficient_roundoff_bound);
    r.metric_j_roundoff_evaluated=true;
    if(r.forward_metric_roundoff_evaluated && r.coupling_roundoff_evaluated) {
        // Exact stored paired map, scientific contract Section 5.38. The
        // combined coefficient enclosure already includes metric arithmetic;
        // adding the metric-only contribution again would count it twice.
        r.paired_j_roundoff_bound=spectral_interval::plus_up(
            spectral_interval::times_up(receipt_.metric_action_adjoint_norm_bound,
                r.forward_metric_coefficient_roundoff_bound),r.coupling_adjoint_roundoff_bound);
        r.paired_energy_roundoff_bound=r.forward_metric_energy_roundoff_bound;
        r.paired_roundoff_evaluated=true;
        r.paired_input_roundoff_included=r.weighted_input_roundoff_evaluated && r.coupling_forward_input_error_included;
    }
}
PeriodicRIFittingBoundPlan plan_periodic_ri_fitting_accuracy(const PeriodicRIQPlan& q,
    const PeriodicRIFittingAccuracyOptions& o,const ResourceBudget& b) {
    using namespace admission;validate_rank_perturbation(o.metric_rank_perturbation);validate_action_targets(o.metric_action_j_error,o.metric_action_energy_error);
    if(o.metric_action_j_error>0. && !q.coupling_norm_supported)throw std::invalid_argument("MANTA metric action targets require an explicit tile or compressed non-pair-tree coupling source");
    if(bool(o.reference)==bool(o.reference_selection) || !std::isfinite(o.j_error) || o.j_error<=0. || !std::isfinite(o.energy_error) || o.energy_error<=0.)
        throw std::invalid_argument("MANTA complete fitting accuracy requires one prepared reference or automatic reference policy and positive finite J/energy targets");
    if(!o.probe_batch_size || o.probe_batch_size>32)throw std::invalid_argument("MANTA fitting probe batch requires 1 to 32 densities");
    if(o.maximum_metric_refinements>31 || !std::isfinite(o.metric_cutoff_reduction) ||
       o.metric_cutoff_reduction<=0. || o.metric_cutoff_reduction>=1. ||
       !std::isfinite(o.minimum_metric_cutoff) || o.minimum_metric_cutoff<0.)
        throw std::invalid_argument("MANTA metric refinement needs at most 31 refinements, a reduction in (0,1), and a finite nonnegative cutoff floor");
    PeriodicRIFittingBoundOptions family;family.density_l1_limit=1.;family.j_error=o.j_error;
    family.energy_error=o.energy_error;family.maximum_probes=o.maximum_probes;family.probe_batch_size=o.probe_batch_size;
    family.metric_rank_perturbation=o.metric_rank_perturbation;
    family.metric_action_j_error=o.metric_action_j_error;family.metric_action_energy_error=o.metric_action_energy_error;
    auto leaf=b;leaf.other_live_bytes=0;
    leaf.memory_bytes=leaf.shell_calls=leaf.primitive_quartets=leaf.scalar_terms=leaf.kernel_terms=std::numeric_limits<Count>::max();
    if(o.reference_selection || o.domain_selection || o.auxiliary_selection) {
        const auto search=periodic_ri_fitting_search_budget(o);
        PeriodicRIFittingBoundPlan p;p.n_basis=q.n_basis;p.momenta=q.momenta;p.spins=q.spins;
        p.maximum_probes=mul(q.momenta,square(q.n_basis));
        if(!o.maximum_probes || p.maximum_probes>o.maximum_probes)
            throw std::invalid_argument("MANTA fitting probe limit exceeded");
        if(o.auxiliary_selection && !o.auxiliary_selection->prepared)
            throw std::invalid_argument("MANTA auxiliary selection requires prepared candidate metadata");
        const auto* sequence=o.auxiliary_selection?&o.auxiliary_selection->prepared->plan():nullptr;
        p.maximum_trials=add(mul(sequence?add(sequence->candidates,1):1,
            mul(add(1,o.domain_selection?o.domain_selection->maximum_refinements:0),add(1,o.maximum_metric_refinements))),
            o.reference_selection?o.reference_selection->maximum_refinements:0);
        // The declared reservation still must admit its value-free minimum:
        // retained Gamma owner, family/history capture and selection setup.
        const Count history=add(sequence?sequence->modeled_peak_bytes:0,add(65536,add(mul(16384,add(p.maximum_trials,1)),mul(128,p.maximum_probes))));
        const Count setup=add(sequence?sequence->work_units:0,add(65536,mul(4096,add(p.maximum_trials,p.maximum_probes))));
        const Count extra_margin=std::max(b.backend_margin_bytes,search.backend_margin_bytes)-b.backend_margin_bytes;
        cap(add(extra_margin,add(history,q.modeled_peak_bytes)),search.memory_bytes,"MANTA reference search setup memory budget exceeded");
        cap(setup,search.scalar_terms,"MANTA reference search setup scalar budget exceeded");
        cap(add(mul(16,setup),sequence?mul(16,sequence->work_units):0),search.kernel_terms,"MANTA reference search setup kernel budget exceeded");
        p.modeled_peak_bytes=add(b.other_live_bytes,search.memory_bytes);
        p.shell_calls=search.shell_calls;p.primitive_products=search.primitive_quartets;
        p.scalar_terms=search.scalar_terms;p.kernel_terms=search.kernel_terms;
        cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA automatic reference memory reservation exceeds budget");
        cap(p.shell_calls,b.shell_calls,"MANTA automatic reference shell reservation exceeds budget");
        cap(p.primitive_products,b.primitive_quartets,"MANTA automatic reference primitive reservation exceeds budget");
        cap(p.scalar_terms,b.scalar_terms,"MANTA automatic reference scalar reservation exceeds budget");
        cap(p.kernel_terms,b.kernel_terms,"MANTA automatic reference kernel reservation exceeds budget");
        return p;
    }
    const auto& ref=o.reference->plan();auto p=PeriodicRIFittingBoundSource::plan_counts(q,ref.bupo,family,leaf);
    const Count extra=o.maximum_metric_refinements;p.maximum_trials=add(1,extra);
    if(extra) {
        // Each rejected probe and each new factorization is charged. Count
        // plans reserve the full auxiliary rank, independent of the cutoff.
        // Both old and new RI owners can overlap during replacement. Native
        // trial receipts, copies and option capture have a bounded allowance.
        p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(q.modeled_peak_bytes,mul(8192,p.maximum_trials)));
        p.shell_calls=add(mul(p.maximum_trials,p.shell_calls),mul(extra,q.shell_calls));
        p.primitive_products=add(mul(p.maximum_trials,p.primitive_products),mul(extra,q.primitive_products));
        // Fock replacement also rechecks Gamma conjugacy, forming two
        // inverse-factor products, and refreshes its source identity. Charge
        // this cubic work separately from the RI factorization reservation.
        const Count m=q.n_auxiliary;
        const Count refresh=mul(extra,mul(512,add(mul(m,square(m)),add(square(m),mul(q.momenta,square(q.n_basis))))));
        p.scalar_terms=add(mul(p.maximum_trials,p.scalar_terms),add(refresh,add(mul(extra,q.scalar_terms),mul(8192,p.maximum_trials))));
        p.kernel_terms=add(mul(p.maximum_trials,p.kernel_terms),add(mul(16,refresh),add(mul(extra,q.kernel_terms),mul(65536,p.maximum_trials))));
    }
    p.modeled_peak_bytes=add(b.other_live_bytes,add(p.modeled_peak_bytes,ref.source_live_bytes));
    p.shell_calls=add(p.shell_calls,ref.shell_calls);p.primitive_products=add(p.primitive_products,ref.primitive_pairs);
    p.scalar_terms=add(p.scalar_terms,ref.scalar_terms);p.kernel_terms=add(p.kernel_terms,ref.kernel_terms);
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA complete fitting memory budget exceeded");
    cap(p.shell_calls,b.shell_calls,"MANTA complete fitting shell budget exceeded");
    cap(p.primitive_products,b.primitive_quartets,"MANTA complete fitting primitive budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA complete fitting scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA complete fitting kernel budget exceeded");return p;
}
PeriodicRIFittingBoundOptions periodic_ri_fitting_family(const PeriodicRIFittingAccuracyOptions& o,
    const BoundedPeriodicDensityEnvelope& density) {
    PeriodicRIFittingBoundOptions family;family.density_l1_limit=density.density_l1_limit;
    family.density_entry_limits=density.weighted_entry_bounds;family.j_error=o.j_error;
    family.energy_error=o.energy_error;family.maximum_probes=o.maximum_probes;family.probe_batch_size=o.probe_batch_size;
    family.metric_rank_perturbation=o.metric_rank_perturbation;
    family.metric_action_j_error=o.metric_action_j_error;family.metric_action_energy_error=o.metric_action_energy_error;return family;
}
void PeriodicRIFittingBoundSource::certify_metric_action(const ResourceBudget& budget) {
    if(!ri_->plan().explicit_coupling_action)
        receipt_.metric_action_coupling="stored-residual-and-compressed-moment-DAG;norm-bound";
    auto action=ri_->inverse_action(budget);if(!action->certificate().qualified)return;
    const auto& source=ri_->coupling_source();const Count n=plan_.n_basis,n2=n*n,nk=plan_.momenta,m=ri_->plan().n_auxiliary,mesh=nk*n2;
    auto entries=options_.density_entry_limits;
    const auto& weights=reference_->pair_source().weights();
    if(!entries.empty())for(Count k=0;k<nk;++k)for(Count i=0;i<n;++i)for(Count j=i;j<n;++j) {
        const Count a=k*n2+i*n+j,b=k*n2+j*n+i;
        entries[a]=entries[b]=weights[k]>0.?std::min(entries[a],entries[b]):0.;
    }
    if(!ri_->plan().explicit_coupling_action) {
        const auto coupling=source.coupling_norm_bounds(budget);
        if(!coupling.metric_norm_qualified)throw std::logic_error("MANTA producer lost compressed metric norm qualification");
        const auto bound=action->norm_family(coupling.metric_column_norm_bounds,options_.density_l1_limit,entries,budget);
        receipt_.metric_action_columns=coupling.metric_column_norm_bounds.size();
        for(double norm:coupling.metric_column_norm_bounds)
            receipt_.metric_action_adjoint_norm_bound=std::max(receipt_.metric_action_adjoint_norm_bound,norm);
        receipt_.metric_action_shell_calls=coupling.receipt.shell_calls;
        receipt_.metric_action_primitive_products=coupling.receipt.primitive_products;
        receipt_.metric_action_qualified=bound.qualified;receipt_.metric_action_j_bound=bound.j_error_bound;
        receipt_.metric_action_energy_bound=bound.energy_error_bound;metric_action_j_bounds_=bound.j_error_bounds;return;
    }
    std::vector<HermitianInverseColumnBound> columns(mesh);
    // Stream one AO shell-pair panel across auxiliary shells. This uses the
    // same producer evaluate API and actual stored entries as the fitted map.
    for(Count im=0;im<source.orbital().nshells();++im)for(Count in=0;in<source.orbital().nshells();++in) {
        Eigen::MatrixXcd panel;Count nm=0,nn=0,m0=0,n0=0;
        for(Count ip=0;ip<source.auxiliary().nshells();++ip) {
            const auto tile=source.evaluate(ip,im,in);
            receipt_.metric_action_shell_calls=admission::add(receipt_.metric_action_shell_calls,tile.receipt.shell_calls);
            receipt_.metric_action_primitive_products=admission::add(receipt_.metric_action_primitive_products,tile.receipt.primitive_products);
            const Count np=tile.shape[1],p0=tile.offsets[1];nm=tile.shape[2];nn=tile.shape[3];m0=tile.offsets[2];n0=tile.offsets[3];
            if(!ip)panel.resize(m,nk*nm*nn);
            for(Count k=0;k<nk;++k)for(Count p=0;p<np;++p)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j)
                panel(p0+p,(k*nm+i)*nn+j)=tile.values[((k*np+p)*nm+i)*nn+j];
        }
        for(Count k=0;k<nk;++k)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j) {
            columns[k*n2+(m0+i)*n+n0+j]=action->column(panel.col((k*nm+i)*nn+j),budget);++receipt_.metric_action_columns;
        }
    }
    for(const auto& column:columns)
        receipt_.metric_action_adjoint_norm_bound=std::max(receipt_.metric_action_adjoint_norm_bound,column.coupling_norm);
    const auto bound=action->family(columns,options_.density_l1_limit,entries,budget);
    receipt_.metric_action_qualified=bound.qualified;receipt_.metric_action_j_bound=bound.j_error_bound;
    receipt_.metric_action_energy_bound=bound.energy_error_bound;metric_action_j_bounds_=bound.j_error_bounds;
}
} // namespace vibeqc::manta
