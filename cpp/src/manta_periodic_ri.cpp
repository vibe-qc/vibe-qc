#include "vibeqc/manta_periodic_ri.hpp"
#include "vibeqc/manta_ri_metric.hpp"
#include "vibeqc/manta_periodic_error.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <Eigen/Eigenvalues>
#include <cstring>
#include <limits>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) { std::array<std::uint8_t,8> b{};for(int i=0;i<8;++i)b[i]=v>>(56-8*i);hash.update(b.data(),8); }
    void text(const std::string& s) { word(s.size());hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double v) { if(v==0)v=0.;Count bits;std::memcpy(&bits,&v,8);word(bits); }
    void value(Complex z) { real(z.real());real(z.imag()); }
    void matrix(const Eigen::MatrixXcd& m) { word(m.rows());word(m.cols());for(Eigen::Index i=0;i<m.rows();++i)for(Eigen::Index j=0;j<m.cols();++j)value(m(i,j)); }
};
std::string matrix_id(const Eigen::MatrixXcd& m) { Digest d;d.text("manta.periodic-ri-matrix.v1");d.matrix(m);return d.hash.finish_hex(); }
std::string frame_id(const PeriodicGDFTiles& s,bool target) {
    Digest d;d.text("manta.periodic-ri-frame.v1");d.text(s.source().short_range.orbital_basis_identity);
    for(int i=0;i<3;++i)for(int j=0;j<3;++j)d.real(s.lattice()(i,j));
    d.word(s.momenta().rows());
    for(Eigen::Index k=0;k<s.momenta().rows();++k)for(int i=0;i<3;++i)d.real(s.momenta()(k,i)-(target?s.transfer()[i]:0.));
    return d.hash.finish_hex();
}
const PeriodicGDFTiles& require_source(const std::shared_ptr<const PeriodicGDFTiles>& s) {
    if(!s) throw std::invalid_argument("MANTA periodic RI requires an immutable GDF tile source");return *s;
}
void add_compensated(Complex v,Complex& sum,Complex& correction) {
    if(!std::isfinite(v.real()) || !std::isfinite(v.imag())) throw std::overflow_error("MANTA periodic RI contribution is nonfinite");
    const auto y=v-correction,t=sum+y;correction=(t-sum)-y;sum=t;
    if(!std::isfinite(sum.real()) || !std::isfinite(sum.imag())) throw std::overflow_error("MANTA periodic RI sum is nonfinite");
}
Count build_workspace(Count m,Count mesh,Count spins,Count block) {
    using namespace admission;
    // Component envelopes retained across both actions, factor propagation
    // vectors and the producer's additional owned/temporary bounds.
    return add(mul(128,add(m,mesh)),
        mul(16,add(add(mul(8,square(m)),mul(64,m)),mul(mesh,add(add(mul(12,spins),8),mul(4,block))))));
}
double hermiticity(const Complex* values,Count n) {
    double scale=1.,error=0.;
    for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
        const auto v=values[i*n+j];
        if(!std::isfinite(v.real()) || !std::isfinite(v.imag())) throw std::invalid_argument("MANTA periodic RI matrix must be finite");
        scale=std::max(scale,std::abs(v));error=std::max(error,std::abs(v-std::conj(values[j*n+i])));
    }
    if(!std::isfinite(scale) || !std::isfinite(error))
        throw std::overflow_error("MANTA periodic RI Hermiticity norm is not representable");
    return error/scale;
}
}

PeriodicRIQPlan plan_periodic_ri_q(const PeriodicGDFTiles& source,Count spins,const RIOptions& o,const ResourceBudget& budget,const PeriodicRIAccuracyOptions& accuracy) {
    if(source.source().metric_source_identity.empty())
        throw std::invalid_argument("MANTA periodic RI requires an admitted metric from the same source");
    return plan_periodic_ri_q(source.plan(),spins,o,budget,accuracy);
}
PeriodicRIQPlan plan_periodic_ri_q(const PeriodicGDFTilesPlan& s,Count spins,const RIOptions& o,const ResourceBudget& budget,const PeriodicRIAccuracyOptions& accuracy) {
    using namespace admission;
    margin(budget);ri_detail::validate_options(o);
    for(double value:{accuracy.j_error,accuracy.energy_error,accuracy.density_l1_limit})
        if(!std::isfinite(value) || value<0.) throw std::invalid_argument("MANTA RI accuracy controls must be finite nonnegative");
    if(accuracy.metric_reference_cutoff && (!std::isfinite(*accuracy.metric_reference_cutoff) || *accuracy.metric_reference_cutoff<0.))
        throw std::invalid_argument("MANTA RI metric reference cutoff must be finite nonnegative");
    if(!accuracy.maximum_attempts || accuracy.maximum_attempts>32)
        throw std::invalid_argument("MANTA RI accuracy maximum_attempts must be from one through 32");
    if(accuracy.enabled() && (accuracy.j_error<=0. || accuracy.energy_error<=0. || accuracy.density_l1_limit<=0. || !s.pair_tree_bytes))
        throw std::invalid_argument("MANTA RI accuracy requires positive J/energy targets, a density L1 limit and an explicit pair-tree multipole source");
    if(spins!=1 && spins!=2) throw std::invalid_argument("MANTA periodic RI requires one or two spin channels");
    if(!s.compute_metric || !s.n_aux || !s.n_orb || !s.momenta)
        throw std::invalid_argument("MANTA periodic RI requires an admitted metric from the same source");
    PeriodicRIQPlan p;p.n_basis=s.n_orb;p.n_auxiliary=s.n_aux;p.momenta=s.momenta;p.spins=spins;
    const Count n=p.n_basis,m=p.n_auxiliary,nk=p.momenta,n2=square(n),m2=square(m);
    if(!accuracy.density_entry_limits.empty() && (!accuracy.enabled() || accuracy.density_entry_limits.size()!=mul(nk,n2)))
        throw std::invalid_argument("MANTA RI density entry-limit dimensions or inactive accuracy");
    for(double value:accuracy.density_entry_limits)if(!std::isfinite(value) || value<0.)
        throw std::invalid_argument("MANTA RI density entry limits must be finite nonnegative");
    p.maximum_coupling_tile_elements=s.maximum_tile_elements;
    p.explicit_coupling_action=!(s.compressed_action_bytes || s.global_action_bytes || s.pair_tree_bytes);
    p.maximum_rank=m;p.mode_block=std::min(o.mode_block,m);p.maximum_coupling_passes=add(m,p.mode_block-1)/p.mode_block;
    if(s.coupling_error_bound>0.) p.maximum_coupling_passes=std::max<Count>(2,p.maximum_coupling_passes);
    const Count metric_bytes=add(mul(512,m),mul(16,add(mul(24,m2),mul(64,m))));
    const Count build_bytes=build_workspace(m,mul(nk,n2),spins,p.mode_block);
    // The source's caller-owned reservation belongs to its construction,
    // not this consumer. Retain its immutable owners/workspaces once and
    // charge this call's external live storage separately.
    const Count selection_memory=accuracy.enabled()?add(mul(2,s.modeled_peak_bytes-s.external_live_bytes),mul(64,mul(nk,n2))):0;
    const Count construction=add(s.metric_peak_bytes-s.external_live_bytes,metric_bytes);
    const Count build=add(s.coupling_peak_bytes-s.external_live_bytes,build_bytes);
    p.modeled_peak_bytes=add(selection_memory,add(std::max(construction,build),add(budget.other_live_bytes,budget.backend_margin_bytes)));
    // Retained discarded-mode factor, its owned export and construction
    // scratch. No full auxiliary/AO-pair coupling tensor is introduced.
    if(accuracy.metric_reference_cutoff) p.modeled_peak_bytes=add(p.modeled_peak_bytes,mul(48,m2));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic RI metric/block memory budget exceeded");
    p.shell_calls=add(s.metric_shell_calls,mul(p.maximum_coupling_passes,s.coupling_shell_calls));
    p.primitive_products=add(s.metric_primitive_products,mul(p.maximum_coupling_passes,s.coupling_primitive_products));
    const Count cubic=mul(m,m2),whiten=mul(m2,mul(nk,n2)),exchange=mul(spins,mul(m,mul(nk,mul(n,n2))));
    const Count scalar=add(mul(16,m2),mul(128,add(cubic,add(whiten,add(exchange,mul(m,mul(nk,n2)))))));
    p.scalar_terms=add(s.metric_scalar_terms,add(mul(p.maximum_coupling_passes,s.coupling_scalar_terms),scalar));
    p.kernel_terms=add(s.metric_kernel_terms,add(mul(p.maximum_coupling_passes,s.coupling_kernel_terms),
        add(mul(8,scalar),mul(4096,cubic))));
    // Outward-rounded residual/orthogonality audit of the existing eigensystem.
    const Count spectral=add(mul(1024,cubic),mul(512,m2));
    p.scalar_terms=add(p.scalar_terms,spectral);p.kernel_terms=add(p.kernel_terms,mul(8,spectral));
    if(accuracy.enabled()) {
        p.accuracy_selection_passes=mul(2,accuracy.maximum_attempts);
        p.shell_calls=add(p.shell_calls,mul(p.accuracy_selection_passes,s.coupling_shell_calls));
        p.primitive_products=add(p.primitive_products,mul(p.accuracy_selection_passes,s.coupling_primitive_products));
        const Count extra=add(mul(256,m2),mul(128,add(mul(m,mul(nk,n2)),add(m,mul(nk,n2)))));
        p.scalar_terms=add(p.scalar_terms,mul(accuracy.maximum_attempts,add(mul(2,s.coupling_scalar_terms),extra)));
        p.kernel_terms=add(p.kernel_terms,mul(accuracy.maximum_attempts,add(mul(2,s.coupling_kernel_terms),mul(8,extra))));
    }
    if(accuracy.metric_reference_cutoff) {
        const Count extra=mul(256,mul(m2,add(accuracy.maximum_attempts,1)));
        p.scalar_terms=add(p.scalar_terms,extra);p.kernel_terms=add(p.kernel_terms,mul(8,extra));
    }
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic RI shell-call budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA periodic RI primitive-product budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic RI scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic RI kernel-work budget exceeded");
    return p;
}

PeriodicRIQBuilder::PeriodicRIQBuilder(std::shared_ptr<const PeriodicGDFTiles> source,Count spins,
    const RIOptions& options,const ResourceBudget& budget,const PeriodicRIAccuracyOptions& accuracy)
    : plan_(plan_periodic_ri_q(require_source(source),spins,options,budget,accuracy)),source_(std::move(source)),options_(options) {
    const Count mesh=admission::mul(plan_.momenta,admission::square(plan_.n_basis));
    coupling_budget_=budget;
    // The original RI workspace already covers input/output/correction and
    // metric buffers. Reserve the remainder around each producer action.
    coupling_budget_.other_live_bytes=admission::add(budget.other_live_bytes,
        build_workspace(plan_.n_auxiliary,mesh,spins,plan_.mode_block)-admission::mul(192,admission::add(mesh,plan_.n_auxiliary)));
    const auto raw=source_->metric();const Count m=plan_.n_auxiliary;
    info_.metric_source_identity=source_->source().metric_source_identity;
    info_.zero_transfer=source_->source().zero_transfer;
    info_.shell_calls=raw.receipt.shell_calls;info_.primitive_products=raw.receipt.primitive_products;
    info_.hermiticity_residual=hermiticity(raw.values.data(),m);
    if(info_.hermiticity_residual>1e-12) throw std::runtime_error("MANTA periodic RI metric is not Hermitian");
    metric_.resize(m,m);
    for(Count i=0;i<m;++i) {
        metric_(i,i)=raw.values[i*m+i].real();
        for(Count j=0;j<i;++j) { metric_(i,j)=raw.values[i*m+j];metric_(j,i)=std::conj(metric_(i,j)); }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> eigen(metric_,Eigen::ComputeEigenvectors);
    if(eigen.info()!=Eigen::Success || !eigen.eigenvalues().allFinite() || !eigen.eigenvectors().allFinite())
        throw std::runtime_error("MANTA periodic RI metric eigensolve failed");
    info_.minimum_eigenvalue=eigen.eigenvalues()[0];info_.maximum_eigenvalue=eigen.eigenvalues()[m-1];
    const double positive=std::max(0.,info_.maximum_eigenvalue);
    info_.cutoff=std::max(options.absolute_cutoff,options.relative_cutoff*positive);
    info_.negative_limit=std::max(options.absolute_cutoff,options.negative_relative_tolerance*positive);
    if(!std::isfinite(info_.cutoff) || !std::isfinite(info_.negative_limit)) throw std::overflow_error("MANTA periodic RI metric cutoff overflow");
    if(info_.minimum_eigenvalue < -info_.negative_limit)
        throw std::runtime_error("MANTA periodic RI metric has a significant negative eigenvalue; converge the finite source");
    for(Count i=0;i<m;++i) if(eigen.eigenvalues()[i]>info_.cutoff) ++info_.rank;
    factor_.resize(m,info_.rank);Count column=0;
    for(Count i=0;i<m;++i) if(eigen.eigenvalues()[i]>info_.cutoff)
        factor_.col(column++)=eigen.eigenvectors().col(i)/std::sqrt(eigen.eigenvalues()[i]);
    projector_=Eigen::MatrixXcd::Zero(m,m);
    if(info_.rank) projector_=eigen.eigenvectors().rightCols(info_.rank)*eigen.eigenvectors().rightCols(info_.rank).adjoint();
    if(!factor_.allFinite() || !projector_.allFinite()) throw std::overflow_error("MANTA periodic RI metric factor is nonfinite");
    info_.inverse_certificate=certify_hermitian_inverse(metric_,eigen.eigenvalues(),eigen.eigenvectors(),factor_,options,budget);
    info_.spectrum=info_.inverse_certificate.spectrum;
    info_.canonical_metric_identity=matrix_id(metric_);info_.factor_identity=matrix_id(factor_);info_.projector_identity=matrix_id(projector_);
    if(accuracy.metric_reference_cutoff) {
        auto& a=info_.accuracy;a.metric_reference_cutoff=*accuracy.metric_reference_cutoff;
        if(a.metric_reference_cutoff>info_.cutoff)
            throw std::invalid_argument("MANTA RI metric reference cutoff exceeds the selected cutoff");
        for(Count i=0;i<m;++i)if(eigen.eigenvalues()[i]>a.metric_reference_cutoff && eigen.eigenvalues()[i]<=info_.cutoff)
            ++a.metric_discarded_rank;
        a.metric_reference_rank=admission::add(info_.rank,a.metric_discarded_rank);
        metric_discarded_factor_.resize(m,a.metric_discarded_rank);Count mode=0;
        for(Count i=0;i<m;++i)if(eigen.eigenvalues()[i]>a.metric_reference_cutoff && eigen.eigenvalues()[i]<=info_.cutoff)
            metric_discarded_factor_.col(mode++)=eigen.eigenvectors().col(i)/std::sqrt(eigen.eigenvalues()[i]);
        if(!metric_discarded_factor_.allFinite())throw std::overflow_error("MANTA RI discarded metric factor is nonfinite");
        a.metric_discarded_factor_identity=matrix_id(metric_discarded_factor_);
        Digest id;id.text("manta.periodic-ri-nested-positive-factor.v1;stored-eigensystem;no-roundoff-proof");
        id.text(info_.canonical_metric_identity);id.text(info_.factor_identity);id.text(a.metric_discarded_factor_identity);
        id.real(a.metric_reference_cutoff);id.real(info_.cutoff);a.metric_reference_identity=id.hash.finish_hex();
        a.metric_truncation_certified=true;
    }
    if(accuracy.enabled()) select_accuracy(accuracy);
    Digest d;d.text("manta.periodic-ri-q.v1;fixed-positive-eigenprojector;lower-triangle");
    d.text(source_->source().source_identity);d.text(info_.canonical_metric_identity);d.text(info_.factor_identity);d.text(info_.projector_identity);
    d.real(options.absolute_cutoff);d.real(options.relative_cutoff);d.real(options.negative_relative_tolerance);
    info_.source_identity=d.hash.finish_hex();source_frame_=frame_id(*source_,false);target_frame_=frame_id(*source_,true);
}

std::shared_ptr<PeriodicRIQBuilder> PeriodicRIQBuilder::refine_metric(double cutoff,const ResourceBudget& budget) const {
    if(info_.accuracy.enabled || !std::isfinite(cutoff) || cutoff<0. || cutoff>=info_.cutoff)
        throw std::invalid_argument("MANTA metric refinement requires a smaller cutoff and no separate component accuracy policy");
    auto options=options_;options.absolute_cutoff=cutoff;options.relative_cutoff=0.;
    return std::make_shared<PeriodicRIQBuilder>(source_,plan_.spins,options,budget);
}

void PeriodicRIQBuilder::select_accuracy(const PeriodicRIAccuracyOptions& o) {
    using namespace admission;
    using periodic_error::upper;
    using periodic_error::positive_add;
    if(!source_->source().zero_transfer || source_->source().kernel!="coulomb")
        throw std::invalid_argument("MANTA RI automatic accuracy requires a Gamma Coulomb J source");
    auto& r=info_.accuracy;r.enabled=true;r.j_target=o.j_error;r.energy_target=o.energy_error;r.density_l1_limit=o.density_l1_limit;
    r.reference_coupling_identity=source_->source().reference_three_center_source_identity;
    const Count m=plan_.n_auxiliary,rank=info_.rank;
    density_entry_limits_=o.density_entry_limits;r.component_density_envelope=!density_entry_limits_.empty();
    std::vector<Complex> envelope(mul(plan_.momenta,square(plan_.n_basis)));
    if(r.component_density_envelope) {
        Digest d;d.text("manta.periodic-ri-weighted-entry-envelope.v1");d.word(density_entry_limits_.size());
        for(Count i=0;i<density_entry_limits_.size();++i) {envelope[i]=density_entry_limits_[i];d.real(density_entry_limits_[i]);}
        r.density_envelope_identity=d.hash.finish_hex();
    }
    // Triangle certificate for EVERY weighted total density with ||X||_1<=L.
    // C bounds exact columns; eps bounds every omitted coupling entry.
    // |b~| <= (C+eps)L, |delta b| <= eps L. Propagate through actual F,
    // then |delta J| <= eps ||c~||_1 + C.|delta c|.
    // The same paired Hermitian Gamma source gives |delta E|<=L|delta J|/2.
    for(Count attempt=0;attempt<o.maximum_attempts;++attempt) {
        const auto probe=source_->forward(envelope,coupling_budget_);++r.attempts;
        r.shell_calls=add(r.shell_calls,probe.receipt.shell_calls);
        r.primitive_products=add(r.primitive_products,probe.receipt.primitive_products);
        const long double eps=probe.receipt.coupling_error_bound,L=o.density_l1_limit;
        std::vector<double> rho(rank),delta(rank),reference(m),omission(m);
        if(r.component_density_envelope) for(Count i=0;i<m;++i) {
            reference[i]=std::min(upper(static_cast<long double>(probe.reference_magnitude_bounds[i])*L),probe.reference_action_bounds[i]);
            omission[i]=std::min(upper(eps*L),probe.approximation_bounds[i]);
        }
        for(Count mode=0;mode<rank;++mode) {
            long double value=0.,error=0.;
            for(Count i=0;i<m;++i) {
                const long double f=upper(std::hypot(static_cast<long double>(factor_(i,mode).real()),static_cast<long double>(factor_(i,mode).imag())));
                if(r.component_density_envelope) {
                    positive_add(f*(static_cast<long double>(reference[i])+omission[i]),value);
                    positive_add(f*omission[i],error);
                } else {
                    positive_add(f*(probe.reference_magnitude_bounds[i]+eps)*L,value);
                    positive_add(f*eps*L,error);
                }
            }
            rho[mode]=upper(value);delta[mode]=upper(error);
        }
        long double bound=0.,adjoint_uniform=0.,propagated=0.;
        std::vector<Complex> coefficient_limits(m);
        for(Count i=0;i<m;++i) {
            long double coefficient=0.,error=0.;
            for(Count mode=0;mode<rank;++mode) {
                const long double f=upper(std::hypot(static_cast<long double>(factor_(i,mode).real()),static_cast<long double>(factor_(i,mode).imag())));
                positive_add(f*rho[mode],coefficient);positive_add(f*delta[mode],error);
            }
            positive_add(eps*upper(coefficient),bound);
            positive_add(static_cast<long double>(probe.reference_magnitude_bounds[i])*upper(error),bound);
            if(r.component_density_envelope) {
                coefficient_limits[i]=upper(coefficient);
                positive_add(eps*upper(coefficient),adjoint_uniform);
                positive_add(static_cast<long double>(probe.reference_magnitude_bounds[i])*upper(error),propagated);
            }
        }
        r.j_bound=upper(bound);r.energy_bound=upper(.5L*L*r.j_bound);
        if(r.component_density_envelope) {
            // Positive source actions bound EVERY input dominated by the
            // fixed envelope. The adjoint omission uses the same partition;
            // no SCF density enters selection. Keep the total-L1 certificate.
            double adjoint_bound=upper(adjoint_uniform);
            if(adjoint_bound>0.) {
                const auto backward=source_->adjoint(coefficient_limits,coupling_budget_);
                r.shell_calls=add(r.shell_calls,backward.receipt.shell_calls);
                r.primitive_products=add(r.primitive_products,backward.receipt.primitive_products);
                adjoint_bound=std::min(adjoint_bound,backward.receipt.approximation_bound);
            }
            r.j_bound=std::min(r.j_bound,upper(static_cast<long double>(adjoint_bound)+propagated));
            long double energy=0.;
            for(Count mode=0;mode<rank;++mode)
                positive_add(static_cast<long double>(rho[mode])*delta[mode]+.5L*delta[mode]*delta[mode],energy);
            r.energy_bound=std::min(upper(.5L*L*r.j_bound),upper(energy));
        }
        r.coupling_j_bound=r.j_bound;r.coupling_energy_bound=r.energy_bound;
        if(r.metric_truncation_certified) {
            // Sun et al., DOI 10.1063/1.4998644, II.B Eqs.20-23: positive
            // metric modes define the finite fitting space. Neese (2025),
            // DOI 10.1021/acs.jpca.4c07415, II.1 Eqs.9-11: paired RI-J map.
            // This derived bound compares nested STORED factors, never an
            // ideal inverse or unfitted Coulomb operator. For removed H,
            // Delta W=conj(H) H^T. C bounds exact coupling columns, R bounds
            // |T_exact^H X| for the admitted weighted total-density family.
            // z=|H|^T R, g=|H|z; |Delta J|<=C.g and Delta E<=z.z/2.
            long double j=0.,energy=0.;
            for(Count mode=0;mode<r.metric_discarded_rank;++mode) {
                long double projected=0.,column=0.;
                for(Count i=0;i<m;++i) {
                    const auto value=metric_discarded_factor_(i,mode);
                    const long double h=upper(std::hypot(static_cast<long double>(value.real()),static_cast<long double>(value.imag())));
                    const double rhs_bound=r.component_density_envelope?reference[i]:upper(static_cast<long double>(probe.reference_magnitude_bounds[i])*L);
                    positive_add(h*rhs_bound,projected);
                    positive_add(h*probe.reference_magnitude_bounds[i],column);
                }
                const double z=upper(projected),c=upper(column);
                positive_add(static_cast<long double>(c)*z,j);
                positive_add(.5L*z*z,energy);
            }
            r.metric_j_bound=upper(j);
            r.metric_energy_bound=std::min(upper(energy),upper(.5L*L*r.metric_j_bound));
            r.j_bound=upper(static_cast<long double>(r.coupling_j_bound)+r.metric_j_bound);
            r.energy_bound=upper(static_cast<long double>(r.coupling_energy_bound)+r.metric_energy_bound);
        }
        r.selected_coupling_error=source_->source().short_range.coupling_multipole_error;
        r.selected_coupling_identity=source_->source().three_center_source_identity;
        if(r.j_bound<=o.j_error && r.energy_bound<=o.energy_error) {r.qualified=true;break;}
        if(attempt+1==o.maximum_attempts) break;
        // The partition changes discretely and its evaluated error can be
        // much smaller than the requested tolerance. Probe above the linear
        // target-ratio estimate to avoid needlessly selecting an exact map;
        // still halve at each failed trial to guarantee progress. Neither
        // linear scaling nor monotone certificates are assumed for acceptance.
        // ONLY the evaluated producer bounds accept a trial, before SCF.
        const double ratio=std::min(o.j_error/r.j_bound,o.energy_error/r.energy_bound);
        const double next=std::max(std::numeric_limits<double>::denorm_min(),
            r.selected_coupling_error*std::min(.5,2.*ratio));
        if(next>=r.selected_coupling_error) break; // No smaller positive double.
        auto refined=source_->refine_multipole(next,coupling_budget_);
        if(refined->source().metric_source_identity!=info_.metric_source_identity ||
           refined->source().reference_three_center_source_identity!=r.reference_coupling_identity)
            throw std::logic_error("MANTA RI accuracy refinement changed its reference metric/source");
        source_=std::move(refined);
    }
    if(!r.qualified) throw std::runtime_error("MANTA RI accuracy targets unmet after admitted selection attempts");
    Digest d;d.text("manta.periodic-ri-density-envelope.v1");d.text(r.selected_coupling_identity);d.text(info_.factor_identity);
    d.real(r.j_target);d.real(r.energy_target);d.real(r.density_l1_limit);d.real(r.j_bound);d.real(r.energy_bound);
    d.real(r.selected_coupling_error);d.word(o.maximum_attempts);
    if(r.component_density_envelope)d.text(r.density_envelope_identity);
    if(r.metric_truncation_certified) {
        d.text(r.metric_reference_identity);d.real(r.metric_j_bound);d.real(r.metric_energy_bound);
        d.real(r.coupling_j_bound);d.real(r.coupling_energy_bound);
    }
    r.certificate_identity=d.hash.finish_hex();
}

void PeriodicRIQBuilder::validate_request(bool make_j,bool make_k) const {
    if(!make_j && !make_k) throw std::invalid_argument("MANTA periodic RI requires a Coulomb or exchange target");
    if(source_->source().coupling_error_bound>0. && (make_k || source_->source().kernel!="coulomb"))
        throw std::invalid_argument("MANTA coupling-only approximation is qualified for Coulomb J only; select an independent exchange source");
    if(info_.accuracy.enabled && (!make_j || make_k)) throw std::invalid_argument("MANTA RI automatic accuracy qualifies J only");
    if(make_j && !info_.zero_transfer) throw std::invalid_argument("MANTA periodic RI Coulomb requires a reciprocal Gamma transfer");
}
PeriodicRIQResult PeriodicRIQBuilder::build(const std::vector<Complex>& d,const std::vector<double>& weights,
    bool make_j,bool make_k) const {
    using namespace admission;
    validate_request(make_j,make_k);
    const Count n=plan_.n_basis,m=plan_.n_auxiliary,nk=plan_.momenta,ns=plan_.spins,n2=square(n),mesh=mul(nk,n2);
    if(d.size()!=mul(ns,mesh) || weights.size()!=nk) throw std::invalid_argument("MANTA periodic RI density/weight dimensions");
    for(Count spin=0;spin<ns;++spin)for(Count k=0;k<nk;++k)
        if(hermiticity(d.data()+(spin*nk+k)*n2,n)>1e-12) throw std::invalid_argument("MANTA periodic RI requires Hermitian spin densities");
    double weight_sum=0.;
    for(double w:weights) { require_finite(w);if(w<0.) throw std::invalid_argument("MANTA periodic RI source weights must be nonnegative");weight_sum+=w; }
    if(!std::isfinite(weight_sum) || weight_sum>1.+1e-12 || (make_j && std::abs(weight_sum-1.)>1e-12))
        throw std::invalid_argument("MANTA periodic RI weights must sum to one for J, or at most one for a partial K channel");
    PeriodicRIQResult result;result.n_basis=n;result.momenta=nk;result.spins=ns;auto& r=result.receipt;
    r.kernel=source_->source().kernel;
    r.physical_omega=source_->source().physical_omega;r.zero_transfer=source_->source().zero_transfer;
    r.physical_zero_included=source_->source().physical_zero_included;
    r.has_j=make_j;r.has_k=make_k;r.source_identity=info_.source_identity;r.projector_identity=info_.projector_identity;
    r.source_frame_identity=source_frame_;r.target_frame_identity=target_frame_;r.retained_rank=info_.rank;r.modeled_peak_bytes=plan_.modeled_peak_bytes;
    Digest density;density.text("manta.periodic-ri-density.v1");density.text(source_frame_);density.word(ns);density.word(nk);density.word(n);
    for(auto v:d)density.value(v);r.density_identity=density.hash.finish_hex();
    Digest weight;weight.text("manta.periodic-ri-weights.v1");weight.word(nk);for(double w:weights)weight.real(w);r.weight_identity=weight.hash.finish_hex();
    Digest snapshot;snapshot.text("manta.periodic-ri-q-snapshot.v1");snapshot.text(r.source_identity);snapshot.text(r.density_identity);
    snapshot.text(r.weight_identity);snapshot.word(make_j);snapshot.word(make_k);r.snapshot_identity=snapshot.hash.finish_hex();
    if(make_j) result.J.assign(mesh,{});if(make_k) result.K.assign(mul(ns,mesh),{});
    std::vector<Complex> j_error(result.J.size()),k_error(result.K.size()),total(mesh),rhs(m),rhs_error(m),rho(info_.rank);
    for(Count spin=0;spin<ns;++spin)for(Count i=0;i<mesh;++i)total[i]+=d[spin*mesh+i];
    if(info_.accuracy.enabled) {
        long double norm=0.;
        for(Count i=0;i<mesh;++i) periodic_error::positive_add(static_cast<long double>(weights[i/n2])*
            periodic_error::upper(std::hypot(static_cast<long double>(total[i].real()),static_cast<long double>(total[i].imag()))),norm);
        r.accuracy_density_l1_bound=periodic_error::upper(norm);
        if(r.accuracy_density_l1_bound>info_.accuracy.density_l1_limit)
            throw std::invalid_argument("MANTA RI density exceeds the fixed accuracy L1 envelope");
        if(!density_entry_limits_.empty())for(Count i=0;i<mesh;++i) {
            const double value=periodic_error::upper(static_cast<long double>(weights[i/n2])*
                periodic_error::upper(std::hypot(static_cast<long double>(total[i].real()),static_cast<long double>(total[i].imag()))));
            if(value>density_entry_limits_[i])throw std::invalid_argument("MANTA RI density exceeds the fixed accuracy entry envelope");
        }
        r.accuracy_certified=true;
    }
    const Count passes_rank=make_j?std::max<Count>(1,info_.rank):info_.rank;
    // The q=0 J map is a paired forward/adjoint over the SAME raw T.
    // Neese et al. (2025), DOI 10.1021/acs.jpca.4c07415, Eqs.29-36,
    // motivates the paired coupling contract; these periodic phases/source
    // domains remain those of Sun et al. (2017), DOI 10.1063/1.4998644.
    // Coupling-only screening keeps this fixed paired map and exact metric.
    // Without screening, use two raw sweeps only when they save work. That choice uses
    // fixed rank/block metadata, never the evolving density. Joint J/K keeps
    // its fused retained-mode sweep and shares no newly reconstructed factor.
    const Count mode_passes=add(passes_rank,plan_.mode_block-1)/plan_.mode_block;
    r.coupling_error_bound=source_->source().coupling_error_bound;
    r.reference_coupling_identity=source_->source().reference_three_center_source_identity;
    const bool screened=r.coupling_error_bound>0.;
    const bool paired=make_j && !make_k && (screened || mode_passes>2);
    if(make_j) r.coulomb_contraction=source_->source().short_range.coupling_multipole_error>0.?"paired-multipole":
        (screened?"paired-screened":(paired?"paired-raw":"retained-mode"));
    if(paired) {
        for(Count k=0;k<nk;++k)for(Count pair=0;pair<n2;++pair)total[k*n2+pair]*=weights[k];
        auto forward=source_->forward(total,coupling_budget_);
        rhs=std::move(forward.values);
        r.coupling_passes=forward.receipt.coupling_passes;
        r.shell_calls=forward.receipt.shell_calls;r.primitive_products=forward.receipt.primitive_products;
        const Eigen::Map<const Eigen::VectorXcd> raw_rhs(rhs.data(),m);
        // B=F^H T implies rho=F^T rhs and c=conj(F) rho. Do not whiten
        // F, assume F^H M F is exactly I, or alter its retained projector.
        const Eigen::VectorXcd projected=factor_.transpose()*raw_rhs;
        const Eigen::VectorXcd coefficients=factor_.conjugate()*projected;
        if(!projected.allFinite() || !coefficients.allFinite())
            throw std::overflow_error("MANTA periodic RI paired metric contraction is nonfinite");
        for(Count mode=0;mode<info_.rank;++mode)rho[mode]=projected[mode];
        r.coulomb_energy=.5*projected.squaredNorm();
        const std::vector<Complex> c(coefficients.data(),coefficients.data()+coefficients.size());
        auto adjoint=source_->adjoint(c,coupling_budget_);
        result.J=std::move(adjoint.values);
        r.coupling_multipole_partition_builds=add(forward.receipt.multipole_partition_builds,adjoint.receipt.multipole_partition_builds);
        r.coupling_multipole_partition_reuses=add(forward.receipt.multipole_partition_reuses,adjoint.receipt.multipole_partition_reuses);
        r.coupling_multipole_partition_bypasses=add(forward.receipt.multipole_partition_bypasses,adjoint.receipt.multipole_partition_bypasses);
        r.coupling_multipole_trials=add(forward.receipt.multipole_trials,adjoint.receipt.multipole_trials);
        r.coupling_multipole_images=add(forward.receipt.multipole_images,adjoint.receipt.multipole_images);
        r.coupling_multipole_tree_nodes=add(forward.receipt.multipole_tree_nodes,adjoint.receipt.multipole_tree_nodes);
        r.coupling_multipole_far_blocks=add(forward.receipt.multipole_far_blocks,adjoint.receipt.multipole_far_blocks);
        r.coupling_multipole_translations=add(forward.receipt.multipole_translations,adjoint.receipt.multipole_translations);
        r.coupling_multipole_moment_panels=add(forward.receipt.multipole_moment_panels,adjoint.receipt.multipole_moment_panels);
        r.coupling_multipole_envelope_evaluations=add(forward.receipt.multipole_envelope_evaluations,adjoint.receipt.multipole_envelope_evaluations);
        r.coupling_multipole_reused_moment_panels=add(forward.receipt.multipole_reused_moment_panels,adjoint.receipt.multipole_reused_moment_panels);
        r.coupling_multipole_reused_extents=add(forward.receipt.multipole_reused_extents,adjoint.receipt.multipole_reused_extents);
        r.coupling_multipole_compressed_pairs=add(forward.receipt.multipole_compressed_pairs,adjoint.receipt.multipole_compressed_pairs);
        r.coupling_multipole_compressed_nodes=add(forward.receipt.multipole_compressed_nodes,adjoint.receipt.multipole_compressed_nodes);
        r.coupling_multipole_global_pairs=add(forward.receipt.multipole_global_pairs,adjoint.receipt.multipole_global_pairs);
        r.coupling_multipole_global_pair_reuses=add(forward.receipt.multipole_global_pair_reuses,adjoint.receipt.multipole_global_pair_reuses);
        r.coupling_multipole_global_auxiliary=add(forward.receipt.multipole_global_auxiliary,adjoint.receipt.multipole_global_auxiliary);
        r.coupling_multipole_global_auxiliary_projections=add(forward.receipt.multipole_global_auxiliary_projections,adjoint.receipt.multipole_global_auxiliary_projections);
        r.coupling_multipole_pair_groups=add(forward.receipt.multipole_pair_groups,adjoint.receipt.multipole_pair_groups);
        r.coupling_multipole_global_projections=add(forward.receipt.multipole_global_projections,adjoint.receipt.multipole_global_projections);
        r.coupling_multipole_action_products=add(forward.receipt.multipole_action_products,adjoint.receipt.multipole_action_products);
        r.coupling_multipole_action_translations=add(forward.receipt.multipole_action_translations,adjoint.receipt.multipole_action_translations);
        r.coupling_passes=add(r.coupling_passes,adjoint.receipt.coupling_passes);
        r.shell_calls=add(r.shell_calls,adjoint.receipt.shell_calls);
        r.primitive_products=add(r.primitive_products,adjoint.receipt.primitive_products);
        if(screened) {
            r.coupling_error_bound=std::max(forward.receipt.coupling_error_bound,adjoint.receipt.coupling_error_bound);
            // Same finite F as the unscreened reference. W=conj(F) F^T;
            // ||W||_2 <= ||F||_F^2, including nonideal finite F^H M F.
            // With e=||delta b||_2, |delta E| <= ||c~|| e + .5 ||W|| e^2.
            // |delta J|_max <= adjoint-error(c~) + sqrt(m) (T~max+eps) ||W|| e.
            // Paired tree actions propagate per-shell-block envelopes in both
            // directions; other producers retain the uniform epsilon bound.
            // RI-basis/domain, metric solve and arithmetic errors are separate.
            using periodic_error::upper;
            const long double fnorm=factor_.stableNorm(), cnorm=coefficients.stableNorm();
            r.metric_inverse_norm_bound=upper(fnorm*fnorm);
            r.forward_error_l2_bound=forward.receipt.approximation_l2_bound;
            const long double e=r.forward_error_l2_bound, w=r.metric_inverse_norm_bound;
            r.approximation_energy_bound=upper(cnorm*e+.5L*w*e*e);
            const long double exact_max=static_cast<long double>(forward.receipt.maximum_coupling)+r.coupling_error_bound;
            r.approximation_j_bound=upper(static_cast<long double>(adjoint.receipt.approximation_bound)
                +std::sqrt(static_cast<long double>(m))*exact_max*w*e);
            // Componentwise triangle propagation through the ACTUAL retained
            // F, with no ideal-metric assumption or source re-partitioning:
            // u=|F|^T e, g=|F|u bound delta rho and delta c; C_P bounds
            // max_pair |T_exact(P,pair)| from the shared coupling producer.
            // |delta E| <= min(|c~|.e, |rho~|.u) + .5 u.u;
            // |delta J|_max <= adjoint-error(c~) + C.g.
            // Keep the independently valid scalar bounds when tighter.
            using periodic_error::positive_add;
            auto magnitude=[](Complex z) {
                return upper(std::hypot(static_cast<long double>(z.real()),static_cast<long double>(z.imag())));
            };
            std::vector<double> projected_bounds(info_.rank);
            long double projected_linear=0.,coefficient_linear=0.,quadratic=0.,propagated=0.;
            for(Count mode=0;mode<info_.rank;++mode) {
                long double bound=0.;
                for(Count i=0;i<m;++i) positive_add(static_cast<long double>(magnitude(factor_(i,mode)))*forward.approximation_bounds[i],bound);
                projected_bounds[mode]=upper(bound);
                positive_add(static_cast<long double>(magnitude(projected[mode]))*projected_bounds[mode],projected_linear);
                positive_add(static_cast<long double>(projected_bounds[mode])*projected_bounds[mode],quadratic);
            }
            for(Count i=0;i<m;++i) {
                long double bound=0.;
                for(Count mode=0;mode<info_.rank;++mode)
                    positive_add(static_cast<long double>(magnitude(factor_(i,mode)))*projected_bounds[mode],bound);
                positive_add(static_cast<long double>(forward.reference_magnitude_bounds[i])*upper(bound),propagated);
                positive_add(static_cast<long double>(magnitude(coefficients[i]))*forward.approximation_bounds[i],coefficient_linear);
            }
            r.approximation_energy_bound=std::min(r.approximation_energy_bound,
                upper(std::min(projected_linear,coefficient_linear)+.5L*quadratic));
            r.approximation_j_bound=std::min(r.approximation_j_bound,
                upper(static_cast<long double>(adjoint.receipt.approximation_bound)+propagated));
        }
        if(info_.accuracy.enabled) {
            r.approximation_j_bound=std::min(r.approximation_j_bound,info_.accuracy.coupling_j_bound);
            r.approximation_energy_bound=std::min(r.approximation_energy_bound,info_.accuracy.coupling_energy_bound);
        }
    } else for(Count first=0;first<passes_rank;first+=plan_.mode_block) {
        const Count modes=std::min(plan_.mode_block,info_.rank-first);
        std::vector<Complex> fitted(mul(modes,mesh)),fitted_error(fitted.size());
        ++r.coupling_passes;
        for(Count ip=0;ip<source_->auxiliary().nshells();++ip)for(Count im=0;im<source_->orbital().nshells();++im)
            for(Count in=0;in<source_->orbital().nshells();++in) {
                const auto tile=source_->evaluate(ip,im,in);
                r.shell_calls=admission::add(r.shell_calls,tile.receipt.shell_calls);
                r.primitive_products=admission::add(r.primitive_products,tile.receipt.primitive_products);
                const Count np=tile.shape[1],nm=tile.shape[2],nn=tile.shape[3],p0=tile.offsets[1],m0=tile.offsets[2],n0=tile.offsets[3];
                for(Count k=0;k<nk;++k)for(Count p=0;p<np;++p)for(Count i=0;i<nm;++i)for(Count j=0;j<nn;++j) {
                    const Count pair=(m0+i)*n+n0+j;const auto value=tile.values[((k*np+p)*nm+i)*nn+j];
                    if(make_j && first==0) add_compensated(weights[k]*std::conj(value)*total[k*n2+pair],rhs[p0+p],rhs_error[p0+p]);
                    for(Count mode=0;mode<modes;++mode) {
                        const Count out=(k*modes+mode)*n2+pair;
                        add_compensated(std::conj(factor_(p0+p,first+mode))*value,fitted[out],fitted_error[out]);
                    }
                }
            }
        for(Count mode=0;mode<modes;++mode) {
            if(make_j) {
                Complex value{},error{};
                for(Count k=0;k<nk;++k)for(Count pair=0;pair<n2;++pair)
                    add_compensated(weights[k]*std::conj(fitted[(k*modes+mode)*n2+pair])*total[k*n2+pair],value,error);
                rho[first+mode]=value;r.coulomb_energy+=.5*std::norm(value);
                for(Count k=0;k<nk;++k)for(Count pair=0;pair<n2;++pair)
                    add_compensated(fitted[(k*modes+mode)*n2+pair]*value,result.J[k*n2+pair],j_error[k*n2+pair]);
            }
            if(make_k) for(Count k=0;k<nk;++k)for(Count spin=0;spin<ns;++spin) {
                const Eigen::Map<const md::ComplexMatrix> B(fitted.data()+(k*modes+mode)*n2,n,n);
                const Eigen::Map<const md::ComplexMatrix> D(d.data()+(spin*nk+k)*n2,n,n);
                const md::ComplexMatrix contribution=(B*D)*B.adjoint();
                for(Count pair=0;pair<n2;++pair) {
                    const Count out=(spin*nk+k)*n2+pair;
                    add_compensated(weights[k]*contribution.data()[pair],result.K[out],k_error[out]);
                }
            }
        }
    }
    if(make_j) {
        const Eigen::Map<const Eigen::VectorXcd> raw_rhs(rhs.data(),m),projected(rho.data(),info_.rank);
        const Eigen::VectorXcd solution=factor_.conjugate()*projected;
        r.metric_rhs_norm=raw_rhs.norm();r.metric_residual_norm=(metric_.conjugate()*solution-raw_rhs).norm();r.metric_residual_evaluated=true;
        require_finite(r.coulomb_energy);require_finite(r.metric_rhs_norm);require_finite(r.metric_residual_norm);
        for(Count k=0;k<nk;++k)r.hermiticity_residual=std::max(r.hermiticity_residual,hermiticity(result.J.data()+k*n2,n));
        if(r.hermiticity_residual>1e-10) throw std::runtime_error("MANTA periodic RI Coulomb source is not Hermitian; qualify the reciprocal domain");
    }
    if(make_k) for(Count spin=0;spin<ns;++spin)for(Count k=0;k<nk;++k)
        r.hermiticity_residual=std::max(r.hermiticity_residual,hermiticity(result.K.data()+(spin*nk+k)*n2,n));
    if(r.hermiticity_residual>1e-10) throw std::runtime_error("MANTA periodic RI response is not Hermitian");
    cap(r.coupling_passes,plan_.maximum_coupling_passes,"MANTA periodic RI executed pass bound exceeded");
    cap(add(info_.shell_calls,r.shell_calls),plan_.shell_calls,"MANTA periodic RI executed shell bound exceeded");
    cap(add(info_.primitive_products,r.primitive_products),plan_.primitive_products,"MANTA periodic RI executed primitive bound exceeded");
    return result;
}
std::shared_ptr<HermitianInverseAction> PeriodicRIQBuilder::inverse_action(const ResourceBudget& b) const {
    return std::shared_ptr<HermitianInverseAction>(new HermitianInverseAction(metric_,factor_,info_.inverse_certificate,options_,b));
}
} // namespace vibeqc::manta
