#pragma once
#include "vibeqc/manta_periodic_ri_fock.hpp"
#include "vibeqc/manta_admission.hpp"
#include "manta_ri_interval.hpp"

namespace vibeqc::manta::assembly_interval {
using namespace spectral_interval;
// Outward elementary binary64 arithmetic, as in the paired metric audit.
// Verified-computation reference: Rump, Acta Numerica 19 (2010), 287-449,
// DOI 10.1017/S096249291000005X. The witnesses are the evaluated Fock and method
// energy, so callback reconstruction and solver projection are included.
inline PeriodicAssemblyPlan plan(Count n,Count nk,Count ns,const ResourceBudget& b,Count exchanges=1) {
    using namespace admission;
    if(!n || !nk || (ns!=1 && ns!=2))throw std::invalid_argument("MANTA assembly requires positive AO/k extents and one or two spins");
    PeriodicAssemblyPlan p;p.n_basis=n;p.momenta=nk;p.spins=ns;
    const Count count=mul(ns,mul(nk,square(n)));
    // Persistent interval F and weighted density plus binding input copies.
    p.modeled_peak_bytes=add(add(4096,mul(256,count)),add(b.other_live_bytes,b.backend_margin_bytes));
    p.scalar_terms=mul(4096,mul(count,add(exchanges,4)));p.kernel_terms=p.scalar_terms;
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA assembly memory budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA assembly scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA assembly work budget exceeded");return p;
}
inline ComplexInterval finite_point(std::complex<double> z) {
    if(!std::isfinite(z.real()) || !std::isfinite(z.imag()))throw std::invalid_argument("MANTA assembly requires finite components and witnesses");
    return point(z);
}
// Stored-overlap projection used by both native HF and KS zero-mode seams.
// No Hermitian substitution: the actual expression is S D S, not S^dagger D S.
inline PeriodicAssemblyPlan correction_plan(Count n,Count nk,Count ns,const ResourceBudget& b) {
    using namespace admission;
    if(!n || !nk || (ns!=1 && ns!=2))throw std::invalid_argument("MANTA correction requires positive AO/k extents and one or two spins");
    PeriodicAssemblyPlan p;p.n_basis=n;p.momenta=nk;p.spins=ns;
    const Count count=mul(ns,mul(nk,square(n)));
    p.modeled_peak_bytes=add(add(4096,add(mul(128,count),mul(64,square(n)))),add(b.other_live_bytes,b.backend_margin_bytes));
    p.scalar_terms=mul(4096,mul(count,n));p.kernel_terms=p.scalar_terms;
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA correction memory budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA correction scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA correction work budget exceeded");return p;
}
class Correction {
    Count n_;
    const std::complex<double>* s_;
    ComplexInterval factor_;
    std::vector<ComplexInterval> sd_;
public:
    Correction(Count n,const std::complex<double>* s,const std::complex<double>* d,double madelung,double scale)
        :n_(n),s_(s),sd_(admission::square(n)) {
        require_arithmetic();
        factor_=product(finite_point(-scale),finite_point(madelung));
        for(Count i=0;i<n_;++i)for(Count j=0;j<n_;++j) {
            ComplexInterval value{};
            for(Count a=0;a<n_;++a)value=add(value,product(finite_point(s[i*n_+a]),finite_point(d[a*n_+j])));
            sd_[i*n_+j]=value;
        }
    }
    double error(Count i,Count j,std::complex<double> actual) const {
        ComplexInterval value{};
        for(Count a=0;a<n_;++a)value=add(value,product(sd_[i*n_+a],finite_point(s_[a*n_+j])));
        return magnitude(sub(finite_point(actual),product(factor_,value)));
    }
};
class Audit {
    Count n_,mesh_,count_;
    std::vector<ComplexInterval> f_,weighted_;
    Interval energy_;
    std::vector<double> exchange_errors_;
    double exchange_energy_error_=0.,fitted_energy_error_=0.;
    bool fitted_included_=false;
    std::vector<double> fitted_errors_;
    bool exchange_included_=false,correction_included_=false;
    std::vector<double> correction_errors_;
    double correction_energy_error_=0.;
    Count transpose(Count q) const {const Count n2=n_*n_,local=q%n2;return q-local+(local%n_)*n_+local/n_;}
public:
    Audit(Count n,Count nk,Count ns,const std::vector<double>& weights,
          const std::vector<std::complex<double>>& h,const std::complex<double>* d,
          const std::vector<std::complex<double>>& j,const std::complex<double>* xc,
          double ej,double exc,double enuc):n_(n),mesh_(admission::mul(nk,admission::square(n))),count_(admission::mul(ns,mesh_)) {
        require_arithmetic();
        if(!n || !nk || (ns!=1 && ns!=2) || weights.size()!=nk || h.size()!=mesh_ || j.size()!=mesh_ || !d)
            throw std::invalid_argument("MANTA assembly component extent mismatch");
        f_.resize(count_);weighted_.resize(count_);
        energy_=add(add(finite_point(ej).real,finite_point(exc).real),finite_point(enuc).real);
        for(Count q=0;q<count_;++q) {
            const Count k=(q%mesh_)/(n*n);
            if(!std::isfinite(weights[k]) || weights[k]<0.)throw std::invalid_argument("MANTA assembly weights must be finite nonnegative");
            weighted_[q]=product(finite_point(d[q]),point(weights[k]));
            f_[q]=add(finite_point(h[q%mesh_]),finite_point(j[q%mesh_]));
            if(xc)f_[q]=add(f_[q],finite_point(xc[q]));
            energy_=add(energy_,product(weighted_[q],finite_point(h[transpose(q)%mesh_])).real);
        }
    }
    void exchange(const std::vector<std::complex<double>>& k,double coefficient,const std::vector<double>& errors={},bool fitted=false) {
        if(k.size()<count_)throw std::invalid_argument("MANTA assembly exchange extent mismatch");
        const auto c=finite_point(coefficient),half=point(.5);
        auto& output_errors=fitted?fitted_errors_:exchange_errors_;
        auto& energy_error=fitted?fitted_energy_error_:exchange_energy_error_;
        if(!errors.empty()) {
            if(errors.size()<count_)throw std::invalid_argument("MANTA assembly exchange-error extent mismatch");
            if(output_errors.empty())output_errors.resize(count_);
            if(fitted)fitted_included_=true;else exchange_included_=true;
        }
        for(Count q=0;q<count_;++q) {
            const auto value=product(c,finite_point(k[q]));
            f_[q]=sub(f_[q],value);
            energy_=sub(energy_,product(half,product(weighted_[transpose(q)],value)).real);
            if(!errors.empty()) {
                if(!std::isfinite(errors[q]) || errors[q]<0.)throw std::invalid_argument("MANTA assembly requires finite nonnegative exchange errors");
                const double error=times_up(std::abs(coefficient),errors[q]);
                output_errors[q]=plus_up(output_errors[q],error);
                energy_error=plus_up(energy_error,times_up(.5,times_up(magnitude(weighted_[transpose(q)]),error)));
            }
        }
    }
    void correction(Count q,std::complex<double> z,double error=0.,bool audited=false) {
        if(q>=count_)throw std::invalid_argument("MANTA assembly correction extent mismatch");
        const auto value=finite_point(z);f_[q]=add(f_[q],value);
        energy_=add(energy_,product(point(.5),product(weighted_[transpose(q)],value)).real);
        if(audited) {
            if(!std::isfinite(error) || error<0.)throw std::invalid_argument("MANTA assembly requires finite nonnegative correction errors");
            if(correction_errors_.empty())correction_errors_.resize(count_);
            correction_included_=true;correction_errors_[q]=plus_up(correction_errors_[q],error);
            correction_energy_error_=plus_up(correction_energy_error_,times_up(.5,times_up(magnitude(weighted_[transpose(q)]),error)));
        }
    }
    PeriodicAssemblyRoundoff finish(const std::complex<double>* actual,Count count,double energy,
                                   const PeriodicRIFockReceipt* paired=nullptr) const {
        if(!actual || count!=count_)throw std::invalid_argument("MANTA assembly witness extent mismatch");
        PeriodicAssemblyRoundoff r;r.evaluated=true;
        for(Count q=0;q<count_;++q)r.fock_bound=std::max(r.fock_bound,magnitude(sub(finite_point(actual[q]),f_[q])));
        const auto error=sub(finite_point(energy).real,energy_);
        r.energy_bound=std::max(std::abs(error.lo),std::abs(error.hi));
        r.combined_fock_bound=r.fock_bound;r.combined_energy_bound=r.energy_bound;
        if(paired && paired->paired_roundoff_evaluated) {
            const double j=paired->paired_j_roundoff_bound,e=paired->paired_energy_roundoff_bound;
            if(!std::isfinite(j) || !std::isfinite(e) || j<0 || e<0)throw std::invalid_argument("MANTA assembly requires finite nonnegative paired bounds");
            r.paired_coulomb_included=true;
            r.combined_fock_bound=plus_up(r.fock_bound,j);r.combined_energy_bound=plus_up(r.energy_bound,e);
        }
        if(paired && paired->retained_j_roundoff_evaluated) {
            if(r.paired_coulomb_included)throw std::logic_error("MANTA conflicting Coulomb arithmetic routes");
            r.retained_coulomb_included=true;
            r.combined_fock_bound=plus_up(r.combined_fock_bound,paired->retained_j_roundoff_bound);
            r.combined_energy_bound=plus_up(r.combined_energy_bound,paired->retained_energy_roundoff_bound);
        }
        r.direct_exchange_roundoff_included=exchange_included_;
        for(auto error:exchange_errors_)r.direct_exchange_fock_bound=std::max(r.direct_exchange_fock_bound,error);
        r.direct_exchange_energy_bound=exchange_energy_error_;
        r.combined_fock_bound=plus_up(r.combined_fock_bound,r.direct_exchange_fock_bound);
        r.combined_energy_bound=plus_up(r.combined_energy_bound,r.direct_exchange_energy_bound);
        r.fitted_exchange_roundoff_included=fitted_included_;
        for(auto error:fitted_errors_)r.fitted_exchange_fock_bound=std::max(r.fitted_exchange_fock_bound,error);
        r.fitted_exchange_energy_bound=fitted_energy_error_;
        r.combined_fock_bound=plus_up(r.combined_fock_bound,r.fitted_exchange_fock_bound);
        r.combined_energy_bound=plus_up(r.combined_energy_bound,r.fitted_exchange_energy_bound);
        r.exchange_correction_roundoff_included=correction_included_;
        for(auto error:correction_errors_)r.exchange_correction_fock_bound=std::max(r.exchange_correction_fock_bound,error);
        r.exchange_correction_energy_bound=correction_energy_error_;
        r.combined_fock_bound=plus_up(r.combined_fock_bound,r.exchange_correction_fock_bound);
        r.combined_energy_bound=plus_up(r.combined_energy_bound,r.exchange_correction_energy_bound);
        return r;
    }
};
} // namespace vibeqc::manta::assembly_interval
