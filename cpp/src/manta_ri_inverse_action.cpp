#include "vibeqc/manta_ri_metric.hpp"
#include "vibeqc/manta_admission.hpp"
#include "manta_ri_interval.hpp"

namespace vibeqc::manta {
using namespace spectral_interval;
HermitianSpectrumPlan plan_hermitian_inverse_action(Count n,const RIOptions& o,const ResourceBudget& b) {
    using namespace admission;auto p=plan_hermitian_inverse(n,o,b);
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(mul(64,square(n)),mul(512,n)));
    p.scalar_terms=add(p.scalar_terms,add(mul(1024,mul(n,square(n))),mul(512,square(n))));
    p.kernel_terms=mul(8,p.scalar_terms);
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA metric action memory budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA metric action scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA metric action kernel budget exceeded");return p;
}
HermitianSpectrumPlan plan_hermitian_inverse_column(Count n,const ResourceBudget& b) {
    using namespace admission;margin(b);if(!n)throw std::invalid_argument("MANTA metric column dimension must be positive");
    HermitianSpectrumPlan p;p.dimension=n;
    p.modeled_peak_bytes=add(add(mul(64,square(n)),mul(1024,n)),add(b.other_live_bytes,b.backend_margin_bytes));
    p.scalar_terms=add(mul(1024,square(n)),mul(512,n));p.kernel_terms=mul(8,p.scalar_terms);
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA metric column memory budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA metric column scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA metric column kernel budget exceeded");return p;
}
HermitianSpectrumPlan plan_hermitian_inverse_family(Count n,Count count,const ResourceBudget& b) {
    using namespace admission;margin(b);if(!n || !count)throw std::invalid_argument("MANTA metric family dimensions must be positive");
    HermitianSpectrumPlan p;p.dimension=n;
    p.modeled_peak_bytes=add(add(mul(64,square(n)),add(mul(1024,n),mul(256,count))),add(b.other_live_bytes,b.backend_margin_bytes));
    Count levels=0;for(Count size=count;size;size>>=1)++levels;
    // Five positive-envelope contractions, each with one bounded heap sort.
    p.scalar_terms=mul(mul(1024,add(levels,4)),count);p.kernel_terms=mul(8,p.scalar_terms);
    cap(p.modeled_peak_bytes,b.memory_bytes,"MANTA metric family memory budget exceeded");
    cap(p.scalar_terms,b.scalar_terms,"MANTA metric family scalar budget exceeded");
    cap(p.kernel_terms,b.kernel_terms,"MANTA metric family kernel budget exceeded");return p;
}
namespace {
// Support of {0<=d_i<=h_i, sum d_i<=L}. For every lambda>=0,
// v.d <= lambda L + sum h_i max(v_i-lambda,0). An outward capacity
// sum selects a near-optimal breakpoint; the dual inequality is valid even
// when rounding selects the neighbouring breakpoint. Never subtract a
// rounded capacity from L as if the remainder were exact.
double density_support(const std::vector<double>& v,double limit,const std::vector<double>& h) {
    double maximum=0.,weighted=0.;
    for(Count i=0;i<v.size();++i) {
        maximum=std::max(maximum,v[i]);
        if(!h.empty())weighted=plus_up(weighted,times_up(h[i],v[i]));
    }
    double bound=times_up(limit,maximum);if(h.empty())return bound;
    bound=std::min(bound,weighted);if(bound==0.)return 0.;
    std::vector<Count> order(v.size());for(Count i=0;i<order.size();++i)order[i]=i;
    auto less=[&](Count a,Count b){return v[a]<v[b];};
    std::make_heap(order.begin(),order.end(),less);
    double capacity=0.,lambda=0.;
    for(Count remaining=order.size();remaining;--remaining) {
        std::pop_heap(order.begin(),order.begin()+remaining,less);
        const Count i=order[remaining-1];
        // Each cap may be clipped to L without changing the feasible set.
        const double cap=std::min(h[i],limit);
        if(capacity>=limit-cap){lambda=v[i];break;}
        capacity=plus_up(capacity,cap);
        if(capacity>=limit){lambda=v[i];break;}
    }
    if(lambda==0.)return bound;
    double dual=times_up(lambda,limit);
    for(Count i=0;i<v.size();++i)if(v[i]>lambda)
        dual=plus_up(dual,times_up(h[i],sub(Interval{v[i],v[i]},Interval{lambda,lambda}).hi));
    return std::min(bound,dual);
}
}
struct HermitianInverseAction::Data {
    Eigen::MatrixXcd factor;
    HermitianInverseCertificate certificate;
    std::vector<ComplexInterval> af;
    std::vector<double> gram_rows;
    double discarded_factor=0.,discarded_metric=0.;
    Data(const Eigen::MatrixXcd& a,const Eigen::MatrixXcd& f,const HermitianInverseCertificate& c)
        :factor(f),certificate(c),af(f.size()),gram_rows(f.cols()) {
        const auto n=f.rows(),rank=f.cols();if(!c.qualified || !rank)return;
        for(Eigen::Index i=0;i<n;++i)for(Eigen::Index j=0;j<rank;++j)
            for(Eigen::Index k=0;k<n;++k)af[i*rank+j]=add(af[i*rank+j],product(a(i,k),f(k,j)));
        // G=F^* A F-I, bounded entrywise. G is exactly Hermitian; use one
        // triangle for symmetric magnitude envelopes and Gershgorin weights.
        for(Eigen::Index i=0;i<rank;++i)for(Eigen::Index j=i;j<rank;++j) {
            ComplexInterval g;
            for(Eigen::Index k=0;k<n;++k)g=add(g,product(point(std::conj(f(k,i))),af[k*rank+j]));
            if(i==j)g.real=sub(g.real,{1.,1.});
            const double bound=magnitude(g);gram_rows[i]=plus_up(gram_rows[i],bound);
            if(i!=j)gram_rows[j]=plus_up(gram_rows[j],bound);
        }
        if(rank<n) {
            // P_d C P_d=0 and ||F F^*-C||<=epsilon imply
            // ||P_d F||<=sqrt(epsilon). No unverified projector is used.
            discarded_factor=sqrt_upper(c.inverse_error_bound);
            for(Eigen::Index i=0;i<n;++i) {
                double row=0.;for(Eigen::Index j=0;j<n;++j)row=plus_up(row,magnitude(point(a(i,j))));
                discarded_metric=std::max(discarded_metric,row);
            }
        }
    }
};
HermitianInverseAction::HermitianInverseAction(const Eigen::MatrixXcd& a,const Eigen::MatrixXcd& f,
    const HermitianInverseCertificate& c,const RIOptions& o,const ResourceBudget& b) {
    plan_hermitian_inverse_action(a.rows(),o,b);require_arithmetic();
    data_=std::make_shared<Data>(a,f,c);
}
HermitianInverseAction::HermitianInverseAction(const Eigen::MatrixXcd& a,const Eigen::VectorXd& w,
    const Eigen::MatrixXcd& q,const Eigen::MatrixXcd& f,const RIOptions& o,const ResourceBudget& b) {
    plan_hermitian_inverse_action(a.rows(),o,b);
    auto c=certify_hermitian_inverse(a,w,q,f,o,b);data_=std::make_shared<Data>(a,f,c);
}
Count HermitianInverseAction::dimension() const { return data_->factor.rows(); }
const HermitianInverseCertificate& HermitianInverseAction::certificate() const { return data_->certificate; }
HermitianInverseColumnBound HermitianInverseAction::column(const Eigen::VectorXcd& t,const ResourceBudget& b) const {
    const auto n=data_->factor.rows(),rank=data_->factor.cols();plan_hermitian_inverse_column(n,b);require_arithmetic();
    if(t.size()!=n || !t.allFinite())throw std::invalid_argument("MANTA metric column requires finite matching coupling vector");
    HermitianInverseColumnBound c;c.owner=data_;if(!certificate().qualified)return c;
    for(Eigen::Index i=0;i<n;++i)c.coupling_norm=plus_up(c.coupling_norm,magnitude(point(t[i])));
    c.qualified=true;if(!rank){c.residual_norm=c.coupling_norm;return c;}
    std::vector<ComplexInterval> projected(rank);double squared=0.,projected_norm=0.;
    for(Eigen::Index j=0;j<rank;++j) {
        for(Eigen::Index i=0;i<n;++i)projected[j]=add(projected[j],product(std::conj(data_->factor(i,j)),t[i]));
        const double v=magnitude(projected[j]);projected_norm=plus_up(projected_norm,v);
        squared=plus_up(squared,times_up(data_->gram_rows[j],times_up(v,v)));
    }
    c.factor_score=sqrt_upper(squared);
    c.discarded_norm=times_up(data_->discarded_factor,projected_norm);
    for(Eigen::Index i=0;i<n;++i) {
        auto residual=point(t[i]);
        for(Eigen::Index j=0;j<rank;++j)residual=sub(residual,product(data_->af[i*rank+j],projected[j]));
        c.residual_norm=plus_up(c.residual_norm,magnitude(residual));
    }
    return c;
}
HermitianInverseFamilyBound HermitianInverseAction::norm_family(const std::vector<double>& norms,
    double limit,const std::vector<double>& entries,const ResourceBudget& b) const {
    plan_hermitian_inverse_family(dimension(),norms.size(),b);require_arithmetic();
    if(!std::isfinite(limit) || limit<=0. || (!entries.empty() && entries.size()!=norms.size()))
        throw std::invalid_argument("MANTA metric norm family requires positive L1 and matching entry limits");
    for(const auto* values:{&norms,&entries})for(double v:*values)
        if(!std::isfinite(v) || v<0.)throw std::invalid_argument("MANTA metric norm family requires finite nonnegative envelopes");
    HermitianInverseFamilyBound result;if(!certificate().qualified)return result;
    const double weighted=density_support(norms,limit,entries);
    result.j_error_bounds.reserve(norms.size());
    for(double t:norms) {
        const double j=times_up(certificate().inverse_error_bound,times_up(t,weighted));
        result.j_error_bounds.push_back(j);result.j_error_bound=std::max(result.j_error_bound,j);
    }
    result.energy_error_bound=times_up(.5,density_support(result.j_error_bounds,limit,entries));
    result.qualified=true;return result;
}
HermitianInverseFamilyBound HermitianInverseAction::family(const std::vector<HermitianInverseColumnBound>& columns,
    double limit,const std::vector<double>& entries,const ResourceBudget& b) const {
    using namespace admission;require_arithmetic();margin(b);const Count count=columns.size();
    plan_hermitian_inverse_family(data_->factor.rows(),count,b);
    if(!count || !std::isfinite(limit) || limit<=0. || (!entries.empty() && entries.size()!=count))
        throw std::invalid_argument("MANTA metric family requires positive finite L1 and matching entry limits");
    for(double h:entries)if(!std::isfinite(h) || h<0.)throw std::invalid_argument("MANTA metric family entry limits must be finite nonnegative");
    for(const auto& c:columns)if(c.owner!=data_)throw std::invalid_argument("MANTA metric columns belong to a different owner");
    HermitianInverseFamilyBound result;
    if(!certificate().qualified || std::any_of(columns.begin(),columns.end(),[](const auto& c){return !c.qualified;}))return result;
    // |Delta K_pq| <= s_p s_q + e_p e_q/alpha + l_p t_q + t_p l_q
    //                + ||A|| l_p l_q. Independent absolute-inverse fallback.
    // Rump (2010), Acta Numerica19, Eq.(10.52), doi:10.1017/S096249291000005X,
    // supplies the inverse-norm/residual principle. This finite quadratic
    // extension also retains discarded-projector terms; no first-order drop.
    // Gershgorin/Cauchy bounds the G term; the rest follows the finite
    // primal/dual residual identity in scientific contract Section 5.34.
    using Member=double HermitianInverseColumnBound::*;
    const std::array<Member,4> fields{&HermitianInverseColumnBound::factor_score,&HermitianInverseColumnBound::residual_norm,
        &HermitianInverseColumnBound::discarded_norm,&HermitianInverseColumnBound::coupling_norm};
    std::array<double,4> maxima{};std::vector<double> values(count);
    for(Count j=0;j<4;++j) {
        for(Count i=0;i<count;++i)values[i]=columns[i].*fields[j];
        maxima[j]=density_support(values,limit,entries);
    }
    result.j_error_bounds.resize(count);
    for(Count i=0;i<count;++i) {
        const auto& c=columns[i];double error=times_up(c.factor_score,maxima[0]);
        error=plus_up(error,times_up(certificate().inverse_norm_bound,times_up(c.residual_norm,maxima[1])));
        error=plus_up(error,plus_up(times_up(c.discarded_norm,maxima[3]),times_up(c.coupling_norm,maxima[2])));
        error=plus_up(error,times_up(data_->discarded_metric,times_up(c.discarded_norm,maxima[2])));
        const double fallback=times_up(certificate().inverse_error_bound,times_up(c.coupling_norm,maxima[3]));
        result.j_error_bounds[i]=std::min(error,fallback);result.j_error_bound=std::max(result.j_error_bound,result.j_error_bounds[i]);
    }
    result.energy_error_bound=times_up(.5,density_support(result.j_error_bounds,limit,entries));
    result.qualified=true;return result;
}
} // namespace vibeqc::manta
