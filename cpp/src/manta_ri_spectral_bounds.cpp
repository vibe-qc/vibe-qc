#include "vibeqc/manta_ri_metric.hpp"
#include "vibeqc/manta_admission.hpp"
#include "manta_ri_interval.hpp"
#include <cfenv>
#include <cstring>
#include <complex>
#include <vector>

#if defined(__FAST_MATH__)
#error "MANTA spectral enclosure requires IEEE floating-point arithmetic"
#endif

namespace vibeqc::manta {
using namespace spectral_interval;
HermitianSpectrumPlan plan_hermitian_spectrum(Count n,const RIOptions& options,const ResourceBudget& budget) {
    using namespace admission;ri_detail::validate_options(options);margin(budget);
    if(!n || n>static_cast<Count>(std::numeric_limits<Eigen::Index>::max()))
        throw std::invalid_argument("MANTA spectral enclosure dimension must be positive and representable");
    HermitianSpectrumPlan p;p.dimension=n;
    // Public entry reserves both copied matrices, eigenvalues and O(n) norms.
    p.modeled_peak_bytes=add(add(mul(64,square(n)),mul(256,n)),add(budget.other_live_bytes,budget.backend_margin_bytes));
    p.scalar_terms=add(mul(512,mul(n,square(n))),mul(256,square(n)));p.kernel_terms=mul(8,p.scalar_terms);
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA spectral enclosure memory budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA spectral enclosure scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA spectral enclosure kernel-work budget exceeded");return p;
}
HermitianSpectrumCertificate certify_hermitian_spectrum(const Eigen::MatrixXcd& a,
    const Eigen::VectorXd& values,const Eigen::MatrixXcd& q,const RIOptions& options,const ResourceBudget& budget) {
    const Eigen::Index n=a.rows();plan_hermitian_spectrum(n,options,budget);require_arithmetic();
    if(a.cols()!=n || q.rows()!=n || q.cols()!=n || values.size()!=n ||
       !a.allFinite() || !q.allFinite() || !values.allFinite())
        throw std::invalid_argument("MANTA spectral enclosure requires finite matching square eigensystem arrays");
    for(Eigen::Index i=0;i<n;++i) {
        if(i && values[i]<values[i-1]) throw std::invalid_argument("MANTA spectral enclosure eigenvalues must be sorted");
        for(Eigen::Index j=0;j<=i;++j) if(a(i,j)!=std::conj(a(j,i)))
            throw std::invalid_argument("MANTA spectral enclosure requires an exactly Hermitian stored matrix");
    }
    HermitianSpectrumCertificate c;
    std::vector<double> residual_rows(n),residual_columns(n),gram_rows(n),gram_columns(n);
    for(Eigen::Index i=0;i<n;++i) for(Eigen::Index j=0;j<n;++j) {
        ComplexInterval residual,gram;
        for(Eigen::Index k=0;k<n;++k) {
            residual=add(residual,product(a(i,k),q(k,j)));
            gram=add(gram,product(std::conj(q(k,i)),q(k,j)));
        }
        residual=sub(residual,product(q(i,j),std::complex<double>(values[j],0.)));
        if(i==j) gram.real=sub(gram.real,{1.,1.});
        const double rb=magnitude(residual),gb=magnitude(gram);
        residual_rows[i]=plus_up(residual_rows[i],rb);residual_columns[j]=plus_up(residual_columns[j],rb);
        gram_rows[i]=plus_up(gram_rows[i],gb);gram_columns[j]=plus_up(gram_columns[j],gb);
    }
    for(Eigen::Index i=0;i<n;++i) {
        c.residual_bound=std::max({c.residual_bound,residual_rows[i],residual_columns[i]});
        c.orthogonality_bound=std::max({c.orthogonality_bound,gram_rows[i],gram_columns[i]});
    }
    const double selected_cutoff=std::max(options.absolute_cutoff,options.relative_cutoff*std::max(0.,values[n-1]));
    if(!std::isfinite(selected_cutoff)) throw std::overflow_error("MANTA spectral enclosure cutoff overflow");
    for(Eigen::Index i=0;i<n;++i) if(values[i]>selected_cutoff) ++c.rank;
    if(c.orthogonality_bound>=1.) return c; // No full-rank Q proof.
    // Rump & Lange, JCAM 434 (2023) 115332, Sections 3 and 4,
    // Eq. (3.2), Eq. (4.2), Theorem 4.1; doi:10.1016/j.cam.2023.115332.
    // All n eigenvalues are paired, not merely covered by per-column intervals.
    // sigma_min(Q)>=sqrt(1-e)>=1-e. max(row norm, column norm) bounds ||.||2.
    volatile double denominator=1.-c.orthogonality_bound;
    const double lower=down(denominator);
    if(lower<=0.) return c;
    volatile double quotient=c.residual_bound/lower;c.eigenvalue_error_bound=up(quotient);
    const auto largest=add(Interval{values[n-1],values[n-1]},{-c.eigenvalue_error_bound,c.eigenvalue_error_bound});
    const auto cutlo=product(options.relative_cutoff,std::max(0.,largest.lo));
    const auto cuthi=product(options.relative_cutoff,std::max(0.,largest.hi));
    c.cutoff_lower=std::max(options.absolute_cutoff,cutlo.lo);
    c.cutoff_upper=std::max(options.absolute_cutoff,cuthi.hi);
    c.spectrum_certified=true;double margin=infinity;
    for(Eigen::Index i=0;i<n;++i) {
        const auto eigen=add(Interval{values[i],values[i]},{-c.eigenvalue_error_bound,c.eigenvalue_error_bound});
        const auto gap=values[i]>selected_cutoff?sub(Interval{eigen.lo,eigen.lo},{c.cutoff_upper,c.cutoff_upper}):
            sub(Interval{c.cutoff_lower,c.cutoff_lower},{eigen.hi,eigen.hi});
        margin=std::min(margin,gap.lo);
    }
    if(margin<=0.) return c;
    // Weyl: an additional Hermitian perturbation of norm t moves each value
    // by <=t, and the adaptive relative cutoff by <=relative_cutoff*t.
    // Strictly t<radius preserves every selected/discarded classification.
    volatile double radius=margin/plus_up(1.,options.relative_cutoff);
    c.rank_stability_radius=std::max(0.,down(radius));c.rank_certified=c.rank_stability_radius>0.;return c;
}

HermitianSpectrumPlan plan_hermitian_inverse(Count n,const RIOptions& options,const ResourceBudget& budget) {
    using namespace admission;
    auto p=plan_hermitian_spectrum(n,options,budget);
    // One additional n-by-r factor copy at the public boundary, O(n) scratch,
    // and the outward-rounded F F^* - Q D Q^* contraction (r <= n).
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(mul(32,square(n)),mul(256,n)));
    p.scalar_terms=add(p.scalar_terms,add(mul(512,mul(n,square(n))),mul(256,square(n))));
    p.kernel_terms=mul(8,p.scalar_terms);
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA inverse enclosure memory budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA inverse enclosure scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA inverse enclosure kernel-work budget exceeded");return p;
}
HermitianInverseCertificate certify_hermitian_inverse(const Eigen::MatrixXcd& a,
    const Eigen::VectorXd& values,const Eigen::MatrixXcd& q,const Eigen::MatrixXcd& factor,
    const RIOptions& options,const ResourceBudget& budget) {
    const Eigen::Index n=a.rows();plan_hermitian_inverse(n,options,budget);
    HermitianInverseCertificate c;c.spectrum=certify_hermitian_spectrum(a,values,q,options,budget);
    const auto rank=static_cast<Eigen::Index>(c.spectrum.rank),first=n-rank;
    if(factor.rows()!=n || factor.cols()!=rank || !factor.allFinite())
        throw std::invalid_argument("MANTA inverse enclosure requires a finite factor matching the selected rank");
    if(!c.spectrum.rank_certified) return c;
    const auto divide_up=[](double x,double y) { if(x==0.) return 0.;volatile double z=x/y;return up(z); };
    const auto multiply_up=[](double x,double y) { return product(x,y).hi; };
    Count root=1;while(admission::square(root)<static_cast<Count>(n)) ++root;
    c.residual_frobenius_bound=multiply_up(static_cast<double>(root),c.spectrum.residual_bound);
    if(!rank) { c.qualified=true;return c; }
    const double delta=c.spectrum.eigenvalue_error_bound;
    c.retained_lower=sub(Interval{values[first],values[first]},{delta,delta}).lo;
    if(c.retained_lower<=0.) return c;
    c.inverse_norm_bound=divide_up(1.,c.retained_lower);
    c.divided_difference_bound=divide_up(c.inverse_norm_bound,values[first]);
    if(first) {
        const double discarded_upper=add(Interval{values[first-1],values[first-1]},{delta,delta}).hi;
        const double left_gap=sub(Interval{c.retained_lower,c.retained_lower},{values[first-1],values[first-1]}).lo;
        const double right_gap=sub(Interval{values[first],values[first]},{discarded_upper,discarded_upper}).lo;
        if(left_gap<=0. || right_gap<=0.) return c;
        c.divided_difference_bound=std::max({c.divided_difference_bound,
            divide_up(c.inverse_norm_bound,left_gap),divide_up(divide_up(1.,values[first]),right_gap)});
    }
    // Finite divided-difference identity, not a first-order perturbation.
    // In an exact eigenbasis V of A: (V^*(C Q-Q D))_ij = f[lambda_i,mu_j] (V^*R)_ij,
    // where C=f(A), D=f(diag(mu)), R=AQ-Q diag(mu). See Higham (2008),
    // Functions of Matrices, Ch. 3, for the spectral divided differences.
    // ||C-Q D Q^*|| <= ||C|| ||I-Q Q^*|| + ||C Q-Q D|| ||Q||,
    // ||I-Q Q^*||=||I-Q^*Q||<=e; ||Q||<=1+e; ||R||F<=ceil(sqrt(n))||R||2.
    // The rank certificate proves both mixed retained/discarded gaps above.
    c.residual_term=multiply_up(plus_up(1.,c.spectrum.orthogonality_bound),
        multiply_up(c.divided_difference_bound,c.residual_frobenius_bound));
    c.orthogonality_term=multiply_up(c.inverse_norm_bound,c.spectrum.orthogonality_bound);
    const auto divide=[](Interval x,double y) {
        if(x.lo==0. && x.hi==0.) return Interval{};
        volatile double lo=x.lo/y,hi=x.hi/y;return Interval{down(lo),up(hi)};
    };
    std::vector<double> rows(n),columns(n);
    for(Eigen::Index i=0;i<n;++i) for(Eigen::Index j=0;j<n;++j) {
        ComplexInterval difference;
        for(Eigen::Index k=0;k<rank;++k) {
            auto ideal=product(q(i,first+k),std::conj(q(j,first+k)));
            ideal={divide(ideal.real,values[first+k]),divide(ideal.imag,values[first+k])};
            difference=add(difference,sub(product(factor(i,k),std::conj(factor(j,k))),ideal));
        }
        const double bound=magnitude(difference);
        rows[i]=plus_up(rows[i],bound);columns[j]=plus_up(columns[j],bound);
    }
    for(Eigen::Index i=0;i<n;++i) c.factor_product_bound=std::max({c.factor_product_bound,rows[i],columns[i]});
    c.inverse_error_bound=plus_up(plus_up(c.residual_term,c.orthogonality_term),c.factor_product_bound);
    c.qualified=true;return c;
}
} // namespace vibeqc::manta
