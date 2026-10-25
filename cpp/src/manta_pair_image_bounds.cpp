#include "manta_pair_image_internal.hpp"
#include "manta_lattice_geometry.hpp"
#include "vibeqc/manta_provider.hpp"
#include <tuple>

namespace vibeqc::manta::pair_image_detail {
// Ewald (1921), IV.1, Eqs.37-40, DOI 10.1002/andp.19213690304 supplies
// the split kernel. The absolute Gaussian lattice envelopes and uniform
// Coulomb bound below are derived in the scientific contract, Section 5.9.
namespace {
using Real=long double;
constexpr Real pi=3.141592653589793238462643383279502884L;
constexpr Real neg=-std::numeric_limits<Real>::infinity();
constexpr Real eps=std::numeric_limits<double>::epsilon();
Real logadd(Real a,Real b) {
    if(a==neg) return b; if(b==neg) return a;
    const Real hi=std::max(a,b); return hi+std::log1p(std::exp(std::min(a,b)-hi));
}
double upper(Real x) {
    if(x==neg) return 0;
    const double value=std::nextafter(static_cast<double>(std::exp(x)*(1+8192*eps)),
        std::numeric_limits<double>::infinity());
    if(!std::isfinite(value)) throw std::overflow_error("MANTA pair-image envelope exceeds binary64");
    return value;
}
Real decay(Real logx) { return std::exp(std::min(logx,std::log(1.e6L))); }
Real representation(const libint2::Shell& s) {
    const auto t=md::representation(s.contr[0].l,s.contr[0].pure); Real norm=0;
    for(Eigen::Index i=0;i<t.rows();++i) {
        Real row=0; for(Eigen::Index j=0;j<t.cols();++j) row+=std::abs(Real(t(i,j)));
        norm=std::max(norm,row);
    }
    return std::log(norm*(1+256*eps));
}
Real angular(int l,Real exponent) {
    return l?Real(l)/2*(std::log(Real(l))-std::log(exponent)-1):0;
}
Real gradient(int l,Real exponent) {
    return logadd(l?std::log(Real(l))+angular(l-1,exponent):neg,
                  std::log(2*exponent)+angular(l+1,exponent));
}
struct Term { Real amplitude,logmu,kinetic=neg; };
// Absorb each nonconstant Cartesian monomial into half its primitive
// Gaussian. Plain s-s masses retain more of the product exponent: no angular
// absorption is needed. Gradient/Fourier envelopes keep their separate
// established half-Gaussian derivations (contract Sections 5.11 and 5.13).
// The full-s ceiling retains the original global kernel bound, so every
// mass and every resulting image bound is no larger than its prior envelope.
std::vector<Term> terms(const libint2::Shell& a,const libint2::Shell& b,double& maximum,bool gradients=false,bool fourier=false,Real full_s_ceiling=0) {
    std::vector<Term> out; const Real norm=representation(a)+representation(b);
    for(std::size_t i=0;i<a.alpha.size();++i) for(std::size_t j=0;j<b.alpha.size();++j) {
        const Real ca=std::abs(Real(a.contr[0].coeff[i])),cb=std::abs(Real(b.contr[0].coeff[j]));
        if(ca==0 || cb==0) continue;
        const Real x=a.alpha[i],y=b.alpha[j];
        const bool constant=!gradients && !fourier && a.contr[0].l==0 && b.contr[0].l==0 && full_s_ceiling>0;
        const Real half_sum=x/2+y/2;
        const Real p=constant?std::min(full_s_ceiling,2*half_sum):half_sum;
        // Scale both primitive exponents by t=p/(x+y), 1/2 <= t <= 1.
        // For constants no prefactor is needed; choose the tightest t that
        // cannot enlarge the original global periodized-kernel ceiling.
        const Real logmu=std::log(std::min(x,y))-std::log1p(std::min(x,y)/std::max(x,y))
            -std::log(2.L)+(constant?std::log(p/half_sum):0)+std::log1p(-256*eps);
        maximum=std::max(maximum,upper(std::log(p)));
        out.push_back({norm+std::log(ca)+std::log(cb)+angular(a.contr[0].l,x)+angular(b.contr[0].l,y)
            +1.5L*(std::log(pi)-std::log(p)),logmu});
        if(gradients)out.back().kinetic=norm+std::log(ca)+std::log(cb)
            +gradient(a.contr[0].l,x)+gradient(b.contr[0].l,y)
            +1.5L*(std::log(pi)-std::log(p))-std::log(2.L);
        if(fourier) {
            // Shift the Gaussian Fourier contour by -iQ/(2P), P=x+y.
            // Expand (|r-A|+|Q|/(2P))^la (|r-B|+|Q|/(2P))^lb,
            // absorb spatial monomials in half Gaussians and Q powers in
            // exp(-Q^2/(8P)). See scientific contract Section 5.13.
            const int la=a.contr[0].l,lb=b.contr[0].l;Real polynomial=neg;
            auto choose=[](int n,int k){Real v=1;for(int j=1;j<=k;++j)v=v*(n-j+1)/j;return v;};
            for(int u=0;u<=la;++u)for(int v=0;v<=lb;++v) {
                const int degree=u+v;const Real P=x+y;
                const Real q=degree?Real(degree)/2*(std::log(4*P*degree)-1)-degree*std::log(2*P):0;
                polynomial=logadd(polynomial,std::log(choose(la,u)*choose(lb,v))
                    +angular(la-u,x)+angular(lb-v,y)+q);
            }
            out.back().amplitude=norm+std::log(ca)+std::log(cb)+polynomial+1.5L*(std::log(pi)-std::log(p));
        }
    }
    return out;
}
bool retained(const std::vector<PeriodicPairEntry>& entries,Count a,Count b,LatticeIndex image) {
    const LatticeIndex inverse{-image[0],-image[1],-image[2]};
    if(a==b && image>inverse) image=inverse;
    const auto key=std::make_tuple(a,b,image);
    const auto it=std::lower_bound(entries.begin(),entries.end(),key,[](const auto& e,const auto& k) {
        return std::tie(e.shells[0],e.shells[1],e.image)<k;
    });
    return it!=entries.end() && std::tie(it->shells[0],it->shells[1],it->image)==key;
}
Real distance_lower(const libint2::Shell& a,const libint2::Shell& b,const Eigen::Matrix3d& cell,const LatticeIndex& n) {
    Real d=0,guard=0;
    for(int i=0;i<3;++i) {
        Real x=Real(a.O[i])-b.O[i],scale=std::abs(Real(a.O[i]))+std::abs(Real(b.O[i]));
        for(int j=0;j<3;++j) { const Real v=Real(cell(i,j))*n[j]; x-=v; scale+=std::abs(v); }
        d=std::hypot(d,x); guard=std::hypot(guard,256*eps*scale);
    }
    if(!std::isfinite(d) || !std::isfinite(guard)) throw std::overflow_error("MANTA pair-image distance overflow");
    return std::max(Real(0),d*(1-256*eps)-guard);
}
Real outside_cube(Real logmu,Real metric,Real displacement,Count radius) {
    const Real loga=logmu+2*std::log(metric);
    // For arbitrary shifts a one-dimensional Gaussian sum is bounded by
    // its two nearest samples plus the full line integral.
    const Real logtheta=logadd(std::log(2.L),.5L*(std::log(pi)-loga));
    const Real u=Real(radius)+1-displacement/metric;
    if(u<=0) return 3*logtheta;
    // Union of three omitted coordinate slabs; each half-line is bounded
    // by its first sample plus its integral (erfc(x) <= exp(-x*x)/(sqrt(pi)*x)).
    const Real tail=std::log(6.L)-decay(loga+2*std::log(u))
        +logadd(0,-std::log(2.L)-loga-std::log(u))+2*logtheta;
    return std::min(3*logtheta,tail);
}
double reciprocal_metric_lower(const Eigen::Matrix3d& reciprocal) {
    // Gershgorin bounds lambda_min(B^T B). Keep the inverse-residual
    // certificate as fallback for cells whose Gram matrix is not diagonally
    // dominant. Positive product guards precede the subtraction.
    Real lower=std::numeric_limits<Real>::infinity();
    for(int i=0;i<3;++i) {
        Real row=0;
        for(int j=0;j<3;++j) {
            Real value=0,absolute=0;
            for(int k=0;k<3;++k){const Real product=Real(reciprocal(k,i))*reciprocal(k,j);value+=product;absolute+=std::abs(product);}
            const Real guard=8192*eps*absolute;
            row+=i==j?value-guard:-std::abs(value)-guard;
        }
        lower=std::min(lower,row);
    }
    const double fallback=gaussian_domain_detail::real_metric_lower(reciprocal);
    if(!std::isfinite(lower)||lower<=0)return fallback;
    return std::max(fallback,std::nextafter(double(std::sqrt(lower)*(1-256*eps)),0.));
}
} // namespace

FourierEnvelope fourier_images(const BasisSet& basis,const Eigen::Matrix3d& cell,
    const std::vector<PeriodicPairEntry>& entries,const Plan& plan) {
    FourierEnvelope out;out.retained=out.omitted=md::Matrix::Zero(basis.nbasis(),basis.nbasis());
    const double metric=gaussian_domain_detail::real_metric_lower(cell);double maximum=0;bool all_s=true;
    const auto& shells=basis.libint();std::vector<Count> offsets;Count offset=0;
    for(const auto& s:shells){offsets.push_back(offset);offset+=s.size();all_s=all_s && s.contr[0].l==0;}
    for(Count a=0;a<shells.size();++a)for(Count b=a;b<shells.size();++b) {
        const auto& sa=shells[a];const auto& sb=shells[b];const auto t=terms(sa,sb,maximum,false,true);
        Real keep=neg,omit=neg,displacement=0,guard=0;
        for(int axis=0;axis<3;++axis) {
            displacement=std::hypot(displacement,Real(sa.O[axis])-sb.O[axis]);
            guard=std::hypot(guard,256*eps*(std::abs(Real(sa.O[axis]))+std::abs(Real(sb.O[axis]))));
        }
        displacement=displacement*(1+256*eps)+guard;
        const auto radius=static_cast<std::int64_t>(plan.radius);
        for(std::int64_t i=-radius;i<=radius;++i)for(std::int64_t j=-radius;j<=radius;++j)for(std::int64_t k=-radius;k<=radius;++k) {
            const LatticeIndex n{i,j,k};const Real d=distance_lower(sa,sb,cell,n);Real mass=neg;
            for(const auto& term:t)mass=logadd(mass,term.amplitude-(d==0?0:decay(term.logmu+2*std::log(d))));
            const bool included=retained(entries,a,b,n);(included?keep:omit)=logadd(included?keep:omit,mass);
        }
        for(const auto& term:t)omit=logadd(omit,term.amplitude+outside_cube(term.logmu,metric,displacement,plan.radius));
        const double kr=upper(keep),ko=upper(omit);
        for(Count i=0;i<sa.size();++i)for(Count j=0;j<sb.size();++j) {
            out.retained(offsets[a]+i,offsets[b]+j)=out.retained(offsets[b]+j,offsets[a]+i)=kr;
            out.omitted(offsets[a]+i,offsets[b]+j)=out.omitted(offsets[b]+j,offsets[a]+i)=ko;
        }
    }
    // maximum is an upper bound on (a+b)/2. For all-s sources no Q
    // polynomial needs absorption, so retain the complete Fourier Gaussian.
    out.decay=std::nextafter(double((1-256*eps)/Real(all_s?8:16)/maximum),0.);
    if(!std::isfinite(out.decay) || out.decay<=0)throw std::overflow_error("MANTA exchange Fourier decay is not representable");
    return out;
}
std::array<double,2> reciprocal_exchange_sums(const Eigen::Matrix3d& cell,const Eigen::Vector3d& delta,
    const std::vector<LatticeIndex>& labels,double fourier_decay,Count radius) {
    const Eigen::Matrix3d reciprocal=2*double(pi)*cell.inverse().transpose();
    const Real metric=reciprocal_metric_lower(reciprocal);
    const Real real_metric=gaussian_domain_detail::real_metric_lower(cell);
    const Real displacement=Real(delta.norm())*(1+256*eps),beta=2*Real(fourier_decay);
    const Real minimum=metric*(Real(radius)+1)-displacement;
    if(minimum<=0)throw std::invalid_argument("MANTA exchange completion cube cannot separate the omitted reciprocal domain from Q=0");
    const Real factor=std::log(4*pi)-3*std::log(real_metric)-std::log1p(-256*eps);
    Real keep=neg,omit=neg;const auto r=static_cast<std::int64_t>(radius);
    for(std::int64_t i=-r;i<=r;++i)for(std::int64_t j=-r;j<=r;++j)for(std::int64_t k=-r;k<=r;++k) {
        const LatticeIndex n{i,j,k};Real norm=0,guard=0;bool zero=true;
        for(int a=0;a<3;++a) {
            Real value=delta[a],scale=std::abs(value);
            for(int b=0;b<3;++b){const Real v=Real(reciprocal(a,b))*n[b];value+=v;scale+=std::abs(v);}
            zero=zero && value==0;norm=std::hypot(norm,value);guard=std::hypot(guard,256*eps*scale);
        }
        if(zero)continue; // Full Coulomb has the same explicitly omitted zero mode.
        norm=norm*(1-256*eps)-guard;
        if(norm<=0)throw std::invalid_argument("MANTA exchange reciprocal point is unresolved from Q=0");
        const Real value=factor-beta*norm*norm-2*std::log(norm);
        const bool included=std::binary_search(labels.begin(),labels.end(),n);
        (included?keep:omit)=logadd(included?keep:omit,value);
    }
    omit=logadd(omit,factor-2*std::log(minimum)+outside_cube(std::log(beta),metric,displacement,radius));
    return {upper(keep),upper(omit)};
}

Plan plan(const BasisSet& basis,const std::vector<PeriodicPairEntry>& entries,const std::vector<LatticeIndex>& images,Count nk,const ResourceBudget& budget) {
    Count radius=0;
    for(const auto& index:images) for(auto x:index) radius=std::max(radius,Count(std::abs(x)));
    return plan(basis,entries.size(),nk,radius,budget);
}
Plan plan(const BasisSet& basis,Count maximum_entries,Count nk,Count radius,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget); const auto c=census(basis,6); Plan out;out.radius=radius;
    const Count width=add(mul(2,out.radius),1); out.cells=mul(square(width),width);
    const Count ns=basis.nshells(),np=mul(ns,add(ns,1))/2,n2=square(basis.nbasis());
    Count maximum_primitives=0;
    for(const auto& s:basis.libint()) maximum_primitives=std::max(maximum_primitives,Count(s.alpha.size()));
    out.shell_calls=mul(np,out.cells);
    out.primitive_products=mul(mul(np,add(out.cells,1)),square(maximum_primitives));
    Count levels=1; for(auto n=maximum_entries;n>1;n>>=1) ++levels;
    out.scalar_terms=add(mul(256,out.primitive_products),add(mul(mul(16,levels),out.shell_calls),mul(64,mul(nk,n2))));
    out.kernel_terms=mul(128,out.scalar_terms);
    out.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),
        add(mul(128,n2),add(mul(128,square(maximum_primitives)),mul(64,ns))));
    cap(out.shell_calls,budget.shell_calls,"MANTA pair-image shell-call budget exceeded");
    cap(out.primitive_products,budget.primitive_quartets,"MANTA pair-image primitive budget exceeded");
    cap(out.scalar_terms,budget.scalar_terms,"MANTA pair-image scalar budget exceeded");
    cap(out.kernel_terms,budget.kernel_terms,"MANTA pair-image work budget exceeded");
    cap(out.modeled_peak_bytes,budget.memory_bytes,"MANTA pair-image memory budget exceeded");
    return out;
}

Envelope produce(const BasisSet& basis,const Eigen::Matrix3d& cell,const std::vector<PeriodicPairEntry>& entries,
    const gaussian_domain_detail::Geometry& geometry,const std::string& pair_identity,const std::string& kernel_identity,
    const Plan& plan) {
    Envelope out; auto& r=out.receipt;
    r.pair_source_identity=pair_identity; r.kernel_identity=kernel_identity;
    r.cube_radius=plan.radius; r.cube_cells=plan.cells;
    r.real_metric_lower=geometry.metric; r.reciprocal_metric_lower=geometry.reciprocal_metric;
    const Real logs=std::log(Real(geometry.metric)),logg=std::log(Real(geometry.reciprocal_metric));
    // det(A) >= sigma_min(A)^3. This lower bound needs no determinant
    // conditioning assumption beyond the already checked inverse residual.
    const Real logv=3*logs+std::log1p(-256*eps);
    r.volume_lower=std::nextafter(static_cast<double>(std::exp(logv)),0.);
    if(r.volume_lower<=0 || !std::isfinite(r.volume_lower)) throw std::overflow_error("MANTA pair-image volume bound is not representable");
    out.retained=out.omitted=md::Matrix::Zero(basis.nbasis(),basis.nbasis());
    const auto& shells=basis.libint(); std::vector<Count> offsets; Count offset=0;
    Real full_s_ceiling=0;
    for(const auto& s:shells) {
        offsets.push_back(offset); offset+=s.size();
        for(std::size_t i=0;i<s.alpha.size();++i)if(s.contr[0].coeff[i]!=0)
            full_s_ceiling=std::max(full_s_ceiling,Real(s.alpha[i]));
    }
    // The largest contributing primitive paired with itself establishes
    // this ceiling in the original half-Gaussian model. It stays there;
    // constant pairs may tighten up to it without raising the common kernel.
    for(Count a=0;a<shells.size();++a) for(Count b=a;b<shells.size();++b) {
        const auto& sa=shells[a]; const auto& sb=shells[b]; const auto t=terms(sa,sb,r.maximum_gaussian_exponent,false,false,full_s_ceiling);
        Real keep=neg,omit=neg,displacement=0,guard=0;
        for(int axis=0;axis<3;++axis) {
            displacement=std::hypot(displacement,Real(sa.O[axis])-sb.O[axis]);
            guard=std::hypot(guard,256*eps*(std::abs(Real(sa.O[axis]))+std::abs(Real(sb.O[axis]))));
        }
        displacement=displacement*(1+256*eps)+guard;
        const auto radius=static_cast<std::int64_t>(plan.radius);
        for(std::int64_t i=-radius;i<=radius;++i) for(std::int64_t j=-radius;j<=radius;++j) for(std::int64_t k=-radius;k<=radius;++k) {
            const LatticeIndex n{i,j,k}; const bool included=retained(entries,a,b,n);
            const Real d=distance_lower(sa,sb,cell,n); Real mass=neg;
            for(const auto& p:t) mass=logadd(mass,p.amplitude-(d==0?0:decay(p.logmu+2*std::log(d))));
            (included?keep:omit)=logadd(included?keep:omit,mass);
            (included?r.retained_ordered_pairs:r.omitted_ordered_pairs_in_cube)+=a==b?1:2;
            ++r.shell_pairs_evaluated; r.primitive_pairs_evaluated+=t.size();
        }
        // Every retained entry lies inside the completion cube. Sum the
        // disjoint missing cells, never subtract unrelated upper bounds.
        for(const auto& p:t) omit=logadd(omit,p.amplitude+outside_cube(p.logmu,geometry.metric,displacement,plan.radius));
        r.primitive_pairs_evaluated+=t.size();
        const double kr=upper(keep),ko=upper(omit);
        for(Count i=0;i<sa.size();++i) for(Count j=0;j<sb.size();++j) {
            out.retained(offsets[a]+i,offsets[b]+j)=out.retained(offsets[b]+j,offsets[a]+i)=kr;
            out.omitted(offsets[a]+i,offsets[b]+j)=out.omitted(offsets[b]+j,offsets[a]+i)=ko;
        }
    }
    // |B(f,g)| <= C Q_f Q_g for all envelope Gaussians and all centers.
    // The real split uses the uniform periodized convolution density and
    // integral erfc(eta*r)/r = pi/eta^2; reciprocal phases have modulus one.
    const Real logeta=std::log(Real(geometry.eta));
    const Real logalpha=r.maximum_gaussian_exponent?std::log(Real(r.maximum_gaussian_exponent)):neg;
    r.real_kernel_bound=upper(std::log(pi)-2*logeta+3*logadd(std::log(2.L)+.5L*(logalpha-std::log(2*pi)),-logs));
    const Real logt=std::log(2.L)+.5L*std::log(pi)+logeta-logg;
    const Real nonzero=logadd(logadd(std::log(3.L)+logt,std::log(3.L)+2*logt),3*logt);
    r.reciprocal_kernel_bound=upper(std::log(4*pi)-logv-2*logg+nonzero);
    r.gauge_kernel_bound=upper(std::log(pi)-2*logeta-logv);
    r.kernel_bound=upper(logadd(logadd(std::log(Real(r.real_kernel_bound)),std::log(Real(r.reciprocal_kernel_bound))),std::log(Real(r.gauge_kernel_bound))));
    r.maximum_retained_pair_mass=out.retained.maxCoeff(); r.maximum_omitted_pair_mass=out.omitted.maxCoeff();
    lattice_detail::Hash h; h.text("manta.periodic-pair-image-envelope.v2;kernel-capped-s-and-half-angular-gaussians;completion-cube;infinite-zero-average");
    h.text(pair_identity); h.text(kernel_identity); h.word(plan.radius);
    for(double x:{r.real_metric_lower,r.reciprocal_metric_lower,r.maximum_gaussian_exponent,r.kernel_bound}) h.number(x);
    for(Eigen::Index i=0;i<out.retained.size();++i) { h.number(out.retained.data()[i]); h.number(out.omitted.data()[i]); }
    r.source_identity=h.value.finish_hex();
    return out;
}
namespace { Real point_log(double x) { return x?std::log(Real(x)):neg; } }
double point_kernel_bound(const Envelope& image,const gaussian_domain_detail::Geometry& g) {
    const Real logalpha=point_log(image.receipt.maximum_gaussian_exponent),logs=point_log(g.metric);
    // A Gaussian convolved with a point has beta=alpha, without the factor
    // one half used for the Gaussian/Gaussian uniform interaction bound.
    const Real real=std::log(pi)-2*point_log(g.eta)
        +3*logadd(std::log(2.L)+.5L*(logalpha-std::log(pi)),-logs);
    return upper(logadd(real,logadd(point_log(image.receipt.reciprocal_kernel_bound),point_log(image.receipt.gauge_kernel_bound))));
}
double density_trace_bound(const md::Matrix& bounds,const std::vector<std::complex<double>>& d,const std::vector<double>& weights) {
    const Count n=bounds.rows();Real result=neg;
    for(Count k=0;k<weights.size();++k)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
        if(!weights[k] || !bounds(i,j))continue;
        const auto a=d[(k*n+i)*n+j],b=d[(k*n+j)*n+i];
        const Real norm=std::max(std::hypot(Real(a.real()),Real(a.imag())),std::hypot(Real(b.real()),Real(b.imag())));
        if(norm)result=logadd(result,std::log(Real(weights[k]))+std::log(Real(bounds(i,j)))+std::log(norm));
    }
    return upper(result);
}
PeriodicOneElectronImageResult one_electron(const BasisSet& basis,const Eigen::Matrix3d& cell,
    const std::vector<PeriodicPairEntry>& entries,const std::string& identity,const Plan& plan) {
    // K_ab = 1/2 integral grad(phi_a).grad(phi_b). Polynomial/Gaussian
    // gradient majorants and the Section 5.9 image sum give Section 5.12's
    // S/T enclosure, with no Coulomb kernel, auxiliary fit or density input.
    PeriodicOneElectronImageResult out;auto& r=out.receipt;r.pair_source_identity=identity;
    r.cube_radius=plan.radius;r.cube_cells=plan.cells;
    r.real_metric_lower=gaussian_domain_detail::real_metric_lower(cell);
    out.overlap_error_bound=out.kinetic_error_bound=md::Matrix::Zero(basis.nbasis(),basis.nbasis());
    const auto& shells=basis.libint();std::vector<Count> offsets;Count offset=0;
    for(const auto& s:shells){offsets.push_back(offset);offset+=s.size();}
    for(Count a=0;a<shells.size();++a)for(Count b=a;b<shells.size();++b) {
        const auto& sa=shells[a];const auto& sb=shells[b];double maximum=0;
        const auto t=terms(sa,sb,maximum,true);Real overlap=neg,kinetic=neg,displacement=0,guard=0;
        for(int axis=0;axis<3;++axis) {
            displacement=std::hypot(displacement,Real(sa.O[axis])-sb.O[axis]);
            guard=std::hypot(guard,256*eps*(std::abs(Real(sa.O[axis]))+std::abs(Real(sb.O[axis]))));
        }
        displacement=displacement*(1+256*eps)+guard;
        const auto radius=static_cast<std::int64_t>(plan.radius);
        for(std::int64_t i=-radius;i<=radius;++i)for(std::int64_t j=-radius;j<=radius;++j)for(std::int64_t k=-radius;k<=radius;++k) {
            const LatticeIndex n{i,j,k};const bool included=retained(entries,a,b,n);
            (included?r.retained_ordered_pairs:r.omitted_ordered_pairs_in_cube)+=a==b?1:2;
            ++r.shell_pairs_evaluated;
            if(included)continue;
            const Real d=distance_lower(sa,sb,cell,n);
            for(const auto& term:t) {
                const Real e=d==0?0:decay(term.logmu+2*std::log(d));
                overlap=logadd(overlap,term.amplitude-e);kinetic=logadd(kinetic,term.kinetic-e);
            }
            r.primitive_pairs_evaluated+=t.size();
        }
        for(const auto& term:t) {
            const Real tail=outside_cube(term.logmu,r.real_metric_lower,displacement,plan.radius);
            overlap=logadd(overlap,term.amplitude+tail);kinetic=logadd(kinetic,term.kinetic+tail);
        }
        r.primitive_pairs_evaluated+=t.size();
        const double s=upper(overlap),tbound=upper(kinetic);
        for(Count i=0;i<sa.size();++i)for(Count j=0;j<sb.size();++j) {
            out.overlap_error_bound(offsets[a]+i,offsets[b]+j)=out.overlap_error_bound(offsets[b]+j,offsets[a]+i)=s;
            out.kinetic_error_bound(offsets[a]+i,offsets[b]+j)=out.kinetic_error_bound(offsets[b]+j,offsets[a]+i)=tbound;
        }
    }
    r.maximum_overlap_error=out.overlap_error_bound.maxCoeff();r.maximum_kinetic_error=out.kinetic_error_bound.maxCoeff();
    lattice_detail::Hash h;h.text("manta.periodic-one-electron-image-bound.v1;half-gaussians;gradient-form;completion-cube");
    h.text(identity);h.word(plan.radius);h.number(r.real_metric_lower);
    for(Eigen::Index i=0;i<out.overlap_error_bound.size();++i){h.number(out.overlap_error_bound.data()[i]);h.number(out.kinetic_error_bound.data()[i]);}
    r.source_identity=h.value.finish_hex();return out;
}
} // namespace vibeqc::manta::pair_image_detail
