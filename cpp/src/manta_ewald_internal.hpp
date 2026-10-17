#pragma once
#include "vibeqc/manta_periodic_ewald.hpp"
#include "vibeqc/manta_admission.hpp"
#include "manta_lattice_geometry.hpp"

namespace vibeqc::manta::ewald_detail {
using namespace admission;
using Complex=std::complex<double>;
using Real=long double;
constexpr Real pi=3.141592653589793238462643383279502884L;
template<class T> void accumulate(T x,T& sum,T& error) {
    const auto y=x-error,next=sum+y; error=(next-sum)-y; sum=next;
}
inline double number(Real x) {
    const double result=static_cast<double>(x);
    if(!std::isfinite(result)) throw std::overflow_error("MANTA periodic Ewald numerical overflow");
    return result;
}
inline void finite(Complex x) { require_finite(x.real()); require_finite(x.imag()); }
inline ResourceBudget unlimited(const ResourceBudget& b) {
    auto r=b; r.other_live_bytes=0;
    r.memory_bytes=r.shell_calls=r.primitive_quartets=r.scalar_terms=r.kernel_terms=std::numeric_limits<Count>::max();
    return r;
}
inline const PeriodicPairSource& required(const std::shared_ptr<const PeriodicPairSource>& p) {
    if(!p) throw std::invalid_argument("MANTA periodic Ewald requires a native periodic pair source");
    return *p;
}
inline Complex phase(const md::Matrix& k,Count i,const md::Point& r) {
    double angle=0; for(int axis=0;axis<3;++axis) angle+=k(i,axis)*r[axis];
    require_finite(angle); return std::exp(Complex(0,angle));
}
inline libint2::Shell shifted(const libint2::Shell& a,const md::Point& shift) {
    auto b=a; for(int axis=0;axis<3;++axis) { b.O[axis]+=shift[axis]; require_finite(b.O[axis]); }
    return b;
}
inline md::Point vector(const Eigen::Matrix3d& cell,const LatticeIndex& n) {
    md::Point r{};
    for(int i=0;i<3;++i) {
        Real x=0; for(int j=0;j<3;++j) x+=Real(cell(i,j))*n[j]; r[i]=number(x);
    }
    return r;
}
struct Coefficient {
    Real normalized=0;
    double scale=0;
    Real times(double integral) const {
        if(normalized==0 || scale==0 || integral==0) return 0;
        int a=0,b=0,c=0;
        const Real x=std::frexp(normalized,&a),y=std::frexp(scale,&b),z=std::frexp(integral,&c);
        const Real out=std::scalbn(x*y*z,a+b+c);
        if(!std::isfinite(out)) throw std::overflow_error("MANTA periodic Ewald coefficient overflow");
        return out;
    }
};
// Same q=0 coefficient and normalization as PeriodicPairSource::contract.
// Keep scale separate until multiplication: small weighted D contributions
// must not underflow before their sum or before a large integral is applied.
inline std::vector<Coefficient> coefficients(const PeriodicPairEntry& e,Count n,
    const md::Matrix& k,const std::vector<double>& weights,const std::vector<Complex>& d) {
    const Count nk=k.rows(); std::vector<Complex> phases(nk);
    for(Count ik=0;ik<nk;++ik) phases[ik]=std::conj(phase(k,ik,e.translation));
    std::vector<Coefficient> out(e.sizes[0]*e.sizes[1]);
    for(Count i=0;i<e.sizes[0];++i) for(Count j=0;j<e.sizes[1];++j) {
        auto& c=out[i*e.sizes[1]+j]; Real error=0;
        for(Count ik=0;ik<nk;++ik) {
            const auto x=d[(ik*n+e.offsets[0]+i)*n+e.offsets[1]+j];
            c.scale=std::max(c.scale,std::max(std::abs(x.real()),std::abs(x.imag())));
        }
        for(Count ik=0;ik<nk && c.scale>0;++ik) {
            const auto x=d[(ik*n+e.offsets[0]+i)*n+e.offsets[1]+j];
            const Real v=Real(weights[ik])*(Real(x.real()/c.scale)*phases[ik].real()
                -Real(x.imag()/c.scale)*phases[ik].imag());
            accumulate(v,c.normalized,error);
        }
        c.normalized*=e.paired?2:1;
    }
    return out;
}
inline void scatter(const PeriodicPairEntry& e,Count n,const md::Matrix& k,
    const std::vector<Real>& values,std::vector<Complex>& j,std::vector<Complex>& error) {
    for(Count ik=0;ik<Count(k.rows());++ik) {
        const auto ph=phase(k,ik,e.translation);
        for(Count a=0;a<e.sizes[0];++a) for(Count b=0;b<e.sizes[1];++b) {
            const auto contribution=ph*number(values[a*e.sizes[1]+b]); finite(contribution);
            const Count ab=(ik*n+e.offsets[0]+a)*n+e.offsets[1]+b;
            accumulate(contribution,j[ab],error[ab]);
            if(e.paired) {
                const Count ba=(ik*n+e.offsets[1]+b)*n+e.offsets[0]+a;
                accumulate(std::conj(contribution),j[ba],error[ba]);
            }
        }
    }
}
inline void validate_density(const PeriodicPairSource& s,const std::vector<Complex>& d,PeriodicEwaldReceipt& r) {
    const auto& p=s.plan(); const Count n=p.n_basis,nk=p.momenta;
    if(d.size()!=nk*n*n) throw std::invalid_argument("MANTA periodic Ewald density shape disagrees with source");
    double scale=0; lattice_detail::Hash hash;
    hash.text("manta.periodic-pair-density.v1"); hash.text(s.receipt().source_identity);
    for(Count k=0;k<nk;++k) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto x=d[(k*n+i)*n+j]; finite(x); require_finite(std::abs(x));
        scale=std::max(scale,std::abs(x));
        r.density_hermiticity_residual=std::max(r.density_hermiticity_residual,std::abs(x-std::conj(d[(k*n+j)*n+i])));
        hash.number(x.real()); hash.number(x.imag());
    }
    require_finite(r.density_hermiticity_residual);
    if(r.density_hermiticity_residual>1e-12*std::max(1.,scale))
        throw std::invalid_argument("MANTA periodic Ewald densities must be Hermitian");
    r.density_identity=hash.value.finish_hex();
}
} // namespace
