#pragma once
// Shared finite complex Bloch AO support. Value and gradient consumers must
// use exactly the same canonical shell images and phase transport.
#include "vibeqc/manta_periodic_local_exchange.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/ao_eval_subset.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <Eigen/LU>
#include <cstring>
namespace vibeqc::manta::periodic_xc_detail {
using Complex=std::complex<double>;
using namespace admission;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void word(Count x) { std::array<std::uint8_t,8> b{}; for(int i=0;i<8;++i) b[i]=x>>(56-8*i); hash.update(b.data(),8); }
    void text(const std::string& s) { word(s.size()); hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double x) { require_finite(x); if(x==0) x=0.; Count b; std::memcpy(&b,&x,8); word(b); }
    void value(Complex x) { real(x.real()); real(x.imag()); }
    void matrix(const Eigen::MatrixXd& a) { word(a.rows()); word(a.cols()); for(Eigen::Index i=0;i<a.rows();++i) for(Eigen::Index j=0;j<a.cols();++j) real(a(i,j)); }
};
struct Sum {
    double value=0.,error=0.;
    void add(double x) { require_finite(x); const auto y=x-error,next=value+y; error=(next-value)-y; value=next; require_finite(value); }
};
inline void finite(Complex x) { require_finite(x.real()); require_finite(x.imag()); }
inline std::vector<Eigen::Vector3d> initialize(const BasisSet& basis,const Eigen::MatrixXd& lattice,
    const Eigen::MatrixXd& images,const Eigen::MatrixXd& k,const std::vector<double>& weights,
    const Eigen::MatrixXd& points,const std::vector<double>& qw,const std::vector<Complex>& d,
    Count ns,const PeriodicLocalExchangePlan& p,PeriodicLocalExchangeReceipt& r,const std::string& source_tag) {
    const Count n=p.n_basis,nk=p.momenta,ng=p.grid_points,ni=p.images,e=mul(mul(ns,nk),square(n));
    if(lattice.rows()!=3 || lattice.cols()!=3 || images.cols()!=3 || k.cols()!=3 || points.cols()!=3
       || weights.size()!=nk || qw.size()!=ng || d.size()!=e)
        throw std::invalid_argument("MANTA local exchange source shapes disagree");
    validate_basis(basis);
    if(!lattice.allFinite() || !images.allFinite() || !k.allFinite() || !points.allFinite())
        throw std::invalid_argument("MANTA local exchange source must be finite");
    const Eigen::Matrix3d a=lattice; const double volume=std::abs(a.determinant());
    const Eigen::Matrix3d inverse=a.inverse();
    if(!std::isfinite(volume) || volume<1e-14 || !inverse.allFinite())
        throw std::invalid_argument("MANTA local exchange requires a nonsingular 3D lattice");
    for(Count i=0;i<ni;++i) {
        const Eigen::Vector3d label=inverse*images.row(i).transpose();
        if(!label.allFinite() || label.cwiseAbs().maxCoeff()>1e9 || (label-label.array().round().matrix()).cwiseAbs().maxCoeff()>1e-10)
            throw std::invalid_argument("MANTA local exchange images must be lattice translations");
        for(Count j=0;j<i;++j) if((inverse*(images.row(i)-images.row(j)).transpose()).cwiseAbs().maxCoeff()<1e-10)
            throw std::invalid_argument("MANTA local exchange images must be unique");
    }
    Sum weight_sum;
    for(double w:weights) { require_finite(w); if(w<=0) throw std::invalid_argument("MANTA local exchange k weights must be positive"); weight_sum.add(w); }
    if(std::abs(weight_sum.value-1)>1e-12) throw std::invalid_argument("MANTA local exchange k weights must sum to one");
    for(double w:qw) { require_finite(w); if(w<0) throw std::invalid_argument("MANTA local exchange quadrature weights must be nonnegative"); }
    for(Count block=0;block<ns*nk;++block) {
        double scale=0.,residual=0.;
        for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
            const auto z=d[(block*n+i)*n+j]; finite(z);
            scale=std::max(scale,std::abs(z)); residual=std::max(residual,std::abs(z-std::conj(d[(block*n+j)*n+i])));
        }
        require_finite(scale); require_finite(residual);
        if(residual>1e-12*std::max(1.,scale)) throw std::invalid_argument("MANTA local exchange density must be Hermitian without repair");
        r.density_hermiticity_residual=std::max(r.density_hermiticity_residual,residual);
    }
    Digest source; source.text(source_tag); source.word(ns);
    source.matrix(lattice); source.matrix(images); source.matrix(k); source.matrix(points);
    for(double w:weights) source.real(w); for(double w:qw) source.real(w);
    source.word(basis.nshells());
    std::vector<Eigen::Vector3d> shifts; shifts.reserve(basis.nshells());
    for(const auto& shell:basis.libint()) {
        source.word(shell.contr[0].l); source.word(shell.contr[0].pure); source.word(shell.alpha.size());
        for(double x:shell.O) source.real(x); for(double x:shell.alpha) source.real(x);
        for(double x:shell.contr[0].coeff) source.real(x);
        Eigen::Vector3d f=inverse*Eigen::Vector3d(shell.O.data());
        for(int axis=0;axis<3;++axis) {
            if(!std::isfinite(f[axis]) || std::abs(f[axis])>1e9) throw std::invalid_argument("MANTA local exchange shell image is not representable");
            // Snap only numerical roundoff at a cell face, then select [0,1).
            if(std::abs(f[axis]-std::round(f[axis]))<1e-10) f[axis]=std::round(f[axis]);
            f[axis]=std::floor(f[axis]);
        }
        shifts.push_back(a*f);
    }
    r.source_identity=source.hash.finish_hex();
    Digest density; density.text(r.source_identity); for(auto x:d) density.value(x); r.density_identity=density.hash.finish_hex();
    return shifts;
}
struct BlochPanel {
    std::vector<Complex> values;
    std::array<std::vector<Complex>,3> gradients;
};
inline BlochPanel evaluate_panel(const BasisSet& basis,const Eigen::MatrixXd& images,
    const Eigen::MatrixXd& k,const Eigen::MatrixXd& points,const std::vector<Eigen::Vector3d>& shifts,
    Count start,Count b,bool gradients,PeriodicLocalExchangeReceipt& r) {
    const Count n=basis.nbasis(),nk=k.rows(),ni=images.rows();
        BlochPanel panel;
        panel.values.assign(mul(mul(b,nk),n),0.);
        if(gradients) for(auto& x:panel.gradients) x.assign(panel.values.size(),0.);
        Eigen::MatrixX3d shifted(b,3); Eigen::MatrixXd ao; std::array<Eigen::MatrixXd,3> grad;
        std::vector<int> shell_list(1);
        Count offset=0;
        for(Count sh=0;sh<basis.nshells();++sh) {
            const auto& shell=basis.libint()[sh]; shell_list[0]=static_cast<int>(sh);
            for(Count image=0;image<ni;++image) {
                const Eigen::Vector3d t=images.row(image).transpose()-shifts[sh];
                for(Count g=0;g<b;++g) shifted.row(g)=points.row(start+g)-t.transpose();
                if(!shifted.allFinite()) throw std::overflow_error("MANTA local exchange shifted grid overflow");
                evaluate_ao_shell_subset(basis,shifted,shell_list,ao,gradients?&grad:nullptr);
                if(gradients) for(const auto& x:grad) if(!x.allFinite()) throw std::overflow_error("MANTA XC AO gradient overflow");
                if(!ao.allFinite()) throw std::overflow_error("MANTA local exchange AO overflow");
                ++r.shell_calls; r.primitive_points+=b*shell.alpha.size();
                for(Count ik=0;ik<nk;++ik) {
                    const double phase=k.row(ik).dot(t); require_finite(phase);
                    const Complex z(std::cos(phase),std::sin(phase));
                    for(Count g=0;g<b;++g) for(Count i=0;i<shell.size();++i) {
                        const Count index=(g*nk+ik)*n+offset+i;
                        panel.values[index]+=z*ao(g,i);
                        if(gradients) for(int axis=0;axis<3;++axis) panel.gradients[axis][index]+=z*grad[axis](g,i);
                    }
                }
            }
            offset+=shell.size();
        }
    return panel;
}
} // namespace vibeqc::manta::periodic_xc_detail
