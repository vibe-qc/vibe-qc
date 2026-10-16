#pragma once
// Spin-independent physical HF source; its numerical identities are shared.
#include "vibeqc/manta_periodic_rhf.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/lattice_pair_cells.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "vibeqc/ewald.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
using namespace admission;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) { std::array<std::uint8_t,8> b{}; for(int i=0;i<8;++i) b[i]=v>>(56-8*i); hash.update(b.data(),8); }
    void text(const std::string& s) { word(s.size()); hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double x) { if(x==0) x=0.; Count bits; std::memcpy(&bits,&x,8); word(bits); }
    void value(Complex z) { real(z.real()); real(z.imag()); }
};
struct Sum {
    double value=0.,error=0.;
    void add(double x) { require_finite(x); const double y=x-error,next=value+y; error=(next-value)-y; value=next; require_finite(value); }
};
ResourceBudget unlimited(const ResourceBudget& budget) {
    auto b=budget; b.other_live_bytes=0;
    b.memory_bytes=b.shell_calls=b.primitive_quartets=b.scalar_terms=b.kernel_terms=std::numeric_limits<Count>::max(); return b;
}
Eigen::Matrix3d supercell(const PeriodicSystem& system,const RegularKMesh& mesh) {
    Eigen::Matrix3d a=system.lattice;
    for(int i=0;i<3;++i) a.col(i)*=mesh.mesh()[i];
    return a;
}
Count ewald_box(const Eigen::Matrix3d& a,double cutoff) {
    // Bound the shared Ewald producer's real pair box (including its padding)
    // and reciprocal origin box before either traversal allocates vectors.
    const double volume=std::abs(a.determinant()); Count count=1;
    if(!a.allFinite() || !std::isfinite(volume) || volume<1e-14 || !a.inverse().allFinite())
        throw std::invalid_argument("MANTA Madelung requires a finite nonsingular supercell");
    for(int i=0;i<3;++i) {
        const double cross=a.col((i+1)%3).cross(a.col((i+2)%3)).norm();
        const double reach=cutoff/(volume/cross);
        if(!std::isfinite(cross) || cross<1e-14 || !std::isfinite(reach) || reach>1e9)
            throw std::invalid_argument("MANTA Madelung Ewald image extent is not supported");
        count=mul(count,add(mul(2,static_cast<Count>(std::ceil(reach))),5));
    }
    // The shared source records at most two million support vectors. Admit
    // its enclosing box so this limit cannot fail after an unbounded reserve.
    cap(count,2000000,"MANTA Madelung Ewald support budget exceeded"); return count;
}
template<class Options>
PeriodicMadelungPlan plan_madelung(const PeriodicSystem& system,const RegularKMesh& mesh,const Options& o) {
    PeriodicMadelungPlan p;
    if(o.exchange_q0=="strict-zero-mode") return p;
    const auto a=supercell(system,mesh); const double volume=std::abs(a.determinant());
    p.alpha=o.madelung_alpha.value_or(std::sqrt(std::acos(-1.))/std::cbrt(volume));
    const double reach=std::sqrt(-std::log(o.madelung_tolerance));
    p.real_cutoff_bohr=reach/p.alpha; p.reciprocal_cutoff_bohr_inv=2*p.alpha*reach;
    if(!std::isfinite(p.alpha) || p.alpha<=0 || !std::isfinite(p.real_cutoff_bohr)
       || !std::isfinite(p.reciprocal_cutoff_bohr_inv) || p.real_cutoff_bohr<=0 || p.reciprocal_cutoff_bohr_inv<=0
       || !std::isfinite(4*p.alpha*p.alpha) || !std::isfinite(1/(4*p.alpha*p.alpha))
       || !std::isfinite(p.real_cutoff_bohr*p.real_cutoff_bohr)
       || !std::isfinite(p.reciprocal_cutoff_bohr_inv*p.reciprocal_cutoff_bohr_inv))
        throw std::invalid_argument("MANTA Madelung Ewald settings are not representable");
    p.real_candidates=ewald_box(a,p.real_cutoff_bohr);
    p.reciprocal_candidates=ewald_box(2*std::acos(-1.)*a.inverse().transpose(),p.reciprocal_cutoff_bohr_inv);
    const Count candidates=add(p.real_candidates,p.reciprocal_candidates);
    p.modeled_peak_bytes=add(4096,mul(64,candidates)); p.kernel_terms=mul(256,candidates); return p;
}
template<class Options,class Receipt>
void compute_madelung(const PeriodicSystem& system,const RegularKMesh& mesh,const Options& o,
                      const PeriodicMadelungPlan& p,Receipt& r) {
    r.exchange_q0=o.exchange_q0;
    if(o.exchange_q0=="strict-zero-mode") return;
    // Probe-charge Ewald: Sundararaman & Arias, PRB 87, 165122 (2013),
    // Sec. II.A. Full reciprocal Coulomb omits Q=0: add xi*S*D*S with no
    // subtraction of pi/(alpha^2 V Nk), which belongs only to an erfc/erf split.
    PeriodicSystem probe; probe.lattice=supercell(system,mesh); probe.unit_cell={{1,{0.,0.,0.}}};
    EwaldOptions options; options.alpha=p.alpha; options.tolerance=o.madelung_tolerance;
    options.real_cutoff_bohr=p.real_cutoff_bohr; options.recip_cutoff_bohr_inv=p.reciprocal_cutoff_bohr_inv;
    const auto result=ewald_nuclear_repulsion_with_support(probe,options);
    r.madelung_real_vectors=result.real_pair_displacements[0].size();
    r.madelung_reciprocal_vectors=result.reciprocal_vectors.size();
    if(r.madelung_real_vectors>p.real_candidates || r.madelung_reciprocal_vectors>p.reciprocal_candidates)
        throw std::logic_error("MANTA Madelung Ewald exceeded its admitted source bound");
    r.madelung=-2*result.energy; require_finite(r.madelung);
    r.boundary="neutral-3d-tinfoil-zero-average;exchange-BvK-ewald";
    Digest source; source.text("manta.periodic-rhf.madelung.v1;shared-ewald-point-probe;full-reciprocal-coulomb");
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) source.real(probe.lattice(i,j));
    source.real(p.alpha); source.real(p.real_cutoff_bohr); source.real(p.reciprocal_cutoff_bohr_inv);
    for(const auto* vectors:{&result.real_pair_displacements[0],&result.reciprocal_vectors}) {
        source.word(vectors->size()); for(const auto& v:*vectors) for(int i=0;i<3;++i) source.real(v[i]);
    }
    source.real(r.madelung); r.exchange_correction_identity=source.hash.finish_hex();
}
void validate_centers(const BasisSet& basis,const PeriodicSystem& system) {
    const auto inverse=system.lattice.inverse();
    for(Count i=0;i<basis.nshells();++i) {
        const int atom=basis.shell_atom_index(i);
        if(atom<0 || static_cast<std::size_t>(atom)>=system.unit_cell.size())
            throw std::invalid_argument("MANTA periodic RHF shell atom mapping is invalid");
        const auto& s=basis.libint()[i]; const auto& a=system.unit_cell[atom];
        const Eigen::Vector3d delta(s.O[0]-a.xyz[0],s.O[1]-a.xyz[1],s.O[2]-a.xyz[2]);
        const Eigen::Vector3d f=inverse*delta;
        for(int d=0;d<3;++d) if(!std::isfinite(f[d]) || std::abs(f[d])>0x1p40 || std::abs(f[d]-std::round(f[d]))>1e-10)
            throw std::invalid_argument("MANTA periodic RHF basis centers must match nuclear images");
    }
}
template<class Options,class Receipt>
void nuclear_energy(const PeriodicSystem& system,const md::Matrix& g,const Options& o,
                    const PeriodicNuclearPlan& nuclear,Receipt& r) {
    // Ewald (1921); zero-average point-charge energy, tin-foil surface and
    // uniform compensating background. Same alpha and explicit G as V.
    const double pi=std::acos(-1.),volume=std::abs(system.lattice.determinant());
    PairNuclearImageSelector selector(system,o.nuclear_cutoff_bohr);
    Sum real,reciprocal,self; double charge=0.;
    Digest source; source.text("manta.periodic-rhf.nuclear-energy.v1;physical-pair-sphere;tinfoil-jellium");
    source.text(r.nuclear_identity); source.real(o.alpha); source.real(o.nuclear_cutoff_bohr);
    for(Count a=0;a<system.unit_cell.size();++a) {
        const auto& atom=system.unit_cell[a]; charge+=atom.Z;
        const Eigen::Vector3d center(atom.xyz[0],atom.xyz[1],atom.xyz[2]);
        std::vector<std::array<int,4>> labels;
        const auto images=selector.select(center,nullptr,&labels);
        if(images.charges.size()>nuclear.maximum_nuclear_candidates_per_pair)
            throw std::logic_error("MANTA periodic RHF nuclear selector exceeded its admitted bound");
        for(Count i=0;i<images.charges.size();++i) {
            const auto& label=labels[i];
            if(label[0]==static_cast<int>(a) && !label[1] && !label[2] && !label[3]) continue;
            const auto& position=images.charges[i].second;
            const Eigen::Vector3d delta=Eigen::Vector3d(position[0],position[1],position[2])-center;
            const double distance=delta.norm();
            if(!std::isfinite(distance) || distance<=0) throw std::invalid_argument("MANTA periodic RHF has coincident nuclear images");
            real.add(.5*atom.Z*images.charges[i].first*std::erfc(o.alpha*distance)/distance); ++r.nuclear_pair_images;
        }
        self.add(-o.alpha/std::sqrt(pi)*atom.Z*atom.Z);
    }
    for(Eigen::Index ig=0;ig<g.rows();++ig) {
        Complex structure=0.;
        for(const auto& atom:system.unit_cell) {
            const double phase=g(ig,0)*atom.xyz[0]+g(ig,1)*atom.xyz[1]+g(ig,2)*atom.xyz[2];
            require_finite(phase); structure+=static_cast<double>(atom.Z)*Complex(std::cos(phase),std::sin(phase));
        }
        const double norm=g.row(ig).squaredNorm(),x=(.5*std::sqrt(norm))/o.alpha;
        reciprocal.add(2*pi/volume*std::exp(-x*x)/norm*std::norm(structure));
    }
    r.nuclear_real=real.value; r.nuclear_reciprocal=reciprocal.value; r.nuclear_self=self.value;
    r.nuclear_background=-((.5*pi/volume/o.alpha)/o.alpha)*charge*charge;
    r.nuclear_energy=r.nuclear_real+r.nuclear_reciprocal+r.nuclear_self+r.nuclear_background;
    require_finite(r.nuclear_energy);
    source.real(r.nuclear_real); source.real(r.nuclear_reciprocal); source.real(r.nuclear_self); source.real(r.nuclear_background);
    r.nuclear_energy_identity=source.hash.finish_hex();
}
} // namespace
} // namespace vibeqc::manta
