#include "vibeqc/manta_periodic_fock.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/detail/sha256.hpp"
#include <cstring>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
struct Digest {
    vibeqc::detail::Sha256 hash;
    void word(Count v) { std::array<std::uint8_t,8> b{}; for(int i=0;i<8;++i) b[i]=v>>(56-8*i); hash.update(b.data(),8); }
    void text(const std::string& s) { word(s.size()); hash.update(reinterpret_cast<const std::uint8_t*>(s.data()),s.size()); }
    void real(double x) { if(x==0) x=0.; Count bits; std::memcpy(&bits,&x,8); word(bits); }
    void value(Complex z) { real(z.real()); real(z.imag()); }
};
void finite(Complex z) { admission::require_finite(z.real()); admission::require_finite(z.imag()); }
void accumulate(Complex x,Complex& sum,Complex& error) { const auto y=x-error,next=sum+y; error=(next-sum)-y; sum=next; }
double hermitian(const std::vector<Complex>& values,Count blocks,Count n,double tolerance) {
    double maximum=0.;
    for(Count block=0;block<blocks;++block) {
        double scale=0.,residual=0.;
        for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
            const auto z=values[(block*n+i)*n+j]; finite(z);
            const double magnitude=std::abs(z),error=std::abs(z-std::conj(values[(block*n+j)*n+i]));
            admission::require_finite(magnitude); admission::require_finite(error);
            scale=std::max(scale,magnitude); residual=std::max(residual,error);
        }
        if(residual>tolerance*std::max(1.,scale)) throw std::invalid_argument("MANTA periodic Fock requires Hermitian blocks without repair");
        maximum=std::max(maximum,residual);
    }
    return maximum;
}
ResourceBudget unlimited(const ResourceBudget& budget) {
    auto b=budget; b.other_live_bytes=0;
    b.memory_bytes=b.shell_calls=b.primitive_quartets=b.scalar_terms=b.kernel_terms=std::numeric_limits<Count>::max(); return b;
}
} // namespace
PeriodicFockPlan plan_periodic_fock(const BasisSet& basis,Count ng,Count nq,Count nc,Count nk,Count batch,
    std::optional<double> cutoff,const ResourceBudget& budget,const std::string& kernel,double omega,double scale) {
    using namespace admission;
    margin(budget); require_finite(scale);
    const auto leaf=unlimited(budget);
    PeriodicFockPlan p; p.n_basis=basis.nbasis(); p.momenta=nk;
    p.exchange=plan_periodic_exchange(basis,nq,nc,nk,2,batch,cutoff,leaf,kernel,omega);
    p.hartree=plan_periodic_hartree(basis,ng,nc,nk,batch,cutoff,leaf);
    const Count elements=mul(nk,square(p.n_basis));
    // Sum both independent leaf peaks conservatively. They inventory caller
    // inputs, native snapshots, component owners/exports and workspaces.
    // Reserve additional H/H-copy, spin F/exports, sum-density/energy scratch
    // and fixed receipt owners. No overlapping owner is omitted.
    p.driver_live_bytes=add(mul(128,elements),8192);
    p.modeled_peak_bytes=add(budget.other_live_bytes,add(p.driver_live_bytes,
        add(p.hartree.modeled_peak_bytes,p.exchange.modeled_peak_bytes)));
    p.shell_calls=add(p.hartree.shell_calls,p.exchange.shell_calls);
    p.primitive_pairs=add(p.hartree.primitive_pairs,p.exchange.primitive_pairs);
    p.scalar_terms=add(mul(128,elements),add(p.hartree.scalar_terms,p.exchange.scalar_terms));
    p.kernel_terms=add(mul(4096,elements),add(p.hartree.kernel_terms,p.exchange.kernel_terms));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic Fock aggregate memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic Fock aggregate shell-call budget exceeded");
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA periodic Fock aggregate primitive-pair budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic Fock aggregate scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic Fock aggregate kernel-work budget exceeded");
    return p;
}
PeriodicFockResult compute_periodic_fock(const BasisSet& basis,const md::Matrix& lattice,
    const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,const md::Matrix& cells,
    const md::Matrix& k,const std::vector<double>& weights,const std::vector<Complex>& h,
    const std::vector<Complex>& d,Count batch,std::optional<double> cutoff,const ResourceBudget& budget,
    const std::string& kernel,double omega,double scale) {
    using namespace admission;
    const Count nk=k.rows(),pairs=square(nk),n=basis.nbasis(),n2=square(n),elements=mul(nk,n2);
    if(offsets.size()!=add(pairs,1) || offsets.front()!=0 || offsets.back()!=static_cast<Count>(q.rows()))
        throw std::invalid_argument("MANTA periodic Fock offsets must delimit every transfer pair");
    Count maximum=0;
    for(Count i=0;i<pairs;++i) {
        if(offsets[i]>offsets[i+1]) throw std::invalid_argument("MANTA periodic Fock offsets must be monotone");
        maximum=std::max(maximum,offsets[i+1]-offsets[i]);
    }
    PeriodicFockResult result;
    result.plan=plan_periodic_fock(basis,g.rows(),maximum,cells.rows(),nk,batch,cutoff,budget,kernel,omega,scale);
    const auto& p=result.plan; auto& r=result.receipt;
    if(h.size()!=elements || d.size()!=mul(2,elements)) throw std::invalid_argument("MANTA periodic Fock H/D extents disagree");
    r.hcore_hermiticity_residual=hermitian(h,nk,n,1e-12);
    hermitian(d,mul(2,nk),n,1e-12);
    std::vector<Complex> total(elements);
    for(Count i=0;i<elements;++i) { total[i]=d[i]+d[elements+i]; finite(total[i]); }
    auto leaf=budget;
    leaf.other_live_bytes=add(budget.other_live_bytes,add(p.driver_live_bytes,p.hartree.modeled_peak_bytes));
    result.exchange=compute_periodic_exchange(basis,lattice,q,offsets,cells,k,weights,d,2,batch,cutoff,leaf,kernel,omega);
    leaf.other_live_bytes=add(budget.other_live_bytes,add(p.driver_live_bytes,p.exchange.modeled_peak_bytes));
    result.hartree=compute_periodic_hartree(basis,lattice,g,cells,k,weights,total,batch,cutoff,leaf);
    result.values.resize(mul(2,elements));
    for(Count spin=0;spin<2;++spin) for(Count i=0;i<elements;++i)
        result.values[spin*elements+i]=h[i]+result.hartree.momenta[i]-scale*result.exchange.values[spin*elements+i];
    r.fock_hermiticity_residual=hermitian(result.values,mul(2,nk),n,1e-10);
    Complex energy=0,error=0;
    for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j)
        accumulate(weights[ik]*total[(ik*n+j)*n+i]*h[(ik*n+i)*n+j],energy,error);
    finite(energy);
    if(std::abs(energy.imag())>1e-10*std::max(1.,std::abs(energy.real())))
        throw std::runtime_error("MANTA periodic Fock one-electron energy is not real");
    r.exchange_scale=scale; r.one_electron_energy=energy.real(); r.coulomb_energy=result.hartree.receipt.energy;
    r.exchange_energy=scale*(result.exchange.receipt.energies[0]+result.exchange.receipt.energies[1]);
    r.electronic_energy=r.one_electron_energy+r.coulomb_energy+r.exchange_energy;
    require_finite(r.exchange_energy); require_finite(r.electronic_energy); r.modeled_peak_bytes=p.modeled_peak_bytes;
    Digest one; one.text("manta.periodic-fock.hcore.v1"); one.word(nk); one.word(n); for(auto z:h) one.value(z);
    r.one_electron_identity=one.hash.finish_hex();
    Digest source; source.text("manta.periodic-fock.source.v1"); source.text(r.one_electron_identity);
    source.text(result.hartree.receipt.source_identity); source.text(result.exchange.receipt.source_identity); source.real(scale);
    r.source_identity=source.hash.finish_hex();
    Digest density; density.text("manta.periodic-fock.density.v1"); density.text(r.source_identity);
    density.text(result.exchange.receipt.density_identity); density.text(result.hartree.receipt.density_identity);
    r.density_identity=density.hash.finish_hex();
    Digest snapshot; snapshot.text("manta.periodic-fock.snapshot.v1"); snapshot.text(r.source_identity); snapshot.text(r.density_identity);
    for(auto z:result.values) snapshot.value(z);
    snapshot.real(r.one_electron_energy); snapshot.real(r.coulomb_energy); snapshot.real(r.exchange_energy);
    r.snapshot_identity=snapshot.hash.finish_hex(); return result;
}
} // namespace vibeqc::manta
