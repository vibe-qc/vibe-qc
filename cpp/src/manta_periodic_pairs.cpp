#include "vibeqc/manta_periodic_pairs.hpp"
#include "vibeqc/manta_admission.hpp"
#include "vibeqc/manta_shells.hpp"
#include "vibeqc/periodic_auxiliary_fourier.hpp"
#include "vibeqc/detail/sha256.hpp"
#include "manta_extent_pair.hpp"
#include <algorithm>
#include <cstring>
#include <tuple>

namespace vibeqc::manta {
namespace {
using Complex=std::complex<double>;
using namespace admission;
struct Hash {
    vibeqc::detail::Sha256 hash;
    void word(Count x) {
        std::array<std::uint8_t,8> b{};
        for(int i=0;i<8;++i) b[i]=x>>(56-8*i);
        hash.update(b.data(),b.size());
    }
    void number(double x) { if(x==0) x=0; Count bits; std::memcpy(&bits,&x,8); word(bits); }
    void text(const std::string& x) { word(x.size()); hash.update(reinterpret_cast<const std::uint8_t*>(x.data()),x.size()); }
    void matrix(const md::Matrix& x) {
        word(x.rows()); word(x.cols());
        for(Eigen::Index i=0;i<x.rows();++i) for(Eigen::Index j=0;j<x.cols();++j) number(x(i,j));
    }
};
ResourceBudget leaf_budget(const ResourceBudget& budget) {
    auto r=budget; r.other_live_bytes=0;
    r.memory_bytes=r.shell_calls=r.primitive_quartets=r.scalar_terms=r.kernel_terms=std::numeric_limits<Count>::max();
    return r;
}
template<class T> void accumulate(T value,T& sum,T& correction) {
    const auto y=value-correction,next=sum+y; correction=(next-sum)-y; sum=next;
}
void finite(Complex z) { require_finite(z.real()); require_finite(z.imag()); }
double upper(long double x) {
    if(x==0) return 0;
    x*=1+64*std::numeric_limits<double>::epsilon();
    if(!std::isfinite(x) || x>std::numeric_limits<double>::max())
        throw std::overflow_error("MANTA periodic pair absolute density bound exceeds binary64");
    const double out=std::nextafter(static_cast<double>(x),std::numeric_limits<double>::infinity());
    if(!std::isfinite(out)) throw std::overflow_error("MANTA periodic pair absolute density bound exceeds binary64");
    return out;
}
long double scaled_product(long double a,double b,double c) {
    if(a==0 || b==0 || c==0) return 0;
    int ea=0,eb=0,ec=0;
    const auto ma=std::frexp(a,&ea),mb=static_cast<long double>(std::frexp(b,&eb)),
               mc=static_cast<long double>(std::frexp(c,&ec));
    return std::scalbn(ma*mb*mc,ea+eb+ec);
}
double upper_product(long double a,double b) {
    if(a==0 || b==0) return 0;
    const auto product=scaled_product(a,b,1.);
    return product==0?std::numeric_limits<double>::denorm_min():upper(product);
}
libint2::Shell shifted(const libint2::Shell& s,const md::Point& r) {
    auto result=s;
    for(int d=0;d<3;++d) { result.O[d]+=r[d]; require_finite(result.O[d]); }
    return result;
}
Complex phase(const md::Matrix& k,Count index,const md::Point& r) {
    double angle=0; for(int d=0;d<3;++d) angle+=k(index,d)*r[d];
    require_finite(angle); return std::exp(Complex(0,angle));
}
} // namespace

PeriodicPairPlan plan_periodic_pairs(const BasisSet& basis,Count nc,Count nk,
    const PeriodicPairOptions& options,const ResourceBudget& budget) {
    using namespace admission;
    margin(budget);
    if(!nc || !(nc%2) || !nk)
        throw std::invalid_argument("MANTA periodic pairs require an odd positive image count and positive momentum count");
    if(options.order<0 || options.order>12)
        throw std::invalid_argument("MANTA periodic pair moment order must be zero through twelve");
    if(options.pair_cutoff_bohr) {
        require_finite(*options.pair_cutoff_bohr);
        if(*options.pair_cutoff_bohr<0) throw std::invalid_argument("MANTA periodic pair cutoff must be nonnegative");
    }
    const Count ns=basis.nshells();
    PeriodicPairPlan p; p.n_basis=basis.nbasis(); p.cells=nc; p.momenta=nk;
    p.maximum_pairs=add(mul(square(ns),nc),ns)/2;
    p.moments=Count(options.order+1)*(options.order+2)*(options.order+3)/6;
    const Count n2=square(p.n_basis),nm=mul(p.maximum_pairs,p.moments),dn=mul(nk,n2);
    cap(nm,std::numeric_limits<Eigen::Index>::max(),"MANTA periodic pair moments are not addressable");
    cap(dn,std::numeric_limits<Eigen::Index>::max(),"MANTA periodic pair density is not addressable");
    p.shell_calls=mul(3,p.maximum_pairs); // extent setup and one forward/adjoint cycle
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic pair shell-call budget exceeded");
    const Count input=add(72,add(mul(24,add(nc,nk)),mul(8,nk)));
    // Retained inventory, constructor copies and independent Python exports.
    p.retained_bytes=add(mul(3,input),mul(p.maximum_pairs,3*(sizeof(PeriodicPairEntry)+16*(options.order+1)+256)));
    p.driver_live_bytes=add(mul(80,dn),add(mul(48,nm),add(mul(64,p.moments),
        add(mul(64,p.maximum_pairs),add(mul(128,nk),65536)))));
    p.modeled_peak_bytes=add(add(budget.other_live_bytes,budget.backend_margin_bytes),add(p.retained_bytes,p.driver_live_bytes));
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic pair driver memory budget exceeded");
    auto leaf=leaf_budget(budget);
    auto ext=options.extents; ext.moment_order=options.order; ext.retain_full_moments=true;
    const auto ep=plan_pair_extents(basis,ext,leaf);
    const auto count=census(basis,6);
    const auto& shells=basis.libint(); Count largest_l=0,max_primitives=0;
    for(Count s=0;s<ns;++s) {
        if(shells[s].contr[0].l>shells[largest_l].contr[0].l) largest_l=s;
        max_primitives=std::max<Count>(max_primitives,shells[s].alpha.size());
    }
    const auto mp=plan_pair_moments(basis,{largest_l,largest_l},options.order,leaf);
    const Count primitive_scale=std::max<Count>(1,add(square(max_primitives),mp.primitive_pairs-1)/mp.primitive_pairs);
    const Count extent_scale=mul(2,nc),panels=mul(2,p.maximum_pairs);
    p.primitive_pairs=add(mul(extent_scale,ep.primitive_pairs),mul(panels,square(max_primitives)));
    const Count panel_scalars=mul(square(count.maximum_size),p.moments);
    p.scalar_terms=add(mul(extent_scale,ep.scalar_terms),add(mul(panels,panel_scalars),
        add(mul(32,mul(nk,mul(p.maximum_pairs,square(count.maximum_size)))),add(mul(16,dn),mul(8,nm)))));
    Count levels=1; for(auto n=nc;n>1;n>>=1) ++levels;
    p.kernel_terms=add(mul(extent_scale,ep.kernel_terms),add(mul(panels,mul(mp.kernel_terms,primitive_scale)),
        add(mul(64,p.scalar_terms),mul(256,mul(nc,levels)))));
    p.retained_bytes=add(p.retained_bytes,mul(3,count.bytes));
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(mul(3,count.bytes),
        std::max(ep.modeled_peak_bytes,mp.modeled_peak_bytes)-budget.backend_margin_bytes));
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA periodic pair primitive budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic pair scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic pair work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic pair memory budget exceeded");
    return p;
}

PeriodicPairSource::PeriodicPairSource(const BasisSet& basis,const Eigen::Matrix3d& lattice,
    const std::vector<LatticeIndex>& images,const md::Matrix& k,const std::vector<double>& weights,
    const PeriodicPairOptions& options,const ResourceBudget& budget)
    :plan_(plan_periodic_pairs(basis,images.size(),k.rows(),options,budget)),basis_(basis),
     lattice_(lattice),momenta_(k),weights_(weights),options_(options) {
    if(k.cols()!=3 || weights.size()!=Count(k.rows()) || !k.allFinite() || !lattice.allFinite())
        throw std::invalid_argument("MANTA periodic pair source requires finite lattice, momenta and matching weights");
    const double volume=std::abs(lattice.determinant());
    if(!std::isfinite(volume) || volume<=0 || !lattice.inverse().allFinite())
        throw std::invalid_argument("MANTA periodic pair source requires a nonsingular 3D lattice");
    double total=0;
    for(double w:weights) { require_finite(w); if(w<0) throw std::invalid_argument("MANTA periodic pair weights must be nonnegative"); total+=w; }
    if(!std::isfinite(total) || std::abs(total-1)>1e-12)
        throw std::invalid_argument("MANTA periodic pair weights must sum to one");
    auto domain=images;
    for(const auto& index:domain) for(auto x:index)
        if(x<-(std::int64_t{1}<<40) || x>(std::int64_t{1}<<40))
            throw std::invalid_argument("MANTA periodic pair image index exceeds the supported finite range");
    std::sort(domain.begin(),domain.end());
    if(!std::binary_search(domain.begin(),domain.end(),LatticeIndex{}))
        throw std::invalid_argument("MANTA periodic pair images must include the origin");
    for(Count i=0;i<domain.size();++i) {
        const auto x=domain[i]; const LatticeIndex negative{-x[0],-x[1],-x[2]};
        if(i && x==domain[i-1]) throw std::invalid_argument("MANTA periodic pair images must be unique");
        if(!std::binary_search(domain.begin(),domain.end(),negative))
            throw std::invalid_argument("MANTA periodic pair images must be inversion closed");
    }
    receipt_.basis_identity=auxiliary_basis_content_identity_sha256(basis_);
    Hash geometry; geometry.text("manta.periodic-pair-domain.v1;canonical-physical-pairs");
    geometry.matrix(lattice); for(const auto& index:domain) for(auto x:index) geometry.word(x);
    geometry.word(options.pair_cutoff_bohr.has_value());
    if(options.pair_cutoff_bohr) geometry.number(*options.pair_cutoff_bohr);
    receipt_.domain_identity=geometry.hash.finish_hex();
    options_.extents.moment_order=options.order; options_.extents.retain_full_moments=true;
    const auto& shells=basis_.libint(); std::vector<Count> offsets; Count offset=0;
    for(const auto& shell:shells) { offsets.push_back(offset); offset+=shell.size(); }
    entries_.reserve(plan_.maximum_pairs);
    for(Count a=0;a<shells.size();++a) for(Count b=0;b<shells.size();++b) for(const auto& index:domain) {
        const LatticeIndex negative{-index[0],-index[1],-index[2]};
        if(std::tie(a,b,index)>std::tie(b,a,negative)) continue;
        PeriodicPairEntry entry; entry.shells={a,b}; entry.offsets={offsets[a],offsets[b]};
        entry.sizes={shells[a].size(),shells[b].size()}; entry.image=index;
        entry.paired=a!=b || index!=LatticeIndex{};
        for(int i=0;i<3;++i) {
            long double x=0; for(int j=0;j<3;++j) x+=static_cast<long double>(lattice(i,j))*index[j];
            entry.translation[i]=static_cast<double>(x); require_finite(entry.translation[i]);
        }
        const auto ket=shifted(shells[b],entry.translation);
        const double distance=std::hypot(shells[a].O[0]-ket.O[0],shells[a].O[1]-ket.O[1],shells[a].O[2]-ket.O[2]);
        require_finite(distance);
        if(options.pair_cutoff_bohr && distance>*options.pair_cutoff_bohr) {
            receipt_.skipped_ordered_pairs+=entry.paired?2:1; continue;
        }
        for(Count ik=0;ik<plan_.momenta;++ik) (void)phase(momenta_,ik,entry.translation);
        entry.extent=detail::pair_extent(shells[a],ket,{a,b},options_.extents,
            receipt_.envelope_evaluations,receipt_.always_near_pairs);
        receipt_.ordered_pairs+=entry.paired?2:1;
        receipt_.primitive_pairs+=shells[a].alpha.size()*shells[b].alpha.size();
        entries_.push_back(std::move(entry));
    }
    receipt_.pairs=entries_.size(); receipt_.modeled_peak_bytes=plan_.modeled_peak_bytes;
    Hash source; source.text("manta.periodic-pair-source.v1;md-fp64;libint-shell-axial-v1;no-screening");
    source.text(receipt_.basis_identity); source.text(receipt_.domain_identity); source.text(receipt_.convention);
    source.matrix(momenta_); for(double w:weights_) source.number(w); source.word(options.order);
    for(double v:{options_.extents.pointwise_tolerance,options_.extents.tail_mass_tolerance,
                  options_.extents.minimum_radius,options_.extents.maximum_radius}) source.number(v);
    for(const auto& entry:entries_) { for(auto s:entry.shells) source.word(s); for(auto x:entry.image) source.word(x); }
    receipt_.source_identity=source.hash.finish_hex();
    images_=std::move(domain);
}

PairExtents PeriodicPairSource::extents() const {
    PairExtents out; out.pairs.reserve(entries_.size());
    for(const auto& e:entries_) out.pairs.push_back(e.extent);
    out.receipt.source_identity=receipt_.source_identity;
    out.receipt.center_policy="canonical-periodic-pair-midpoint";
    out.receipt.shell_pairs=receipt_.pairs; out.receipt.primitive_pairs=receipt_.primitive_pairs;
    out.receipt.envelope_evaluations=receipt_.envelope_evaluations;
    out.receipt.always_near_pairs=receipt_.always_near_pairs; out.receipt.modeled_peak_bytes=plan_.modeled_peak_bytes;
    return out;
}

PairMomentResult PeriodicPairSource::moment_panel(Count pair) const {
    if(pair>=entries_.size()) throw std::out_of_range("MANTA periodic pair index");
    const auto& e=entries_[pair]; const auto& a=basis_.libint()[e.shells[0]];
    const auto b=shifted(basis_.libint()[e.shells[1]],e.translation);
    PairMomentResult out; out.values=md::shell_moments(a,b,e.extent.center,options_.order);
    if(!out.values.allFinite()) throw std::overflow_error("MANTA periodic pair moment overflow");
    out.indices=md::hermites(options_.order); out.ao_shape=e.sizes;
    out.receipt.primitive_pairs=a.alpha.size()*b.alpha.size();
    out.receipt.primitive_expansions=out.receipt.primitive_pairs;
    out.receipt.primitive_moment_products=out.receipt.primitive_pairs*plan_.moments;
    out.receipt.modeled_peak_bytes=plan_.modeled_peak_bytes;
    Hash hash; hash.text(receipt_.source_identity); hash.text("moment-panel"); hash.word(pair);
    out.receipt.source_identity=hash.hash.finish_hex(); return out;
}

PeriodicPairDensity PeriodicPairSource::contract(const std::vector<Complex>& d) const {
    const Count n=plan_.n_basis,nk=plan_.momenta,n2=n*n;
    if(d.size()!=nk*n2) throw std::invalid_argument("MANTA periodic pair density shape disagrees with source");
    PeriodicPairDensity out; auto& r=out.receipt;
    r.source_identity=receipt_.source_identity; r.modeled_peak_bytes=plan_.modeled_peak_bytes;
    double scale=0;
    Hash hash; hash.text("manta.periodic-pair-density.v1"); hash.text(receipt_.source_identity);
    for(Count k=0;k<nk;++k) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto x=d[(k*n+i)*n+j]; finite(x); require_finite(std::abs(x));
        scale=std::max(scale,std::abs(x));
        r.density_hermiticity_residual=std::max(r.density_hermiticity_residual,std::abs(x-std::conj(d[(k*n+j)*n+i])));
        hash.number(x.real()); hash.number(x.imag());
    }
    require_finite(r.density_hermiticity_residual);
    if(r.density_hermiticity_residual>1e-12*std::max(1.,scale))
        throw std::invalid_argument("MANTA periodic pair densities must be Hermitian");
    r.density_identity=hash.hash.finish_hex();
    out.moments=md::Matrix::Zero(entries_.size(),plan_.moments); out.pair_coefficient_l1.resize(entries_.size());
    std::vector<Complex> phases(nk); long double norm=0,absolute_charge=0; double charge_error=0;
    // Sun et al., JCP147,164119 (2017), doi:10.1063/1.4998644,
    // Eqs.16-17: exp(+ik.R) AO products and the q=0 reciprocal contraction.
    // (a,b,R) and (b,a,-R) differ by a lattice translation. For a periodic
    // potential their Hermitian density coefficients add to 2*Re(D*exp(-ik.R)).
    // This does not relate densities at different momenta or assume TR symmetry.
    for(Count p=0;p<entries_.size();++p) {
        const auto& e=entries_[p]; const auto panel=moment_panel(p);
        ++r.shell_calls; r.primitive_pairs+=panel.receipt.primitive_pairs;
        for(Count k=0;k<nk;++k) { phases[k]=std::conj(phase(momenta_,k,e.translation)); ++r.phase_products; }
        std::vector<long double> sums(plan_.moments),correction(plan_.moments); long double pair_norm=0;
        for(Count i=0;i<e.sizes[0];++i) for(Count j=0;j<e.sizes[1];++j) {
            double density_scale=0;
            for(Count k=0;k<nk;++k) {
                const auto value=d[(k*n+e.offsets[0]+i)*n+e.offsets[1]+j];
                density_scale=std::max(density_scale,std::max(std::abs(value.real()),std::abs(value.imag())));
                ++r.contraction_terms;
            }
            long double coefficient=0,error=0;
            for(Count k=0;k<nk && density_scale>0;++k) {
                const auto value=d[(k*n+e.offsets[0]+i)*n+e.offsets[1]+j];
                // Normalize before weighting, also where long double has the
                // binary64 range. Equal subnormal densities must not vanish
                // merely because each individual w_k*D rounded to zero.
                const long double x=static_cast<long double>(weights_[k])*
                    (static_cast<long double>(value.real()/density_scale)*phases[k].real()
                     -static_cast<long double>(value.imag()/density_scale)*phases[k].imag());
                accumulate(x,coefficient,error); ++r.contraction_terms;
            }
            coefficient*=e.paired?2:1; pair_norm+=upper_product(std::abs(coefficient),density_scale);
            for(Count m=0;m<plan_.moments;++m) {
                const long double value=scaled_product(coefficient,panel.values(i*e.sizes[1]+j,m),density_scale);
                accumulate(value,sums[m],correction[m]); ++r.contraction_terms;
            }
        }
        for(Count m=0;m<plan_.moments;++m) out.moments(p,m)=static_cast<double>(sums[m]);
        out.pair_coefficient_l1[p]=upper(pair_norm); norm+=pair_norm;
        absolute_charge+=upper_product(out.pair_coefficient_l1[p],e.extent.full_moment_bounds[0]);
        accumulate(out.moments(p,0),r.charge,charge_error);
    }
    if(!out.moments.allFinite()) throw std::overflow_error("MANTA periodic pair contracted moment overflow");
    require_finite(r.charge); r.coefficient_l1=upper(norm); r.absolute_charge_bound=upper(absolute_charge);
    return out;
}

PeriodicPairAdjoint PeriodicPairSource::adjoint(const md::Matrix& local) const {
    if(local.rows()!=static_cast<Eigen::Index>(entries_.size()) || local.cols()!=static_cast<Eigen::Index>(plan_.moments) || !local.allFinite())
        throw std::invalid_argument("MANTA periodic pair local potentials must be finite and match (pairs,moments)");
    const Count n=plan_.n_basis,nk=plan_.momenta;
    PeriodicPairAdjoint out; out.n_basis=n; out.momenta=nk; out.source_identity=receipt_.source_identity;
    Hash hash; hash.text("manta.periodic-pair-potential.v1"); hash.text(receipt_.source_identity); hash.matrix(local);
    out.potential_identity=hash.hash.finish_hex();
    out.values.resize(nk*n*n); std::vector<Complex> correction(out.values.size()),phases(nk);
    for(Count p=0;p<entries_.size();++p) {
        const auto& e=entries_[p]; const auto panel=moment_panel(p);
        ++out.shell_calls; out.primitive_pairs+=panel.receipt.primitive_pairs;
        for(Count k=0;k<nk;++k) { phases[k]=phase(momenta_,k,e.translation); ++out.phase_products; }
        for(Count i=0;i<e.sizes[0];++i) for(Count j=0;j<e.sizes[1];++j) {
            double value=0,error=0;
            for(Count m=0;m<plan_.moments;++m) {
                const double x=local(p,m)*panel.values(i*e.sizes[1]+j,m); require_finite(x);
                accumulate(x,value,error); ++out.contraction_terms;
            }
            require_finite(value);
            for(Count k=0;k<nk;++k) {
                const Count ij=(k*n+e.offsets[0]+i)*n+e.offsets[1]+j;
                const Complex contribution=phases[k]*value;
                accumulate(contribution,out.values[ij],correction[ij]); ++out.contraction_terms;
                if(e.paired) {
                    const Count ji=(k*n+e.offsets[1]+j)*n+e.offsets[0]+i;
                    accumulate(std::conj(contribution),out.values[ji],correction[ji]); ++out.contraction_terms;
                }
            }
        }
    }
    double scale=0;
    for(Count k=0;k<nk;++k) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto x=out.values[(k*n+i)*n+j]; finite(x); require_finite(std::abs(x)); scale=std::max(scale,std::abs(x));
        const double residual=std::abs(x-std::conj(out.values[(k*n+j)*n+i])); require_finite(residual);
        out.hermiticity_residual=std::max(out.hermiticity_residual,residual);
    }
    if(out.hermiticity_residual>1e-11*std::max(1.,scale))
        throw std::runtime_error("MANTA periodic pair adjoint violates Hermiticity");
    return out;
}
} // namespace vibeqc::manta
