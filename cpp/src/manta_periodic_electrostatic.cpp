#include "vibeqc/manta_periodic_electrostatic.hpp"
#include "vibeqc/manta_one_electron.hpp"
#include "vibeqc/manta_fourier.hpp"
#include "manta_ewald_internal.hpp"
#include "manta_pair_image_internal.hpp"

namespace vibeqc::manta {
using namespace ewald_detail;
// Ewald (1921), IV.1 Eqs.37-40, DOI 10.1002/andp.19213690304.
// All three charge components share one finite kernel. Its constant already
// supplies -pi*Q_total^2/(2*eta^2*volume); never add a second background term.
namespace {
using namespace ewald_detail;
constexpr Real neg=-std::numeric_limits<Real>::infinity();
Real logadd(Real a,Real b) {
    if(a==neg) return b; if(b==neg) return a;
    const auto hi=std::max(a,b); return hi+std::log1p(std::exp(std::min(a,b)-hi));
}
Real logarithm(double x) { return x?std::log(Real(x)):neg; }
double upper(Real log) {
    if(log==neg) return 0;
    const auto x=std::nextafter(number(std::exp(log)*(1+8192*std::numeric_limits<double>::epsilon())),
        std::numeric_limits<double>::infinity());
    if(!std::isfinite(x)) throw std::overflow_error("MANTA electrostatic bound exceeds binary64");
    return x;
}
double add_bound(double a,double b) { return upper(logadd(logarithm(a),logarithm(b))); }
double product(double a,double b) { return a && b?upper(logarithm(a)+logarithm(b)):0; }
const PeriodicBupoSource& required(const std::shared_ptr<const PeriodicBupoSource>& p) {
    if(!p) throw std::invalid_argument("MANTA electrostatics requires a native periodic BUPO source");
    return *p;
}
double trace(const std::vector<Complex>& d,const std::vector<Complex>& a,Count n,const std::vector<double>& w) {
    Real sum=0,error=0;
    for(Count k=0;k<w.size();++k) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
        const auto x=d[(k*n+j)*n+i],y=a[(k*n+i)*n+j];
        accumulate(Real(w[k])*(Real(x.real())*y.real()-Real(x.imag())*y.imag()),sum,error);
    }
    return number(sum);
}
void scatter_error(const PeriodicPairEntry& e,double error,md::Matrix& m) {
    for(Count i=0;i<e.sizes[0];++i) for(Count j=0;j<e.sizes[1];++j) {
        auto& a=m(e.offsets[0]+i,e.offsets[1]+j); a=add_bound(a,error);
        if(e.paired) { auto& b=m(e.offsets[1]+j,e.offsets[0]+i); b=add_bound(b,error); }
    }
}
double omitted_charge(const std::vector<Complex>& d,Count n,const std::vector<double>& w,const md::Matrix& omitted) {
    Real sum=neg;
    for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) if(omitted(i,j)) for(Count k=0;k<w.size();++k) if(w[k]) {
        const auto a=d[(k*n+i)*n+j],b=d[(k*n+j)*n+i];
        const Real norm=std::max(std::hypot(Real(a.real()),Real(a.imag())),std::hypot(Real(b.real()),Real(b.imag())));
        if(norm) sum=logadd(sum,logarithm(omitted(i,j))+logarithm(w[k])+std::log(norm));
    }
    return upper(sum);
}
double point_kernel(const pair_image_detail::Envelope& image,const gaussian_domain_detail::Geometry& g) {
    const Real logalpha=logarithm(image.receipt.maximum_gaussian_exponent),logs=logarithm(g.metric);
    // A Gaussian convolved with a point has beta=alpha, without the factor
    // one half used for the Gaussian/Gaussian uniform interaction bound.
    const Real real=std::log(pi)-2*logarithm(g.eta)
        +3*logadd(std::log(2.L)+.5L*(logalpha-std::log(pi)),-logs);
    return upper(logadd(real,logadd(logarithm(image.receipt.reciprocal_kernel_bound),logarithm(image.receipt.gauge_kernel_bound))));
}
} // namespace

PeriodicElectrostaticPlan PeriodicElectrostaticSource::plan_source(const PeriodicBupoSource& b,Count np,
    const PeriodicElectrostaticOptions& o,const ResourceBudget& budget) {
    margin(budget); require_finite(o.charge_tolerance);
    if((o.charge_policy!="neutral" && o.charge_policy!="uniform-background") || o.charge_tolerance<=0)
        throw std::invalid_argument("MANTA electrostatic charge policy requires neutral or uniform-background and a positive tolerance");
    if(!b.domain_receipt_ || !b.image_envelope_)
        throw std::invalid_argument("MANTA electrostatics requires both infinite-domain and pair-image bounds");
    const auto& direct=*b.direct_; const auto& s=*direct.source_; const auto& d=direct.plan_;
    PeriodicElectrostaticPlan p; p.n_basis=b.plan_.n_basis; p.momenta=b.plan_.momenta;p.points=np;p.pairs=b.plan_.pairs;
    const Count dn=mul(p.momenta,square(p.n_basis));
    p.nuclear_shell_calls=mul(p.pairs,d.real_cells); p.overlap_shell_calls=p.pairs;
    p.fourier_shell_calls=mul(p.pairs,d.reciprocal_batches); p.point_pair_terms=mul(square(np),d.real_cells);
    p.shell_calls=add(b.plan_.shell_calls,add(p.nuclear_shell_calls,add(p.overlap_shell_calls,p.fourier_shell_calls)));
    const auto leaf=unlimited(budget); OneElectronRequest req;req.operation="potential";req.kernel="erfc";req.omega=1.;
    req.charges={{1.,{0.,0.,0.}}}; const auto one=plan_one_electron(s.basis_,s.basis_,req,leaf);
    const auto fourier=plan_fourier(s.basis_,&s.basis_,std::max<Count>(1,d.reciprocal_batch),leaf);
    const Count bounds=add(square(np),mul(np,p.pairs));
    p.primitive_products=add(b.plan_.primitive_products,mul(one.primitive_pairs,
        add(add(mul(p.nuclear_shell_calls,std::max<Count>(1,np)),p.overlap_shell_calls),add(p.fourier_shell_calls,bounds))));
    p.scalar_terms=add(b.plan_.scalar_terms,add(mul(256,dn),add(mul(256,mul(bounds,one.primitive_pairs)),
        mul(128,add(p.point_pair_terms,add(mul(np,d.reciprocal_vectors),mul(p.pairs,mul(d.reciprocal_vectors,square(census(s.basis_,6).maximum_size)))))))));
    p.kernel_terms=add(b.plan_.kernel_terms,add(mul(one.kernel_terms,mul(p.nuclear_shell_calls,std::max<Count>(1,np))),
        add(mul(p.fourier_shell_calls,fourier.kernel_terms),mul(128,p.scalar_terms-b.plan_.scalar_terms))));
    p.modeled_peak_bytes=add(budget.other_live_bytes,add(b.plan_.modeled_peak_bytes,add(std::max(one.modeled_peak_bytes,fourier.modeled_peak_bytes),
        add(mul(256,dn),add(mul(64,square(p.n_basis)),add(mul(256,np),add(mul(128,d.reciprocal_batch),mul(64,p.pairs))))))));
    cap(p.shell_calls,budget.shell_calls,"MANTA electrostatic shell-call budget exceeded");
    cap(p.primitive_products,budget.primitive_quartets,"MANTA electrostatic primitive budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA electrostatic scalar budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA electrostatic work budget exceeded");
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA electrostatic memory budget exceeded");
    return p;
}

PeriodicElectrostaticSource::PeriodicElectrostaticSource(std::shared_ptr<const PeriodicBupoSource> b,
    const md::Matrix& positions,const std::vector<double>& charges,const PeriodicElectrostaticOptions& options,const ResourceBudget& budget)
    :plan_(plan_source(required(b),charges.size(),options,budget)),electronic_(std::move(b)),options_(options) {
    if(positions.cols()!=3 || Count(positions.rows())!=plan_.points || !positions.allFinite())
        throw std::invalid_argument("MANTA electrostatic point positions must be finite and match the charges");
    auto& r=receipt_; const auto& e=*electronic_; const auto& direct=*e.direct_; const auto& s=*direct.source_;
    const auto& geometry=*e.domain_geometry_; const Count n=plan_.n_basis,dn=plan_.momenta*n*n;
    r.electronic_source_identity=e.source_identity();r.pair_source_identity=s.receipt_.source_identity;r.kernel_identity=direct.kernel_identity();
    r.charge_policy=options.charge_policy;r.charge_tolerance=options.charge_tolerance;r.modeled_peak_bytes=plan_.modeled_peak_bytes;
    md::Charges points; points.reserve(plan_.points); Real charge=0,charge_error=0,absolute=neg,self=0,self_error=0;
    lattice_detail::Hash identity;identity.text("manta.electrostatic-points.v1;signed;point-self-removed");identity.text(r.kernel_identity);identity.word(plan_.points);
    for(Count i=0;i<plan_.points;++i) {
        require_finite(charges[i]);md::Point center{positions(i,0),positions(i,1),positions(i,2)};
        points.push_back({charges[i],center});identity.number(charges[i]);for(auto x:center)identity.number(x);
        accumulate(Real(charges[i]),charge,charge_error);
        if(charges[i])absolute=logadd(absolute,std::log(std::abs(Real(charges[i]))));
        accumulate(-Real(direct.receipt_.eta)/std::sqrt(pi)*charges[i]*charges[i],self,self_error);
    }
    r.point_source_identity=identity.value.finish_hex();r.nuclear_charge=number(charge);r.nuclear_absolute_charge=upper(absolute);r.nuclear_self=number(self);
    const Eigen::Matrix3d inverse=s.lattice_.inverse();
    for(Count a=0;a<plan_.points;++a) for(Count b=a+1;b<plan_.points;++b) if(charges[a] && charges[b]) {
        const Eigen::Vector3d delta=positions.row(a).transpose()-positions.row(b).transpose();
        Eigen::Vector3d fractional=inverse*delta;fractional-=fractional.array().round().matrix();
        const double separation=(s.lattice_*fractional).norm();
        const double guard=256*std::numeric_limits<double>::epsilon()*(s.lattice_.norm()+positions.row(a).norm()+positions.row(b).norm());
        if(!std::isfinite(separation) || separation<=guard)
            throw std::invalid_argument("MANTA electrostatics has coincident or unresolved periodic point charges");
    }
    std::array<Real,3> nn{},nn_error{};
    for(Count a=0;a<plan_.points;++a) for(Count b=0;b<plan_.points;++b) {
        if(!charges[a] || !charges[b])continue;
        const auto domain=gaussian_domain_detail::point_pair(geometry,points[a].second,points[b].second);
        const auto factor=std::abs(Real(charges[a]))*std::abs(Real(charges[b]));
        r.nuclear_domain_error_bound=add_bound(r.nuclear_domain_error_bound,
            factor?upper(logarithm(domain.total)+std::log(factor)-std::log(2.L)):0);
        ++r.point_pair_bounds;
        for(const auto& index:direct.real_) {
            if(a==b && index==LatticeIndex{})continue;
            const auto shift=vector(s.lattice_,index);Real distance=0;
            for(int axis=0;axis<3;++axis)distance=std::hypot(distance,Real(points[a].second[axis])-points[b].second[axis]+shift[axis]);
            if(distance<=0 || !std::isfinite(distance))throw std::overflow_error("MANTA nuclear point distance is not representable");
            accumulate(Real(.5)*charges[a]*charges[b]*std::erfc(Real(geometry.eta)*distance)/distance,nn[0],nn_error[0]);++r.point_pair_terms;
        }
    }
    overlap_.resize(dn);nuclear_.resize(dn);for(auto& v:attraction_)v.resize(dn);
    std::array<std::vector<Complex>,3> corrections;for(auto& v:corrections)v.resize(dn);
    std::vector<Complex> overlap_error(dn);pair_domain_error_.resize(plan_.pairs);
    domain_j_bound_=image_j_bound_=md::Matrix::Zero(n,n);
    PeriodicEwaldReceipt work;
    for(Count p=0;p<plan_.pairs;++p) {
        const auto& entry=s.entries_[p];const Count size=entry.sizes[0]*entry.sizes[1];
        const auto& a=s.basis_.libint()[entry.shells[0]];const auto ket=shifted(s.basis_.libint()[entry.shells[1]],entry.translation);
        std::vector<Real> v(size),error(size),overlap(size),gauge(size);
        for(const auto& index:direct.real_) {
            const auto shift=vector(s.lattice_,index);md::Charges translated;
            for(const auto& atom:points)if(atom.first) {
                auto moved=atom;for(int axis=0;axis<3;++axis)moved.second[axis]-=shift[axis];
                translated.push_back(moved);
            }
            const auto tile=md::shell_one_electron(a,ket,"potential",translated,{}, {},md::RadialKernel::Erfc,geometry.eta);
            if(!tile.allFinite())throw std::overflow_error("MANTA electrostatic attraction integral overflow");
            for(Count i=0;i<size;++i)accumulate(Real(tile.data()[i]),v[i],error[i]);++r.nuclear_shell_calls;
        }
        scatter(entry,n,s.momenta_,v,attraction_[0],corrections[0]);
        const auto sp=direct.overlap_panel(p,work);
        for(Count i=0;i<size;++i) {overlap[i]=sp.data()[i];gauge[i]=-Real(direct.receipt_.gauge_constant)*r.nuclear_charge*overlap[i];}
        scatter(entry,n,s.momenta_,overlap,overlap_,overlap_error);scatter(entry,n,s.momenta_,gauge,attraction_[2],corrections[2]);
        for(const auto& atom:points) if(atom.first) {
            const auto bound=gaussian_domain_detail::gaussian_point(geometry,e.majorants_[p],atom.second);
            pair_domain_error_[p]=add_bound(pair_domain_error_[p],product(bound.total,std::abs(atom.first)));++r.gaussian_point_bounds;
        }
        scatter_error(entry,pair_domain_error_[p],domain_j_bound_);
    }
    for(Count start=0;start<direct.reciprocal_.size();start+=direct.plan_.reciprocal_batch) {
        const Count count=std::min<Count>(direct.plan_.reciprocal_batch,direct.reciprocal_.size()-start);
        std::vector<double> lambda;const auto g=direct.reciprocal_tile(start,count,lambda);
        std::vector<Complex> rho(count);
        for(Count ig=0;ig<count;++ig) {
            Complex correction=0;
            for(const auto& atom:points) if(atom.first) {
                Real angle=0;for(int axis=0;axis<3;++axis)angle+=Real(g(ig,axis))*atom.second[axis];
                accumulate(atom.first*std::exp(Complex(0,-number(angle))),rho[ig],correction);
            }
            finite(rho[ig]);
            accumulate(Real(.5)*lambda[ig]*(Real(rho[ig].real())*rho[ig].real()+Real(rho[ig].imag())*rho[ig].imag()),nn[1],nn_error[1]);
        }
        for(Count p=0;p<plan_.pairs;++p) {
            const auto f=direct.fourier_panel(p,g,work);std::vector<Real> v(f.rows()),error(f.rows());
            for(Eigen::Index i=0;i<f.rows();++i)for(Count ig=0;ig<count;++ig)
                accumulate(-Real(lambda[ig])*(Real(f(i,ig).real())*rho[ig].real()+Real(f(i,ig).imag())*rho[ig].imag()),v[i],error[i]);
            scatter(s.entries_[p],n,s.momenta_,v,attraction_[1],corrections[1]);
        }
    }
    r.overlap_shell_calls=work.overlap_shell_calls;r.fourier_shell_calls=work.fourier_shell_calls;
    r.nuclear_real=number(nn[0]);r.nuclear_reciprocal=number(nn[1]);
    r.nuclear_gauge=number(Real(.5)*direct.receipt_.gauge_constant*r.nuclear_charge*r.nuclear_charge);
    r.nuclear_energy=number(nn[0]+nn[1]+Real(r.nuclear_self)+r.nuclear_gauge);
    r.point_kernel_bound=point_kernel(*e.image_envelope_,geometry);
    for(Count i=0;i<dn;++i) {nuclear_[i]=attraction_[0][i]+attraction_[1][i]+attraction_[2][i];finite(nuclear_[i]);}
    for(Count i=0;i<n;++i)for(Count j=0;j<n;++j)
        image_j_bound_(i,j)=product(product(r.point_kernel_bound,r.nuclear_absolute_charge),e.image_envelope_->omitted(i,j));
    lattice_detail::Hash h;h.text("manta.periodic-electrostatics.v1;shared-finite-kernel;full-source-errors;point-self-only");
    h.text(r.electronic_source_identity);h.text(r.point_source_identity);h.text(r.charge_policy);h.number(r.charge_tolerance);
    r.source_identity=h.value.finish_hex();
}

PeriodicElectrostaticResult PeriodicElectrostaticSource::build(const std::vector<Complex>& d) const {
    const auto& e=*electronic_;const auto& direct=*e.direct_;const auto& s=*direct.source_;
    PeriodicEwaldReceipt checked;validate_density(s,d,checked);
    PeriodicElectrostaticResult out;out.plan=plan_;out.receipt=receipt_;auto& r=out.receipt;
    const Count n=plan_.n_basis;r.density_identity=checked.density_identity;
    r.electron_charge=trace(d,overlap_,n,s.weights_);r.retained_net_charge=number(Real(r.nuclear_charge)-r.electron_charge);
    r.charge_uncertainty_bound=omitted_charge(d,n,s.weights_,e.image_envelope_->omitted);
    r.maximum_net_charge_bound=add_bound(std::abs(r.retained_net_charge),r.charge_uncertainty_bound);
    if(options_.charge_policy=="neutral" && r.maximum_net_charge_bound>options_.charge_tolerance)
        throw std::invalid_argument("MANTA neutral electrostatics requires net charge plus omitted-image uncertainty within tolerance; enlarge the pair source or select uniform-background");
    if(options_.charge_policy=="uniform-background")r.background_charge=-r.retained_net_charge;
    out.electronic=e.build(d);r.electronic_energy=out.electronic.receipt.energy;
    r.attraction_real=trace(d,attraction_[0],n,s.weights_);r.attraction_reciprocal=trace(d,attraction_[1],n,s.weights_);
    r.attraction_gauge=trace(d,attraction_[2],n,s.weights_);
    r.attraction_energy=number(Real(r.attraction_real)+r.attraction_reciprocal+r.attraction_gauge);
    r.energy=number(Real(r.electronic_energy)+r.attraction_energy+r.nuclear_energy);
    r.trace_energy=number(Real(.5)*trace(d,out.electronic.values,n,s.weights_)+trace(d,nuclear_,n,s.weights_)+r.nuclear_energy);
    r.electronic_gauge=number(Real(.5)*direct.receipt_.gauge_constant*r.electron_charge*r.electron_charge);
    r.total_gauge=number(Real(.5)*direct.receipt_.gauge_constant*r.retained_net_charge*r.retained_net_charge);
    for(Count p=0;p<plan_.pairs;++p) {
        Real norm=neg;
        for(const auto& c:coefficients(s.entries_[p],n,s.momenta_,s.weights_,d))
            if(c.normalized!=0 && c.scale!=0)norm=logadd(norm,std::log(std::abs(c.normalized))+logarithm(c.scale));
        r.attraction_domain_error_bound=add_bound(r.attraction_domain_error_bound,upper(norm+logarithm(pair_domain_error_[p])));
    }
    r.attraction_image_error_bound=product(product(r.point_kernel_bound,r.nuclear_absolute_charge),r.charge_uncertainty_bound);
    r.energy_error_bound=add_bound(out.electronic.receipt.energy_error_bound,add_bound(r.nuclear_domain_error_bound,
        add_bound(r.attraction_domain_error_bound,r.attraction_image_error_bound)));
    out.nuclear_j_error_bound=domain_j_bound_;out.j_error_bound=out.electronic.j_error_bound;
    for(Eigen::Index i=0;i<out.j_error_bound.size();++i) {
        out.nuclear_j_error_bound.data()[i]=add_bound(out.nuclear_j_error_bound.data()[i],image_j_bound_.data()[i]);
        out.j_error_bound.data()[i]=add_bound(out.j_error_bound.data()[i],out.nuclear_j_error_bound.data()[i]);
    }
    r.maximum_j_error_bound=out.j_error_bound.maxCoeff();out.values=out.electronic.values;
    for(Count i=0;i<out.values.size();++i) {out.values[i]+=nuclear_[i];finite(out.values[i]);}
    double scale=0;
    for(Count k=0;k<plan_.momenta;++k)for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
        const auto x=out.values[(k*n+i)*n+j];scale=std::max(scale,std::abs(x));
        r.hermiticity_residual=std::max(r.hermiticity_residual,std::abs(x-std::conj(out.values[(k*n+j)*n+i])));
    }
    if(!std::isfinite(scale) || r.hermiticity_residual>1e-11*std::max(1.,scale))throw std::runtime_error("MANTA electrostatic response violates Hermiticity");
    return out;
}
} // namespace vibeqc::manta
