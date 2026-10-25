#include "vibeqc/manta_periodic_semilocal_xc.hpp"
#include "manta_periodic_xc_source.hpp"
#include "manta_periodic_xc_component.hpp"
#include <xc.h>
#include <map>

namespace vibeqc::manta {
namespace {
using namespace periodic_xc_detail;
using Kind=periodic_xc_component::Kind;
void admit_functional(const Functional& f,Kind kind) {
    for(const auto& component:f.components()) require_finite(component.second);
    if(f.is_external() || f.kind()==XCKind::MGGA || (kind==Kind::Pure && f.is_hybrid())
       || (kind!=Kind::RangeSeparated && f.is_range_separated())
       || f.is_double_hybrid() || f.needs_vv10())
        throw std::invalid_argument(kind==Kind::RangeSeparated ? "MANTA range-separated KS requires complete native LDA/GGA components" : kind==Kind::GlobalHybrid ? "MANTA global hybrid KS requires a complete native global LDA/GGA hybrid"
            : "MANTA semilocal XC requires a complete native pure LDA/GGA functional");
    const auto components=f.component_capabilities();
    if(components.empty()) throw std::invalid_argument("MANTA semilocal XC requires nonempty functional components");
    for(const auto& c:components) {
        const auto& kind=std::get<1>(c);
        if((kind!="exchange" && kind!="correlation" && kind!="exchange-correlation")
           || !std::get<2>(c) || !std::get<3>(c) || !std::get<4>(c) || std::get<5>(c))
            throw std::invalid_argument("MANTA semilocal XC component lacks a 3D exchange/correlation energy and potential");
    }
}
void caps(const PeriodicSemilocalXCPlan& p,const ResourceBudget& budget) {
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA semilocal XC memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA semilocal XC shell-call budget exceeded");
    cap(p.primitive_points,budget.primitive_quartets,"MANTA semilocal XC primitive-point budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA semilocal XC scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA semilocal XC kernel-work budget exceeded");
}
}
double periodic_xc_component::global_hybrid_fraction(const Functional& f) {
    admit_functional(f,Kind::GlobalHybrid);
    // Resolve method-owned components, never infer fractions from an alias.
    double fraction=0.;
    for(const auto& [kernel,coefficient,range]:f.exchange_components()) {
        if(kernel!="coulomb" || !std::isfinite(coefficient) || !std::isfinite(range) || range!=0.)
            throw std::invalid_argument("MANTA global hybrid KS requires finite full-Coulomb exchange components");
        fraction+=coefficient;
    }
    if(!std::isfinite(fraction) || fraction==0. || !std::isfinite(f.cam_alpha())
       || std::abs(fraction-f.cam_alpha())>1e-12 || f.cam_beta()!=0. || !f.is_hybrid())
        throw std::invalid_argument("MANTA global hybrid KS requires a nonzero resolved global exchange fraction");
    return fraction;
}
std::vector<std::tuple<std::string,double,double>> periodic_xc_component::range_separated_components(const Functional& f) {
    admit_functional(f,Kind::RangeSeparated);
    // Functional owns the exact-exchange decomposition. Libxc's CAM form is
    // a/r+b*erfc(omega*r)/r; retain distinct ranges instead of using aggregate
    // CAM metadata, which cannot represent a mixture of different omegas.
    double full=0.;std::map<double,double> screened;bool saw_range=false;
    for(const auto& [kernel,coefficient,omega]:f.exchange_components()) {
        require_finite(coefficient);require_finite(omega);
        if(omega<0 || (kernel=="coulomb" && omega!=0)
           || (kernel!="coulomb" && kernel!="erfc" && kernel!="erf"))
            throw std::invalid_argument("MANTA range-separated KS requires finite Coulomb/erf/erfc exchange components");
        if(kernel=="coulomb") full+=coefficient;
        else {
            saw_range=true;
            if(omega==0) { if(kernel=="erfc") full+=coefficient; }
            else if(kernel=="erfc") screened[omega]+=coefficient;
            else { full+=coefficient;screened[omega]-=coefficient; }
        }
        require_finite(full);
    }
    if(!saw_range) throw std::invalid_argument("MANTA range-separated KS requires a resolved screened exchange component");
    std::vector<std::tuple<std::string,double,double>> out;
    if(full!=0.) out.emplace_back("coulomb",full,0.);
    for(const auto& [omega,coefficient]:screened) {
        require_finite(coefficient);
        if(coefficient!=0.) out.emplace_back("erfc",coefficient,omega);
    }
    return out;
}
PeriodicSemilocalXCPlan periodic_xc_component::plan(const BasisSet& basis,const Functional& f,
    Count ni,Count nk,Count ng,Count batch,const ResourceBudget& budget,Kind kind) {
    using namespace periodic_xc_detail;
    if(kind==Kind::GlobalHybrid) global_hybrid_fraction(f);
    else if(kind==Kind::RangeSeparated) range_separated_components(f);
    else admit_functional(f,Kind::Pure);
    auto unlimited=budget;
    unlimited.memory_bytes=unlimited.shell_calls=unlimited.primitive_quartets=unlimited.scalar_terms=unlimited.kernel_terms=std::numeric_limits<Count>::max();
    PeriodicSemilocalXCPlan p;
    static_cast<PeriodicLocalExchangePlan&>(p)=plan_periodic_local_exchange(basis,ni,nk,ng,f.spin(),batch,unlimited);
    p.gradients=f.kind()==XCKind::GGA; p.functional_components=f.components().size();
    // libxc vector API uses an int point count. Keep all component work in a
    // bounded panel. Conservative allowance includes its transient interleaved
    // spin arrays, returned vectors, functional metadata and per-component
    // backend state; it is not a process RSS or allocator certificate.
    cap(p.grid_batch,std::numeric_limits<int>::max(),"MANTA semilocal XC batch exceeds libxc indexing");
    p.functional_workspace_bytes=add(mul(4,f.name().size()),add(mul(1024,p.grid_batch),mul(Count{1}<<20,add(p.functional_components,1))));
    const Count extra=p.gradients?mul(3,p.panel_bytes):0;
    p.panel_bytes=add(p.panel_bytes,extra);
    p.modeled_peak_bytes=add(p.modeled_peak_bytes,add(extra,p.functional_workspace_bytes));
    const Count functional_work=mul(100000,mul(ng,p.functional_components));
    p.scalar_terms=add(mul(p.gradients?8:2,p.scalar_terms),functional_work);
    p.kernel_terms=add(mul(p.gradients?8:2,p.kernel_terms),functional_work);
    caps(p,budget); return p;
}
PeriodicSemilocalXCResult periodic_xc_component::compute(const BasisSet& basis,const Functional& f,
    const Eigen::MatrixXd& lattice,const Eigen::MatrixXd& images,const Eigen::MatrixXd& k,
    const std::vector<double>& weights,const Eigen::MatrixXd& points,const std::vector<double>& qw,
    const std::vector<std::complex<double>>& d,Count batch,const ResourceBudget& budget,Kind kind) {
    using namespace periodic_xc_detail;
    PeriodicSemilocalXCResult result;
    result.plan=plan(basis,f,images.rows(),k.rows(),points.rows(),batch,budget,kind);
    const auto& p=result.plan;auto& r=result.receipt;
    const Count n=p.n_basis,nk=p.momenta,ng=p.grid_points,ns=p.spins,e=mul(mul(ns,nk),square(n));
    const auto shifts=initialize(basis,lattice,images,k,weights,points,qw,d,ns,p,r,"manta.periodic-semilocal-xc.geometry.v1");
    r.algorithm="finite-complex-bloch-semilocal-xc";r.functional=f.name();r.backend_version=xc_version_string();
    r.convention="dE=sum_s,k w_k Re Tr(dD_s,k V_s,k); LDA/GGA density-gradient adjoint";
    Digest source;source.text("manta.periodic-semilocal-xc.source.v1");source.text(r.source_identity);
    source.text(r.functional);source.text(r.backend_version);source.word(p.gradients);
    for(const auto& component:f.components()) {
        r.component_ids.push_back(component.first);r.component_weights.push_back(component.second);
        source.word(component.first);source.real(component.second);
    }
    if(kind==Kind::GlobalHybrid) {
        r.semilocal_component_only=true;r.required_exact_exchange_fraction=global_hybrid_fraction(f);
        r.algorithm="finite-complex-bloch-global-hybrid-semilocal-component";
        source.text("semilocal-component-only;requires-full-coulomb-exchange");
        source.real(r.required_exact_exchange_fraction);
    } else if(kind==Kind::RangeSeparated) {
        r.semilocal_component_only=true;r.required_exchange_components=range_separated_components(f);
        r.algorithm="finite-complex-bloch-range-separated-semilocal-component";
        source.text("semilocal-component-only;requires-resolved-exchange-kernels");
        for(const auto& [kernel,coefficient,omega]:r.required_exchange_components) {
            source.text(kernel);source.real(coefficient);source.real(omega);
        }
    }
    r.source_identity=source.hash.finish_hex();
    Digest density;density.text(r.source_identity);for(auto z:d) density.value(z);r.density_identity=density.hash.finish_hex();
    r.minimum_density=std::numeric_limits<double>::infinity();result.values.assign(e,0.);
    std::vector<Complex> errors(e,0.);std::vector<Sum> electrons(ns);Sum energy;
    for(Count start=0;start<ng;start+=p.grid_batch) {
        const Count b=std::min(p.grid_batch,ng-start);
        const auto panel=evaluate_panel(basis,images,k,points,shifts,start,b,p.gradients,r);
        std::array<Eigen::VectorXd,2> rho,vr,active;
        std::array<std::array<Eigen::VectorXd,3>,2> grad,flow;
        for(Count spin=0;spin<ns;++spin) {
            rho[spin].resize(b);active[spin]=Eigen::VectorXd::Ones(b);
            if(p.gradients) for(int axis=0;axis<3;++axis) { grad[spin][axis].resize(b);flow[spin][axis].resize(b); }
            for(Count g=0;g<b;++g) {
                Sum value,absolute;std::array<Sum,3> derivative;
                for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                    const Count left=(g*nk+ik)*n+i,right=(g*nk+ik)*n+j;
                    const auto density_value=weights[ik]*d[((spin*nk+ik)*n+i)*n+j];
                    const auto term=panel.values[left]*density_value*std::conj(panel.values[right]);
                    finite(term);value.add(term.real());absolute.add(std::abs(term));
                    if(p.gradients) for(int axis=0;axis<3;++axis) {
                        const auto x=density_value*(panel.gradients[axis][left]*std::conj(panel.values[right])
                            +panel.values[left]*std::conj(panel.gradients[axis][right]));
                        finite(x);derivative[axis].add(x.real());
                    }
                }
                r.minimum_density=std::min(r.minimum_density,value.value);
                if(value.value < -1e-12*std::max(1.,absolute.value))
                    throw std::invalid_argument("MANTA semilocal XC requires nonnegative grid spin density");
                if(value.value<0) { ++r.rounded_negative_points;active[spin][g]=0.; }
                rho[spin][g]=std::max(0.,value.value);electrons[spin].add(qw[start+g]*value.value);
                if(p.gradients) for(int axis=0;axis<3;++axis) grad[spin][axis][g]=active[spin][g]*derivative[axis].value;
            }
        }
        std::array<Eigen::VectorXd,3> sigma,vs;Eigen::VectorXd exc;
        if(p.gradients) {
            for(auto& x:sigma) x=Eigen::VectorXd::Zero(b);
            for(int axis=0;axis<3;++axis) {
                sigma[0].array()+=grad[0][axis].array().square();
                if(ns==2) {
                    sigma[1].array()+=grad[0][axis].array()*grad[1][axis].array();
                    sigma[2].array()+=grad[1][axis].array().square();
                }
            }
            for(const auto& x:sigma) if(!x.allFinite()) throw std::overflow_error("MANTA XC gradient invariant overflow");
        }
        // libxc supplies integrand energy and derivatives (Lehtola et al.,
        // SoftwareX 7, 1 (2018), doi:10.1016/j.softx.2017.11.002).
        if(ns==1) f.eval_unpolarised(rho[0],sigma[0],exc,vr[0],vs[0]);
        else f.eval_polarised(rho[0],rho[1],sigma[0],sigma[1],sigma[2],exc,vr[0],vr[1],vs[0],vs[1],vs[2]);
        if(exc.size()!=static_cast<Eigen::Index>(b) || !exc.allFinite()) throw std::runtime_error("MANTA XC backend energy is incomplete or nonfinite");
        for(Count spin=0;spin<ns;++spin) if(vr[spin].size()!=static_cast<Eigen::Index>(b) || !vr[spin].allFinite())
            throw std::runtime_error("MANTA XC backend density potential is incomplete or nonfinite");
        if(p.gradients) {
            for(Count channel=0;channel<(ns==1?1:3);++channel) if(vs[channel].size()!=static_cast<Eigen::Index>(b) || !vs[channel].allFinite())
                throw std::runtime_error("MANTA XC backend gradient potential is incomplete or nonfinite");
            for(int axis=0;axis<3;++axis) {
                flow[0][axis]=2*vs[0].array()*grad[0][axis].array();
                if(ns==2) {
                    flow[0][axis].array()+=vs[1].array()*grad[1][axis].array();
                    flow[1][axis]=2*vs[2].array()*grad[1][axis].array()+vs[1].array()*grad[0][axis].array();
                }
            }
        }
        for(Count g=0;g<b;++g) {
            energy.add(qw[start+g]*exc[g]);
            for(Count spin=0;spin<ns;++spin) for(Count ik=0;ik<nk;++ik) for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                const Count left=(g*nk+ik)*n+i,right=(g*nk+ik)*n+j,index=((spin*nk+ik)*n+i)*n+j;
                Complex term=vr[spin][g]*std::conj(panel.values[left])*panel.values[right];
                if(p.gradients) for(int axis=0;axis<3;++axis)
                    term+=flow[spin][axis][g]*(std::conj(panel.gradients[axis][left])*panel.values[right]
                        +std::conj(panel.values[left])*panel.gradients[axis][right]);
                term*=qw[start+g]*active[spin][g];finite(term);
                const auto y=term-errors[index],next=result.values[index]+y;
                errors[index]=(next-result.values[index])-y;result.values[index]=next;finite(next);
            }
        }
        ++r.panel_calls;++r.functional_batches;
    }
    r.energy=energy.value;for(auto x:electrons) r.electrons_grid.push_back(x.value);r.modeled_peak_bytes=p.modeled_peak_bytes;
    Digest snapshot;snapshot.text(r.density_identity);snapshot.real(r.energy);
    for(double x:r.electrons_grid) snapshot.real(x);for(auto z:result.values) snapshot.value(z);
    r.snapshot_identity=snapshot.hash.finish_hex();return result;
}
PeriodicSemilocalXCPlan plan_periodic_semilocal_xc(const BasisSet& basis,const Functional& f,
    Count ni,Count nk,Count ng,Count batch,const ResourceBudget& budget) {
    return periodic_xc_component::plan(basis,f,ni,nk,ng,batch,budget,Kind::Pure);
}
PeriodicSemilocalXCResult compute_periodic_semilocal_xc(const BasisSet& basis,const Functional& f,
    const Eigen::MatrixXd& lattice,const Eigen::MatrixXd& images,const Eigen::MatrixXd& k,
    const std::vector<double>& weights,const Eigen::MatrixXd& points,const std::vector<double>& qw,
    const std::vector<std::complex<double>>& d,Count batch,const ResourceBudget& budget) {
    return periodic_xc_component::compute(basis,f,lattice,images,k,weights,points,qw,d,batch,budget,Kind::Pure);
}
} // namespace vibeqc::manta
