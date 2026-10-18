#include "vibeqc/manta_periodic_ks.hpp"
#include "manta_periodic_hf_source.hpp"
#include "manta_periodic_xc_component.hpp"
#include <memory>

namespace vibeqc::manta {
namespace {
using XCKind=periodic_xc_component::Kind;
struct HybridControls {
    std::string exchange_q0;
    std::optional<double> madelung_alpha;
    double madelung_tolerance;
};
template<class Options> HybridControls hybrid_controls(const Options& o) {
    return {o.exchange_q0,o.madelung_alpha,o.madelung_tolerance};
}
struct FittingControls {
    const BasisSet& auxiliary;
    const PeriodicGDFShortRangeOptions& fitting;
    const RIOptions& ri;
    const PeriodicRIAccuracyOptions& accuracy;
    bool make_j=true,make_k=false;
};
template<class Options> FittingControls fitting_controls(const BasisSet& auxiliary,const Options& o) {
    if((o.coulomb_algorithm!="ri" && o.coulomb_algorithm!="exact") ||
       (o.exchange_algorithm!="ri" && o.exchange_algorithm!="exact") ||
       (o.coulomb_algorithm!="ri" && o.exchange_algorithm!="ri"))
        throw std::invalid_argument("MANTA RI hybrid KS requires exact/ri J and K with at least one RI target");
    return {auxiliary,o.fitting,o.ri,o.ri_accuracy,o.coulomb_algorithm=="ri",o.exchange_algorithm=="ri"};
}
template<class Snapshot> struct Trace {
    std::vector<Snapshot>& rows;
    Count capacity;
    static bool append(const Snapshot& snapshot,void* pointer) {
        auto& self=*static_cast<Trace*>(pointer);
        if(self.rows.size()>=self.capacity || snapshot.evaluated_iteration!=self.rows.size()+1)
            throw std::logic_error("MANTA periodic KS trace exceeded admitted extent");
        self.rows.push_back(snapshot); return true;
    }
};
Count validate(const BasisSet& basis,const PeriodicSystem& system,const Functional& f,const PeriodicKSOptions& o) {
    if(system.dim!=3 || system.charge!=0 || system.multiplicity<1 || system.unit_cell.empty())
        throw std::invalid_argument("MANTA periodic KS requires neutral all-electron 3D nuclei and positive multiplicity");
    if(!std::isfinite(o.confirmation_fock_tolerance) || o.confirmation_fock_tolerance<=0
       || !std::isfinite(o.confirmation_energy_tolerance) || o.confirmation_energy_tolerance<=0)
        throw std::invalid_argument("MANTA periodic KS confirmation tolerances must be finite positive");
    Count electrons=0;
    for(const auto& atom:system.unit_cell) {
        if(atom.Z<=0) throw std::invalid_argument("MANTA periodic KS requires positive all-electron nuclear charges");
        electrons=add(electrons,static_cast<Count>(atom.Z));
    }
    const Count unpaired=system.multiplicity-1;
    if(unpaired>electrons || (electrons-unpaired)%2 || add(electrons,unpaired)/2>=basis.nbasis()
       || (f.spin()==1 && system.multiplicity!=1))
        throw std::invalid_argument("MANTA periodic KS requires compatible spin populations, singlet RKS and one virtual per spin");
    return electrons;
}
struct Adapter {
    const BasisSet& basis; const PeriodicSystem& system; const Functional& functional;
    const md::Matrix &g,&cells,&k; const Eigen::MatrixXd &lattice,&images,&points,&xc_k;
    const std::vector<double> &weights,&qw; const std::vector<Complex>& h;
    const PeriodicKSOptions& options; const PeriodicKSPlan& plan;
    ResourceBudget budget; PeriodicKSReceipt& receipt;
    std::string source_identity;
    const md::Matrix* exchange_vectors=nullptr;
    const std::vector<Count>* transfer_offsets=nullptr;
    const std::vector<Complex>* overlap=nullptr;
    const md::Matrix* screened_vectors=nullptr;
    const std::vector<Count>* screened_offsets=nullptr;
    const PeriodicRIFockBuilder* fitted=nullptr;
    const std::vector<std::unique_ptr<PeriodicRIFockBuilder>>* range_fitted=nullptr;
    void add_exchange(const std::vector<Complex>& d,std::vector<Complex>& values) {
        if(!plan.exchange && !plan.range_separated && plan.exact_exchange_fraction==0.) return;
        // RKS supplies the TOTAL density: -a*K[D]/2 and -a*Tr(D K[D])/4.
        // UKS supplies each spin: -a*K[Ds] and -a*Tr(Ds K[Ds])/2.
        const double factor=plan.spins==1?2.:1.;
        const double scale=plan.long_range_exchange_fraction/factor;
        const bool fit_k=plan.fitting && plan.fitting->has_k;
        if(!fit_k) receipt.uncorrected_exchange_energy=0.;
        auto component=[&](const PeriodicExchangePlan& p,const std::string& kernel,double coefficient,double omega) {
            auto leaf=budget;leaf.other_live_bytes=plan.modeled_peak_bytes-p.modeled_peak_bytes;
            const bool screened=kernel=="erfc";
            auto exchange=compute_periodic_exchange(basis,system.lattice,
                screened?*screened_vectors:*exchange_vectors,screened?*screened_offsets:*transfer_offsets,cells,k,
                weights,d,plan.spins,options.reciprocal_batch,options.pair_cutoff_bohr,leaf,kernel,omega);
            double energy=0.;const double scale=coefficient/factor;
            for(double value:exchange.receipt.energies) energy+=scale*value;
            receipt.uncorrected_exchange_energy+=energy;
            for(Count i=0;i<values.size();++i) values[i]-=scale*exchange.values[i];
            receipt.shell_calls=add(receipt.shell_calls,exchange.receipt.shell_calls);
            receipt.primitive_pairs=add(receipt.primitive_pairs,exchange.receipt.primitive_pairs);
            if(plan.range_separated) receipt.exchange_components.push_back({kernel,coefficient,omega,energy,std::move(exchange.receipt)});
            else receipt.exchange=std::move(exchange.receipt);
        };
        if(plan.range_separated) {
            receipt.exchange_components.clear();
            for(Count index=0;index<plan.exchange_components.size();++index) {
                const auto& p=plan.exchange_components[index];
                if(!p.fitting) { component(p.exchange,p.kernel,p.coefficient,p.omega);continue; }
                const Count e=h.size();std::vector<Complex> spin(mul(2,e));
                if(plan.spins==1) for(Count i=0;i<e;++i) spin[i]=spin[e+i]=.5*d[i];
                else std::copy(d.begin(),d.end(),spin.begin());
                const auto& builder=*(*range_fitted)[index];
                auto result=builder.build(h,spin,p.coefficient);
                if(plan.spins==1) for(Count i=0;i<e;++i) if(result.K[i]!=result.K[e+i])
                    throw std::runtime_error("MANTA RI range RKS exchange spin responses disagree");
                for(Count i=0;i<values.size();++i) values[i]-=p.coefficient*result.K[i];
                receipt.uncorrected_exchange_energy+=result.receipt.exchange_energy;
                receipt.shell_calls=add(receipt.shell_calls,result.receipt.shell_calls);
                receipt.primitive_pairs=add(receipt.primitive_pairs,result.receipt.primitive_products);
                receipt.exchange_components.push_back({p.kernel,p.coefficient,p.omega,result.receipt.exchange_energy,
                    {},builder.info(),std::move(result.receipt)});
            }
        } else if(!fit_k) {
            component(*plan.exchange,"coulomb",plan.exact_exchange_fraction,0.);
            receipt.exchange_identity=receipt.exchange->source_identity;
        }
        receipt.exchange_correction_energy=0.;
        if(receipt.exchange_q0=="BvK-ewald") {
            const Count n=plan.n_basis,n2=square(n),e=mul(plan.momenta,n2);
            std::vector<Complex> product(n2);Sum energy;
            auto accumulate=[](Complex x,Complex& sum,Complex& error) {
                const auto y=x-error,next=sum+y;error=(next-sum)-y;sum=next;
            };
            // Same probe-charge source as RHF/UHF, scaled by the method's
            // exchange fraction. The overlap belongs to the source AO frame.
            for(Count spin=0;spin<plan.spins;++spin) for(Count ik=0;ik<plan.momenta;++ik) {
                const Count offset=ik*n2,block=spin*e+offset;
                for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                    Complex value=0,error=0;
                    for(Count a=0;a<n;++a) accumulate((*overlap)[offset+i*n+a]*d[block+a*n+j],value,error);
                    product[i*n+j]=value;
                }
                Complex trace=0,error=0;
                for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                    Complex value=0,rounding=0;
                    for(Count a=0;a<n;++a) accumulate(product[i*n+a]*(*overlap)[offset+a*n+j],value,rounding);
                    const auto correction=-scale*receipt.madelung*value;
                    require_finite(correction.real());require_finite(correction.imag());
                    values[block+i*n+j]+=correction;
                    accumulate(.5*weights[ik]*d[block+j*n+i]*correction,trace,error);
                }
                require_finite(trace.real());require_finite(trace.imag());
                if(std::abs(trace.imag())>1e-10*std::max(1.,std::abs(trace.real())))
                    throw std::runtime_error("MANTA hybrid Madelung energy is not real");
                energy.add(trace.real());
            }
            receipt.exchange_correction_energy=energy.value;
        }
        receipt.exchange_energy=receipt.uncorrected_exchange_energy+receipt.exchange_correction_energy;
        require_finite(receipt.exchange_energy);
    }
    std::vector<Complex> build(const Complex* d,Count count) {
        const Count e=h.size(),ns=plan.spins;
        if(count!=mul(ns,e)) throw std::logic_error("MANTA periodic KS density extent mismatch");
        std::vector<Complex> spin(d,d+count),total(e);
        for(Count i=0;i<e;++i) total[i]=spin[i]+(ns==2?spin[e+i]:Complex{});
        auto leaf=budget;
        std::vector<Complex> j;
        std::optional<PeriodicRIFockResult> fit;
        if(fitted) {
            std::vector<Complex> fit_spin(mul(2,e));
            if(ns==1) for(Count i=0;i<e;++i) fit_spin[i]=fit_spin[e+i]=.5*spin[i];
            else std::copy(spin.begin(),spin.end(),fit_spin.begin());
            fit.emplace(fitted->build(h,fit_spin,plan.exact_exchange_fraction));
            receipt.fitted_fock=fit->receipt;
            receipt.shell_calls=add(receipt.shell_calls,fit->receipt.shell_calls);
            receipt.primitive_pairs=add(receipt.primitive_pairs,fit->receipt.primitive_products);
            if(plan.fitting->has_j) {
                j=std::move(fit->J);receipt.coulomb_energy=fit->receipt.coulomb_energy;
            }
        }
        if(!plan.fitting || !plan.fitting->has_j) {
            leaf.other_live_bytes=plan.modeled_peak_bytes-plan.hartree.modeled_peak_bytes;
            auto direct=compute_periodic_hartree(basis,system.lattice,g,cells,k,weights,total,
                options.reciprocal_batch,options.pair_cutoff_bohr,leaf);
            receipt.hartree=direct.receipt;receipt.coulomb_identity=direct.receipt.source_identity;
            receipt.coulomb_energy=direct.receipt.energy;j=std::move(direct.momenta);
            receipt.shell_calls=add(receipt.shell_calls,direct.receipt.shell_calls);
            receipt.primitive_pairs=add(receipt.primitive_pairs,direct.receipt.primitive_pairs);
        }
        leaf.other_live_bytes=plan.modeled_peak_bytes-plan.xc.modeled_peak_bytes;
        auto xc=periodic_xc_component::compute(basis,functional,lattice,images,xc_k,weights,points,qw,spin,
            options.grid_batch,leaf,plan.range_separated?XCKind::RangeSeparated:plan.exact_exchange_fraction!=0.?XCKind::GlobalHybrid:XCKind::Pure);
        ++receipt.fock_calls;
        receipt.shell_calls=add(receipt.shell_calls,xc.receipt.shell_calls);
        receipt.primitive_points=add(receipt.primitive_points,xc.receipt.primitive_points);
        receipt.xc=std::move(xc.receipt);
        if(fit && plan.fitting->has_k) {
            receipt.uncorrected_exchange_energy=fit->receipt.exchange_energy;
            if(ns==1) for(Count i=0;i<e;++i) if(fit->K[i]!=fit->K[e+i])
                throw std::runtime_error("MANTA RI RKS exchange spin responses disagree");
            for(Count i=0;i<count;++i) xc.values[i]-=plan.exact_exchange_fraction*fit->K[i];
        }
        add_exchange(spin,xc.values);
        receipt.xc_energy=receipt.xc.energy;
        Sum e1;
        const Count n=plan.n_basis,n2=square(n);
        for(Count ik=0;ik<plan.momenta;++ik) for(Count i=0;i<n;++i) for(Count a=0;a<n;++a)
            e1.add(weights[ik]*(total[ik*n2+i*n+a]*h[ik*n2+a*n+i]).real());
        receipt.one_electron_energy=e1.value;
        receipt.energy=receipt.nuclear_energy+e1.value+receipt.coulomb_energy+receipt.xc_energy+receipt.exchange_energy;
        require_finite(receipt.energy);
        receipt.maximum_grid_population_error=0.;
        for(Count s=0;s<ns;++s) {
            const double expected=ns==1?plan.electrons:(s==0?plan.n_alpha:plan.n_beta);
            receipt.maximum_grid_population_error=std::max(receipt.maximum_grid_population_error,std::abs(receipt.xc.electrons_grid[s]-expected));
        }
        for(Count i=0;i<count;++i) {
            xc.values[i]+=j[i%e];
            require_finite(xc.values[i].real());require_finite(xc.values[i].imag());
        }
        Digest source;source.text(fitted?"manta.periodic-ks.interaction.v2;fixed-ri-semilocal":"manta.periodic-ks.interaction.v1;joint-hartree-semilocal");
        source.text(receipt.coulomb_identity);source.text(receipt.xc.source_identity);
        if(plan.exact_exchange_fraction!=0. || plan.range_separated) {
            if(plan.range_separated) {
                source.text("range-separated;canonical-coulomb-erfc;finite-screened-q0");
                for(const auto& c:receipt.exchange_components) {
                    source.text(c.kernel);source.real(c.coefficient);source.real(c.omega);
                    if(c.fitted_fock) source.text(c.fitted_fock->source_identity);else source.text(c.exchange.source_identity);
                }
            } else {
                source.text("global-hybrid;method-owned-exact-exchange");source.real(plan.exact_exchange_fraction);
                source.text(receipt.exchange_identity);
            }
            source.text(receipt.exchange_q0);
            if(receipt.exchange_q0=="BvK-ewald") {
                source.text(receipt.overlap_identity);source.text(receipt.exchange_correction_identity);
            }
        }
        source_identity=source.hash.finish_hex();
        // A source change does not create a new density epoch or AO frame.
        Digest density;density.text("manta.periodic-ks.evaluated-density.v2");
        density.text(receipt.overlap_identity);density.word(plan.spins);
        for(double weight:weights) density.real(weight);
        for(Count i=0;i<count;++i) density.value(d[i]);
        receipt.final_density_identity=density.hash.finish_hex();
        Digest snapshot;snapshot.text("manta.periodic-ks.snapshot.v2");snapshot.text(source_identity);
        snapshot.text(receipt.kinetic_identity);snapshot.text(receipt.nuclear_identity);snapshot.text(receipt.nuclear_energy_identity);
        snapshot.text(receipt.final_density_identity);
        for(Count i=0;i<count;++i) snapshot.value(h[i%e]+xc.values[i]);
        snapshot.real(receipt.energy);receipt.final_fock_identity=snapshot.hash.finish_hex();
        return std::move(xc.values);
    }
    static void evaluate(const Complex* d,std::size_t dn,Complex* out,std::size_t on,double* energy,void* pointer) {
        auto& self=*static_cast<Adapter*>(pointer);
        if(on!=dn) throw std::logic_error("MANTA periodic KS callback extent mismatch");
        const auto values=self.build(d,dn);std::copy(values.begin(),values.end(),out);
        *energy=self.receipt.coulomb_energy+self.receipt.xc_energy+self.receipt.exchange_energy;
    }
};
template<class Numerical> void confirm(const Numerical& numerical,Adapter& adapter) {
    auto& r=adapter.receipt;
    if(!numerical.converged()) return;
    const Count e=adapter.h.size(),count=mul(adapter.plan.spins,e);
    const auto g=adapter.build(numerical.density_data(),count);
    for(Count i=0;i<count;++i)
        r.confirmation_fock_error=std::max(r.confirmation_fock_error,std::abs(adapter.h[i%e]+g[i]-numerical.fock_data()[i]));
    r.confirmation_energy_error=std::abs(r.energy-numerical.final_snapshot().energy_per_cell);
    r.confirmed=r.confirmation_fock_error<=adapter.options.confirmation_fock_tolerance
        && r.confirmation_energy_error<=adapter.options.confirmation_energy_tolerance;
}
PeriodicKSPlan plan_ks(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,
    Count ng,Count nc,Count ni,Count ngrid,const PeriodicKSOptions& o,const ResourceBudget& budget,
    const HybridControls* hybrid=nullptr,Count nq=0,bool range_separated=false,const FittingControls* fitting=nullptr) {
    PeriodicKSPlan p;p.electrons=validate(basis,system,f,o);p.spins=f.spin();p.n_basis=basis.nbasis();p.momenta=mesh.size();
    p.n_beta=(p.electrons-static_cast<Count>(system.multiplicity-1))/2;p.n_alpha=p.electrons-p.n_beta;
    const auto leaf=unlimited(budget);const Count e=mul(p.momenta,square(p.n_basis));
    p.range_separated=range_separated;
    p.xc=periodic_xc_component::plan(basis,f,ni,p.momenta,ngrid,o.grid_batch,leaf,
        range_separated?XCKind::RangeSeparated:hybrid?XCKind::GlobalHybrid:XCKind::Pure);
    if(fitting) {
        if(fitting->accuracy.enabled() && (!fitting->make_j || fitting->make_k))
            throw std::invalid_argument("MANTA RI automatic accuracy requires J-only fitting and independent exchange");
        if(!o.pair_cutoff_bohr || *o.pair_cutoff_bohr!=fitting->fitting.pair_cutoff_bohr)
            throw std::invalid_argument("MANTA RI KS requires identical explicit KS/fitting pair cutoffs");
        if(!range_separated || fitting->make_j)
            p.fitting=plan_periodic_ri_fock(basis,fitting->auxiliary,system,ng,!range_separated && fitting->make_k?nq:0,p.momenta,
                fitting->fitting,fitting->ri,leaf,fitting->make_j,!range_separated && fitting->make_k,{},fitting->accuracy.for_method_plan(mul(p.momenta,square(p.n_basis))));
    }
    PeriodicExchangePlan exchange_work;
    if(hybrid) {
        if(hybrid->exchange_q0!="strict-zero-mode" && hybrid->exchange_q0!="BvK-ewald")
            throw std::invalid_argument("MANTA hybrid KS exchange_q0 must be strict-zero-mode or BvK-ewald");
        if(!std::isfinite(hybrid->madelung_tolerance) || hybrid->madelung_tolerance<=0 || hybrid->madelung_tolerance>=1
           || (hybrid->madelung_alpha && (!std::isfinite(*hybrid->madelung_alpha) || *hybrid->madelung_alpha<=0)))
            throw std::invalid_argument("MANTA hybrid KS Madelung alpha must be positive and tolerance in (0,1)");
        if(range_separated) {
            for(const auto& [kernel,coefficient,omega]:periodic_xc_component::range_separated_components(f)) {
                PeriodicKSExchangeComponentPlan part;part.kernel=kernel;part.coefficient=coefficient;part.omega=omega;
                if(fitting && fitting->make_k) {
                    part.fitting=plan_periodic_ri_fock(basis,fitting->auxiliary,system,ng,nq,p.momenta,
                        fitting->fitting,fitting->ri,leaf,false,true,{kernel,omega});
                } else part.exchange=plan_periodic_exchange(basis,add(nq,kernel=="erfc"),nc,p.momenta,p.spins,
                    o.reciprocal_batch,o.pair_cutoff_bohr,leaf,kernel,omega);
                const auto& child=part.exchange;
                if(kernel=="coulomb") p.long_range_exchange_fraction+=coefficient;
                p.short_range_exchange_fraction+=coefficient;
                exchange_work.modeled_peak_bytes=add(exchange_work.modeled_peak_bytes,child.modeled_peak_bytes);
                exchange_work.shell_calls=add(exchange_work.shell_calls,child.shell_calls);
                exchange_work.primitive_pairs=add(exchange_work.primitive_pairs,child.primitive_pairs);
                exchange_work.scalar_terms=add(exchange_work.scalar_terms,child.scalar_terms);
                exchange_work.kernel_terms=add(exchange_work.kernel_terms,child.kernel_terms);
                p.exchange_components.push_back(std::move(part));
            }
            require_finite(p.short_range_exchange_fraction);require_finite(p.long_range_exchange_fraction);
        } else {
            p.exact_exchange_fraction=periodic_xc_component::global_hybrid_fraction(f);
            p.long_range_exchange_fraction=p.short_range_exchange_fraction=p.exact_exchange_fraction;
            if(!fitting || !fitting->make_k) {
                p.exchange=plan_periodic_exchange(basis,nq,nc,p.momenta,p.spins,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
                exchange_work=*p.exchange;
            }
        }
        if(p.long_range_exchange_fraction!=0.) p.madelung=plan_madelung(system,mesh,*hybrid);
        if(p.long_range_exchange_fraction!=0. && hybrid->exchange_q0=="BvK-ewald")
            p.exchange_correction_kernel_terms_per_call=mul(512,mul(mul(p.spins,e),p.n_basis));
    }
    p.overlap=plan_bloch_one_electron(basis,"overlap",nc,p.momenta,o.pair_cutoff_bohr,leaf);
    p.kinetic=plan_bloch_one_electron(basis,"kinetic",nc,p.momenta,o.pair_cutoff_bohr,leaf);
    p.nuclear=plan_periodic_nuclear(basis,system,ng,nc,p.momenta,o.alpha,o.nuclear_cutoff_bohr,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
    if(!fitting || !fitting->make_j)
        p.hartree=plan_periodic_hartree(basis,ng,nc,p.momenta,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
    validate_centers(basis,system);
    // Concurrent J/XC results, copied spin and summed density, H/S, exports,
    // generated k in both layouts, receipts and nuclear selector labels.
    p.driver_live_bytes=add(32768,add(mul(384,e),add(mul(128,p.nuclear.maximum_nuclear_candidates_per_pair),
        add(mul(256,system.unit_cell.size()),mul(128,p.momenta)))));
    p.provider_workspace_bytes=add(p.hartree.modeled_peak_bytes,p.xc.modeled_peak_bytes);
    Count fit_retained=0,fit_work=0,fit_scalars=0,fit_setup_calls=0,fit_build_calls=0,fit_setup_primitives=0,fit_build_primitives=0;
    auto admit_fitting=[&](const PeriodicRIFockPlan& fit) {
        // Fitting result, RKS equal-spin embedding, fixed metric receipts and
        // empty RI-J offset domain coexist with the XC and direct responses.
        p.driver_live_bytes=add(p.driver_live_bytes,add(32768,add(mul(256,e),
            add(mul(4096,fit.channels),mul(64,add(square(p.momenta),1))))));
        p.provider_workspace_bytes=add(p.provider_workspace_bytes,fit.modeled_peak_bytes-fit.retained_source_bytes);
        fit_retained=add(fit_retained,fit.retained_source_bytes);
        fit_work=add(fit_work,add(fit.kernel_terms,mul(8192,e)));
        fit_scalars=add(fit_scalars,add(fit.scalar_terms,mul(512,e)));
        fit_setup_calls=add(fit_setup_calls,fit.metric_shell_calls);fit_build_calls=add(fit_build_calls,fit.build_shell_calls);
        fit_setup_primitives=add(fit_setup_primitives,fit.metric_primitive_products);
        fit_build_primitives=add(fit_build_primitives,fit.build_primitive_products);
    };
    if(p.fitting) admit_fitting(*p.fitting);
    for(const auto& part:p.exchange_components) if(part.fitting) admit_fitting(*part.fitting);
    if(hybrid) {
        // Copied Q/offsets, component receipt copies and Madelung matrix scratch.
        p.driver_live_bytes=add(p.driver_live_bytes,add(mul(64,add(mul(square(p.momenta),add(nq,1)),1)),
            add(mul(64,square(p.n_basis)),p.madelung.modeled_peak_bytes)));
        p.provider_workspace_bytes=add(p.provider_workspace_bytes,exchange_work.modeled_peak_bytes);
        if(range_separated) {
            // Retained Q=0-expanded source, offsets, component plans/receipts
            // and binding exports coexist with every evaluated D/F snapshot.
            p.driver_live_bytes=add(p.driver_live_bytes,add(mul(96,add(mul(square(p.momenta),add(nq,2)),1)),
                mul(8192,add(p.exchange_components.size(),1))));
        }
    }
    const Count per_call=add(fit_work,add(p.hartree.kernel_terms,add(p.xc.kernel_terms,add(mul(512,mul(p.spins,e)),
        add(exchange_work.kernel_terms,p.exchange_correction_kernel_terms_per_call)))));
    p.provider_kernel_terms_per_call=per_call;
    const BoundedPeriodicFunctionalProvider provider{Adapter::evaluate,nullptr,fit_retained,p.provider_workspace_bytes,per_call};
    if(p.spins==1) p.solver=plan_bounded_periodic_rks_solver(p.momenta,p.n_basis,p.electrons,p.n_basis,provider,o.scf,0);
    else p.solver=plan_bounded_periodic_uks_solver(p.momenta,p.n_basis,p.n_alpha,p.n_beta,p.n_basis,provider,o.scf,0);
    if(fitting && fitting->accuracy.derives_density_limit()) {
        p.density_envelope=plan_bounded_periodic_density_envelope(p.momenta,p.n_basis,p.n_alpha,p.n_beta,o.scf);
        p.driver_live_bytes=add(p.driver_live_bytes,p.density_envelope->peak_owned_numerical_bytes);
    }
    p.history_capacity=o.scf.maximum_iterations;
    p.history_live_bytes=mul(p.history_capacity,add(mul(3,sizeof(BoundedPeriodicUHFSnapshot)),256));
    p.driver_live_bytes=add(p.driver_live_bytes,p.history_live_bytes);
    p.modeled_peak_bytes=add(budget.other_live_bytes,add(p.driver_live_bytes,add(p.solver.peak_owned_numerical_bytes,
        add(p.solver.provider_retained_bytes,add(p.provider_workspace_bytes,
        add(p.overlap.modeled_peak_bytes,add(p.kinetic.modeled_peak_bytes,p.nuclear.modeled_peak_bytes)))))));
    p.fock_calls_upper_bound=add(o.scf.maximum_iterations,1);
    p.shell_calls=add(p.overlap.shell_calls,add(p.kinetic.shell_calls,add(p.nuclear.shell_calls,
        mul(p.fock_calls_upper_bound,add(p.hartree.shell_calls,add(p.xc.shell_calls,exchange_work.shell_calls))))));
    p.primitive_pairs=add(p.overlap.primitive_pairs,add(p.kinetic.primitive_pairs,add(p.nuclear.primitive_pairs,
        mul(p.fock_calls_upper_bound,add(p.hartree.primitive_pairs,exchange_work.primitive_pairs)))));
    p.primitive_points=mul(p.fock_calls_upper_bound,p.xc.primitive_points);
    p.shell_calls=add(p.shell_calls,add(fit_setup_calls,mul(p.fock_calls_upper_bound,fit_build_calls)));
    p.primitive_pairs=add(p.primitive_pairs,add(fit_setup_primitives,mul(p.fock_calls_upper_bound,fit_build_primitives)));
    p.primitive_work=add(p.primitive_pairs,p.primitive_points);
    p.nuclear_image_candidates_upper_bound=mul(system.unit_cell.size(),p.nuclear.maximum_nuclear_candidates_per_pair);
    // Q validation/copying and zero expansion still run when a functional's
    // exchange components cancel. Charge that source traversal independently
    // of the exchange leaves rather than relying on their contraction work.
    const Count range_source_work=range_separated?mul(128,add(mul(square(p.momenta),add(nq,2)),1)):0;
    const Count driver=add(range_source_work,add(mul(512,mul(mul(p.spins,e),p.fock_calls_upper_bound)),
        add(mul(128,p.nuclear_image_candidates_upper_bound),mul(128,mul(ng,system.unit_cell.size())))));
    p.scalar_terms=add(driver,add(p.overlap.scalar_terms,add(p.kinetic.scalar_terms,add(p.nuclear.scalar_terms,
        add(mul(p.fock_calls_upper_bound,add(p.hartree.scalar_terms,add(p.xc.scalar_terms,
            add(exchange_work.scalar_terms,p.exchange_correction_kernel_terms_per_call)))),p.solver.maximum_work_units)))));
    p.kernel_terms=add(mul(32,driver),add(p.overlap.kernel_terms,add(p.kinetic.kernel_terms,
        add(p.nuclear.kernel_terms,add(per_call,p.solver.maximum_work_units)))));
    p.scalar_terms=add(p.scalar_terms,p.madelung.kernel_terms);
    p.scalar_terms=add(p.scalar_terms,mul(p.fock_calls_upper_bound,fit_scalars));
    p.kernel_terms=add(p.kernel_terms,p.madelung.kernel_terms);
    if(p.density_envelope) {
        p.scalar_terms=add(p.scalar_terms,p.density_envelope->maximum_work_units);
        p.kernel_terms=add(p.kernel_terms,p.density_envelope->maximum_work_units);
    }
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic KS aggregate memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic KS aggregate shell-call budget exceeded");
    cap(p.primitive_work,budget.primitive_quartets,"MANTA periodic KS aggregate primitive-work budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic KS aggregate scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic KS aggregate kernel-work budget exceeded");return p;
}
PeriodicKSResult run_ks(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,
    const md::Matrix& g,const md::Matrix& cells,const Eigen::MatrixXd& images,const Eigen::MatrixXd& points,
    const std::vector<double>& qw,const PeriodicKSOptions& options,const ResourceBudget& budget,
    const HybridControls* hybrid=nullptr,const md::Matrix* q=nullptr,const std::vector<Count>* offsets=nullptr,
    bool range_separated=false,const FittingControls* fitting=nullptr) {
    const auto o=options;
    Count nq=0;
    if(hybrid) {
        // Admit metadata traversal before reading any offset payload.
        plan_ks(basis,system,mesh,f,g.rows(),cells.rows(),images.rows(),points.rows(),o,budget,hybrid,
            fitting && fitting->make_k && mesh.size()>1?1:0,range_separated,fitting);
        if(offsets->size()!=add(square(mesh.size()),1) || offsets->front()!=0 || offsets->back()!=static_cast<Count>(q->rows()))
            throw std::invalid_argument("MANTA hybrid KS exchange offset extent mismatch");
        for(Count i=1;i<offsets->size();++i) {
            if((*offsets)[i]<(*offsets)[i-1] || (*offsets)[i]>static_cast<Count>(q->rows()))
                throw std::invalid_argument("MANTA hybrid KS exchange offsets must be monotone and bounded");
            nq=std::max(nq,(*offsets)[i]-(*offsets)[i-1]);
        }
    }
    PeriodicKSResult result;result.plan=plan_ks(basis,system,mesh,f,g.rows(),cells.rows(),images.rows(),points.rows(),o,budget,hybrid,nq,range_separated,fitting);
    const auto& p=result.plan;auto& r=result.receipt;r.modeled_peak_bytes=p.modeled_peak_bytes;
    const Count nk=p.momenta,e=mul(nk,square(p.n_basis));
    md::Matrix screened;std::vector<Count> screened_offsets;
    if(range_separated && !(fitting && fitting->make_k)) {
        // The analytic erfc limit is pi/omega^2. Adding it at the diagonal
        // transfer preserves its ordinary 1/Nk quadrature weight. It receives
        // no Coulomb singular correction (Sundararaman & Arias, PRB 87,
        // 165122 (2013), Sec. II.A); that belongs to the resolved 1/r term.
        if(!q->allFinite()) throw std::invalid_argument("MANTA range-separated KS requires finite nonzero base Q vectors");
        for(Eigen::Index i=0;i<q->rows();++i) if(q->row(i).isZero(0))
            throw std::invalid_argument("MANTA range-separated KS base Q vectors must omit Q=0; finite erfc zero modes are added by the method");
        screened.resize(add(q->rows(),nk),3);screened_offsets.reserve(offsets->size());screened_offsets.push_back(0);
        Count next=0;
        for(Count t=0;t<nk;++t) for(Count s=0;s<nk;++s) {
            const Count pair=t*nk+s,count=(*offsets)[pair+1]-(*offsets)[pair];
            screened.middleRows(next,count)=q->middleRows((*offsets)[pair],count);next+=count;
            if(t==s) screened.row(next++).setZero();
            screened_offsets.push_back(next);
        }
    }
    if(range_separated) { r.range_separated=true;r.exchange_components.reserve(p.exchange_components.size()); }
    if(range_separated && fitting && fitting->make_k) {
        if(!q->allFinite()) throw std::invalid_argument("MANTA RI range KS requires finite exchange vectors");
        for(Count t=0;t<nk;++t) for(Count s=0;s<nk;++s) {
            const Count pair=t*nk+s,length=(*offsets)[pair+1]-(*offsets)[pair];
            if((t==s && length) || (t!=s && !length))
                throw std::invalid_argument("MANTA RI range KS requires empty diagonal and nonempty off-diagonal pair segments");
        }
    }
    if(p.spins==1) result.rks_history.reserve(p.history_capacity);else result.uks_history.reserve(p.history_capacity);
    md::Matrix k(nk,3);std::vector<double> weights(nk,1./nk);
    for(Count i=0;i<nk;++i) k.row(i)=(system.reciprocal_lattice()*mesh.fractional_at(i)).transpose();
    const Eigen::MatrixXd xc_k=k,lattice=system.lattice;
    auto leaf=budget;leaf.other_live_bytes=p.modeled_peak_bytes-p.overlap.modeled_peak_bytes;
    { auto s=compute_bloch_one_electron(basis,"overlap",cells,k,o.pair_cutoff_bohr,leaf);
      r.overlap_identity=s.receipt.source_identity;r.shell_calls+=s.receipt.shell_calls;r.primitive_pairs+=s.receipt.primitive_pairs;result.overlap=std::move(s.values); }
    leaf.other_live_bytes=p.modeled_peak_bytes-p.kinetic.modeled_peak_bytes;
    { auto t=compute_bloch_one_electron(basis,"kinetic",cells,k,o.pair_cutoff_bohr,leaf);
      r.kinetic_identity=t.receipt.source_identity;r.shell_calls+=t.receipt.shell_calls;r.primitive_pairs+=t.receipt.primitive_pairs;result.hcore=std::move(t.values); }
    leaf.other_live_bytes=p.modeled_peak_bytes-p.nuclear.modeled_peak_bytes;
    { auto v=compute_periodic_nuclear(basis,system,g,cells,k,o.alpha,o.nuclear_cutoff_bohr,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
      r.nuclear_identity=v.receipt.source_identity;r.shell_calls+=v.receipt.nuclear_shell_calls+v.receipt.overlap_shell_calls+v.receipt.fourier_shell_calls;
      r.primitive_pairs+=v.receipt.primitive_pairs;for(Count i=0;i<e;++i) result.hcore[i]+=v.momenta[i]; }
    nuclear_energy(system,g,o,p.nuclear,r);
    if(hybrid) {
        r.exchange_algorithm="exact";
        r.algorithm=range_separated?"finite-3d-range-separated-ks":"finite-3d-global-hybrid-ks";
        r.boundary="neutral-3d-tinfoil-zero-average;exchange-q0-omitted";
        r.exact_exchange_fraction=p.exact_exchange_fraction;
        r.long_range_exchange_fraction=p.long_range_exchange_fraction;r.short_range_exchange_fraction=p.short_range_exchange_fraction;
        r.exchange_q0_requested=hybrid->exchange_q0;
        if(p.long_range_exchange_fraction!=0.) compute_madelung(system,mesh,*hybrid,p.madelung,r);
        else r.boundary="neutral-3d-tinfoil-zero-average;no-singular-exchange";
        if(range_separated) r.boundary+=";range-separated;erfc-finite-q0";
    }
    std::unique_ptr<PeriodicRIFockBuilder> fitted;
    if(p.fitting) {
        leaf.other_live_bytes=p.modeled_peak_bytes-p.fitting->modeled_peak_bytes;
        const md::Matrix empty(0,3);std::vector<Count> empty_offsets;
        if(!p.fitting->has_k) empty_offsets.resize(add(square(nk),1));
        auto accuracy=fitting->accuracy;
        if(p.density_envelope) {
            const auto& ep=*p.density_envelope;BoundedPeriodicRHFCaps ec;
            ec.maximum_owned_numerical_bytes=ep.peak_owned_numerical_bytes;ec.maximum_total_numerical_bytes=p.modeled_peak_bytes;
            ec.maximum_work_units=ep.maximum_work_units;
            ec.other_live_numerical_bytes=p.modeled_peak_bytes-ep.peak_owned_numerical_bytes-ep.borrowed_input_bytes;
            r.accuracy_density=bound_bounded_periodic_density_envelope(nk,p.n_basis,p.n_alpha,p.n_beta,result.overlap.data(),e,o.scf,ec);
            accuracy.density_l1_limit=r.accuracy_density->density_l1_limit;
            accuracy.density_entry_limits=r.accuracy_density->weighted_entry_bounds;
        }
        fitted=std::make_unique<PeriodicRIFockBuilder>(basis,fitting->auxiliary,system,k,weights,g,
            p.fitting->has_k?*q:empty,p.fitting->has_k?*offsets:empty_offsets,
            fitting->fitting,fitting->ri,leaf,p.fitting->has_j,p.fitting->has_k,PeriodicGDFKernelOptions{},accuracy);
        r.fitting_info=fitted->info();
        r.shell_calls=add(r.shell_calls,r.fitting_info->metric_shell_calls);
        r.primitive_pairs=add(r.primitive_pairs,r.fitting_info->metric_primitive_products);
        r.shell_calls=add(r.shell_calls,r.fitting_info->accuracy.shell_calls);
        r.primitive_pairs=add(r.primitive_pairs,r.fitting_info->accuracy.primitive_products);
        r.algorithm=hybrid?"finite-3d-ri-global-hybrid-ks":"finite-3d-ri-semilocal-ks";
        if(fitting->make_j) {
            r.coulomb_algorithm="ri";Digest id;id.text("manta.periodic-ri-J.v1");
            id.text(r.fitting_info->source_identity);r.coulomb_identity=id.hash.finish_hex();
        }
        if(p.fitting->has_k) {
            r.exchange_algorithm="ri";Digest id;id.text("manta.periodic-ri-K.v1");
            id.text(r.fitting_info->source_identity);r.exchange_identity=id.hash.finish_hex();
        }
    }
    std::vector<std::unique_ptr<PeriodicRIFockBuilder>> range_fitted(p.exchange_components.size());
    if(range_separated && fitting) {
        r.algorithm="finite-3d-ri-range-separated-ks";
        if(fitting->make_k) r.exchange_algorithm="ri";
        for(Count index=0;index<p.exchange_components.size();++index) {
            const auto& part=p.exchange_components[index];if(!part.fitting) continue;
            leaf.other_live_bytes=p.modeled_peak_bytes-part.fitting->modeled_peak_bytes;
            auto builder=std::make_unique<PeriodicRIFockBuilder>(basis,fitting->auxiliary,system,k,weights,g,*q,*offsets,
                fitting->fitting,fitting->ri,leaf,false,true,PeriodicGDFKernelOptions{part.kernel,part.omega});
            r.shell_calls=add(r.shell_calls,builder->info().metric_shell_calls);
            r.primitive_pairs=add(r.primitive_pairs,builder->info().metric_primitive_products);
            range_fitted[index]=std::move(builder);
        }
    }
    Adapter adapter{basis,system,f,g,cells,k,lattice,images,points,xc_k,weights,qw,result.hcore,o,p,budget,r,{},q,offsets,&result.overlap,
        range_separated?&screened:nullptr,range_separated?&screened_offsets:nullptr,fitted.get(),&range_fitted};
    const Count per_call=p.provider_kernel_terms_per_call;
    Count retained=p.fitting?p.fitting->retained_source_bytes:0;
    for(const auto& part:p.exchange_components) if(part.fitting) retained=add(retained,part.fitting->retained_source_bytes);
    // The functional solver adds its own energy scalar to this count.
    const BoundedPeriodicFunctionalProvider provider{Adapter::evaluate,&adapter,retained,p.provider_workspace_bytes,per_call};
    BoundedPeriodicRHFCaps caps;caps.maximum_owned_numerical_bytes=p.solver.peak_owned_numerical_bytes;
    caps.maximum_total_numerical_bytes=p.modeled_peak_bytes;caps.maximum_work_units=p.solver.maximum_work_units;
    caps.maximum_fock_calls=o.scf.maximum_iterations;
    caps.other_live_numerical_bytes=p.modeled_peak_bytes-p.solver.peak_owned_numerical_bytes-p.provider_workspace_bytes
        -p.solver.provider_retained_bytes-p.solver.borrowed_input_bytes;
    if(p.spins==1) {
        Trace<BoundedPeriodicRHFSnapshot> trace{result.rks_history,p.history_capacity};
        result.rks.emplace(solve_bounded_periodic_rks({nk,p.n_basis,p.electrons,r.nuclear_energy,
            result.overlap.data(),static_cast<std::size_t>(e),result.hcore.data(),static_cast<std::size_t>(e)},provider,o.scf,caps,decltype(trace)::append,&trace));
        confirm(*result.rks,adapter);
    } else {
        Trace<BoundedPeriodicUHFSnapshot> trace{result.uks_history,p.history_capacity};
        result.uks.emplace(solve_bounded_periodic_uks({nk,p.n_basis,p.n_alpha,p.n_beta,r.nuclear_energy,
            result.overlap.data(),static_cast<std::size_t>(e),result.hcore.data(),static_cast<std::size_t>(e)},provider,o.scf,caps,decltype(trace)::append,&trace));
        confirm(*result.uks,adapter);
    }
    Digest source;source.text(range_separated?"manta.periodic-ks.source.v3;neutral-all-electron;uniform-full-mesh;range-separated":hybrid?"manta.periodic-ks.source.v2;neutral-all-electron;uniform-full-mesh;global-hybrid"
        :"manta.periodic-ks.source.v1;neutral-all-electron;uniform-full-mesh;pure-semilocal");
    source.text(r.overlap_identity);source.text(r.kinetic_identity);source.text(r.nuclear_identity);source.text(r.nuclear_energy_identity);
    source.text(adapter.source_identity);source.word(p.spins);source.word(p.n_alpha);source.word(p.n_beta);
    for(auto v:mesh.mesh()) source.word(v);for(auto v:mesh.is_shift()) source.word(v);
    r.source_identity=source.hash.finish_hex();return result;
}
} // namespace
PeriodicKSPlan plan_periodic_ks(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,
    Count ng,Count nc,Count ni,Count ngrid,const PeriodicKSOptions& o,const ResourceBudget& budget) {
    return plan_ks(basis,system,mesh,f,ng,nc,ni,ngrid,o,budget);
}
PeriodicKSResult run_periodic_ks(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,
    const md::Matrix& g,const md::Matrix& cells,const Eigen::MatrixXd& images,const Eigen::MatrixXd& points,
    const std::vector<double>& qw,const PeriodicKSOptions& o,const ResourceBudget& budget) {
    return run_ks(basis,system,mesh,f,g,cells,images,points,qw,o,budget);
}
PeriodicKSPlan plan_periodic_global_hybrid_ks(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,
    Count ng,Count nq,Count nc,Count ni,Count ngrid,const PeriodicGlobalHybridKSOptions& o,const ResourceBudget& budget) {
    const auto controls=hybrid_controls(o);
    return plan_ks(basis,system,mesh,f,ng,nc,ni,ngrid,o.ks,budget,&controls,nq);
}
PeriodicKSResult run_periodic_global_hybrid_ks(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,
    const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,const md::Matrix& cells,
    const Eigen::MatrixXd& images,const Eigen::MatrixXd& points,const std::vector<double>& qw,
    const PeriodicGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    const auto o=options;const auto controls=hybrid_controls(o);
    return run_ks(basis,system,mesh,f,g,cells,images,points,qw,o.ks,budget,&controls,&q,&offsets);
}
PeriodicKSPlan plan_periodic_range_separated_ks(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,
    Count ng,Count nq,Count nc,Count ni,Count ngrid,const PeriodicRangeSeparatedKSOptions& o,const ResourceBudget& budget) {
    const auto controls=hybrid_controls(o);
    return plan_ks(basis,system,mesh,f,ng,nc,ni,ngrid,o.ks,budget,&controls,nq,true);
}
PeriodicKSResult run_periodic_range_separated_ks(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,const Functional& f,
    const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,const md::Matrix& cells,
    const Eigen::MatrixXd& images,const Eigen::MatrixXd& points,const std::vector<double>& qw,
    const PeriodicRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    const auto o=options;
    const auto controls=hybrid_controls(o);
    return run_ks(basis,system,mesh,f,g,cells,images,points,qw,o.ks,budget,&controls,&q,&offsets,true);
}
PeriodicKSPlan plan_periodic_ri_ks(const BasisSet& basis,const BasisSet& auxiliary,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,Count ng,Count nc,Count ni,Count ngrid,
    const PeriodicRIKSOptions& o,const ResourceBudget& budget) {
    const FittingControls fitting{auxiliary,o.fitting,o.ri,o.ri_accuracy,true,false};
    return plan_ks(basis,system,mesh,f,ng,nc,ni,ngrid,o.ks,budget,nullptr,0,false,&fitting);
}
PeriodicKSResult run_periodic_ri_ks(const BasisSet& basis,const BasisSet& auxiliary,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,const md::Matrix& g,const md::Matrix& cells,
    const Eigen::MatrixXd& images,const Eigen::MatrixXd& points,const std::vector<double>& qw,
    const PeriodicRIKSOptions& options,const ResourceBudget& budget) {
    const auto o=options;const FittingControls fitting{auxiliary,o.fitting,o.ri,o.ri_accuracy,true,false};
    return run_ks(basis,system,mesh,f,g,cells,images,points,qw,o.ks,budget,nullptr,nullptr,nullptr,false,&fitting);
}
PeriodicKSPlan plan_periodic_ri_global_hybrid_ks(const BasisSet& basis,const BasisSet& auxiliary,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,Count ng,Count nq,Count nc,Count ni,Count ngrid,
    const PeriodicRIGlobalHybridKSOptions& o,const ResourceBudget& budget) {
    const auto hybrid=hybrid_controls(o);const auto fitting=fitting_controls(auxiliary,o);
    return plan_ks(basis,system,mesh,f,ng,nc,ni,ngrid,o.ks,budget,&hybrid,nq,false,&fitting);
}
PeriodicKSResult run_periodic_ri_global_hybrid_ks(const BasisSet& basis,const BasisSet& auxiliary,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,
    const md::Matrix& cells,const Eigen::MatrixXd& images,const Eigen::MatrixXd& points,const std::vector<double>& qw,
    const PeriodicRIGlobalHybridKSOptions& options,const ResourceBudget& budget) {
    const auto o=options;const auto hybrid=hybrid_controls(o);const auto fitting=fitting_controls(auxiliary,o);
    return run_ks(basis,system,mesh,f,g,cells,images,points,qw,o.ks,budget,&hybrid,&q,&offsets,false,&fitting);
}
PeriodicKSPlan plan_periodic_ri_range_separated_ks(const BasisSet& basis,const BasisSet& auxiliary,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,Count ng,Count nq,Count nc,Count ni,Count ngrid,
    const PeriodicRIRangeSeparatedKSOptions& o,const ResourceBudget& budget) {
    const auto hybrid=hybrid_controls(o);const auto fitting=fitting_controls(auxiliary,o);
    return plan_ks(basis,system,mesh,f,ng,nc,ni,ngrid,o.ks,budget,&hybrid,nq,true,&fitting);
}
PeriodicKSResult run_periodic_ri_range_separated_ks(const BasisSet& basis,const BasisSet& auxiliary,const PeriodicSystem& system,
    const RegularKMesh& mesh,const Functional& f,const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,
    const md::Matrix& cells,const Eigen::MatrixXd& images,const Eigen::MatrixXd& points,const std::vector<double>& qw,
    const PeriodicRIRangeSeparatedKSOptions& options,const ResourceBudget& budget) {
    const auto o=options;const auto hybrid=hybrid_controls(o);const auto fitting=fitting_controls(auxiliary,o);
    return run_ks(basis,system,mesh,f,g,cells,images,points,qw,o.ks,budget,&hybrid,&q,&offsets,true,&fitting);
}
} // namespace vibeqc::manta
