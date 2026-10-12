#include "vibeqc/manta_periodic_ri_uhf.hpp"
#include "vibeqc/manta_periodic_ri_rhf.hpp"
#include <algorithm>
#include <type_traits>
#include "manta_periodic_hf_source.hpp"

namespace vibeqc::manta {
namespace {
// Both method adapters own the same physical source and fixed projectors.
// Only the numerical leaf, density normalization and method receipt differ.
template<bool Restricted> using HFOptions=std::conditional_t<Restricted,PeriodicRIRHFOptions,PeriodicRIUHFOptions>;
template<bool Restricted> using HFPlan=std::conditional_t<Restricted,PeriodicRIRHFPlan,PeriodicRIUHFPlan>;
template<bool Restricted> using HFReceipt=std::conditional_t<Restricted,PeriodicRIRHFReceipt,PeriodicRIUHFReceipt>;
template<bool Restricted> using HFResult=std::conditional_t<Restricted,PeriodicRIRHFResult,PeriodicRIUHFResult>;
template<bool Restricted> using Snapshot=std::conditional_t<Restricted,BoundedPeriodicRHFSnapshot,BoundedPeriodicUHFSnapshot>;
template<bool Restricted> using Provider=std::conditional_t<Restricted,BoundedPeriodicRHFTwoElectronProvider,BoundedPeriodicUHFTwoElectronProvider>;
template<bool Restricted> using SolverCaps=std::conditional_t<Restricted,BoundedPeriodicRHFCaps,BoundedPeriodicUHFCaps>;
template<bool Restricted> struct Trace {
    std::vector<Snapshot<Restricted>>& rows;
    Count capacity;
    static bool append(const Snapshot<Restricted>& snapshot,void* pointer) {
        auto& trace=*static_cast<Trace*>(pointer);
        if(trace.rows.size()>=trace.capacity || snapshot.evaluated_iteration!=trace.rows.size()+1)
            throw std::logic_error("MANTA periodic RI HF trace exceeded its admitted iteration extent");
        trace.rows.push_back(snapshot);
        return true;
    }
};
template<bool Restricted> Count validate(const BasisSet& basis,const PeriodicSystem& system,const HFOptions<Restricted>& o) {
    if((o.coulomb_algorithm!="ri" && o.coulomb_algorithm!="exact") ||
       (o.exchange_algorithm!="ri" && o.exchange_algorithm!="exact") ||
       (o.coulomb_algorithm!="ri" && o.exchange_algorithm!="ri"))
        throw std::invalid_argument("MANTA periodic RI HF requires exact/ri J and K with at least one RI target");
    if(!o.pair_cutoff_bohr || *o.pair_cutoff_bohr!=o.fitting.pair_cutoff_bohr)
        throw std::invalid_argument("MANTA RI HF requires identical explicit HF and fitting pair cutoffs");
    if(system.dim!=3 || system.charge!=0 || system.multiplicity<1 || system.unit_cell.empty())
        throw std::invalid_argument("MANTA periodic RI HF requires a neutral all-electron 3D system with positive multiplicity and nuclei");
    if(!std::isfinite(o.confirmation_fock_tolerance) || o.confirmation_fock_tolerance<=0
       || !std::isfinite(o.confirmation_energy_tolerance) || o.confirmation_energy_tolerance<=0)
        throw std::invalid_argument("MANTA periodic RI HF confirmation tolerances must be finite positive");
    if(o.exchange_q0!="strict-zero-mode" && o.exchange_q0!="BvK-ewald")
        throw std::invalid_argument("MANTA periodic RI HF exchange_q0 must be strict-zero-mode or BvK-ewald");
    if(!std::isfinite(o.madelung_tolerance) || o.madelung_tolerance<=0 || o.madelung_tolerance>=1
       || (o.madelung_alpha && (!std::isfinite(*o.madelung_alpha) || *o.madelung_alpha<=0)))
        throw std::invalid_argument("MANTA periodic RI HF Madelung alpha must be positive and tolerance in (0,1)");
    Count electrons=0;
    for(const auto& a:system.unit_cell) {
        if(a.Z<=0) throw std::invalid_argument("MANTA periodic RI HF requires positive all-electron nuclear charges");
        electrons=add(electrons,static_cast<Count>(a.Z));
    }
    if constexpr(Restricted) {
        if(system.multiplicity!=1)
            throw std::invalid_argument("MANTA periodic RI RHF requires a singlet");
    }
    const Count unpaired=static_cast<Count>(system.multiplicity-1);
    if(unpaired>electrons || (electrons-unpaired)%2 || add(electrons,unpaired)/2>=basis.nbasis())
        throw std::invalid_argument("MANTA periodic RI HF requires compatible spin populations and at least one virtual orbital per spin");
    return electrons;
}
template<bool Restricted> struct Adapter {
    const BasisSet& basis; const PeriodicRIFockBuilder& fitted;
    const std::vector<double>& weights;
    const std::vector<Complex>& h; const std::vector<Complex>& overlap; const HFOptions<Restricted>& options;
    HFReceipt<Restricted>& receipt; PeriodicFockReceipt last;
    const PeriodicSystem& system; const md::Matrix& k; const md::Matrix& g; const md::Matrix& q;
    const std::vector<Count>& offsets; const md::Matrix& cells; const HFPlan<Restricted>& plan; ResourceBudget budget;
    void compose(PeriodicRIFockResult& result,const std::vector<Complex>& spin) {
        if(options.coulomb_algorithm=="ri" && options.exchange_algorithm=="ri") return;
        auto& r=result.receipt; const Count n=basis.nbasis(),n2=square(n),nk=weights.size();
        auto leaf=budget;
        if(options.coulomb_algorithm=="exact") {
            std::vector<Complex> total(h.size());
            for(Count i=0;i<h.size();++i)total[i]=spin[i]+spin[h.size()+i];
            leaf.other_live_bytes=plan.modeled_peak_bytes-plan.exact_hartree.modeled_peak_bytes;
            auto direct=compute_periodic_hartree(basis,system.lattice,g,cells,k,weights,total,
                options.reciprocal_batch,options.pair_cutoff_bohr,leaf);
            receipt.exact_hartree=direct.receipt;receipt.coulomb_identity=direct.receipt.source_identity;
            r.coulomb_energy=direct.receipt.energy;result.J=std::move(direct.momenta);
            for(Count s=0;s<2;++s)for(Count i=0;i<h.size();++i)result.values[s*h.size()+i]+=result.J[i];
            r.shell_calls=add(r.shell_calls,direct.receipt.shell_calls);
            r.primitive_products=add(r.primitive_products,direct.receipt.primitive_pairs);
        } else {
            leaf.other_live_bytes=plan.modeled_peak_bytes-plan.exact_exchange.modeled_peak_bytes;
            auto direct=compute_periodic_exchange(basis,system.lattice,q,offsets,cells,k,weights,spin,2,
                options.reciprocal_batch,options.pair_cutoff_bohr,leaf);
            receipt.exact_exchange=direct.receipt;receipt.exchange_identity=direct.receipt.source_identity;
            r.exchange_energy=direct.receipt.energies[0]+direct.receipt.energies[1];result.K=std::move(direct.values);
            for(Count i=0;i<spin.size();++i)result.values[i]-=result.K[i];
            r.shell_calls=add(r.shell_calls,direct.receipt.shell_calls);
            r.primitive_products=add(r.primitive_products,direct.receipt.primitive_pairs);
        }
        r.has_j=r.has_k=true;r.exchange_scale=1.;
        r.electronic_energy=r.one_electron_energy+r.coulomb_energy+r.exchange_energy;
        require_finite(r.electronic_energy);
        for(Count block=0;block<2*nk;++block) {
            double scale=1.,residual=0.;
            for(Count i=0;i<n;++i)for(Count j=0;j<n;++j) {
                const auto value=result.values[block*n2+i*n+j];require_finite(value.real());require_finite(value.imag());
                scale=std::max(scale,std::abs(value));
                residual=std::max(residual,std::abs(value-std::conj(result.values[block*n2+j*n+i])));
            }
            require_finite(scale);require_finite(residual);
            if(residual>1e-10*scale)throw std::runtime_error("MANTA mixed periodic HF Fock is not Hermitian");
            r.fock_hermiticity_residual=std::max(r.fock_hermiticity_residual,residual);
        }
        Digest source;source.text("manta.periodic-mixed-hf-fock.v1");source.text(r.one_electron_identity);
        source.text(receipt.coulomb_identity);source.text(receipt.exchange_identity);r.source_identity=source.hash.finish_hex();
        // The complementary source changes the operator, never the density
        // epoch or frame already recorded by the fixed fitted consumer.
        Digest snapshot;snapshot.text("manta.periodic-mixed-hf-snapshot.v1");snapshot.text(r.source_identity);snapshot.text(r.density_identity);
        for(auto z:result.values)snapshot.value(z);
        snapshot.real(r.one_electron_energy);snapshot.real(r.coulomb_energy);snapshot.real(r.exchange_energy);
        r.snapshot_identity=snapshot.hash.finish_hex();
    }
    void correct(PeriodicRIFockResult& result,const Complex* d) {
        receipt.uncorrected_exchange_energy=result.receipt.exchange_energy;
        if(options.exchange_q0=="strict-zero-mode") return;
        const Count n=basis.nbasis(),n2=square(n),nk=weights.size(),elements=mul(nk,n2);
        std::vector<Complex> product(n2); Sum energy;
        auto accumulate=[](Complex x,Complex& sum,Complex& error) { const auto y=x-error,next=sum+y; error=(next-sum)-y; sum=next; };
        for(Count spin=0;spin<2;++spin) for(Count ik=0;ik<nk;++ik) {
            const Count offset=ik*n2,block=spin*elements+offset;
            for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                Complex value=0,error=0;
                for(Count a=0;a<n;++a) accumulate(overlap[offset+i*n+a]*d[block+a*n+j],value,error);
                product[i*n+j]=value;
            }
            Complex trace=0,trace_error=0;
            for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                Complex value=0,error=0;
                for(Count a=0;a<n;++a) accumulate(product[i*n+a]*overlap[offset+a*n+j],value,error);
                const auto correction=-receipt.madelung*value;
                require_finite(correction.real()); require_finite(correction.imag());
                result.values[block+i*n+j]+=correction;
                accumulate(.5*weights[ik]*d[block+j*n+i]*correction,trace,trace_error);
            }
            require_finite(trace.real()); require_finite(trace.imag());
            if(std::abs(trace.imag())>1e-10*std::max(1.,std::abs(trace.real())))
                throw std::runtime_error("MANTA Madelung exchange energy is not real");
            energy.add(trace.real());
            double scale=1.,residual=0.;
            for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                const auto value=result.values[block+i*n+j];
                require_finite(value.real()); require_finite(value.imag());
                const double magnitude=std::abs(value),error=std::abs(value-std::conj(result.values[block+j*n+i]));
                require_finite(magnitude); require_finite(error);
                scale=std::max(scale,magnitude); residual=std::max(residual,error);
            }
            if(residual>1e-10*scale) throw std::runtime_error("MANTA corrected periodic Fock is not Hermitian");
            result.receipt.fock_hermiticity_residual=std::max(result.receipt.fock_hermiticity_residual,residual);
        }
        receipt.exchange_correction_energy=energy.value;
        auto& r=result.receipt; r.exchange_energy+=energy.value; r.electronic_energy+=energy.value;
        require_finite(r.exchange_energy); require_finite(r.electronic_energy);
        Digest source; source.text(Restricted ? "manta.periodic-ri-rhf.corrected-fock.v1" : "manta.periodic-ri-uhf.corrected-fock.v1"); source.text(r.source_identity);
        source.text(receipt.overlap_identity); source.text(receipt.exchange_correction_identity); r.source_identity=source.hash.finish_hex();
        // Correction changes the physical operator source, not the evaluated
        // density or its AO/Bloch frame. Keep those identities independent.
        Digest snapshot; snapshot.text(Restricted ? "manta.periodic-ri-rhf.corrected-snapshot.v1" : "manta.periodic-ri-uhf.corrected-snapshot.v1"); snapshot.text(r.source_identity); snapshot.text(r.density_identity);
        for(auto z:result.values) snapshot.value(z);
        snapshot.real(r.one_electron_energy); snapshot.real(r.coulomb_energy); snapshot.real(r.exchange_energy); r.snapshot_identity=snapshot.hash.finish_hex();
    }
    PeriodicRIFockResult build(const Complex* d,Count elements) {
        if(elements!=mul(Restricted ? 1 : 2,h.size())) throw std::logic_error("MANTA periodic RI HF density extent mismatch");
        std::vector<Complex> spin(mul(2,h.size()));
        if constexpr(Restricted) {
            for(Count i=0;i<elements;++i) spin[i]=spin[elements+i]=.5*d[i];
        } else std::copy(d,d+elements,spin.begin());
        auto result=fitted.build(h,spin);
        compose(result,spin);
        ++receipt.fock_calls; receipt.shell_calls+=result.receipt.shell_calls;
        receipt.primitive_pairs+=result.receipt.primitive_products;
        correct(result,spin.data());
        if constexpr(Restricted) {
            // The two restricted channels must already agree. Do not average
            // or repair unequal responses into a purported restricted Fock.
            for(Count i=0;i<elements;++i) if(result.values[i]!=result.values[elements+i])
                throw std::runtime_error("MANTA periodic RI RHF spin responses disagree");
        }
        last=result.receipt; return result;
    }
    static void evaluate(const Complex* d,std::size_t dn,Complex* out,std::size_t on,void* pointer) {
        auto& a=*static_cast<Adapter*>(pointer);
        if(dn!=mul(Restricted ? 1 : 2,a.h.size()) || on!=dn) throw std::logic_error("MANTA periodic RI HF callback extent mismatch");
        const auto result=a.build(d,dn);
        for(Count i=0;i<dn;++i) out[i]=result.values[i]-a.h[i%a.h.size()];
    }
};
template<bool Restricted> HFPlan<Restricted> plan_ri_hf(const BasisSet& basis,const BasisSet& auxiliary,const PeriodicSystem& system,const RegularKMesh& mesh,
    Count ng,Count nq,Count nc,const HFOptions<Restricted>& o,const ResourceBudget& budget) {
    HFPlan<Restricted> p; p.electrons=validate<Restricted>(basis,system,o); p.n_basis=basis.nbasis(); p.momenta=mesh.size();
    if constexpr(!Restricted) {
        p.n_beta=(p.electrons-static_cast<Count>(system.multiplicity-1))/2; p.n_alpha=p.electrons-p.n_beta;
    }
    const Count elements=mul(p.momenta,square(p.n_basis)); const auto leaf=unlimited(budget);
    p.overlap=plan_bloch_one_electron(basis,"overlap",nc,p.momenta,o.pair_cutoff_bohr,leaf);
    p.kinetic=plan_bloch_one_electron(basis,"kinetic",nc,p.momenta,o.pair_cutoff_bohr,leaf);
    p.nuclear=plan_periodic_nuclear(basis,system,ng,nc,p.momenta,o.alpha,o.nuclear_cutoff_bohr,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
    const bool fit_j=o.coulomb_algorithm=="ri",fit_k=o.exchange_algorithm=="ri";
    p.fock=plan_periodic_ri_fock(basis,auxiliary,system,ng,fit_k?nq:0,p.momenta,o.fitting,o.ri,leaf,fit_j,fit_k);
    if(!fit_j)p.exact_hartree=plan_periodic_hartree(basis,ng,nc,p.momenta,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
    if(!fit_k)p.exact_exchange=plan_periodic_exchange(basis,nq,nc,p.momenta,2,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
    const bool mixed=!fit_j || !fit_k;
    // Native complementary-source response/receipt, density sum, empty RI
    // offset domain and composition scratch live beside the fixed fit.
    if(mixed)p.mixed_driver_live_bytes=add(16384,add(mul(256,elements),mul(32,add(1,square(p.momenta)))));
    p.two_electron_peak_bytes=add(p.fock.modeled_peak_bytes,add(p.mixed_driver_live_bytes,
        add(p.exact_hartree.modeled_peak_bytes,p.exact_exchange.modeled_peak_bytes)));
    const Count mixed_scalar=mixed?mul(1024,elements):0;
    const Count direct_scalar=add(mixed_scalar,add(p.exact_hartree.scalar_terms,p.exact_exchange.scalar_terms));
    const Count direct_kernel=add(mul(16,mixed_scalar),add(p.exact_hartree.kernel_terms,p.exact_exchange.kernel_terms));
    p.madelung=plan_madelung(system,mesh,o);
    if(o.exchange_q0=="BvK-ewald")
        p.exchange_correction_kernel_terms_per_call=mul(512,mul(elements,p.n_basis));
    validate_centers(basis,system);
    // Include H/S + exports, an extra spin-density owner in the callback,
    // scalar source controls, owned k/weights and a k export, nuclear labels.
    p.driver_live_bytes=add(32768,add(mul(320,elements),add(mul(128,p.nuclear.maximum_nuclear_candidates_per_pair),
        add(mul(256,system.unit_cell.size()),mul(64,p.momenta)))));
    if(o.exchange_q0=="BvK-ewald") p.driver_live_bytes=add(p.driver_live_bytes,add(p.madelung.modeled_peak_bytes,mul(64,square(p.n_basis))));
    const Count fock_work=add(add(p.fock.kernel_terms,direct_kernel),p.exchange_correction_kernel_terms_per_call);
    p.two_electron_kernel_terms_per_call=fock_work;
    const Provider<Restricted> provider{Adapter<Restricted>::evaluate,nullptr,p.fock.retained_source_bytes,p.two_electron_peak_bytes-p.fock.retained_source_bytes,fock_work};
    if constexpr(Restricted)
        p.solver=plan_bounded_periodic_rhf_solver(p.momenta,p.n_basis,p.electrons,p.n_basis,provider,o.scf,0);
    else p.solver=plan_bounded_periodic_uhf_solver(p.momenta,p.n_basis,p.n_alpha,p.n_beta,p.n_basis,provider,o.scf,0);
    p.history_capacity=o.scf.maximum_iterations;
    // Native retained rows, transient vector export and copied Python scalar
    // objects can overlap. Higher-level dictionaries/serialization are charged
    // by the job adapter, independently of these typed scalar snapshots.
    p.history_live_bytes=mul(p.history_capacity,add(mul(3,sizeof(Snapshot<Restricted>)),256));
    p.driver_live_bytes=add(p.driver_live_bytes,add(p.history_live_bytes,mul(4096,p.fock.channels)));
    p.modeled_peak_bytes=add(budget.other_live_bytes,add(p.driver_live_bytes,add(p.solver.peak_owned_numerical_bytes,
        add(p.overlap.modeled_peak_bytes,add(p.kinetic.modeled_peak_bytes,add(p.nuclear.modeled_peak_bytes,p.two_electron_peak_bytes))))));
    p.fock_calls_upper_bound=add(o.scf.maximum_iterations,1);
    p.shell_calls=add(p.overlap.shell_calls,add(p.kinetic.shell_calls,add(p.nuclear.shell_calls,add(p.fock.metric_shell_calls,mul(p.fock_calls_upper_bound,p.fock.build_shell_calls)))));
    p.primitive_pairs=add(p.overlap.primitive_pairs,add(p.kinetic.primitive_pairs,add(p.nuclear.primitive_pairs,add(p.fock.metric_primitive_products,mul(p.fock_calls_upper_bound,p.fock.build_primitive_products)))));
    p.shell_calls=add(p.shell_calls,mul(p.fock_calls_upper_bound,add(p.exact_hartree.shell_calls,p.exact_exchange.shell_calls)));
    p.primitive_pairs=add(p.primitive_pairs,mul(p.fock_calls_upper_bound,add(p.exact_hartree.primitive_pairs,p.exact_exchange.primitive_pairs)));
    p.nuclear_image_candidates_upper_bound=mul(system.unit_cell.size(),p.nuclear.maximum_nuclear_candidates_per_pair);
    const Count driver_work=add(mul(512,mul(elements,p.fock_calls_upper_bound)),add(mul(128,p.nuclear_image_candidates_upper_bound),mul(128,mul(ng,system.unit_cell.size()))));
    p.scalar_terms=add(driver_work,add(p.overlap.scalar_terms,add(p.kinetic.scalar_terms,add(p.nuclear.scalar_terms,
        add(mul(p.fock_calls_upper_bound,add(p.fock.scalar_terms,direct_scalar)),p.solver.maximum_work_units)))));
    p.kernel_terms=add(mul(32,driver_work),add(p.overlap.kernel_terms,add(p.kinetic.kernel_terms,add(p.nuclear.kernel_terms,
        add(fock_work,p.solver.maximum_work_units)))));
    p.scalar_terms=add(p.scalar_terms,add(p.madelung.kernel_terms,mul(p.fock_calls_upper_bound,p.exchange_correction_kernel_terms_per_call)));
    p.kernel_terms=add(p.kernel_terms,p.madelung.kernel_terms);
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic RI HF aggregate memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic RI HF aggregate shell-call budget exceeded");
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA periodic RI HF aggregate primitive-pair budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic RI HF aggregate scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic RI HF aggregate kernel-work budget exceeded");
    return p;
}
template<bool Restricted> HFResult<Restricted> run_ri_hf(const BasisSet& basis,const BasisSet& auxiliary,const PeriodicSystem& system,const RegularKMesh& mesh,
    const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,const md::Matrix& cells,
    const HFOptions<Restricted>& options,const ResourceBudget& budget) {
    const auto o=options; const Count nk=mesh.size(),pairs=square(nk);
    if(offsets.size()!=add(pairs,1) || offsets.front()!=0 || offsets.back()!=static_cast<Count>(q.rows()))
        throw std::invalid_argument("MANTA periodic RI HF offsets must delimit every transfer pair");
    Count maximum=0;
    for(Count i=0;i<pairs;++i) {
        if(offsets[i]>offsets[i+1] || offsets[i+1]>static_cast<Count>(q.rows()))
            throw std::invalid_argument("MANTA periodic RI HF offsets must be monotone and bounded");
        const Count length=offsets[i+1]-offsets[i];
        if(o.exchange_algorithm=="ri" ? ((i/nk==i%nk && length) || (i/nk!=i%nk && !length)) : (i/nk==i%nk && !length))
            throw std::invalid_argument("MANTA periodic RI HF pair segments must match the selected exchange source");
        maximum=std::max(maximum,length);
    }
    HFResult<Restricted> result; result.plan=plan_ri_hf<Restricted>(basis,auxiliary,system,mesh,g.rows(),maximum,cells.rows(),o,budget);
    const auto& p=result.plan; auto& r=result.receipt; r.modeled_peak_bytes=p.modeled_peak_bytes;
    r.coulomb_algorithm=o.coulomb_algorithm;r.exchange_algorithm=o.exchange_algorithm;
    if(o.coulomb_algorithm!="ri" || o.exchange_algorithm!="ri")
        r.algorithm=Restricted?"finite-3d-mixed-ri-rhf":"finite-3d-mixed-ri-uhf";
    if constexpr(!Restricted) { r.n_alpha=p.n_alpha; r.n_beta=p.n_beta; }
    result.scf_history.reserve(p.history_capacity);
    r.domain_translations=cells.rows(); r.domain_hartree_vectors=g.rows(); r.domain_exchange_vectors=q.rows();
    const Count elements=mul(nk,square(p.n_basis));
    auto& k=result.momenta; k.resize(nk,3); std::vector<double> weights(nk,1./nk);
    for(Count i=0;i<nk;++i) k.row(i)=(system.reciprocal_lattice()*mesh.fractional_at(i)).transpose();
    auto leaf=budget;
    leaf.other_live_bytes=p.modeled_peak_bytes-p.overlap.modeled_peak_bytes;
    { auto s=compute_bloch_one_electron(basis,"overlap",cells,k,o.pair_cutoff_bohr,leaf);
      r.overlap_identity=s.receipt.source_identity; r.shell_calls+=s.receipt.shell_calls; r.primitive_pairs+=s.receipt.primitive_pairs; result.overlap=std::move(s.values); }
    leaf.other_live_bytes=p.modeled_peak_bytes-p.kinetic.modeled_peak_bytes;
    { auto t=compute_bloch_one_electron(basis,"kinetic",cells,k,o.pair_cutoff_bohr,leaf);
      r.kinetic_identity=t.receipt.source_identity; r.shell_calls+=t.receipt.shell_calls; r.primitive_pairs+=t.receipt.primitive_pairs; result.hcore=std::move(t.values); }
    leaf.other_live_bytes=p.modeled_peak_bytes-p.nuclear.modeled_peak_bytes;
    { auto v=compute_periodic_nuclear(basis,system,g,cells,k,o.alpha,o.nuclear_cutoff_bohr,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
      r.nuclear_identity=v.receipt.source_identity; r.shell_calls+=v.receipt.nuclear_shell_calls+v.receipt.overlap_shell_calls+v.receipt.fourier_shell_calls;
      r.primitive_pairs+=v.receipt.primitive_pairs;
      for(Count i=0;i<elements;++i) result.hcore[i]+=v.momenta[i]; }
    nuclear_energy(system,g,o,p.nuclear,r);
    compute_madelung(system,mesh,o,p.madelung,r);
    leaf.other_live_bytes=p.modeled_peak_bytes-p.fock.modeled_peak_bytes;
    const bool fit_j=o.coulomb_algorithm=="ri",fit_k=o.exchange_algorithm=="ri";
    const md::Matrix empty(0,3);std::vector<Count> empty_offsets;
    if(!fit_k)empty_offsets.resize(add(pairs,1));
    PeriodicRIFockBuilder fitted(basis,auxiliary,system,k,weights,g,fit_k?q:empty,fit_k?offsets:empty_offsets,o.fitting,o.ri,leaf,fit_j,fit_k);
    r.fitting_info=fitted.info();r.shell_calls+=r.fitting_info.metric_shell_calls;
    r.primitive_pairs+=r.fitting_info.metric_primitive_products;
    if(fit_j) {Digest id;id.text("manta.periodic-ri-J.v1");id.text(r.fitting_info.source_identity);r.coulomb_identity=id.hash.finish_hex();}
    if(fit_k) {Digest id;id.text("manta.periodic-ri-K.v1");id.text(r.fitting_info.source_identity);r.exchange_identity=id.hash.finish_hex();}
    Adapter<Restricted> adapter{basis,fitted,weights,result.hcore,result.overlap,o,r,{},system,k,g,q,offsets,cells,p,budget};
    const Provider<Restricted> provider{Adapter<Restricted>::evaluate,&adapter,p.fock.retained_source_bytes,p.two_electron_peak_bytes-p.fock.retained_source_bytes,
        p.two_electron_kernel_terms_per_call};
    SolverCaps<Restricted> caps;
    caps.maximum_owned_numerical_bytes=p.solver.peak_owned_numerical_bytes; caps.maximum_total_numerical_bytes=p.modeled_peak_bytes;
    caps.maximum_work_units=p.solver.maximum_work_units; caps.maximum_fock_calls=o.scf.maximum_iterations;
    caps.other_live_numerical_bytes=p.modeled_peak_bytes-p.solver.peak_owned_numerical_bytes-p.two_electron_peak_bytes-mul(32,elements);
    Trace<Restricted> trace{result.scf_history,p.history_capacity};
    if constexpr(Restricted)
        result.numerical.emplace(solve_bounded_periodic_rhf({nk,p.n_basis,p.electrons,r.nuclear_energy,
            result.overlap.data(),static_cast<std::size_t>(elements),result.hcore.data(),static_cast<std::size_t>(elements)},
            provider,o.scf,caps,Trace<Restricted>::append,&trace));
    else result.numerical.emplace(solve_bounded_periodic_uhf({nk,p.n_basis,p.n_alpha,p.n_beta,r.nuclear_energy,
        result.overlap.data(),static_cast<std::size_t>(elements),result.hcore.data(),static_cast<std::size_t>(elements)},
        provider,o.scf,caps,Trace<Restricted>::append,&trace));
    if(result.numerical->converged()) {
        const Count density_elements=mul(Restricted ? 1 : 2,elements);
        const auto rebuilt=adapter.build(result.numerical->density_data(),density_elements);
        for(Count i=0;i<density_elements;++i) r.confirmation_fock_error=std::max(r.confirmation_fock_error,std::abs(rebuilt.values[i]-result.numerical->fock_data()[i]));
        r.confirmation_energy_error=std::abs(r.nuclear_energy+rebuilt.receipt.electronic_energy-result.numerical->final_snapshot().energy_per_cell);
        r.confirmed=r.confirmation_fock_error<=o.confirmation_fock_tolerance && r.confirmation_energy_error<=o.confirmation_energy_tolerance;
    }
    r.one_electron_energy=adapter.last.one_electron_energy; r.coulomb_energy=adapter.last.coulomb_energy;
    r.exchange_energy=adapter.last.exchange_energy; r.energy=r.nuclear_energy+adapter.last.electronic_energy;
    r.final_density_identity=adapter.last.density_identity; r.final_fock_identity=adapter.last.snapshot_identity;
    Digest source;
    if constexpr(Restricted) source.text(o.exchange_q0=="strict-zero-mode"
        ? "manta.periodic-ri-rhf.source.v1;neutral-all-electron;full-regular-mesh;exchange-q0-omitted"
        : "manta.periodic-ri-rhf.source.v1;neutral-all-electron;full-regular-mesh;exchange-BvK-ewald");
    else source.text(o.exchange_q0=="strict-zero-mode"
        ? "manta.periodic-ri-uhf.source.v1;neutral-all-electron;full-regular-mesh;exchange-q0-omitted"
        : "manta.periodic-ri-uhf.source.v2;neutral-all-electron;full-regular-mesh;exchange-BvK-ewald");
    source.text(r.overlap_identity); source.text(r.kinetic_identity); source.text(r.nuclear_identity); source.text(r.nuclear_energy_identity);
    source.text(adapter.last.source_identity);
    if constexpr(Restricted) source.word(p.electrons);
    else { source.word(p.n_alpha); source.word(p.n_beta); }
    for(auto n:mesh.mesh()) source.word(n); for(auto n:mesh.is_shift()) source.word(n);
    r.source_identity=source.hash.finish_hex(); return result;
}
} // namespace
PeriodicRIUHFPlan plan_periodic_ri_uhf(const BasisSet& b,const BasisSet& a,const PeriodicSystem& s,const RegularKMesh& k,
    Count ng,Count nq,Count nc,const PeriodicRIUHFOptions& o,const ResourceBudget& budget) {
    return plan_ri_hf<false>(b,a,s,k,ng,nq,nc,o,budget);
}
PeriodicRIRHFPlan plan_periodic_ri_rhf(const BasisSet& b,const BasisSet& a,const PeriodicSystem& s,const RegularKMesh& k,
    Count ng,Count nq,Count nc,const PeriodicRIRHFOptions& o,const ResourceBudget& budget) {
    return plan_ri_hf<true>(b,a,s,k,ng,nq,nc,o,budget);
}
PeriodicRIUHFResult run_periodic_ri_uhf(const BasisSet& b,const BasisSet& a,const PeriodicSystem& s,const RegularKMesh& k,
    const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,const md::Matrix& cells,
    const PeriodicRIUHFOptions& o,const ResourceBudget& budget) {
    return run_ri_hf<false>(b,a,s,k,g,q,offsets,cells,o,budget);
}
PeriodicRIRHFResult run_periodic_ri_rhf(const BasisSet& b,const BasisSet& a,const PeriodicSystem& s,const RegularKMesh& k,
    const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,const md::Matrix& cells,
    const PeriodicRIRHFOptions& o,const ResourceBudget& budget) {
    return run_ri_hf<true>(b,a,s,k,g,q,offsets,cells,o,budget);
}
} // namespace vibeqc::manta
