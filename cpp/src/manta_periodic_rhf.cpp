#include "manta_periodic_hf_source.hpp"

namespace vibeqc::manta {
namespace {
struct Trace {
    std::vector<BoundedPeriodicRHFSnapshot>& rows;
    Count capacity;
    static bool append(const BoundedPeriodicRHFSnapshot& snapshot,void* pointer) {
        auto& trace=*static_cast<Trace*>(pointer);
        if(trace.rows.size()>=trace.capacity || snapshot.evaluated_iteration!=trace.rows.size()+1)
            throw std::logic_error("MANTA periodic RHF trace exceeded its admitted iteration extent");
        trace.rows.push_back(snapshot);
        return true;
    }
};
Count validate(const BasisSet& basis,const PeriodicSystem& system,const PeriodicRHFOptions& o) {
    if(system.dim!=3 || system.charge!=0 || system.multiplicity!=1 || system.unit_cell.empty())
        throw std::invalid_argument("MANTA periodic RHF requires a neutral all-electron 3D singlet with nuclei");
    if(!std::isfinite(o.confirmation_fock_tolerance) || o.confirmation_fock_tolerance<=0
       || !std::isfinite(o.confirmation_energy_tolerance) || o.confirmation_energy_tolerance<=0)
        throw std::invalid_argument("MANTA periodic RHF confirmation tolerances must be finite positive");
    if(o.exchange_q0!="strict-zero-mode" && o.exchange_q0!="BvK-ewald")
        throw std::invalid_argument("MANTA periodic RHF exchange_q0 must be strict-zero-mode or BvK-ewald");
    if(!std::isfinite(o.madelung_tolerance) || o.madelung_tolerance<=0 || o.madelung_tolerance>=1
       || (o.madelung_alpha && (!std::isfinite(*o.madelung_alpha) || *o.madelung_alpha<=0)))
        throw std::invalid_argument("MANTA periodic RHF Madelung alpha must be positive and tolerance in (0,1)");
    Count electrons=0;
    for(const auto& a:system.unit_cell) {
        if(a.Z<=0) throw std::invalid_argument("MANTA periodic RHF requires positive all-electron nuclear charges");
        electrons=add(electrons,static_cast<Count>(a.Z));
    }
    if(electrons%2 || electrons/2>=basis.nbasis())
        throw std::invalid_argument("MANTA periodic RHF requires even electrons and at least one virtual orbital");
    return electrons;
}
struct Adapter {
    const BasisSet& basis; const PeriodicSystem& system;
    const md::Matrix& g; const md::Matrix& q; const std::vector<Count>& offsets;
    const md::Matrix& cells; const md::Matrix& k; const std::vector<double>& weights;
    const std::vector<Complex>& h; const std::vector<Complex>& overlap; const PeriodicRHFOptions& options;
    ResourceBudget budget; PeriodicRHFReceipt& receipt; PeriodicFockReceipt last;
    void correct(PeriodicFockResult& result,const Complex* d) {
        receipt.uncorrected_exchange_energy=result.receipt.exchange_energy;
        if(options.exchange_q0=="strict-zero-mode") return;
        const Count n=basis.nbasis(),n2=square(n),nk=weights.size(),elements=mul(nk,n2);
        std::vector<Complex> product(n2); Sum energy;
        auto accumulate=[](Complex x,Complex& sum,Complex& error) { const auto y=x-error,next=sum+y; error=(next-sum)-y; sum=next; };
        for(Count ik=0;ik<nk;++ik) {
            const Count offset=ik*n2;
            for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                Complex value=0,error=0;
                for(Count a=0;a<n;++a) accumulate(overlap[offset+i*n+a]*d[offset+a*n+j],value,error);
                product[i*n+j]=value;
            }
            Complex trace=0,trace_error=0;
            for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                Complex value=0,error=0;
                for(Count a=0;a<n;++a) accumulate(product[i*n+a]*overlap[offset+a*n+j],value,error);
                const auto correction=-.5*receipt.madelung*value;
                require_finite(correction.real()); require_finite(correction.imag());
                result.values[offset+i*n+j]+=correction;
                result.values[elements+offset+i*n+j]+=correction;
                accumulate(.5*weights[ik]*d[offset+j*n+i]*correction,trace,trace_error);
            }
            require_finite(trace.real()); require_finite(trace.imag());
            if(std::abs(trace.imag())>1e-10*std::max(1.,std::abs(trace.real())))
                throw std::runtime_error("MANTA Madelung exchange energy is not real");
            energy.add(trace.real());
            double scale=1.,residual=0.;
            for(Count i=0;i<n;++i) for(Count j=0;j<n;++j) {
                const auto value=result.values[offset+i*n+j];
                require_finite(value.real()); require_finite(value.imag());
                const double magnitude=std::abs(value),error=std::abs(value-std::conj(result.values[offset+j*n+i]));
                require_finite(magnitude); require_finite(error);
                scale=std::max(scale,magnitude); residual=std::max(residual,error);
            }
            if(residual>1e-10*scale) throw std::runtime_error("MANTA corrected periodic Fock is not Hermitian");
            result.receipt.fock_hermiticity_residual=std::max(result.receipt.fock_hermiticity_residual,residual);
        }
        receipt.exchange_correction_energy=energy.value;
        auto& r=result.receipt; r.exchange_energy+=energy.value; r.electronic_energy+=energy.value;
        require_finite(r.exchange_energy); require_finite(r.electronic_energy);
        Digest source; source.text("manta.periodic-rhf.corrected-fock.v1"); source.text(r.source_identity);
        source.text(receipt.overlap_identity); source.text(receipt.exchange_correction_identity); r.source_identity=source.hash.finish_hex();
        Digest density; density.text(r.source_identity); density.text(r.density_identity); r.density_identity=density.hash.finish_hex();
        Digest snapshot; snapshot.text("manta.periodic-rhf.corrected-snapshot.v1"); snapshot.text(r.source_identity); snapshot.text(r.density_identity);
        for(auto z:result.values) snapshot.value(z);
        snapshot.real(r.one_electron_energy); snapshot.real(r.coulomb_energy); snapshot.real(r.exchange_energy); r.snapshot_identity=snapshot.hash.finish_hex();
    }
    PeriodicFockResult build(const Complex* d,Count elements) {
        std::vector<Complex> spin(mul(2,elements));
        for(Count i=0;i<elements;++i) spin[i]=spin[elements+i]=.5*d[i];
        auto result=compute_periodic_fock(basis,system.lattice,g,q,offsets,cells,k,weights,h,spin,
            options.reciprocal_batch,options.pair_cutoff_bohr,budget);
        ++receipt.fock_calls; receipt.shell_calls+=result.hartree.receipt.shell_calls+result.exchange.receipt.shell_calls;
        receipt.primitive_pairs+=result.hartree.receipt.primitive_pairs+result.exchange.receipt.primitive_pairs;
        correct(result,d);
        last=result.receipt; return result;
    }
    static void evaluate(const Complex* d,std::size_t dn,Complex* out,std::size_t on,void* pointer) {
        auto& a=*static_cast<Adapter*>(pointer);
        if(dn!=a.h.size() || on!=dn) throw std::logic_error("MANTA periodic RHF callback extent mismatch");
        const auto result=a.build(d,dn);
        for(Count i=0;i<dn;++i) out[i]=result.values[i]-a.h[i];
    }
};
} // namespace
PeriodicRHFOptions::PeriodicRHFOptions() {
    scf.maximum_iterations=64; scf.maximum_diis_history=0; scf.jacobi_max_sweeps=64;
    scf.jacobi_relative_tolerance=2e-15;
    scf.overlap_rank_absolute_floor=scf.overlap_rank_relative_floor=1e-12;
    scf.overlap_negative_absolute_tolerance=scf.overlap_negative_relative_tolerance=1e-12;
    scf.hermitian_absolute_tolerance=scf.hermitian_relative_tolerance=2e-12;
    scf.maximum_overlap_projection_error=scf.maximum_hcore_projection_error=scf.maximum_fock_projection_error=1e-11;
    scf.algebra_absolute_tolerance=scf.algebra_relative_tolerance=2e-11;
    scf.eigen_relative_tolerance=2e-11; scf.commutator_tolerance=1e-9;
    scf.density_closure_absolute_tolerance=scf.density_closure_relative_tolerance=1e-10;
    scf.energy_change_tolerance=1e-12; scf.minimum_band_gap_hartree=1e-6;
}
PeriodicRHFPlan plan_periodic_rhf(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,
    Count ng,Count nq,Count nc,const PeriodicRHFOptions& o,const ResourceBudget& budget) {
    PeriodicRHFPlan p; p.electrons=validate(basis,system,o); p.n_basis=basis.nbasis(); p.momenta=mesh.size();
    const Count elements=mul(p.momenta,square(p.n_basis)); const auto leaf=unlimited(budget);
    p.overlap=plan_bloch_one_electron(basis,"overlap",nc,p.momenta,o.pair_cutoff_bohr,leaf);
    p.kinetic=plan_bloch_one_electron(basis,"kinetic",nc,p.momenta,o.pair_cutoff_bohr,leaf);
    p.nuclear=plan_periodic_nuclear(basis,system,ng,nc,p.momenta,o.alpha,o.nuclear_cutoff_bohr,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
    p.fock=plan_periodic_fock(basis,ng,nq,nc,p.momenta,o.reciprocal_batch,o.pair_cutoff_bohr,leaf);
    p.madelung=plan_madelung(system,mesh,o);
    if(o.exchange_q0=="BvK-ewald")
        p.exchange_correction_kernel_terms_per_call=mul(256,mul(elements,p.n_basis));
    validate_centers(basis,system);
    // Include H/S + exports, an extra spin-density owner in the callback,
    // scalar source controls, generated k/weights and nuclear selector labels.
    p.driver_live_bytes=add(32768,add(mul(192,elements),add(mul(128,p.nuclear.maximum_nuclear_candidates_per_pair),
        add(mul(256,system.unit_cell.size()),mul(64,p.momenta)))));
    if(o.exchange_q0=="BvK-ewald") p.driver_live_bytes=add(p.driver_live_bytes,add(p.madelung.modeled_peak_bytes,mul(64,square(p.n_basis))));
    const Count fock_work=add(p.fock.kernel_terms,p.exchange_correction_kernel_terms_per_call);
    const BoundedPeriodicRHFTwoElectronProvider provider{Adapter::evaluate,nullptr,0,p.fock.modeled_peak_bytes,fock_work};
    p.solver=plan_bounded_periodic_rhf_solver(p.momenta,p.n_basis,p.electrons,p.n_basis,provider,o.scf,0);
    p.history_capacity=o.scf.maximum_iterations;
    // Native retained rows, transient vector export and copied Python scalar
    // objects can overlap. Higher-level dictionaries/serialization are charged
    // by the job adapter, independently of these typed scalar snapshots.
    p.history_live_bytes=mul(p.history_capacity,add(mul(3,sizeof(BoundedPeriodicRHFSnapshot)),256));
    p.driver_live_bytes=add(p.driver_live_bytes,p.history_live_bytes);
    p.modeled_peak_bytes=add(budget.other_live_bytes,add(p.driver_live_bytes,add(p.solver.peak_owned_numerical_bytes,
        add(p.overlap.modeled_peak_bytes,add(p.kinetic.modeled_peak_bytes,add(p.nuclear.modeled_peak_bytes,p.fock.modeled_peak_bytes))))));
    p.fock_calls_upper_bound=add(o.scf.maximum_iterations,1);
    p.shell_calls=add(p.overlap.shell_calls,add(p.kinetic.shell_calls,add(p.nuclear.shell_calls,mul(p.fock_calls_upper_bound,p.fock.shell_calls))));
    p.primitive_pairs=add(p.overlap.primitive_pairs,add(p.kinetic.primitive_pairs,add(p.nuclear.primitive_pairs,mul(p.fock_calls_upper_bound,p.fock.primitive_pairs))));
    p.nuclear_image_candidates_upper_bound=mul(system.unit_cell.size(),p.nuclear.maximum_nuclear_candidates_per_pair);
    const Count driver_work=add(mul(256,mul(elements,p.fock_calls_upper_bound)),add(mul(128,p.nuclear_image_candidates_upper_bound),mul(128,mul(ng,system.unit_cell.size()))));
    p.scalar_terms=add(driver_work,add(p.overlap.scalar_terms,add(p.kinetic.scalar_terms,add(p.nuclear.scalar_terms,
        add(mul(p.fock_calls_upper_bound,p.fock.scalar_terms),p.solver.maximum_work_units)))));
    p.kernel_terms=add(mul(32,driver_work),add(p.overlap.kernel_terms,add(p.kinetic.kernel_terms,add(p.nuclear.kernel_terms,
        add(fock_work,p.solver.maximum_work_units)))));
    p.scalar_terms=add(p.scalar_terms,add(p.madelung.kernel_terms,mul(p.fock_calls_upper_bound,p.exchange_correction_kernel_terms_per_call)));
    p.kernel_terms=add(p.kernel_terms,p.madelung.kernel_terms);
    cap(p.modeled_peak_bytes,budget.memory_bytes,"MANTA periodic RHF aggregate memory budget exceeded");
    cap(p.shell_calls,budget.shell_calls,"MANTA periodic RHF aggregate shell-call budget exceeded");
    cap(p.primitive_pairs,budget.primitive_quartets,"MANTA periodic RHF aggregate primitive-pair budget exceeded");
    cap(p.scalar_terms,budget.scalar_terms,"MANTA periodic RHF aggregate scalar-work budget exceeded");
    cap(p.kernel_terms,budget.kernel_terms,"MANTA periodic RHF aggregate kernel-work budget exceeded");
    return p;
}
PeriodicRHFResult run_periodic_rhf(const BasisSet& basis,const PeriodicSystem& system,const RegularKMesh& mesh,
    const md::Matrix& g,const md::Matrix& q,const std::vector<Count>& offsets,const md::Matrix& cells,
    const PeriodicRHFOptions& options,const ResourceBudget& budget) {
    const auto o=options; const Count nk=mesh.size(),pairs=square(nk);
    if(offsets.size()!=add(pairs,1) || offsets.front()!=0 || offsets.back()!=static_cast<Count>(q.rows()))
        throw std::invalid_argument("MANTA periodic RHF offsets must delimit every transfer pair");
    Count maximum=0;
    for(Count i=0;i<pairs;++i) { if(offsets[i]>offsets[i+1]) throw std::invalid_argument("MANTA periodic RHF offsets must be monotone"); maximum=std::max(maximum,offsets[i+1]-offsets[i]); }
    PeriodicRHFResult result; result.plan=plan_periodic_rhf(basis,system,mesh,g.rows(),maximum,cells.rows(),o,budget);
    const auto& p=result.plan; auto& r=result.receipt; r.modeled_peak_bytes=p.modeled_peak_bytes;
    result.scf_history.reserve(p.history_capacity);
    r.domain_translations=cells.rows(); r.domain_hartree_vectors=g.rows(); r.domain_exchange_vectors=q.rows();
    const Count elements=mul(nk,square(p.n_basis));
    md::Matrix k(nk,3); std::vector<double> weights(nk,1./nk);
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
    Adapter adapter{basis,system,g,q,offsets,cells,k,weights,result.hcore,result.overlap,o,leaf,r,{}};
    const BoundedPeriodicRHFTwoElectronProvider provider{Adapter::evaluate,&adapter,0,p.fock.modeled_peak_bytes,
        add(p.fock.kernel_terms,p.exchange_correction_kernel_terms_per_call)};
    BoundedPeriodicRHFCaps caps;
    caps.maximum_owned_numerical_bytes=p.solver.peak_owned_numerical_bytes; caps.maximum_total_numerical_bytes=p.modeled_peak_bytes;
    caps.maximum_work_units=p.solver.maximum_work_units; caps.maximum_fock_calls=o.scf.maximum_iterations;
    caps.other_live_numerical_bytes=p.modeled_peak_bytes-p.solver.peak_owned_numerical_bytes-p.fock.modeled_peak_bytes-mul(32,elements);
    Trace trace{result.scf_history,p.history_capacity};
    result.numerical.emplace(solve_bounded_periodic_rhf({nk,p.n_basis,p.electrons,r.nuclear_energy,
        result.overlap.data(),static_cast<std::size_t>(elements),result.hcore.data(),static_cast<std::size_t>(elements)},
        provider,o.scf,caps,Trace::append,&trace));
    if(result.numerical->converged()) {
        const auto rebuilt=adapter.build(result.numerical->density_data(),elements);
        for(Count i=0;i<elements;++i) r.confirmation_fock_error=std::max(r.confirmation_fock_error,std::abs(rebuilt.values[i]-result.numerical->fock_data()[i]));
        r.confirmation_energy_error=std::abs(r.nuclear_energy+rebuilt.receipt.electronic_energy-result.numerical->final_snapshot().energy_per_cell);
        r.confirmed=r.confirmation_fock_error<=o.confirmation_fock_tolerance && r.confirmation_energy_error<=o.confirmation_energy_tolerance;
    }
    r.one_electron_energy=adapter.last.one_electron_energy; r.coulomb_energy=adapter.last.coulomb_energy;
    r.exchange_energy=adapter.last.exchange_energy; r.energy=r.nuclear_energy+adapter.last.electronic_energy;
    r.final_density_identity=adapter.last.density_identity; r.final_fock_identity=adapter.last.snapshot_identity;
    Digest source; source.text(o.exchange_q0=="strict-zero-mode"
        ? "manta.periodic-rhf.source.v1;neutral-all-electron;full-regular-mesh;exchange-q0-omitted"
        : "manta.periodic-rhf.source.v2;neutral-all-electron;full-regular-mesh;exchange-BvK-ewald");
    source.text(r.overlap_identity); source.text(r.kinetic_identity); source.text(r.nuclear_identity); source.text(r.nuclear_energy_identity);
    source.text(adapter.last.source_identity); for(auto n:mesh.mesh()) source.word(n); for(auto n:mesh.is_shift()) source.word(n);
    r.source_identity=source.hash.finish_hex(); return result;
}
} // namespace vibeqc::manta
