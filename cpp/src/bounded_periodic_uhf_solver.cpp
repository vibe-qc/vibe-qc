#include "bounded_periodic_scf_kernel.hpp"
#include "vibeqc/bounded_periodic_uhf_solver.hpp"
#include "bounded_periodic_scf_numerics.hpp"

namespace vibeqc {
BoundedPeriodicUHFMemoryPlan plan_bounded_periodic_uhf_solver(
    I k,I n,I alpha,I beta,I r,const BoundedPeriodicUHFTwoElectronProvider& provider,const Options& o,I other) {
    controls(o);
    const I occupied=std::max(alpha,beta);
    if (!k || !n || !occupied || occupied>=n || r>n
        || (r && r<=occupied)) throw std::invalid_argument("bounded periodic UHF requires fixed-spin dimensions and at least one virtual band");
    if (!provider.evaluate || !provider.maximum_work_units_per_call)
        throw std::invalid_argument("bounded periodic UHF requires a bounded native two-electron callback");
    BoundedPeriodicUHFMemoryPlan m; m.n_kpoints=k; m.n_basis=n; m.n_occupied=occupied; m.n_alpha=alpha; m.n_beta=beta; m.retained_rank=r;
    const I nn=mul(n,n),knn=mul(k,nn),nr=mul(n,r),knr=mul(k,nr),kr=mul(k,r);
    m.jacobi_workspace_bytes=add(mul(32,nn),mul(8,n));
    m.overlap_discovery_owned_bytes=m.jacobi_workspace_bytes;
    m.borrowed_input_bytes=mul(32,knn); m.provider_retained_bytes=provider.retained_numerical_bytes;
    m.provider_workspace_bytes=provider.peak_workspace_numerical_bytes; m.other_live_numerical_bytes=other;
    m.input_validation_work_units=mul(512,add(knn,1));
    const I n3=mul(nn,n),r3=mul(mul(r,r),r);
    m.overlap_discovery_work_units=mul(1024,mul(k,add(mul(add(o.jacobi_max_sweeps,2),n3),add(nn,1))));
    m.peak_owned_numerical_bytes=m.overlap_discovery_owned_bytes;
    m.maximum_work_units=add(m.input_validation_work_units,m.overlap_discovery_work_units);
    if (r) {
        m.orthogonalizer_bytes=mul(16,knr); m.density_bytes=mul(32,knn); m.fock_bytes=m.density_bytes;
        m.coefficient_bytes=mul(32,knr); m.orbital_energy_bytes=mul(16,kr);
        m.contraction_workspace_bytes=mul(32,nr);
        m.output_numerical_bytes=add(add(m.density_bytes,m.fock_bytes),add(m.coefficient_bytes,m.orbital_energy_bytes));
        m.peak_owned_numerical_bytes=add(m.output_numerical_bytes,
            add(m.orthogonalizer_bytes,add(m.jacobi_workspace_bytes,m.contraction_workspace_bytes)));
        const I contraction=add(mul(nn,r),mul(n,mul(r,r)));
        m.orthogonalizer_work_units=add(m.overlap_discovery_work_units,mul(1024,mul(k,add(contraction,1))));
        const I iteration_math=add(mul(add(o.jacobi_max_sweeps,2),r3),
            add(mul(8,contraction),add(mul(nn,m.n_occupied),add(nn,add(nr,1)))));
        m.initial_density_work_units=mul(4096,mul(k,iteration_math));
        m.work_units_per_evaluated_iteration=add(m.initial_density_work_units,provider.maximum_work_units_per_call);
        m.maximum_work_units=add(m.maximum_work_units,add(m.orthogonalizer_work_units,
            add(m.initial_density_work_units,mul(o.maximum_iterations,m.work_units_per_evaluated_iteration))));
    }
    m.total_numerical_bytes=add(add(m.borrowed_input_bytes,m.provider_retained_bytes),
        add(other,add(m.peak_owned_numerical_bytes,r?m.provider_workspace_bytes:0)));
    extent(mul(2,knn),mul(32,knn)); extent(mul(2,knr),mul(32,knr)); extent(nn,mul(16,nn)); extent(nr,mul(16,nr));
    extent(mul(2,kr),mul(16,kr)); extent(n,mul(8,n));
    return m;
}

void BoundedPeriodicUHFResult::require_live() const {
    if (!snapshot_.evaluated_iteration || density_.size()!=memory_.density_bytes/16
        || fock_.size()!=memory_.fock_bytes/16 || coefficients_.size()!=memory_.coefficient_bytes/16
        || energies_.size()!=memory_.orbital_energy_bytes/8)
        throw std::logic_error("bounded periodic UHF result is consumed or not evaluated");
}
const Z* BoundedPeriodicUHFResult::density_data() const { require_live(); return density_.data(); }
const Z* BoundedPeriodicUHFResult::fock_data() const { require_live(); return fock_.data(); }
const Z* BoundedPeriodicUHFResult::coefficients_data() const { require_live(); return coefficients_.data(); }
const double* BoundedPeriodicUHFResult::orbital_energies_data() const { require_live(); return energies_.data(); }

BoundedPeriodicUHFResult detail::BoundedPeriodicSCFKernel::uhf(
    const BoundedPeriodicUHFInput& input,const BoundedPeriodicUHFTwoElectronProvider& provider,
    const Options& o,const BoundedPeriodicUHFCaps& caps,
    BoundedPeriodicUHFProgressCallback progress,void* progress_context,const double* interaction_energy) {
    for (I x:{caps.maximum_owned_numerical_bytes,caps.maximum_total_numerical_bytes,caps.maximum_work_units,caps.maximum_fock_calls})
        if (!x) throw std::invalid_argument("bounded periodic UHF caps must be positive");
    const I k=input.n_kpoints,n=input.n_basis;
    auto memory=plan_bounded_periodic_uhf_solver(k,n,input.n_alpha,input.n_beta,0,provider,o,caps.other_live_numerical_bytes);
    admit(memory,caps); limit(memory.maximum_work_units,caps.maximum_work_units,"bounded periodic UHF overlap preflight work cap exceeded");
    const I nn=mul(n,n),knn=mul(k,nn); view(input.overlap,input.overlap_elements,knn); view(input.hcore,input.hcore_elements,knn);
    finite(input.nuclear_repulsion_energy_per_cell);
    BoundedPeriodicUHFResult out; auto& d=out.diagnostics_;
    d.minimum_overlap_eigenvalue=std::numeric_limits<double>::infinity();
    d.minimum_retained_overlap_eigenvalue=std::numeric_limits<double>::infinity();
    I work=memory.input_validation_work_units;
    for (I point=0;point<k;++point) {
        audit(input.overlap+point*nn,n,o.hermitian_absolute_tolerance,o.hermitian_relative_tolerance,
            o.maximum_overlap_projection_error,d.maximum_raw_overlap_hermitian_defect,d.maximum_overlap_projection_frobenius);
        audit(input.hcore+point*nn,n,o.hermitian_absolute_tolerance,o.hermitian_relative_tolerance,
            o.maximum_hcore_projection_error,d.maximum_raw_hcore_hermitian_defect,d.maximum_hcore_projection_frobenius);
    }
    work=add(work,memory.overlap_discovery_work_units);
    std::vector<Z> matrix(nn),vectors(nn);
    std::vector<double> values(n);
    I r=0;
    for (I point=0;point<k;++point) {
        const I rank=overlap_eigen(input.overlap+point*nn,n,matrix.data(),vectors.data(),values.data(),o,d);
        if (rank<=memory.n_occupied) throw std::invalid_argument("bounded periodic UHF retained overlap rank lacks an occupied-plus-virtual space");
        if (point && rank!=r) throw std::invalid_argument("bounded periodic UHF overlap retained rank differs across k points");
        r=rank;
    }
    memory=plan_bounded_periodic_uhf_solver(k,n,input.n_alpha,input.n_beta,r,provider,o,caps.other_live_numerical_bytes);
    admit(memory,caps);
    const I preparation=add(memory.orthogonalizer_work_units,memory.initial_density_work_units);
    limit(add(work,add(preparation,memory.work_units_per_evaluated_iteration)),caps.maximum_work_units,
          "bounded periodic UHF initial complete evaluation work cap exceeded");
    out.memory_=memory;
    const I nr=mul(n,r),knr=mul(k,nr);
    std::vector<Z> x(knr),a(nr),b(nr);
    out.density_.resize(mul(2,knn)); out.fock_.resize(mul(2,knn)); out.coefficients_.resize(mul(2,knr)); out.energies_.resize(mul(2,mul(k,r)));
    const std::array<I,2> occupations{input.n_alpha,input.n_beta};
    work=add(work,preparation);
    for (I point=0;point<k;++point) {
        const Z* s=input.overlap+point*nn; const Z* h=input.hcore+point*nn; Z* xp=x.data()+point*nr;
        const I rank=overlap_eigen(s,n,matrix.data(),vectors.data(),values.data(),o,d);
        if (rank!=r) throw std::logic_error("bounded periodic UHF overlap rank changed between immutable-input passes");
        for (I col=0;col<r;++col) {
            const double lambda=values[n-r+col];
            d.minimum_retained_overlap_eigenvalue=std::min(d.minimum_retained_overlap_eigenvalue,lambda);
            const double root=finite(std::sqrt(lambda));
            for (I row=0;row<n;++row) xp[row*r+col]=finite(vectors[row*n+n-r+col]/root);
        }
        apply(n,r,[s,n](I i,I j){return hermitian(s,n,i,j);},xp,a.data());
        left_product(n,r,xp,a.data(),matrix.data());
        for (I i=0;i<r;++i) for (I j=0;j<r;++j)
            check(matrix[i*r+j],double(i==j),o,d.maximum_orthogonalizer_metric_error,
                  "bounded periodic UHF canonical orthogonalizer failed original metric audit");
        for(I spin=0;spin<2;++spin) {
            const I block=spin*k+point;
            diagonalize(n,r,[h,n](I i,I j){return hermitian(h,n,i,j);},xp,matrix.data(),vectors.data(),
                a.data(),b.data(),values.data(),out.coefficients_.data()+block*nr,out.energies_.data()+block*r,o,d);
            density_from_coefficients(n,r,occupations[spin],out.coefficients_.data()+block*nr,out.density_.data()+block*nn,1.0);
        }
    }
    double previous_energy=0;
    for (I iteration=1;iteration<=o.maximum_iterations;++iteration) {
        work=add(work,memory.work_units_per_evaluated_iteration);
        limit(work,caps.maximum_work_units,"bounded periodic UHF iteration work reservation failed");
        std::fill(out.fock_.begin(),out.fock_.end(),Z(std::numeric_limits<double>::quiet_NaN(),0));
        provider.evaluate(out.density_.data(),out.density_.size(),out.fock_.data(),out.fock_.size(),provider.context);
        const double interaction=interaction_energy ? finite(*interaction_energy) : 0.;
        BoundedPeriodicUHFSnapshot snap; snap.evaluated_iteration=iteration; snap.fock_calls=iteration; snap.charged_work_units=work;
        snap.global_homo=-std::numeric_limits<double>::infinity(); snap.global_lumo=std::numeric_limits<double>::infinity();
        snap.spin_homo.fill(-std::numeric_limits<double>::infinity());
        snap.spin_lumo.fill(std::numeric_limits<double>::infinity());
        RealSum energy,raw_energy; double commutator=0; bool density_closed=true;
        const double weight=1.0/static_cast<double>(k);
        for (I spin=0;spin<2;++spin) for (I point=0;point<k;++point) {
            const I block=spin*k+point,occupied=occupations[spin];
            const Z* s=input.overlap+point*nn; const Z* h=input.hcore+point*nn;
            Z* f=out.fock_.data()+block*nn; const Z* density=out.density_.data()+block*nn;
            for (I ij=0;ij<nn;++ij) f[ij]=finite(finite(f[ij])+h[ij]);
            for (I i=0;i<n;++i) for (I j=0;j<n;++j)
                raw_energy.push(interaction_energy
                    ? finite(weight*finite(density[i*n+j]*h[j*n+i]).real())
                    : finite(0.5*weight*finite(density[i*n+j]*(h[j*n+i]+f[j*n+i])).real()));
            audit(f,n,o.hermitian_absolute_tolerance,o.hermitian_relative_tolerance,o.maximum_fock_projection_error,
                d.maximum_raw_fock_hermitian_defect,d.maximum_fock_projection_frobenius);
            project_in_place(f,n);
            for (I i=0;i<n;++i) for (I j=0;j<n;++j)
                energy.push(interaction_energy
                    ? finite(weight*finite(density[i*n+j]*hermitian(h,n,j,i)).real())
                    : finite(0.5*weight*finite(density[i*n+j]*(hermitian(h,n,j,i)+f[j*n+i])).real()));
            Z* c=out.coefficients_.data()+block*nr; double* eps=out.energies_.data()+block*r;
            snap.maximum_projected_eigen_relative_residual=std::max(snap.maximum_projected_eigen_relative_residual,
                diagonalize(n,r,[f,n](I i,I j){return f[i*n+j];},x.data()+point*nr,matrix.data(),vectors.data(),
                    a.data(),b.data(),values.data(),c,eps,o,d));
            density_audits(n,r,occupied,s,f,density,x.data()+point*nr,c,eps,
                matrix.data(),vectors.data(),a.data(),b.data(),o,snap,commutator,density_closed,1.0);
            if(occupied) snap.spin_homo[spin]=std::max(snap.spin_homo[spin],eps[occupied-1]);
            snap.spin_lumo[spin]=std::min(snap.spin_lumo[spin],eps[occupied]);
        }
        snap.global_band_gap=std::numeric_limits<double>::infinity();
        for(I spin=0;spin<2;++spin) {
            snap.spin_band_gap[spin]=occupations[spin] ? finite(snap.spin_lumo[spin]-snap.spin_homo[spin])
                : std::numeric_limits<double>::infinity();
            if(occupations[spin]) snap.global_homo=std::max(snap.global_homo,snap.spin_homo[spin]);
            else snap.spin_homo[spin]=0.;
            snap.global_lumo=std::min(snap.global_lumo,snap.spin_lumo[spin]);
            snap.global_band_gap=std::min(snap.global_band_gap,snap.spin_band_gap[spin]);
        }
        snap.electronic_energy_per_cell=interaction_energy ? finite(energy.get()+interaction) : energy.get();
        snap.energy_per_cell=finite(snap.electronic_energy_per_cell+input.nuclear_repulsion_energy_per_cell);
        snap.raw_energy_per_cell=finite((interaction_energy ? finite(raw_energy.get()+interaction) : raw_energy.get())+input.nuclear_repulsion_energy_per_cell);
        snap.energy_projection_change=finite(std::abs(snap.energy_per_cell-snap.raw_energy_per_cell));
        snap.commutator_frobenius_rms=finite(commutator/std::sqrt(static_cast<double>(k)));
        snap.has_energy_change=iteration>1;
        snap.energy_change=iteration>1?finite(snap.energy_per_cell-previous_energy):0.0;
        const bool stationary=snap.has_energy_change && std::abs(snap.energy_change)<=o.energy_change_tolerance
            && snap.commutator_frobenius_rms<=o.commutator_tolerance && density_closed;
        bool stop=false;
        if (stationary) {
            snap.converged=snap.global_band_gap>o.minimum_band_gap_hartree;
            snap.status=snap.converged?Status::Converged:Status::NonInsulating; stop=true;
        } else if (iteration==o.maximum_iterations) { snap.status=Status::IterationLimit; stop=true; }
        else if (iteration==caps.maximum_fock_calls) { snap.status=Status::FockCallLimit; stop=true; }
        else if (memory.work_units_per_evaluated_iteration>caps.maximum_work_units-work) { snap.status=Status::WorkLimit; stop=true; }
        if (progress && !progress(snap,progress_context)) { snap.status=Status::Cancelled; snap.converged=false; stop=true; }
        out.snapshot_=snap;
        if (stop) break;
        previous_energy=snap.energy_per_cell;
        // This next density is never returned without a complete new F[D]
        // evaluation, eigen/closure audit and same-D energy calculation.
        for (I spin=0;spin<2;++spin) for (I point=0;point<k;++point) {
            const I block=spin*k+point;
            density_from_coefficients(n,r,occupations[spin],out.coefficients_.data()+block*nr,out.density_.data()+block*nn,1.0);
        }
    }
    return out;
}
BoundedPeriodicUHFResult solve_bounded_periodic_uhf(
    const BoundedPeriodicUHFInput& input,const BoundedPeriodicUHFTwoElectronProvider& provider,
    const BoundedPeriodicUHFOptions& options,const BoundedPeriodicUHFCaps& caps,
    BoundedPeriodicUHFProgressCallback progress,void* context) {
    return detail::BoundedPeriodicSCFKernel::uhf(input,provider,options,caps,progress,context,nullptr);
}
} // namespace vibeqc
