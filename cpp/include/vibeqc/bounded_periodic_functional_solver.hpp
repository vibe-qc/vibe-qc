#pragma once
#include "vibeqc/bounded_periodic_uhf_solver.hpp"

namespace vibeqc {
// Numerical functional contract only, not a quadrature or physical KS source.
// Joint same-density callback: write G[D] = F[D]-H and E_interaction[D].
// E_interaction includes all Hartree/exchange/XC terms, excludes one-electron
// and nuclear energy, and is per primitive cell with weights 1/K. For RKS D
// is spin summed [K,n,n]; for UKS it is [2,K,n,n], with alpha then beta.
// The functional derivative convention is dE = sum_spin,k Re Tr(dD G)/K.
// The caller must qualify that variational relation independently. The solver
// checks finite complete outputs, not whether a claimed functional is correct.
using BoundedPeriodicFunctionalCallback = void (*)(
    const std::complex<double>*,std::size_t,std::complex<double>*,std::size_t,
    double* interaction_energy_per_cell,void*);
struct BoundedPeriodicFunctionalProvider {
    BoundedPeriodicFunctionalCallback evaluate=nullptr;
    void* context=nullptr;
    std::uint64_t retained_numerical_bytes=0,peak_workspace_numerical_bytes=0;
    // Includes joint energy and potential work; borrowed D/G are excluded
    // from retained/workspace bytes. The adapter admits its scalar separately.
    std::uint64_t maximum_work_units_per_call=0;
};
BoundedPeriodicRHFMemoryPlan plan_bounded_periodic_rks_solver(
    std::uint64_t k,std::uint64_t n,std::uint64_t electrons,std::uint64_t rank,
    const BoundedPeriodicFunctionalProvider&,const BoundedPeriodicRHFOptions&,std::uint64_t other);
BoundedPeriodicUHFMemoryPlan plan_bounded_periodic_uks_solver(
    std::uint64_t k,std::uint64_t n,std::uint64_t alpha,std::uint64_t beta,std::uint64_t rank,
    const BoundedPeriodicFunctionalProvider&,const BoundedPeriodicUHFOptions&,std::uint64_t other);
// Existing fixed-occupation eigensolve, rank, gap and stopping contracts apply.
// E = E_nuclear + sum_spin,k Re Tr(D H)/K + E_interaction[D], never the HF
// half-trace of H+F. Potential Hermitian repair is diagnosed independently;
// it does not alter the provider's energy. Only H repair changes raw energy.
BoundedPeriodicRHFResult solve_bounded_periodic_rks(const BoundedPeriodicRHFInput&,
    const BoundedPeriodicFunctionalProvider&,const BoundedPeriodicRHFOptions&,const BoundedPeriodicRHFCaps&,
    BoundedPeriodicRHFProgressCallback=nullptr,void* progress_context=nullptr);
BoundedPeriodicUHFResult solve_bounded_periodic_uks(const BoundedPeriodicUHFInput&,
    const BoundedPeriodicFunctionalProvider&,const BoundedPeriodicUHFOptions&,const BoundedPeriodicUHFCaps&,
    BoundedPeriodicUHFProgressCallback=nullptr,void* progress_context=nullptr);
} // namespace vibeqc
