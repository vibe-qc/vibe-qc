#pragma once
#include "vibeqc/bounded_periodic_rhf_solver.hpp"
#include <array>

namespace vibeqc {
namespace detail { struct BoundedPeriodicSCFKernel; }
// Uniform full-k, collinear, fixed-spin UHF numerical leaf. The source owns
// G_sigma[D_alpha,D_beta] = J[D_alpha+D_beta]-K[D_sigma]. Neither spin is
// a k point. One callback evaluates BOTH spins from the same frozen epoch.
// Options, caps, algebra diagnostics and stopping codes reuse the RHF leaf.
using BoundedPeriodicUHFOptions = BoundedPeriodicRHFOptions;
using BoundedPeriodicUHFCaps = BoundedPeriodicRHFCaps;
using BoundedPeriodicUHFDiagnostics = BoundedPeriodicRHFDiagnostics;
using BoundedPeriodicUHFStatus = BoundedPeriodicRHFStatus;
using BoundedPeriodicUHFTwoElectronProvider = BoundedPeriodicRHFTwoElectronProvider;
struct BoundedPeriodicUHFInput {
    std::uint64_t n_kpoints=0,n_basis=0,n_alpha=0,n_beta=0;
    double nuclear_repulsion_energy_per_cell=0.;
    // Shared immutable row-major S/H[K,n,n]. Callback D/G are [2,K,n,n].
    const std::complex<double>* overlap=nullptr;
    std::size_t overlap_elements=0;
    const std::complex<double>* hcore=nullptr;
    std::size_t hcore_elements=0;
};
struct BoundedPeriodicUHFMemoryPlan : BoundedPeriodicRHFMemoryPlan {
    // Inherited n_occupied is max(n_alpha,n_beta), used for admission only.
    std::uint64_t n_alpha=0,n_beta=0;
};
struct BoundedPeriodicUHFSnapshot : BoundedPeriodicRHFSnapshot {
    // Commutator RMS = sqrt(sum_spin,k ||e_spin,k||_F^2 / K).
    // This sums the two spin residuals; it is not an average over 2K blocks.
    // Empty spin: homo=0 and gap=+infinity (no occupied/virtual constraint).
    // global_band_gap is min(spin_band_gap), appropriate to FIXED spin counts;
    // global_homo/lumo are diagnostics and need not differ by global_band_gap.
    std::array<double,2> spin_homo{},spin_lumo{},spin_band_gap{};
};
using BoundedPeriodicUHFProgressCallback = bool (*)(const BoundedPeriodicUHFSnapshot&,void*);
class BoundedPeriodicUHFResult {
public:
    BoundedPeriodicUHFResult(const BoundedPeriodicUHFResult&)=delete;
    BoundedPeriodicUHFResult& operator=(const BoundedPeriodicUHFResult&)=delete;
    BoundedPeriodicUHFResult(BoundedPeriodicUHFResult&&) noexcept=default;
    BoundedPeriodicUHFResult& operator=(BoundedPeriodicUHFResult&&) noexcept=default;
    const BoundedPeriodicUHFMemoryPlan& memory() const noexcept { return memory_; }
    const BoundedPeriodicUHFDiagnostics& diagnostics() const noexcept { return diagnostics_; }
    const BoundedPeriodicUHFSnapshot& final_snapshot() const noexcept { return snapshot_; }
    bool converged() const noexcept { return snapshot_.converged; }
    // Last evaluated D/F[2,K,n,n], canonical C[2,K,n,r], eps[2,K,r].
    // Terminal C need not reconstruct the evaluated D on a capped exit.
    const std::complex<double>* density_data() const;
    const std::complex<double>* fock_data() const;
    const std::complex<double>* coefficients_data() const;
    const double* orbital_energies_data() const;
private:
    friend struct detail::BoundedPeriodicSCFKernel;
    BoundedPeriodicUHFResult()=default;
    void require_live() const;
    BoundedPeriodicUHFMemoryPlan memory_;
    BoundedPeriodicUHFDiagnostics diagnostics_;
    BoundedPeriodicUHFSnapshot snapshot_;
    std::vector<std::complex<double>> density_,fock_,coefficients_;
    std::vector<double> energies_;
    friend BoundedPeriodicUHFResult solve_bounded_periodic_uhf(
        const BoundedPeriodicUHFInput&,const BoundedPeriodicUHFTwoElectronProvider&,
        const BoundedPeriodicUHFOptions&,const BoundedPeriodicUHFCaps&,
        BoundedPeriodicUHFProgressCallback,void*);
};
// Count-only admission; one virtual per spin, at least one total electron,
// fixed occupations across k, and the same retained overlap rank at every k.
BoundedPeriodicUHFMemoryPlan plan_bounded_periodic_uhf_solver(
    std::uint64_t k,std::uint64_t n,std::uint64_t n_alpha,std::uint64_t n_beta,
    std::uint64_t retained_rank,const BoundedPeriodicUHFTwoElectronProvider&,
    const BoundedPeriodicUHFOptions&,std::uint64_t other_live_numerical_bytes);
BoundedPeriodicUHFResult solve_bounded_periodic_uhf(
    const BoundedPeriodicUHFInput&,const BoundedPeriodicUHFTwoElectronProvider&,
    const BoundedPeriodicUHFOptions&,const BoundedPeriodicUHFCaps&,
    BoundedPeriodicUHFProgressCallback=nullptr,void* progress_context=nullptr);
} // namespace vibeqc
