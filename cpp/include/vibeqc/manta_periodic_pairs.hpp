#pragma once
#include "vibeqc/manta_extents.hpp"
#include "vibeqc/manta_lattice_tail.hpp"
#include "vibeqc/manta_moments.hpp"
#include <optional>

namespace vibeqc::manta {
struct PeriodicPairOptions {
    int order=6;
    ExtentOptions extents;
    std::optional<double> pair_cutoff_bohr;
};
struct PeriodicPairPlan {
    Count n_basis=0,cells=0,momenta=0,maximum_pairs=0,moments=0;
    Count shell_calls=0,primitive_pairs=0,scalar_terms=0,kernel_terms=0;
    Count retained_bytes=0,driver_live_bytes=0,modeled_peak_bytes=0;
};
struct PeriodicPairEntry {
    std::array<Count,2> shells{},offsets{},sizes{};
    LatticeIndex image{};
    md::Point translation{};
    bool paired=false;
    PairExtent extent;
};
struct PeriodicPairReceipt {
    std::string source_identity,basis_identity,domain_identity;
    std::string convention="q=0;canonical-(a,b,R)/(b,a,-R);real-conjugate-density-pairing";
    std::string qualification="finite-pair-image-moments-and-adjoint";
    Count pairs=0,ordered_pairs=0,skipped_ordered_pairs=0,primitive_pairs=0;
    Count envelope_evaluations=0,always_near_pairs=0,modeled_peak_bytes=0;
};
struct PeriodicPairDensityReceipt {
    std::string source_identity,density_identity;
    Count shell_calls=0,primitive_pairs=0,phase_products=0,contraction_terms=0,modeled_peak_bytes=0;
    double charge=0.,coefficient_l1=0.,absolute_charge_bound=0.,density_hermiticity_residual=0.;
};
struct PeriodicPairDensity {
    md::Matrix moments;
    std::vector<double> pair_coefficient_l1;
    PeriodicPairDensityReceipt receipt;
};
struct PeriodicPairAdjoint {
    std::vector<std::complex<double>> values;
    Count n_basis=0,momenta=0,shell_calls=0,primitive_pairs=0,phase_products=0,contraction_terms=0;
    double hermiticity_residual=0.;
    std::string source_identity,potential_identity;
};
PeriodicPairPlan plan_periodic_pairs(const BasisSet&,Count cells,Count momenta,
    const PeriodicPairOptions&,const ResourceBudget&);

// q=0 density source, not a Coulomb kernel or an exchange source. The image
// involution is exact for periodic potentials; no k/-k state relation or
// spatial symmetry is assumed. A finite nonperiodic kernel cannot inherit
// translation covariance merely by consuming this canonical inventory.
class PeriodicPairSource {
public:
    PeriodicPairSource(const BasisSet&,const Eigen::Matrix3d&,const std::vector<LatticeIndex>&,
        const md::Matrix& momenta,const std::vector<double>& weights,
        const PeriodicPairOptions&,const ResourceBudget&);
    const PeriodicPairPlan& plan() const { return plan_; }
    const PeriodicPairReceipt& receipt() const { return receipt_; }
    int order() const { return options_.order; }
    const Eigen::Matrix3d& lattice() const { return lattice_; }
    const md::Matrix& momenta() const { return momenta_; }
    const std::vector<double>& weights() const { return weights_; }
    const std::vector<PeriodicPairEntry>& entries() const { return entries_; }
    PairExtents extents() const;
    PairMomentResult moment_panel(Count pair) const;
    PeriodicPairDensity contract(const std::vector<std::complex<double>>&) const;
    PeriodicPairAdjoint adjoint(const md::Matrix& local_potentials) const;
private:
    friend class PeriodicExchangeBoundSource;
    friend struct PeriodicRIExchangeFittingDriver;
    friend struct PeriodicOneElectronImageDriver;
    friend struct PeriodicBupoHFDriver;
    friend class PeriodicEwaldSource;
    friend class PeriodicBupoSource;
    friend class PeriodicElectrostaticSource;
    // Plan before any retained source copy.
    PeriodicPairPlan plan_;
    BasisSet basis_;
    Eigen::Matrix3d lattice_;
    md::Matrix momenta_;
    std::vector<double> weights_;
    PeriodicPairOptions options_;
    PeriodicPairReceipt receipt_;
    std::vector<PeriodicPairEntry> entries_;
    std::vector<LatticeIndex> images_; // Original ordered support, already admitted in retained input storage.
};
} // namespace vibeqc::manta
