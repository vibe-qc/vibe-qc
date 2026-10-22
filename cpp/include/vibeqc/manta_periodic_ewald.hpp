#pragma once
#include "vibeqc/manta_periodic_pairs.hpp"

namespace vibeqc::manta {
struct PeriodicEwaldPlan {
    Count n_basis=0,momenta=0,pairs=0,pair_interactions=0;
    Count real_cells=0,reciprocal_vectors=0,reciprocal_batch=0,reciprocal_batches=0;
    Count real_shell_calls=0,fourier_shell_calls=0,overlap_shell_calls=0,shell_calls=0;
    Count primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
// Counts reserve cold scalar work; receipt counts describe actual shared work.
struct PeriodicDensityBatchPlan {
    Count densities=0,source_passes=0,shell_calls=0,primitive_products=0;
    Count scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicDensityBatchReceipt {
    std::string source_identity;
    Count source_passes=0,real_shell_calls=0,fourier_shell_calls=0,overlap_shell_calls=0;
    Count primitive_products=0,vector_primitive_products=0,coefficient_panels=0;
    Count domain_pair_interactions=0,domain_majorant_pairs_evaluated=0;
};
Count periodic_density_batch_workspace(Count n,Count nk,Count pairs,Count reciprocal_batch);
struct PeriodicEwaldReceipt {
    std::string provider="manta",provider_version="md-fp64-v1";
    std::string normalization="libint-shell-axial-v1";
    std::string algorithm="canonical-pair-direct-ewald";
    std::string boundary="3d-zero-average-conducting";
    std::string qualification="finite-canonical-pair-ewald-j";
    std::string source_identity,pair_source_identity,kernel_identity,density_identity;
    std::string real_domain_identity,reciprocal_domain_identity;
    Count real_shell_calls=0,fourier_shell_calls=0,overlap_shell_calls=0;
    Count primitive_products=0,vector_primitive_products=0,coefficient_panels=0;
    Count modeled_peak_bytes=0;
    double eta=0,cell_volume=0,gauge_constant=0,charge=0;
    double real_energy=0,reciprocal_energy=0,gauge_energy=0,energy=0,trace_energy=0;
    double density_hermiticity_residual=0,response_hermiticity_residual=0;
};
struct PeriodicEwaldResult {
    PeriodicEwaldPlan plan;
    PeriodicEwaldReceipt receipt;
    std::vector<std::complex<double>> values;
};

struct PeriodicEwaldBatchResult {
    PeriodicDensityBatchPlan plan;
    PeriodicDensityBatchReceipt receipt;
    // Per-density receipts retain logical scalar work counts for comparison;
    // actual shared evaluations are recorded exactly once in receipt above.
    std::vector<PeriodicEwaldResult> results;
};

// Direct fallback for the same immutable canonical pair/image source used by
// PeriodicBubbleTree. Ewald1921, IV.1 Eqs.37-40, with the explicitly declared
// zero-average electronic-component convention. This is a finite numerical
// quadratic form: it does not certify omitted AO images or infinite domains.
// No Gaussian electronic self subtraction, exchange or nuclear energy.
// Canonical folding is exact for the limiting periodic kernel; finite real
// domains retain a covariance error. The weighted derivative is exact for
// the declared finite quadratic form, without relating D(k) to D(-k)*.
class PeriodicEwaldSource {
public:
    static PeriodicEwaldPlan plan_source(const PeriodicPairSource&,Count real_cells,
        Count reciprocal_vectors,Count reciprocal_batch,const ResourceBudget&);
    PeriodicEwaldSource(std::shared_ptr<const PeriodicPairSource>,double eta,
        const std::vector<LatticeIndex>& real,const std::vector<LatticeIndex>& reciprocal,
        Count reciprocal_batch,const ResourceBudget&);
    const PeriodicEwaldPlan& plan() const { return plan_; }
    const std::string& source_identity() const { return receipt_.source_identity; }
    const std::string& pair_source_identity() const { return receipt_.pair_source_identity; }
    const std::string& kernel_identity() const { return receipt_.kernel_identity; }
    PeriodicEwaldResult build(const std::vector<std::complex<double>>&) const;
    PeriodicDensityBatchPlan plan_batch(Count,const ResourceBudget&) const;
    PeriodicEwaldBatchResult build_batch(const std::vector<std::vector<std::complex<double>>>&,const ResourceBudget&) const;
private:
    friend struct PeriodicBupoHFDriver;
    friend class PeriodicBupoSource;
    friend class PeriodicElectrostaticSource;
    md::Matrix real_tile(Count,Count,const LatticeIndex&,PeriodicEwaldReceipt&) const;
    md::ComplexMatrix fourier_panel(Count,const md::Matrix&,PeriodicEwaldReceipt&) const;
    md::Matrix overlap_panel(Count,PeriodicEwaldReceipt&) const;
    md::Matrix reciprocal_tile(Count,Count,std::vector<double>&) const;
    std::array<md::Matrix,3> interaction(Count,Count,PeriodicEwaldReceipt&) const;
    PeriodicEwaldPlan plan_;
    std::shared_ptr<const PeriodicPairSource> source_;
    std::vector<LatticeIndex> real_,reciprocal_;
    PeriodicEwaldReceipt receipt_;
};
} // namespace vibeqc::manta
