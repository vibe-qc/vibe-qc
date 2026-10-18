#pragma once
#include "vibeqc/manta_periodic_ri.hpp"
#include "vibeqc/manta_periodic_fock.hpp"

namespace vibeqc::manta {
struct PeriodicRIFockPlan {
    bool has_j=true,has_k=true;
    Count n_basis=0,n_auxiliary=0,momenta=0,channels=0;
    Count retained_source_bytes=0,driver_live_bytes=0,modeled_peak_bytes=0;
    Count metric_shell_calls=0,metric_primitive_products=0,build_shell_calls=0,build_primitive_products=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0;
    PeriodicRIQPlan gamma,pair;
};
struct PeriodicRIFockInfo {
    bool has_j=true,has_k=true;
    std::string source_identity,frame_identity,weight_identity;
    std::string exchange_kernel="coulomb";
    double physical_omega=0.;
    Count metric_shell_calls=0,metric_primitive_products=0;
    double reverse_metric_residual=0.,reverse_projector_residual=0.,reverse_inverse_residual=0.;
    // Gamma first; then, if K is selected, off-diagonal (target,source) pairs.
    std::vector<PeriodicRIQMetricInfo> metrics;
    PeriodicRIAccuracyReceipt accuracy;
};
struct PeriodicRIFockReceipt : PeriodicFockReceipt {
    PeriodicRIFockReceipt() { algorithm="streamed-periodic-ri-spin-fock"; }
    bool has_j=true,has_k=true;
    std::string frame_identity,weight_identity;
    std::string exchange_kernel="coulomb";
    double physical_omega=0.;
    Count shell_calls=0,primitive_products=0,coupling_passes=0;
    Count coupling_multipole_trials=0,coupling_multipole_images=0;
    Count coupling_multipole_tree_nodes=0,coupling_multipole_far_blocks=0,coupling_multipole_translations=0;
    Count coupling_multipole_moment_panels=0,coupling_multipole_envelope_evaluations=0;
    Count coupling_multipole_reused_moment_panels=0,coupling_multipole_reused_extents=0;
    Count coupling_multipole_global_pairs=0,coupling_multipole_global_pair_reuses=0,coupling_multipole_global_auxiliary=0,coupling_multipole_global_projections=0,coupling_multipole_global_auxiliary_projections=0,coupling_multipole_pair_groups=0;
    Count coupling_multipole_partition_builds=0,coupling_multipole_partition_reuses=0,coupling_multipole_partition_bypasses=0;
    Count coupling_multipole_compressed_pairs=0,coupling_multipole_compressed_nodes=0,coupling_multipole_action_products=0,coupling_multipole_action_translations=0;
    std::string reference_coupling_identity;
    double coupling_error_bound=0.,approximation_j_bound=0.,approximation_energy_bound=0.;
};
struct PeriodicRIFockResult {
    Count n_basis=0,momenta=0;
    PeriodicRIFockReceipt receipt;
    std::vector<PeriodicRIQReceipt> components;
    std::vector<std::complex<double>> values,J,K;
};
PeriodicRIFockPlan plan_periodic_ri_fock(const BasisSet&,const BasisSet&,const PeriodicSystem&,
    Count gamma_vectors,Count maximum_pair_vectors,Count momenta,
    const PeriodicGDFShortRangeOptions&,const RIOptions&,const ResourceBudget&,
    bool make_j=true,bool make_k=true,const PeriodicGDFKernelOptions& exchange_kernel = {},const PeriodicRIAccuracyOptions& accuracy = {});

// Complete explicit-mesh RI J/K, with fixed metric/projector owners. Pair
// vectors belong to G+k_source-k_target. Diagonal pair segments are empty:
// the Gamma source supplies J and every diagonal K contribution together.
// J-only requires an empty pair domain and owns only the Gamma projector.
// K-only retains Gamma for the diagonal contributions, with no J contraction.
// A physical erf/erfc source requires K-only selection. Its metric cannot
// silently replace the separate bare-Coulomb J source of a physical method.
// H is method-supplied; no nuclear/XC/exchange-singularity source is inferred.
// Sun et al., JCP 147,164119 (2017), q-dependent periodic Gaussian fitting;
// Neese, JCC 44,381 (2023), retained-factor spin J/K contractions.
class PeriodicRIFockBuilder {
public:
    PeriodicRIFockBuilder(const BasisSet&,const BasisSet&,const PeriodicSystem&,
        const md::Matrix& momenta,const std::vector<double>& weights,
        const md::Matrix& gamma_vectors,const md::Matrix& pair_vectors,const std::vector<Count>& offsets,
        const PeriodicGDFShortRangeOptions&,const RIOptions&,const ResourceBudget&,
        bool make_j=true,bool make_k=true,const PeriodicGDFKernelOptions& exchange_kernel = {},const PeriodicRIAccuracyOptions& accuracy = {});
    const PeriodicRIFockPlan& plan() const { return plan_; }
    const PeriodicRIFockInfo& info() const { return info_; }
    PeriodicRIFockResult build(const std::vector<std::complex<double>>& hcore,
        const std::vector<std::complex<double>>& spin_densities,double exchange_scale=1.) const;
private:
    PeriodicRIFockPlan plan_;
    PeriodicRIFockInfo info_;
    std::vector<double> weights_;
    std::unique_ptr<PeriodicRIQBuilder> gamma_;
    std::vector<std::unique_ptr<PeriodicRIQBuilder>> pairs_;
};
} // namespace vibeqc::manta
