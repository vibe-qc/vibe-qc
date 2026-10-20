#pragma once
#include "vibeqc/manta_periodic_fitting.hpp"
#include "vibeqc/manta_ri.hpp"
#include <memory>

namespace vibeqc::manta {
// Targets apply only to coupling error relative to the SAME finite RI source.
// Both targets are required. Methods derive the weighted spin-total density
// L1 limit from their fixed overlap when zero; raw sources require it explicitly.
// Select once before SCF, then enforce the resolved limit on every call.
struct PeriodicRIAccuracyOptions {
    double j_error=0.,energy_error=0.,density_l1_limit=0.;
    Count maximum_attempts=8;
    // Optional additional bound on |w(k) sum_spin D(k,i,j)|, row-major.
    // Complete methods derive this with L when L=0; raw callers declare it.
    // Every build enforces both bounds. Empty retains the L1-only contract.
    std::vector<double> density_entry_limits;
    bool enabled() const { return j_error>0. || energy_error>0. || density_l1_limit>0.; }
    bool derives_density_limit() const { return enabled() && density_l1_limit==0.; }
    // A method's count-only admission reserves selection before S is available.
    // This placeholder never reaches a numerical builder or a certificate.
    PeriodicRIAccuracyOptions for_method_plan(Count entries) const {
        if(!density_entry_limits.empty() && (!enabled() || density_entry_limits.size()!=entries))
            throw std::invalid_argument("RI density entry-limit dimensions or inactive accuracy");
        for(double value:density_entry_limits)if(!std::isfinite(value) || value<0.)
            throw std::invalid_argument("RI density entry limits must be finite nonnegative");
        if(derives_density_limit() && !density_entry_limits.empty())
            throw std::invalid_argument("derived RI density limits cannot override entry limits");
        // No vector copy/allocation in count-only admission. Every automatic
        // plan reserves the possible entrywise contract and paired probes.
        PeriodicRIAccuracyOptions copy;
        copy.j_error=j_error;copy.energy_error=energy_error;copy.maximum_attempts=maximum_attempts;
        copy.density_l1_limit=derives_density_limit()?1.:density_l1_limit;return copy;
    }
};
struct PeriodicRIAccuracyReceipt {
    bool enabled=false,qualified=false,component_density_envelope=false;
    double j_target=0.,energy_target=0.,density_l1_limit=0.;
    double selected_coupling_error=0.,j_bound=0.,energy_bound=0.;
    Count attempts=0,shell_calls=0,primitive_products=0;
    std::string reference_coupling_identity,selected_coupling_identity,certificate_identity,density_envelope_identity;
};
struct PeriodicRIQPlan {
    Count n_basis=0,n_auxiliary=0,momenta=0,spins=0,maximum_rank=0,mode_block=0,maximum_coupling_passes=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
    Count accuracy_selection_passes=0;
};
struct PeriodicRIQMetricInfo {
    Count rank=0,shell_calls=0,primitive_products=0;
    PeriodicRIAccuracyReceipt accuracy;
    double cutoff=0.,negative_limit=0.,minimum_eigenvalue=0.,maximum_eigenvalue=0.,hermiticity_residual=0.;
    bool zero_transfer=false;
    std::string metric_source_identity,canonical_metric_identity,factor_identity,projector_identity,source_identity;
};
struct PeriodicRIQReceipt {
    std::string provider="manta",algorithm="streamed-periodic-ri-q",kernel="coulomb";
    std::string convention="fixed-positive-eigenprojector;lower-triangle;J=q0-whole-mesh;K=weighted-q-channel;no-exchange-correction";
    std::string coulomb_contraction="none"; // retained-mode, paired-raw, paired-screened or paired-multipole
    std::string source_identity,projector_identity,density_identity,weight_identity,snapshot_identity;
    std::string source_frame_identity,target_frame_identity;
    Count retained_rank=0,coupling_passes=0,shell_calls=0,primitive_products=0,modeled_peak_bytes=0;
    double coulomb_energy=0.,metric_rhs_norm=0.,metric_residual_norm=0.,hermiticity_residual=0.;
    double physical_omega=0.;
    Count coupling_multipole_trials=0,coupling_multipole_images=0;
    Count coupling_multipole_tree_nodes=0,coupling_multipole_far_blocks=0,coupling_multipole_translations=0;
    Count coupling_multipole_moment_panels=0,coupling_multipole_envelope_evaluations=0;
    Count coupling_multipole_reused_moment_panels=0,coupling_multipole_reused_extents=0;
    Count coupling_multipole_global_pairs=0,coupling_multipole_global_pair_reuses=0,coupling_multipole_global_auxiliary=0,coupling_multipole_global_projections=0,coupling_multipole_global_auxiliary_projections=0,coupling_multipole_pair_groups=0;
    Count coupling_multipole_partition_builds=0,coupling_multipole_partition_reuses=0,coupling_multipole_partition_bypasses=0;
    Count coupling_multipole_compressed_pairs=0,coupling_multipole_compressed_nodes=0,coupling_multipole_action_products=0,coupling_multipole_action_translations=0;
    std::string reference_coupling_identity;
    double coupling_error_bound=0.,metric_inverse_norm_bound=0.,forward_error_l2_bound=0.;
    double approximation_j_bound=0.,approximation_energy_bound=0.;
    bool zero_transfer=false,physical_zero_included=false;
    bool has_j=false,has_k=false,metric_residual_evaluated=false;
    bool accuracy_certified=false;
    double accuracy_density_l1_bound=0.;
};
struct PeriodicRIQResult {
    Count n_basis=0,momenta=0,spins=0;
    std::vector<std::complex<double>> J,K;
    PeriodicRIQReceipt receipt;
};
PeriodicRIQPlan plan_periodic_ri_q(const PeriodicGDFTiles&,Count spins,const RIOptions&,const ResourceBudget&,const PeriodicRIAccuracyOptions& = {});
// Count-only composition before a parent allocates any source snapshots.
PeriodicRIQPlan plan_periodic_ri_q(const PeriodicGDFTilesPlan&,Count spins,const RIOptions&,const ResourceBudget&,const PeriodicRIAccuracyOptions& = {});

// Fixed-q fitted component source. B=F^H T with F=U_keep/sqrt(lambda_keep).
// J is admitted only at reciprocal Gamma, using the whole supplied k mesh.
// K_sigma(k-q;k)=w(k) sum_L B_L(k) D_sigma(k) B_L(k)^H is a contribution
// at the *bra* k-q, not a complete exchange/Fock build or an SCF state.
// Sun et al., JCP 147,164119 (2017); Neese, JCC 44,381 (2023), RI contractions.
class PeriodicRIQBuilder {
public:
    PeriodicRIQBuilder(std::shared_ptr<const PeriodicGDFTiles>,Count spins,const RIOptions&,const ResourceBudget&,const PeriodicRIAccuracyOptions& = {});
    const PeriodicRIQPlan& plan() const { return plan_; }
    const PeriodicRIQMetricInfo& metric_info() const { return info_; }
    const Eigen::MatrixXcd& metric() const { return metric_; }
    const Eigen::MatrixXcd& factor() const { return factor_; }
    const Eigen::MatrixXcd& projector() const { return projector_; }
    const std::string& source_frame_identity() const { return source_frame_; }
    void validate_request(bool make_j,bool make_k) const;
    PeriodicRIQResult build(const std::vector<std::complex<double>>& densities,
        const std::vector<double>& weights,bool make_j=true,bool make_k=true) const;
private:
    PeriodicRIQPlan plan_;
    std::shared_ptr<const PeriodicGDFTiles> source_;
    RIOptions options_;
    ResourceBudget coupling_budget_; // Jointly reserved producer action workspace.
    PeriodicRIQMetricInfo info_;
    Eigen::MatrixXcd metric_,factor_,projector_;
    std::string source_frame_,target_frame_;
    std::vector<double> density_entry_limits_; // Immutable caller-independent accuracy contract.
    void select_accuracy(const PeriodicRIAccuracyOptions&);
};
} // namespace vibeqc::manta
