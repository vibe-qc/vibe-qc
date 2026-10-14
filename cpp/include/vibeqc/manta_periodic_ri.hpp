#pragma once
#include "vibeqc/manta_periodic_fitting.hpp"
#include "vibeqc/manta_ri.hpp"
#include <memory>

namespace vibeqc::manta {
struct PeriodicRIQPlan {
    Count n_basis=0,n_auxiliary=0,momenta=0,spins=0,maximum_rank=0,mode_block=0,maximum_coupling_passes=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicRIQMetricInfo {
    Count rank=0,shell_calls=0,primitive_products=0;
    double cutoff=0.,negative_limit=0.,minimum_eigenvalue=0.,maximum_eigenvalue=0.,hermiticity_residual=0.;
    bool zero_transfer=false;
    std::string metric_source_identity,canonical_metric_identity,factor_identity,projector_identity,source_identity;
};
struct PeriodicRIQReceipt {
    std::string provider="manta",algorithm="streamed-periodic-ri-q",kernel="coulomb";
    std::string convention="fixed-positive-eigenprojector;lower-triangle;J=q0-whole-mesh;K=weighted-q-channel;no-exchange-correction";
    std::string coulomb_contraction="none"; // retained-mode, paired-raw or paired-screened
    std::string source_identity,projector_identity,density_identity,weight_identity,snapshot_identity;
    std::string source_frame_identity,target_frame_identity;
    Count retained_rank=0,coupling_passes=0,shell_calls=0,primitive_products=0,modeled_peak_bytes=0;
    double coulomb_energy=0.,metric_rhs_norm=0.,metric_residual_norm=0.,hermiticity_residual=0.;
    double physical_omega=0.;
    std::string reference_coupling_identity;
    double coupling_error_bound=0.,metric_inverse_norm_bound=0.,forward_error_l2_bound=0.;
    double approximation_j_bound=0.,approximation_energy_bound=0.;
    bool zero_transfer=false,physical_zero_included=false;
    bool has_j=false,has_k=false,metric_residual_evaluated=false;
};
struct PeriodicRIQResult {
    Count n_basis=0,momenta=0,spins=0;
    std::vector<std::complex<double>> J,K;
    PeriodicRIQReceipt receipt;
};
PeriodicRIQPlan plan_periodic_ri_q(const PeriodicGDFTiles&,Count spins,const RIOptions&,const ResourceBudget&);
// Count-only composition before a parent allocates any source snapshots.
PeriodicRIQPlan plan_periodic_ri_q(const PeriodicGDFTilesPlan&,Count spins,const RIOptions&,const ResourceBudget&);

// Fixed-q fitted component source. B=F^H T with F=U_keep/sqrt(lambda_keep).
// J is admitted only at reciprocal Gamma, using the whole supplied k mesh.
// K_sigma(k-q;k)=w(k) sum_L B_L(k) D_sigma(k) B_L(k)^H is a contribution
// at the *bra* k-q, not a complete exchange/Fock build or an SCF state.
// Sun et al., JCP 147,164119 (2017); Neese, JCC 44,381 (2023), RI contractions.
class PeriodicRIQBuilder {
public:
    PeriodicRIQBuilder(std::shared_ptr<const PeriodicGDFTiles>,Count spins,const RIOptions&,const ResourceBudget&);
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
};
} // namespace vibeqc::manta
