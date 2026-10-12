#pragma once
#include "vibeqc/manta_periodic_quadrature.hpp"
#include "vibeqc/manta_periodic_semilocal_xc.hpp"

namespace vibeqc::manta {
struct PeriodicXCRefinementPlan {
    bool semilocal_component_only=false;
    std::vector<PeriodicQuadraturePlan> quadrature;
    std::vector<PeriodicSemilocalXCPlan> xc;
    Count spins=0,momenta=0,n_basis=0,reference_index=0;
    Count driver_live_bytes=0,modeled_peak_bytes=0;
    Count shell_calls=0,primitive_points=0,scalar_terms=0,kernel_terms=0;
};
struct PeriodicXCRefinementStep {
    PeriodicQuadratureOptions options;
    PeriodicQuadratureReceipt quadrature;
    PeriodicSemilocalXCReceipt xc;
    double energy_difference=0.;
    // Ordered [spin,k] norms in the unchanged source AO frame. No k weights
    // belong inside these matrix norms. Population differences are per spin.
    std::vector<double> potential_max_abs,potential_frobenius,population_differences;
};
struct PeriodicXCRefinementResult {
    PeriodicXCRefinementPlan plan;
    std::vector<PeriodicXCRefinementStep> steps;
    std::vector<std::complex<double>> reference_values;
    std::string source_identity,density_identity;
    std::string comparison="frozen-density;explicit-reference-grid;not-an-error-bound";
    Count shell_calls=0,primitive_points=0;
};
// All grid generations and same-density XC evaluations are admitted jointly.
// Evaluate the selected reference first, retain one reference potential and
// stream other grids. Differences measure this reference, not the continuum
// error. Supports, functional, spin/k state and AO frame remain fixed.
PeriodicXCRefinementPlan plan_periodic_xc_refinement(const BasisSet&,const PeriodicSystem&,
    const Functional&,Count images,Count momenta,const std::vector<PeriodicQuadratureOptions>&,
    Count reference_index,Count grid_batch,const ResourceBudget&);
PeriodicXCRefinementResult assess_periodic_xc_refinement(const BasisSet&,const PeriodicSystem&,
    const Functional&,const Eigen::MatrixXd& images,const Eigen::MatrixXd& momenta,
    const std::vector<double>& weights,const std::vector<std::complex<double>>& densities,
    const std::vector<PeriodicQuadratureOptions>&,Count reference_index,Count grid_batch,const ResourceBudget&);
// Explicit partial-component diagnostic for complete native global or
// range-separated LDA/GGA hybrid descriptors. Exchange is not evaluated.
// Every child receipt records its required exchange fraction/kernel list.
PeriodicXCRefinementPlan plan_periodic_hybrid_xc_refinement(const BasisSet&,const PeriodicSystem&,
    const Functional&,Count images,Count momenta,const std::vector<PeriodicQuadratureOptions>&,
    Count reference_index,Count grid_batch,const ResourceBudget&);
PeriodicXCRefinementResult assess_periodic_hybrid_xc_refinement(const BasisSet&,const PeriodicSystem&,
    const Functional&,const Eigen::MatrixXd& images,const Eigen::MatrixXd& momenta,
    const std::vector<double>& weights,const std::vector<std::complex<double>>& densities,
    const std::vector<PeriodicQuadratureOptions>&,Count reference_index,Count grid_batch,const ResourceBudget&);
} // namespace vibeqc::manta
