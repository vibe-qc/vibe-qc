#pragma once
#include "vibeqc/manta_periodic_local_exchange.hpp"
#include "vibeqc/xc.hpp"

namespace vibeqc::manta {
struct PeriodicSemilocalXCPlan : PeriodicLocalExchangePlan {
    bool gradients=false;
    Count functional_components=0,functional_workspace_bytes=0;
};
struct PeriodicSemilocalXCReceipt : PeriodicLocalExchangeReceipt {
    bool semilocal_component_only=false;
    double required_exact_exchange_fraction=0.;
    // A range-separated component requires the complete canonical kernel list;
    // there is no single global fraction. Entries are kernel, coefficient, omega.
    std::vector<std::tuple<std::string,double,double>> required_exchange_components;
    std::string backend_version;
    std::vector<int> component_ids;
    std::vector<double> component_weights;
    Count functional_batches=0;
};
struct PeriodicSemilocalXCResult {
    PeriodicSemilocalXCPlan plan;
    PeriodicSemilocalXCReceipt receipt;
    std::vector<std::complex<double>> values;
};
// Same finite complex AO-image/grid and spin conventions as local exchange.
// Only complete native libxc LDA/GGA functionals are admitted; no external,
// hybrid, double-hybrid, nonlocal or kinetic-density contribution is omitted.
// GGA V includes the adjoint of the same complex density-gradient contraction.
// Functional and numerical sources are borrowed, immutable during the call.
PeriodicSemilocalXCPlan plan_periodic_semilocal_xc(const BasisSet&,const Functional&,Count images,
    Count momenta,Count grid_points,Count grid_batch,const ResourceBudget&);
PeriodicSemilocalXCResult compute_periodic_semilocal_xc(const BasisSet&,const Functional&,
    const Eigen::MatrixXd& lattice,const Eigen::MatrixXd& images,const Eigen::MatrixXd& momenta,
    const std::vector<double>& weights,const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,
    const std::vector<std::complex<double>>& densities,Count grid_batch,const ResourceBudget&);
} // namespace vibeqc::manta
