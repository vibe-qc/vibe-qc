#pragma once
#include "vibeqc/manta_periodic_semilocal_xc.hpp"

// Internal component contract. The complete hybrid adapter and explicitly
// partial frozen-density diagnostics may request a component evaluation;
// the public complete semilocal leaf still refuses all hybrids.
namespace vibeqc::manta::periodic_xc_component {
enum class Kind { Pure,GlobalHybrid,RangeSeparated };
double global_hybrid_fraction(const Functional&);
std::vector<std::tuple<std::string,double,double>> range_separated_components(const Functional&);
PeriodicSemilocalXCPlan plan(const BasisSet&,const Functional&,Count images,Count momenta,
    Count grid_points,Count grid_batch,const ResourceBudget&,Kind);
PeriodicSemilocalXCResult compute(const BasisSet&,const Functional&,const Eigen::MatrixXd& lattice,
    const Eigen::MatrixXd& images,const Eigen::MatrixXd& momenta,const std::vector<double>& weights,
    const Eigen::MatrixXd& points,const std::vector<double>& quadrature_weights,
    const std::vector<std::complex<double>>& densities,Count grid_batch,const ResourceBudget&,Kind);
}
