#pragma once
#include "vibeqc/manta_periodic_bupo.hpp"

namespace vibeqc::manta {
struct PeriodicElectrostaticOptions {
    std::string charge_policy="neutral";
    double charge_tolerance=1e-10;
};
struct PeriodicElectrostaticPlan {
    Count n_basis=0,momenta=0,points=0,pairs=0;
    Count nuclear_shell_calls=0,overlap_shell_calls=0,fourier_shell_calls=0,point_pair_terms=0;
    Count shell_calls=0,primitive_products=0,scalar_terms=0,kernel_terms=0,modeled_peak_bytes=0;
};
struct PeriodicElectrostaticReceipt {
    std::string source_identity,electronic_source_identity,pair_source_identity,kernel_identity,point_source_identity,density_identity;
    std::string qualification="full-pair-image-electronic-point-electrostatics";
    std::string boundary="3d-zero-average-conducting",charge_policy;
    double charge_tolerance=0,nuclear_charge=0,nuclear_absolute_charge=0,electron_charge=0;
    double retained_net_charge=0,charge_uncertainty_bound=0,maximum_net_charge_bound=0,background_charge=0;
    double nuclear_real=0,nuclear_reciprocal=0,nuclear_self=0,nuclear_gauge=0,nuclear_energy=0;
    double attraction_real=0,attraction_reciprocal=0,attraction_gauge=0,attraction_energy=0;
    double electronic_energy=0,electronic_gauge=0,total_gauge=0,energy=0,trace_energy=0;
    double nuclear_domain_error_bound=0,attraction_domain_error_bound=0,attraction_image_error_bound=0;
    double point_kernel_bound=0,energy_error_bound=0,maximum_j_error_bound=0;
    double hermiticity_residual=0;
    Count nuclear_shell_calls=0,fourier_shell_calls=0,overlap_shell_calls=0,point_pair_terms=0;
    Count gaussian_point_bounds=0,point_pair_bounds=0,modeled_peak_bytes=0;
    bool roundoff_included=false;
};
struct PeriodicElectrostaticResult {
    PeriodicElectrostaticPlan plan;
    PeriodicElectrostaticReceipt receipt;
    PeriodicBupoResult electronic;
    // Weighted derivative of the same energy: J[D] + V_nuc, excluding T/K/XC.
    std::vector<std::complex<double>> values;
    md::Matrix nuclear_j_error_bound,j_error_bound;
};
class PeriodicElectrostaticSource {
public:
    static PeriodicElectrostaticPlan plan_source(const PeriodicBupoSource&,Count points,
        const PeriodicElectrostaticOptions&,const ResourceBudget&);
    PeriodicElectrostaticSource(std::shared_ptr<const PeriodicBupoSource>,const md::Matrix& positions,
        const std::vector<double>& charges,const PeriodicElectrostaticOptions&,const ResourceBudget&);
    const PeriodicElectrostaticPlan& plan() const { return plan_; }
    const std::string& source_identity() const { return receipt_.source_identity; }
    const std::vector<std::complex<double>>& overlap() const { return overlap_; }
    const std::vector<std::complex<double>>& nuclear_values() const { return nuclear_; }
    PeriodicElectrostaticResult build(const std::vector<std::complex<double>>&) const;
private:
    friend struct PeriodicBupoHFDriver;
    PeriodicElectrostaticPlan plan_;
    std::shared_ptr<const PeriodicBupoSource> electronic_;
    PeriodicElectrostaticReceipt receipt_;
    PeriodicElectrostaticOptions options_;
    std::vector<std::complex<double>> overlap_,nuclear_;
    std::array<std::vector<std::complex<double>>,3> attraction_;
    std::vector<double> pair_domain_error_;
    md::Matrix domain_j_bound_,image_j_bound_;
};
} // namespace vibeqc::manta
