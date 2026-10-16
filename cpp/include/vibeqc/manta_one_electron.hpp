#pragma once

#include "vibeqc/manta.hpp"
#include "vibeqc/manta_shells.hpp"

namespace vibeqc::manta {
struct OneElectronRequest {
    std::string operation = "overlap";
    md::Charges charges;
    md::Point origin{};
    md::Index powers{};
    std::string kernel = "coulomb";
    double omega = 0.;
};
struct OneElectronDerivative {
    std::string target = "bra";
    int axis = 0;
    std::size_t charge_index = 0;
};
struct OneElectronPlan {
    std::array<Count,2> shape{};
    Count shell_calls = 0, primitive_pairs = 0, kernel_terms = 0;
    Count primitive_evaluations_upper_bound = 0;
    Count tile_elements = 0, output_bytes = 0, modeled_peak_bytes = 0;
};
struct OneElectronReceipt {
    std::string provider = "manta", provider_version = "md-fp64-v1";
    std::string normalization = "libint-shell-axial-v1";
    std::string operation, source_identity;
    std::string kernel = "coulomb";
    double omega = 0.;
    std::string derivative_target = "none";
    int derivative_axis = -1;
    std::size_t derivative_charge_index = 0;
    Count shell_calls = 0, primitive_pairs = 0, integral_elements = 0;
    Count primitive_integrals_evaluated = 0;
    Count modeled_peak_bytes = 0;
};
struct OneElectronResult {
    Eigen::MatrixXd values;
    OneElectronReceipt receipt;
};
OneElectronPlan plan_one_electron(const BasisSet& bra,const BasisSet& ket,
                                  const OneElectronRequest&,const ResourceBudget&);
OneElectronResult compute_one_electron(const BasisSet& bra,const BasisSet& ket,
                                       const OneElectronRequest&,const ResourceBudget&);
OneElectronPlan plan_one_electron_derivative(const BasisSet& bra,const BasisSet& ket,
                                  const OneElectronRequest&,const OneElectronDerivative&,const ResourceBudget&);
OneElectronResult compute_one_electron_derivative(const BasisSet& bra,const BasisSet& ket,
                                  const OneElectronRequest&,const OneElectronDerivative&,const ResourceBudget&);
} // namespace vibeqc::manta
