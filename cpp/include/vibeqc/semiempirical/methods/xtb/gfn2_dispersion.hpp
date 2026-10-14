#pragma once

#include <Eigen/Dense>
#include <string>
#include "vibeqc/molecule.hpp"

namespace vibeqc::semiempirical::xtb {

struct GFN2D4Evaluation {
    double energy = 0.;
    Eigen::VectorXd potential;
    Eigen::MatrixXd gradient;
};

// Immutable, geometry-bound operands of E_D4(q) = 1/2 W(q)^T K W(q)
// plus neutral-charge ATM. Bannwarth et al., JCTC 15, 1652 (2019),
// doi:10.1021/acs.jctc.8b01176; Caldeweyher et al., JCP 150, 154122
// (2019), doi:10.1063/1.5090222. The caller's reference/CN convention
// is part of these operands, not a claim of published-parameter parity.
class GFN2D4Data {
public:
    GFN2D4Data(Eigen::VectorXi numbers, Eigen::MatrixXd positions,
        Eigen::VectorXi owners, Eigen::VectorXd shifted_reference_charges,
        Eigen::VectorXd effective_charges, Eigen::VectorXd hardness,
        Eigen::VectorXd cn_weights, Eigen::MatrixXd kernel,
        double neutral_atm_energy, Eigen::VectorXd cn_weight_derivatives,
        Eigen::MatrixXd cn_jacobian, Eigen::MatrixXd radial_kernel,
        Eigen::MatrixXd neutral_atm_gradient);
    GFN2D4Evaluation evaluate(const Eigen::VectorXd& charges,
                             bool with_gradient = false) const;
    void require_geometry(const Molecule& mol) const;
    const std::string& sha256() const { return sha256_; }

private:
    Eigen::VectorXi numbers_, owners_;
    Eigen::MatrixXd positions_, kernel_, cn_jacobian_, radial_kernel_, atm_gradient_;
    Eigen::VectorXd qref_, zeff_, hardness_, weights_, dweights_;
    double atm_energy_;
    std::string sha256_;
};

} // namespace vibeqc::semiempirical::xtb
