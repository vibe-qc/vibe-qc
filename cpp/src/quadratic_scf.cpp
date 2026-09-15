#include "vibeqc/quadratic_scf.hpp"

#include <algorithm>

namespace vibeqc {

Eigen::MatrixXd expm_skew(const Eigen::MatrixXd& K,
                          double max_norm_per_step) {
    const Eigen::Index n = K.rows();

    // Scaling: find s with ‖K / 2^s‖ ≤ max_norm_per_step.
    int s = 0;
    Eigen::MatrixXd K_scaled = K;
    while (K_scaled.norm() > max_norm_per_step) {
        K_scaled *= 0.5;
        ++s;
    }

    // Truncated Taylor series. Order 12 is conservative for ‖K‖ ≤ 0.5
    // (order-12 term bounded by 0.5^12 / 12! ≈ 6e-13) but trivially
    // cheap for the small SCF matrices we work with (n_kept ≲ 1000).
    Eigen::MatrixXd out = Eigen::MatrixXd::Identity(n, n);
    Eigen::MatrixXd term = Eigen::MatrixXd::Identity(n, n);
    for (int k = 1; k < 13; ++k) {
        term = (term * K_scaled) / static_cast<double>(k);
        out += term;
    }

    // Squaring: exp(K) = exp(K / 2^s)^(2^s).
    for (int i = 0; i < s; ++i) {
        out = out * out;
    }

    return out;
}

QuadraticStepResult quadratic_step(const Eigen::MatrixXd& F,
                                   const Eigen::MatrixXd& C_prev,
                                   const Eigen::VectorXd& eps_prev,
                                   int n_occ,
                                   double shift,
                                   double max_step) {
    const Eigen::Index n_kept = C_prev.cols();
    const Eigen::Index n_vir = n_kept - n_occ;

    if (n_vir <= 0) {
        // No virtual subspace; rotation is the identity. Refresh ε
        // from the diagonal of F in the current MO basis but leave C
        // unchanged.
        QuadraticStepResult out;
        out.C = C_prev;
        out.eps = (C_prev.transpose() * F * C_prev).diagonal();
        return out;
    }

    // F in current MO basis.
    Eigen::MatrixXd F_mo = C_prev.transpose() * F * C_prev;
    F_mo = 0.5 * (F_mo + F_mo.transpose());   // symmetrize residual drift

    // Occ-vir block: F_mo[a, i] for i ∈ occ, a ∈ vir.
    Eigen::MatrixXd F_ov = F_mo.bottomLeftCorner(n_vir, n_occ);

    // Diagonal Hessian denominator + Newton step.
    Eigen::MatrixXd kappa_ov(n_vir, n_occ);
    for (Eigen::Index a = 0; a < n_vir; ++a) {
        const double eps_a = eps_prev(n_occ + a);
        for (Eigen::Index i = 0; i < n_occ; ++i) {
            const double eps_i = eps_prev(i);
            const double Delta = (eps_a - eps_i) + shift;
            kappa_ov(a, i) = -F_ov(a, i) / Delta;
        }
    }

    // Trust region: cap ‖κ_ov‖_F at max_step.
    const double kappa_norm = kappa_ov.norm();
    if (kappa_norm > max_step) {
        kappa_ov *= (max_step / kappa_norm);
    }

    // Build full skew-symmetric κ (n_kept × n_kept).
    Eigen::MatrixXd kappa = Eigen::MatrixXd::Zero(n_kept, n_kept);
    kappa.bottomLeftCorner(n_vir, n_occ) = kappa_ov;
    kappa.topRightCorner(n_occ, n_vir) = -kappa_ov.transpose();

    // Apply orbital rotation: C_new = C_prev · exp(κ).
    const Eigen::MatrixXd U = expm_skew(kappa);

    QuadraticStepResult out;
    out.C = C_prev * U;
    out.eps = (out.C.transpose() * F * out.C).diagonal();
    return out;
}

}  // namespace vibeqc
