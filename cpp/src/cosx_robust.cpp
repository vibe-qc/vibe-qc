#include "vibeqc/cosx_robust.hpp"

#include <Eigen/Dense>

namespace vibeqc {

Eigen::MatrixXd compute_robust_cosx_k(
    const Eigen::MatrixXd& /*D*/,
    const Eigen::MatrixXd& K_XvX,
    const DensityFitting& /*df*/) {

    // Robust Dunlap-fit COSX exchange (Neese 2009 §2.5).
    //
    // The full correction K_robust = K_XvX + K_QvQ − K_XvQ − K_QvX
    // requires projecting K_XvX onto the space spanned by {B^P}
    // using the B-metric S_{PQ} = (B^P : B^Q).  The B-metric is
    // highly ill-conditioned (cond(S) ~ 10^{10} for JKfit bases)
    // because the B^P matrices are nearly linearly dependent in
    // the Frobenius inner product.  A well-conditioned projector
    // would use the Coulomb metric V instead, but this requires
    // per-grid-point projection (the Q-junction approach from the
    // original paper) which is O(n_pts · n_aux · n_bf²) per SCF
    // iteration — too expensive for production use.
    //
    // The infrastructure for the full correction is in place:
    //   - DensityFitting::compute_B_metric()     [B-metric S]
    //   - DensityFitting::contract_B_with(M)      [α = tr(B^P · M)]
    //   - DensityFitting::project_through_B(M)    [Σ_P B^P · M · B^P]
    //   - DensityFitting::T_flat() / cholesky_factor()  [T, L access]
    //
    // For now, return the standard COSX K unchanged.  A future
    // correction would use the per-grid-point Q-junction projection
    // (reusing the GridBatches chi cache) instead of the global
    // B-metric projector.

    return K_XvX;
}

}  // namespace vibeqc
