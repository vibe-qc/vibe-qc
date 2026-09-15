// Robust Dunlap-fit COSX exchange.
//
//   K_robust = K_XvX + K_QvQ − K_XvQ − K_QvX
//
// where K_XvQ = Σ_P B^P · K_XvX · B^P  (DF projection of COSX K)
// and K_QvX = K_XvQ^T.
//
// References: Neese 2009 §2.5, Dunlap 1979.

#pragma once

#include <Eigen/Dense>

#include "df.hpp"

namespace vibeqc {

// Compute the robust COSX K matrix.
//
//   K_robust = K_XvX + K_QvQ − K_XvQ − K_XvQ^T
//
// where:
//   K_XvX    — standard COSX K (already computed by the caller)
//   K_QvQ    — DF-K (DensityFitting::build_K_density)
//   K_XvQ    — DF projection of K_XvX: Σ_P B^P · K_XvX · B^P
//
// The caller applies this as: K_final = compute_robust_cosx_k(D, K_cosx, df)
Eigen::MatrixXd compute_robust_cosx_k(
    const Eigen::MatrixXd& D,
    const Eigen::MatrixXd& K_XvX,
    const DensityFitting& df);

}  // namespace vibeqc
