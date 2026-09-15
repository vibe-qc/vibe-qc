// VV10 nonlocal correlation (Vydrov & Van Voorhis, J. Chem. Phys. 133,
// 244103 (2010), doi:10.1063/1.3521275).
//
// libxc supplies the *semilocal* part of every VV10-paired functional
// (ωB97X-V, ωB97M-V, the standalone VV10, B97M-V, …) and exposes the
// two empirical parameters (b, C) via ``xc_nlc_coef``, but it does NOT
// evaluate the nonlocal double-integral correlation energy — that is the
// host code's job. This module is that evaluator.
//
// The nonlocal correlation energy is the double space integral
//
//   E_c^nl = ∫ dr ρ(r) [ β + ½ ∫ dr' ρ(r') Φ(r, r') ]
//
// with the VV10 kernel (Vydrov-Van Voorhis 2010, Eqs. 8-12)
//
//   Φ(r, r') = -3 / [ 2 g g' (g + g') ]
//   g  = ω₀(r)  · |r-r'|² + κ(r)
//   g' = ω₀(r') · |r-r'|² + κ(r')
//   ω₀(r) = sqrt( C·|∇ρ|⁴/ρ⁴ + (4π/3)·ρ )       (ω_g² + ω_p²/3)
//   κ(r)  = b · (3π/2)^(1/6) · ρ(r)^(1/6)
//          = [ b · 1.5 · π · (9π)^(-1/6) ] · ρ(r)^(1/6)
//   β     = (1/32) · (3 / b²)^(3/4)
//
// On a quadrature grid {(r_g, w_g)} this becomes the O(N²) double sum
//
//   E_c^nl = β Σ_i w_i ρ_i  +  ½ Σ_i Σ_j (w_i ρ_i)(w_j ρ_j) Φ_ij.
//
// The functional derivative (self-consistent VV10 potential) follows the
// Vydrov-Van Voorhis derivation; the implementation cross-checks against
// the reference formulas in PySCF's ``dft.numint._vv10nlc`` (read as an
// independent reference, not imported — vibe-qc has its own evaluator
// per the no-external-QC-program rule). Returned per grid point in the
// libxc ``vrho`` / ``vsigma`` convention so the RKS / UKS V_xc assembly
// folds them in alongside the semilocal pieces.

#pragma once

#include <Eigen/Dense>

namespace vibeqc {

// Result of a VV10 nonlocal-correlation evaluation on a grid.
//
//   energy   — the scalar E_c^nl (already weight-summed over the grid).
//   v_rho    — per grid point ∂(ρ ε_nl)/∂ρ        (libxc vrho convention)
//   v_sigma  — per grid point ∂(ρ ε_nl)/∂σ        (libxc vsigma convention,
//              σ = |∇ρ|²); folds into the GGA gradient term of V_xc.
//
// ``v_rho`` / ``v_sigma`` are NOT pre-multiplied by the grid weights —
// the RKS / UKS V_xc builder applies the weights when assembling the
// matrix, exactly as it does for the libxc semilocal v_rho / v_sigma.
struct VV10Result {
    double energy = 0.0;
    Eigen::VectorXd v_rho;
    Eigen::VectorXd v_sigma;
};

// Evaluate the VV10 nonlocal correlation energy and self-consistent
// potential on a molecular quadrature grid.
//
// Inputs (all length n_pts, except ``points`` which is n_pts × 3):
//   points   — grid coordinates r_g (bohr)
//   weights  — quadrature weights w_g (bohr³)
//   rho      — total density ρ(r_g)
//   sigma    — squared density gradient σ(r_g) = |∇ρ(r_g)|²
//   b, C     — the VV10 empirical parameters (from xc_nlc_coef; e.g.
//              b = 6.0, C = 0.01 for ωB97X-V / ωB97M-V; b = 5.9,
//              C = 0.0093 for the original VV10 functional).
//
// Grid points with ρ below an internal threshold (1e-8) contribute
// nothing and are skipped (matches the standard implementation; the
// kernel is singular as ρ → 0 through ω_p). The double sum is
// OpenMP-parallel over the outer grid index.
//
// ``cutoff`` (bohr): distance beyond which a pair (i,j) is skipped.
// The kernel decays as ~1/R⁶; 50 bohr is safe for sub-µHa accuracy.
// Pass 0.0 or a large value to disable screening.
VV10Result compute_vv10(const Eigen::MatrixX3d& points,
                        const Eigen::VectorXd& weights,
                        const Eigen::VectorXd& rho,
                        const Eigen::VectorXd& sigma,
                        double b, double C,
                        double cutoff = 50.0);

}  // namespace vibeqc
