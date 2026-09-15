// Geometry optimisation performance kernels.
//
// These functions replace the Python/numpy hot paths in
// python/vibeqc/geomopt/optimizers.py, trust.py, and ts.py
// with Eigen-powered C++ implementations.  They are exposed
// to Python via pybind11 in cpp/src/bindings.cpp.
//
// Kernels:
//   lbfgs_two_loop   — Nocedal 1980 L-BFGS step computation
//   bfgs_update       — Sherman-Morrison inverse-Hessian update
//   hessian_update_*  — SR1 / Powell / Bofill / MS Hessian updates
//   trust_region_step — More-Sorensen secular-equation solver
//   convergence_metrics — max|g|, RMS|g|, max Δx, RMS Δx

#pragma once

#include <Eigen/Dense>
#include <vector>
#include <utility>
#include <tuple>

namespace vibeqc {
namespace geomopt {

// ---------------------------------------------------------------------------
// L-BFGS two-loop recursion (Nocedal 1980)
// ---------------------------------------------------------------------------

/// Compute the L-BFGS search direction p = -H·g without forming H.
///
/// s_list, y_list: correction pairs (most recent first).
/// g: current gradient.
/// gamma0: initial Hessian scaling (H₀ = gamma0·I).
/// Returns (direction, success).  direction is -H·g.
std::pair<Eigen::VectorXd, bool> lbfgs_two_loop(
    const std::vector<Eigen::VectorXd>& s_list,
    const std::vector<Eigen::VectorXd>& y_list,
    const Eigen::VectorXd& g,
    double gamma0
);

// ---------------------------------------------------------------------------
// BFGS inverse-Hessian update (Sherman-Morrison form)
// ---------------------------------------------------------------------------

/// Update H_inv using the BFGS formula.
///
/// H_new = (I - rho·s·y^T)·H·(I - rho·y·s^T) + rho·s·s^T
/// where rho = 1/(s^T y).  Returns the updated matrix.
Eigen::MatrixXd bfgs_inverse_update(
    const Eigen::MatrixXd& H_inv,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
);

// ---------------------------------------------------------------------------
// Hessian (not inverse) update families
// ---------------------------------------------------------------------------

/// BFGS Hessian update: H_new = H + y·y^T/(y^T s) - (H·s)(H·s)^T/(s^T H s)
Eigen::MatrixXd hessian_update_bfgs(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
);

/// SR1 Hessian update: H_new = H + (y - H·s)(y - H·s)^T / ((y - H·s)^T s)
Eigen::MatrixXd hessian_update_sr1(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
);

/// Powell-symmetric-Broyden (PSB) Hessian update.
Eigen::MatrixXd hessian_update_powell(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
);

/// Bofill weighted Hessian update (convex combination of PSB + SR1).
Eigen::MatrixXd hessian_update_bofill(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
);

/// Murtagh-Sargent symmetric rank-2 update.
Eigen::MatrixXd hessian_update_ms(
    const Eigen::MatrixXd& H,
    const Eigen::VectorXd& s,
    const Eigen::VectorXd& y
);

// ---------------------------------------------------------------------------
// Trust-region subproblem solver (More-Sorensen)
// ---------------------------------------------------------------------------

/// Solve min_p  g^T p + 0.5 p^T B p  subject to ||p|| ≤ delta.
///
/// Uses the secular equation: ||p(λ)|| = delta where p(λ) = -(B + λI)^{-1} g.
/// Returns (step, level_shift, interior_flag).
/// interior_flag = true means the unconstrained Newton step fit inside delta.
std::tuple<Eigen::VectorXd, double, bool> trust_region_step(
    const Eigen::VectorXd& g,
    const Eigen::MatrixXd& B,
    double delta,
    int max_iter = 30,
    double tol = 1e-8
);

// ---------------------------------------------------------------------------
// Convergence metrics
// ---------------------------------------------------------------------------

/// Max absolute component of a vector (∞-norm).
double max_abs_component(const Eigen::VectorXd& v);

/// Root-mean-square of a vector.
double rms(const Eigen::VectorXd& v);

/// Max atomic displacement from two flat coordinate vectors.
double max_atom_displacement(
    const Eigen::VectorXd& x_old,
    const Eigen::VectorXd& x_new
);

/// RMS displacement from two flat coordinate vectors.
double rms_displacement(
    const Eigen::VectorXd& x_old,
    const Eigen::VectorXd& x_new
);

} // namespace geomopt
} // namespace vibeqc
