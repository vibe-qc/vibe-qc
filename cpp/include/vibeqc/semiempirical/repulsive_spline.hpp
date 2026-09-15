// Repulsive spline potential for production DFTB.
//
// Replaces the simple R⁻¹² placeholder with cubic-spline-fitted
// short-range repulsive potentials of the form:
//
//   V_rep(R) = Σ_{k=0}^{N-1} c_k · (R_cut − R)^k   for R < R_cut
//   V_rep(R) = 0                                      for R ≥ R_cut
//
// where coefficients c_k are fitted to DFT reference data.
// Continuity at R_cut requires c_0 = c_1 = 0 (value and slope zero).
//
// For simplicity, we store (R_i, V_i) knot points and use cubic
// spline interpolation with natural boundary conditions.
// The gradient dV/dR is available analytically from the spline.

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Cubic spline repulsive potential
// ---------------------------------------------------------------------------

class RepulsiveSpline {
public:
    RepulsiveSpline() = default;

    // Build from knot points (R_i, V_i). R must be strictly increasing.
    // Uses natural cubic spline (y'' = 0 at endpoints).
    void build(const std::vector<double>& R, const std::vector<double>& V);

    // Evaluate V_rep(R). Returns 0 for R >= R_max (last knot).
    double evaluate(double R) const;

    // Evaluate dV_rep/dR. Returns 0 for R >= R_max.
    double derivative(double R) const;

    // Number of knots.
    std::size_t n_knots() const noexcept { return R_.size(); }

    // Raw knot data used to construct the spline.  Parameter identity hashes
    // these inputs, not the derived interpolation coefficients.
    const std::vector<double>& knot_positions() const noexcept { return R_; }
    const std::vector<double>& knot_values() const noexcept { return V_; }

    // Cutoff distance (last knot).
    double cutoff() const noexcept {
        return R_.empty() ? 0.0 : R_.back();
    }

    bool empty() const noexcept { return R_.empty(); }

private:
    std::vector<double> R_;     // knot positions (bohr), strictly increasing
    std::vector<double> V_;     // V_rep at knots (Hartree)
    std::vector<double> a_, b_, c_, d_;  // spline coefficients per interval
};

// ---------------------------------------------------------------------------
// Compact repulsive: may be spline (N>0 knots) or analytic R⁻¹² (B=0)
// or exponential (B>0). The factory methods select the form.
// ---------------------------------------------------------------------------

struct RepulsivePairV2 {
    double A = 0.0;           // prefactor for analytic forms
    double B = 0.0;           // decay for exponential; 0 = R⁻¹²
    RepulsiveSpline spline;   // if non-empty, use spline (ignores A, B)
};

// Evaluate V_rep(R) for a RepulsivePairV2.
inline double eval_repulsive(const RepulsivePairV2& rp, double R) {
    if (R < 1e-12) return 0.0;
    if (!rp.spline.empty()) {
        return rp.spline.evaluate(R);
    }
    if (rp.B > 0.0) {
        return rp.A * std::exp(-rp.B * R);
    }
    return rp.A / std::pow(R, 12);
}

// Evaluate dV_rep/dR for a RepulsivePairV2.
inline double eval_repulsive_derivative(const RepulsivePairV2& rp, double R) {
    if (R < 1e-12) return 0.0;
    if (!rp.spline.empty()) {
        return rp.spline.derivative(R);
    }
    if (rp.B > 0.0) {
        return -rp.A * rp.B * std::exp(-rp.B * R);
    }
    return -12.0 * rp.A / std::pow(R, 13);
}

}  // namespace semiempirical
}  // namespace vibeqc
