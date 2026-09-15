#include "vibeqc/vv10.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace vibeqc {

namespace {
// Points with ρ below this are skipped: the kernel is singular as ρ → 0
// (through ω_p² = 4πρ → 0 in the denominator) and the contribution is
// negligible. Standard choice, matches the reference implementations.
constexpr double kRhoThresh = 1e-8;
}  // namespace

// Eq. references are to Vydrov & Van Voorhis, J. Chem. Phys. 133, 244103
// (2010). The derivative bookkeeping (dW0dR / dKdR / dW0dG carry an extra
// factor of ρ so v_rho / v_sigma come out in the libxc integrand
// convention) is cross-checked against PySCF's dft.numint._vv10nlc.
//
// Distance screening: the VV10 kernel Φ(r, r') decays as ~1/R⁶ for large
// separations (g ≈ ω₀·R²).  Pairs with R² > cutoff2 are skipped — the
// asymptotic error is O(1/R⁶) and a cutoff of 50 bohr contributes < 10⁻¹⁰
// Ha per pair.  The cutoff is a hard distance check before the expensive
// kernel evaluation; for small molecules this is a no-op (all pairs are
// within range), but the real speedup comes from the separate coarser
// VV10 grid (see GridOptions::vv10_grid_factor).
VV10Result compute_vv10(const Eigen::MatrixX3d& points,
                        const Eigen::VectorXd& weights,
                        const Eigen::VectorXd& rho,
                        const Eigen::VectorXd& sigma,
                        double b, double C,
                        double cutoff) {
    const Eigen::Index n = rho.size();
    if (points.rows() != n || weights.size() != n || sigma.size() != n) {
        throw std::invalid_argument(
            "compute_vv10: points / weights / rho / sigma must share the "
            "grid length");
    }
    if (!(b > 0.0) || !(C > 0.0)) {
        throw std::invalid_argument(
            "compute_vv10: VV10 parameters b and C must be positive");
    }

    VV10Result out;
    out.v_rho = Eigen::VectorXd::Zero(n);
    out.v_sigma = Eigen::VectorXd::Zero(n);

    // Constants (Vydrov-Van Voorhis 2010): the κ prefactor
    // Kvv = b·(3π/2)^(1/6) written as b·1.5·π·(9π)^(-1/6), and the
    // self-energy constant β = (1/32)(3/b²)^(3/4).
    const double Pi = M_PI;
    const double Pi43 = 4.0 * Pi / 3.0;
    const double Kvv = b * 1.5 * Pi * std::pow(9.0 * Pi, -1.0 / 6.0);
    const double Beta = std::pow(3.0 / (b * b), 0.75) / 32.0;

    // Build the active-point list (ρ ≥ threshold) with all per-point
    // quantities precomputed once. ``idx`` maps back to the full grid so
    // v_rho / v_sigma land in the right slot.
    struct Pt {
        double x, y, z;     // position (bohr)
        double rho;         // ρ
        double w;           // quadrature weight
        double W0;          // ω₀
        double K;           // κ
        double dW0dR;       // ρ · ∂ω₀/∂ρ
        double dKdR;        // ρ · ∂κ/∂ρ
        double dW0dG;       // ρ · ∂ω₀/∂σ
        double rw;          // ρ · w   (inner-grid weight)
        Eigen::Index idx;   // index into the full grid
    };
    std::vector<Pt> pts;
    pts.reserve(static_cast<std::size_t>(n));
    for (Eigen::Index g = 0; g < n; ++g) {
        const double r = rho(g);
        if (r < kRhoThresh) continue;
        const double s = sigma(g);
        // ω_g² = C·(|∇ρ|²/ρ²)² = C·σ²/ρ⁴
        const double w0sq_grad = C * (s / (r * r)) * (s / (r * r));
        const double W0 = std::sqrt(w0sq_grad + Pi43 * r);
        const double K = Kvv * std::pow(r, 1.0 / 6.0);
        // ρ·∂ω₀/∂ρ = (½·(4π/3)·ρ − 2·ω_g²)/ω₀
        const double dW0dR = (0.5 * Pi43 * r - 2.0 * w0sq_grad) / W0;
        // ρ·∂κ/∂ρ = κ/6
        const double dKdR = K / 6.0;
        // ρ·∂ω₀/∂σ = C·σ/(ρ³·ω₀)   (0 as σ → 0; no 0/0)
        const double dW0dG = C * s / (r * r * r * W0);
        Pt p;
        p.x = points(g, 0);
        p.y = points(g, 1);
        p.z = points(g, 2);
        p.rho = r;
        p.w = weights(g);
        p.W0 = W0;
        p.K = K;
        p.dW0dR = dW0dR;
        p.dKdR = dKdR;
        p.dW0dG = dW0dG;
        p.rw = r * weights(g);
        p.idx = g;
        pts.push_back(p);
    }

    const std::ptrdiff_t m = static_cast<std::ptrdiff_t>(pts.size());
    double energy = 0.0;
    // cutoff ≤ 0 disables screening (use effectively infinite cutoff).
    const double cutoff2 = (cutoff > 0.0) ? cutoff * cutoff
                                           : std::numeric_limits<double>::max();

    // Outer loop over active points i; inner sum over active points j.
    // Each i owns a distinct (v_rho[idx], v_sigma[idx]) slot, so the
    // OpenMP parallelisation is race-free; the energy reduces.
    // ``schedule(dynamic, 16)`` with a smaller chunk size than the
    // previous (64) because the inner loop can skip many pairs with the
    // distance cutoff, making per-i work more variable.
    #pragma omp parallel for schedule(dynamic, 16) reduction(+ : energy)
    for (std::ptrdiff_t i = 0; i < m; ++i) {
        const Pt& pi = pts[i];
        double F = 0.0;   // Σ_j RpW_j / (g·g'·(g+g'))
        double U = 0.0;   // Σ_j T_j · (1/g + 1/gt)
        double W = 0.0;   // Σ_j T_j · (1/g + 1/gt) · R²
        for (std::ptrdiff_t j = 0; j < m; ++j) {
            const Pt& pj = pts[j];
            const double dx = pj.x - pi.x;
            const double dy = pj.y - pi.y;
            const double dz = pj.z - pi.z;
            const double R2 = dx * dx + dy * dy + dz * dz;
            // Distance screening: the kernel Φ ∝ 1/(g·g'·(g+g')) and
            // g, g' grow with R², so Φ ~ 1/R⁶ asymptotically.  Pairs
            // beyond the cutoff contribute < 10⁻¹⁰ Ha and are skipped.
            if (R2 > cutoff2) continue;
            const double g = pi.W0 * R2 + pi.K;        // g(i; r_ij)
            const double gp = pj.W0 * R2 + pj.K;       // g'(j; r_ij)
            const double gt = g + gp;
            const double T = pj.rw / (g * gp * gt);
            F += T;
            const double Tk = T * (1.0 / g + 1.0 / gt);
            U += Tk;
            W += Tk * R2;
        }
        F *= -1.5;   // Φ = -3/2 · 1/(g g' (g+g'))
        // Per-electron nonlocal energy density ε_i = β + ½ F_i.
        energy += pi.w * pi.rho * (Beta + 0.5 * F);
        // Self-consistent potential (libxc integrand convention).
        out.v_rho(pi.idx) = Beta + F + 1.5 * (U * pi.dKdR + W * pi.dW0dR);
        out.v_sigma(pi.idx) = 1.5 * W * pi.dW0dG;
    }

    out.energy = energy;
    return out;
}

}  // namespace vibeqc
