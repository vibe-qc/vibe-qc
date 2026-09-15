// SPDX-License-Identifier: MPL-2.0
//
// EEQ atomic-charge solve. Equation numbers refer to eeq_charges.hpp.
//
// Structure of this file (top to bottom):
//
//   1. eeq_coordination_numbers — Eq. 4 + Eq. 5 (CN counting and clip)
//   2. build_coulomb_matrix     — Eqs. 2a, 2b assembled into the upper
//                                 left n×n block + the Lagrange row/col
//   3. build_rhs_vector         — Eq. 3 assembled with Q_total in the
//                                 final slot
//   4. eeq_atomic_charges       — orchestrator: validate input → CN →
//                                 matrix → RHS → LDLᵀ solve → unpack
//
// Each builder is a pure function in its own anonymous-namespace
// helper so they can be unit-tested by mocking the parameter table.
// We do not factor the matrix incrementally (each call is one full
// solve); the n×n LDLᵀ cost is negligible vs. the Coulomb-matrix build
// for n up to ~10⁴.

#include "vibeqc/eeq_charges.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace vibeqc {

// =============================================================================
// Numerical constants (kept inline so callers can see the magic numbers
// without chasing a header).
// =============================================================================

namespace {

// √(2/π) — the "Gaussian self-Coulomb prefactor" in Eq. 2a.
constexpr double kInvSqrtPi_times_sqrt2 = 0.7978845608028654;

// CN regularisation added under the square root in Eq. 3 to keep
// d(√CN)/dCN finite as CN → 0. ε = 10⁻¹⁴ matches the cited reference.
constexpr double kCnRegEps = 1.0e-14;

// D3/D4 covalent-radius convention. The CN counting function (Eq. 4)
// uses the *scaled* covalent radius 4/3 · R_cov, not the bare
// Pyykkö-Atsumi 2009 value. This factor is the long-standing D3
// convention (Grimme et al., J. Chem. Phys. 132, 154104 (2010),
// Eq. 17 surrounding text) and is carried forward unchanged into the
// D4 / EEQ-2019 coordination number. mctc-lib's get_covalent_rad()
// bakes the 4/3 into its stored table; vibe-qc keeps
// eeq_covalent_radius() returning the bare Pyykkö value (honest to
// the cited primary source) and applies the convention explicitly
// here, where it belongs.
//
// Empirically: with the bare radius the EEQ charges are ~10 % off the
// dftd4 reference; with the 4/3 scaling they agree to <1e-3 e across
// H2O / CH4 / NH3 / HF (the counting function then yields the
// expected integer-like coordination numbers — CN(O)≈2, CN(H)≈1 in
// water).
constexpr double kCovalentRadiusScale = 4.0 / 3.0;

// =============================================================================
// 1. Coordination number — Eq. 4 + Eq. 5.
// =============================================================================
//
// The counting function f(R) is monotone in R: 1 at R = 0, ½ at
// R = R0, → 0 as R → ∞. erf gives a smooth interpolation between those
// limits with steepness kcn. We sum f over all neighbour pairs and
// then pass through the soft-cap clip[].
//
// We do not differentiate here — the analytic dCN/dR contributions to
// dq/dR for the dispersion gradient are a separate routine (not yet
// landed; see header scope note).

inline double erf_count(double r, double r0, double kcn) {
    // The counting function in Eq. 4, evaluated for one neighbour pair.
    return 0.5 * (1.0 + std::erf(-kcn * (r - r0) / r0));
}

inline double cn_softcap(double cn_raw, double cnmax) {
    // Eq. 5. log1p(exp(.)) is the numerically-stable form of log(1+exp(.))
    // and handles the cn_raw >> cnmax limit gracefully (the second
    // log1p underflows to ~0, giving CN → log1p(exp(cnmax))).
    return std::log1p(std::exp(cnmax)) - std::log1p(std::exp(cnmax - cn_raw));
}

}  // namespace


Eigen::VectorXd eeq_coordination_numbers(const Molecule& mol,
                                          const EEQOptions& opts) {
    const auto& atoms = mol.atoms();
    const auto n = static_cast<Eigen::Index>(atoms.size());
    const double cutoff2 = opts.cn_cutoff * opts.cn_cutoff;

    Eigen::VectorXd cn = Eigen::VectorXd::Zero(n);

    // Pair sum. We touch each unordered pair (A,B) once and add the
    // counting-function value to both CN_A and CN_B (CN is symmetric
    // in the pair).
    for (Eigen::Index A = 0; A < n; ++A) {
        const double rA = eeq_covalent_radius(atoms[A].Z);
        for (Eigen::Index B = A + 1; B < n; ++B) {
            const double dx = atoms[A].xyz[0] - atoms[B].xyz[0];
            const double dy = atoms[A].xyz[1] - atoms[B].xyz[1];
            const double dz = atoms[A].xyz[2] - atoms[B].xyz[2];
            const double r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > cutoff2 || r2 < 1.0e-20) continue;
            const double rAB = std::sqrt(r2);

            // R0_AB = 4/3 · (R_cov(A) + R_cov(B)) — D3/D4 convention,
            // see kCovalentRadiusScale.
            const double r0 = kCovalentRadiusScale
                            * (rA + eeq_covalent_radius(atoms[B].Z));

            const double f = erf_count(rAB, r0, opts.cn_exp);
            cn(A) += f;
            cn(B) += f;
        }
    }

    // Apply the per-atom soft cap (Eq. 5). Has negligible effect for
    // CN ≪ cnmax (which is the common case for organic-chemistry
    // coordination numbers, 0..4 vs cnmax = 8) but keeps the model
    // bounded for over-coordinated transition-metal cases.
    for (Eigen::Index A = 0; A < n; ++A) {
        cn(A) = cn_softcap(cn(A), opts.cn_max);
    }

    return cn;
}


// =============================================================================
// 2. Coulomb matrix builder — Eqs. 2a, 2b plus the constraint row/col.
// =============================================================================

namespace {

void build_coulomb_matrix(const Molecule& mol,
                           Eigen::Ref<Eigen::MatrixXd> A) {
    const auto& atoms = mol.atoms();
    const auto n = static_cast<Eigen::Index>(atoms.size());

    A.setZero();

    // Off-diagonal: Eq. 2b. Both upper and lower triangles populated
    // (LDLᵀ in Eigen reads either; we fill both to keep the matrix
    // explicitly symmetric for debugging / introspection).
    for (Eigen::Index i = 0; i < n; ++i) {
        const double gi = eeq_gaussian_width(atoms[i].Z);
        for (Eigen::Index j = 0; j < i; ++j) {
            const double gj = eeq_gaussian_width(atoms[j].Z);

            const double dx = atoms[i].xyz[0] - atoms[j].xyz[0];
            const double dy = atoms[i].xyz[1] - atoms[j].xyz[1];
            const double dz = atoms[i].xyz[2] - atoms[j].xyz[2];
            const double r2 = dx*dx + dy*dy + dz*dz;
            const double r  = std::sqrt(r2);
            if (r < 1.0e-12) {
                throw std::invalid_argument(
                    "eeq_atomic_charges: coincident atoms (indices "
                    + std::to_string(i) + " and " + std::to_string(j) +
                    "; r ≈ 0). Check the input geometry.");
            }

            // κ_AB = 1/sqrt(γ_A² + γ_B²)  — see Eq. 2b text
            const double kappa = 1.0 / std::sqrt(gi*gi + gj*gj);
            const double v = std::erf(r * kappa) / r;
            A(i, j) = v;
            A(j, i) = v;
        }
    }

    // Diagonal: Eq. 2a.
    for (Eigen::Index i = 0; i < n; ++i) {
        A(i, i) = eeq_chemical_hardness(atoms[i].Z)
                + kInvSqrtPi_times_sqrt2 / eeq_gaussian_width(atoms[i].Z);
    }

    // Constraint row/column for the charge-conservation Lagrange
    // multiplier (Eq. 1). The bottom-right zero corresponds to the
    // fact that the constraint doesn't couple to itself.
    for (Eigen::Index i = 0; i < n; ++i) {
        A(n, i) = 1.0;
        A(i, n) = 1.0;
    }
    A(n, n) = 0.0;
}

}  // namespace


// =============================================================================
// 3. Right-hand side builder — Eq. 3 + the Q_total slot.
// =============================================================================

namespace {

void build_rhs_vector(const Molecule& mol,
                       const Eigen::VectorXd& cn,
                       double total_charge,
                       Eigen::Ref<Eigen::VectorXd> b) {
    const auto& atoms = mol.atoms();
    const auto n = static_cast<Eigen::Index>(atoms.size());

    for (Eigen::Index i = 0; i < n; ++i) {
        const int Z = atoms[i].Z;
        // χ_eff(A) = χ_A − κχ_A · √(CN_A + ε)   (Eq. 3 with regularisation
        // matching the published reference). The system RHS is −χ_eff
        // (Eq. 1, first n components), so we form it directly with the
        // opposite sign.
        b(i) = -eeq_electronegativity(Z)
             + eeq_cn_scaling(Z) * cn(i) / std::sqrt(cn(i) + kCnRegEps);
    }
    b(n) = total_charge;
}

}  // namespace


// =============================================================================
// 4. Orchestrator — input validation, CN, matrix, RHS, factorise, unpack.
// =============================================================================

EEQResult eeq_atomic_charges(const Molecule& mol,
                              double total_charge,
                              const EEQOptions& opts) {
    const auto& atoms = mol.atoms();
    const auto n = static_cast<Eigen::Index>(atoms.size());

    // Element-coverage check — fail before we burn time computing CN /
    // building the matrix.
    const int maxz = eeq_max_z();
    for (const auto& a : atoms) {
        if (a.Z < 1 || a.Z > maxz) {
            throw std::invalid_argument(
                "eeq_atomic_charges: element Z=" + std::to_string(a.Z) +
                " outside the EEQ parameter table (max Z=" +
                std::to_string(maxz) + ").");
        }
    }

    EEQResult result;
    if (n == 0) {
        result.charges = Eigen::VectorXd();
        result.chemical_potential = 0.0;
        return result;
    }

    // Build everything.
    const Eigen::VectorXd cn = eeq_coordination_numbers(mol, opts);

    const auto ndim = n + 1;
    Eigen::MatrixXd A(ndim, ndim);
    build_coulomb_matrix(mol, A);

    Eigen::VectorXd b(ndim);
    build_rhs_vector(mol, cn, total_charge, b);

    // Symmetric LDLᵀ solve (Eigen's robust path for indefinite
    // symmetric systems; the (n+1)st row/col with the zero bottom-right
    // makes the augmented matrix indefinite even when the n×n block is
    // positive definite).
    Eigen::LDLT<Eigen::MatrixXd> ldlt(A);
    if (ldlt.info() != Eigen::Success) {
        throw std::runtime_error(
            "eeq_atomic_charges: LDLᵀ factorisation of the EEQ Coulomb "
            "matrix failed — system singular or non-PSD. Likely cause: "
            "degenerate input geometry not caught by the coincident-"
            "atoms check.");
    }
    const Eigen::VectorXd x = ldlt.solve(b);

    result.charges = x.head(n);
    result.chemical_potential = x(n);
    return result;
}

}  // namespace vibeqc
