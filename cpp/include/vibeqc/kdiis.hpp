// KDIIS — Kollmar's DIIS (orbital-rotation gradient error vector).
//
// Reference:
//   * C. Kollmar, "Convergence optimization of restricted open-shell
//     self-consistent field calculations", Int. J. Quantum Chem. 62,
//     617-637 (1997),
//     doi:10.1002/(SICI)1097-461X(1997)62:6<617::AID-QUA5>3.0.CO;2-Z
//     Exposed by ORCA as the opt-in `!KDIIS` keyword (ORCA manual,
//     "SCF Convergence"); reported robust on transition-metal complexes
//     and open-shell systems where the AO-basis commutator DIIS error is
//     dominated by the wrong eigenmodes.
//
// Deviation from Kollmar's original. Kollmar pairs the orbital-gradient
// DIIS with a first-order-perturbation orbital update (energy
// denominators of C. Kollmar, J. Chem. Phys. 105, 8204 (1996),
// doi:10.1063/1.472674), which makes his scheme diagonalization-free.
// vibe-qc adopts only the error-vector choice: the extrapolated Fock
// below is handed back to the driver and diagonalized as usual.
//
// Where Pulay's DIIS uses the AO-basis commutator e = F D S − S D F as
// the error vector, KDIIS uses the orbital-rotation gradient
//   g_{ai} = F^MO_{ai}   (occ-vir block of F in the canonical MO basis)
// At the SCF fixed point both vanish (the Brillouin condition is
// equivalent to ‖F^MO_{ov}‖ = 0 on the converged density). The
// extrapolation machinery — Σ_i c_i F_i with constraint Σ_i c_i = 1
// minimising ‖Σ_i c_i e_i‖_F — is the same Pulay-style least-squares
// problem; only the error metric differs.
//
// Closed-shell stores (F, g_ov) per iterate. Open-shell stores
// (F_α, F_β, g_α, g_β) per iterate; the B-matrix is built from
// concatenated (g_α, g_β) errors and a single coefficient set
// extrapolates both Fock matrices.
//
// Storage. ``max_subspace`` caps the rolling history. The B-matrix
// least-squares is the standard Pulay (n+1)-dim system with
// full-pivot LU (handles the mild ill-conditioning that appears when
// errors near-collapse onto a low-rank subspace at convergence).

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <deque>
#include <utility>

namespace vibeqc {

class KDIIS {
public:
    explicit KDIIS(std::size_t max_subspace = 8);

    // Closed-shell (RHF / RKS) extrapolation. Builds the orbital-
    // rotation gradient g_{ai} = F^MO_{ai} from (F, C, eps, n_occ),
    // pushes (F, g) onto the rolling history, solves the Pulay
    // least-squares on the stored history, and returns the extrapolated
    // Fock in the AO basis. Before n_history >= 2 (need at least two
    // iterates to extrapolate) returns ``fock`` unchanged.
    Eigen::MatrixXd extrapolate(const Eigen::MatrixXd& fock,
                                 const Eigen::MatrixXd& C,
                                 const Eigen::VectorXd& eps,
                                 int n_occ);

    // Open-shell (UHF / UKS) extrapolation. Builds per-spin orbital-
    // gradient errors, concatenates them into a single error matrix for
    // the B-matrix solve, and returns the extrapolated (F_α, F_β) pair.
    // The same coefficient set applies to both spins (spin-coupled
    // extrapolation, matching the EDIIS open-shell convention).
    std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
    extrapolate(const Eigen::MatrixXd& fock_alpha,
                const Eigen::MatrixXd& fock_beta,
                const Eigen::MatrixXd& C_alpha,
                const Eigen::MatrixXd& C_beta,
                const Eigen::VectorXd& eps_alpha,
                const Eigen::VectorXd& eps_beta,
                int n_alpha,
                int n_beta);

    std::size_t subspace_size() const noexcept {
        return fock_history_.size();
    }
    void clear();

private:
    std::size_t max_subspace_;

    // Closed-shell history: one Fock + one error per iterate.
    // Open-shell history: per-iterate concatenation of (F_α, F_β) and
    // the stacked error (g_α, g_β). For simplicity we use parallel
    // deques and store F_β as an empty matrix in closed-shell mode.
    std::deque<Eigen::MatrixXd> fock_history_;       // F (closed) or F_α (open)
    std::deque<Eigen::MatrixXd> fock_beta_history_;  // F_β (open) / empty
    std::deque<Eigen::MatrixXd> error_history_;      // g (closed) or [g_α; g_β] (open)
    bool open_shell_ = false;
};

}  // namespace vibeqc
