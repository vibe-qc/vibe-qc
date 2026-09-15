#include "vibeqc/hessian_integrals.hpp"

#include <libint2/atom.h>
#include <libint2/engine.h>
#include <array>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#include "vibeqc/init.hpp"
#include "vibeqc/thread_pool.hpp"

namespace vibeqc {

namespace {

std::vector<libint2::Atom> to_libint_atoms(const Molecule& mol) {
    std::vector<libint2::Atom> atoms;
    atoms.reserve(mol.atoms().size());
    for (const auto& a : mol.atoms()) {
        libint2::Atom la;
        la.atomic_number = a.Z;
        la.x = a.xyz[0];
        la.y = a.xyz[1];
        la.z = a.xyz[2];
        atoms.push_back(la);
    }
    return atoms;
}

std::vector<long> shell_to_atom(const BasisSet& basis, const Molecule& mol) {
    return basis.libint().shell2atom(to_libint_atoms(mol));
}

// libint orders 2nd-derivative buffers as upper-triangle of the
// perturbation-index pair: for n_pert perturbations, buffer index for
// (i, j) with i ≤ j is i·(2·n_pert − i + 1)/2 + (j − i).
inline std::size_t triu_index(int i, int j, int n_pert) {
    // i <= j
    return static_cast<std::size_t>(
        i * (2 * n_pert - i + 1) / 2 + (j - i));
}

// Scatter the contribution of one libint buffer (∂²I / ∂c_i ∂c_j with
// i_pert ≤ j_pert from the upper-triangle ordering) into the 3N × 3N
// Hessian. The Hessian formula sums over *ordered* perturbation
// pairs: H[a, b] = Σ_{i_pert, j_pert ordered} ∂²I/∂c_i∂c_j · 1[i↦a, j↦b].
// Reducing to the upper-triangle ordering picks up a factor of 2 for
// every (i, j) with i ≠ j (since (i, j) and (j, i) give the same value
// by Schwarz). The four cases:
//
//   1. libint_diagonal (i == j) → contributes once. Only writes to
//      H[a, a] since dof(i) == dof(j) by construction.
//
//   2. libint_offdiagonal (i < j) AND global DOFs differ (a ≠ b) →
//      writes to both halves of the symmetric Hessian: H[a, b] += val
//      and H[b, a] += val. Combined ordered-pair contribution = 2·val.
//
//   3. libint_offdiagonal (i < j) AND global DOFs collide (a == b) →
//      both ordered pairs (i, j) and (j, i) hit H[a, a]. Add 2·val.
//      This case occurs when two libint perturbations map to the
//      same global DOF, e.g. shell-1 and shell-2 sit on the same
//      atom (overlap/kinetic), or all four ERI shells share an atom
//      (and the integral is then translation-invariant under uniform
//      atom motion, so the four diagonal-libint contributions plus
//      these double-counted off-diagonals must sum to zero — which
//      is exactly the translational sum rule).
inline void scatter_hessian(Eigen::MatrixXd& H, int a, int b, double val,
                             bool libint_diagonal) {
    if (libint_diagonal) {
        H(a, a) += val;
    } else {
        if (a == b) {
            H(a, a) += 2.0 * val;
        } else {
            H(a, b) += val;
            H(b, a) += val;
        }
    }
}

}  // namespace

// ----------------------------------------------------------------------
// Overlap second-derivative contraction with W.
// ----------------------------------------------------------------------

Eigen::MatrixXd compute_overlap_hessian_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& W) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N = mol.atoms().size();
    const Eigen::Index Ndof = static_cast<Eigen::Index>(3 * N);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(Ndof, Ndof);

    // Overlap: only basis centers contribute (no nucleus dependence).
    // 2 centers × 3 cartesians = 6 perturbations → 21 upper-triangle buffers.
    libint2::Engine prototype(libint2::Operator::overlap,
                              shells.max_nprim(), shells.max_l(),
                              2 /*deriv_order*/);
    auto engines = make_engine_pool(prototype);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> H_tls(n_threads,
                                        Eigen::MatrixXd::Zero(Ndof, Ndof));

    const int n_shells = static_cast<int>(shells.size());
    constexpr int n_pert = 6;

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& H_local = H_tls[tid];

        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const long atom1 = s2a[s1];
        for (int s2 = 0; s2 < n_shells; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            const long atom2 = s2a[s2];

            engine.compute(shells[s1], shells[s2]);

            // pert_index → (atom, dim): 0..2 = atom1, 3..5 = atom2.
            auto pert_to_dof = [&](int p) -> int {
                const long atom = (p < 3) ? atom1 : atom2;
                const int dim = p % 3;
                return static_cast<int>(3 * atom + dim);
            };

            for (int i = 0; i < n_pert; ++i) {
                for (int j = i; j < n_pert; ++j) {
                    const std::size_t k = triu_index(i, j, n_pert);
                    const double* block = buf[k];
                    if (!block) continue;

                    double acc = 0.0;
                    for (std::size_t a = 0; a < n1; ++a) {
                        for (std::size_t b = 0; b < n2; ++b) {
                            acc += W(bf1 + a, bf2 + b) * block[a * n2 + b];
                        }
                    }
                    // E ⊃ −Σ W S; Hessian picks up the −sign.
                    const double val = -acc;
                    const int dof_i = pert_to_dof(i);
                    const int dof_j = pert_to_dof(j);
                    scatter_hessian(H_local, dof_i, dof_j, val, /*libint_diagonal=*/i == j);
                }
            }
        }
    }

    for (const auto& Ht : H_tls) H += Ht;
    // Final symmetrize — guards against residual asymmetry from the
    // double-bookkeeping in scatter_hessian when (i_pert, j_pert) and
    // (j_pert, i_pert) come from different shell pairs and write to
    // the same global (a, b) DOF in different orders.
    H = 0.5 * (H + H.transpose());
    return H;
}

// ----------------------------------------------------------------------
// (Kinetic + Nuclear) second-derivative contraction with D.
// ----------------------------------------------------------------------

Eigen::MatrixXd compute_kinetic_nuclear_hessian_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& D) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N = mol.atoms().size();
    const Eigen::Index Ndof = static_cast<Eigen::Index>(3 * N);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(Ndof, Ndof);

    const int n_threads = omp_max_threads();
    const int n_shells = static_cast<int>(shells.size());

    // ---- Kinetic (basis-only, n_pert = 6) ----
    {
        libint2::Engine prototype(libint2::Operator::kinetic,
                                  shells.max_nprim(), shells.max_l(), 2);
        auto engines = make_engine_pool(prototype);
        std::vector<Eigen::MatrixXd> H_tls(n_threads,
                                            Eigen::MatrixXd::Zero(Ndof, Ndof));
        constexpr int n_pert = 6;

        #pragma omp parallel for schedule(dynamic)
        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& H_local = H_tls[tid];

            const auto bf1 = shell2bf[s1];
            const auto n1 = shells[s1].size();
            const long atom1 = s2a[s1];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells[s2].size();
                const long atom2 = s2a[s2];

                engine.compute(shells[s1], shells[s2]);

                auto pert_to_dof = [&](int p) -> int {
                    const long atom = (p < 3) ? atom1 : atom2;
                    const int dim = p % 3;
                    return static_cast<int>(3 * atom + dim);
                };

                for (int i = 0; i < n_pert; ++i) {
                    for (int j = i; j < n_pert; ++j) {
                        const std::size_t k = triu_index(i, j, n_pert);
                        const double* block = buf[k];
                        if (!block) continue;
                        double acc = 0.0;
                        for (std::size_t a = 0; a < n1; ++a) {
                            for (std::size_t b = 0; b < n2; ++b) {
                                acc += D(bf1 + a, bf2 + b) * block[a * n2 + b];
                            }
                        }
                        // E ⊃ +Σ D T → Hessian +acc
                        scatter_hessian(H_local, pert_to_dof(i),
                                         pert_to_dof(j), acc,
                                         /*libint_diagonal=*/i == j);
                    }
                }
            }
        }
        for (const auto& Ht : H_tls) H += Ht;
    }

    // ---- Nuclear attraction (basis 2 + N nuclei = 2+N centers) ----
    {
        std::vector<std::pair<double, std::array<double, 3>>> q;
        q.reserve(N);
        for (const auto& a : mol.atoms()) {
            q.emplace_back(static_cast<double>(a.Z), a.xyz);
        }
        libint2::Engine prototype(libint2::Operator::nuclear,
                                  shells.max_nprim(), shells.max_l(), 2);
        prototype.set_params(q);
        auto engines = make_engine_pool(prototype);

        std::vector<Eigen::MatrixXd> H_tls(n_threads,
                                            Eigen::MatrixXd::Zero(Ndof, Ndof));
        const int ncenters = 2 + static_cast<int>(N);
        const int n_pert = 3 * ncenters;

        #pragma omp parallel for schedule(dynamic)
        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto tid = static_cast<std::size_t>(omp_thread_index());
            auto& engine = engines[tid];
            const auto& buf = engine.results();
            auto& H_local = H_tls[tid];

            const auto bf1 = shell2bf[s1];
            const auto n1 = shells[s1].size();
            const long atom1 = s2a[s1];
            for (int s2 = 0; s2 < n_shells; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells[s2].size();
                const long atom2 = s2a[s2];

                engine.compute(shells[s1], shells[s2]);

                auto pert_to_dof = [&](int p) -> int {
                    const int center = p / 3;
                    const int dim = p % 3;
                    long atom;
                    if (center == 0) atom = atom1;
                    else if (center == 1) atom = atom2;
                    else atom = static_cast<long>(center - 2);
                    return static_cast<int>(3 * atom + dim);
                };

                for (int i = 0; i < n_pert; ++i) {
                    for (int j = i; j < n_pert; ++j) {
                        const std::size_t k = triu_index(i, j, n_pert);
                        const double* block = buf[k];
                        if (!block) continue;
                        double acc = 0.0;
                        for (std::size_t a = 0; a < n1; ++a) {
                            for (std::size_t b = 0; b < n2; ++b) {
                                acc += D(bf1 + a, bf2 + b) * block[a * n2 + b];
                            }
                        }
                        scatter_hessian(H_local, pert_to_dof(i),
                                         pert_to_dof(j), acc,
                                         /*libint_diagonal=*/i == j);
                    }
                }
            }
        }
        for (const auto& Ht : H_tls) H += Ht;
    }

    H = 0.5 * (H + H.transpose());
    return H;
}

// ----------------------------------------------------------------------
// ERI second-derivative contraction with the two-particle density.
// ----------------------------------------------------------------------

Eigen::MatrixXd compute_eri_hessian_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& D,
    double alpha_hf) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N = mol.atoms().size();
    const Eigen::Index Ndof = static_cast<Eigen::Index>(3 * N);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(Ndof, Ndof);

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells.max_nprim(), shells.max_l(), 2);
    auto engines = make_engine_pool(prototype);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> H_tls(n_threads,
                                        Eigen::MatrixXd::Zero(Ndof, Ndof));

    // Two-particle density: Γ_μνλσ = (1/2) D_μν D_λσ − (α_HF / 4) D_μλ D_νσ
    const double exchange_coeff = 0.25 * alpha_hf;
    const int n_shells = static_cast<int>(shells.size());
    constexpr int n_pert = 12;  // 4 centers × 3 cartesians

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& H_local = H_tls[tid];

        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const long atom1 = s2a[s1];
        for (int s2 = 0; s2 < n_shells; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            const long atom2 = s2a[s2];
            for (int s3 = 0; s3 < n_shells; ++s3) {
                const auto bf3 = shell2bf[s3];
                const auto n3 = shells[s3].size();
                const long atom3 = s2a[s3];
                for (int s4 = 0; s4 < n_shells; ++s4) {
                    const auto bf4 = shell2bf[s4];
                    const auto n4 = shells[s4].size();
                    const long atom4 = s2a[s4];

                    engine.compute(shells[s1], shells[s2],
                                   shells[s3], shells[s4]);

                    const long centers[4] = {atom1, atom2, atom3, atom4};
                    auto pert_to_dof = [&](int p) -> int {
                        const int center = p / 3;
                        const int dim = p % 3;
                        return static_cast<int>(3 * centers[center] + dim);
                    };

                    for (int i = 0; i < n_pert; ++i) {
                        for (int j = i; j < n_pert; ++j) {
                            const std::size_t k = triu_index(i, j, n_pert);
                            const double* block = buf[k];
                            if (!block) continue;

                            double acc = 0.0;
                            for (std::size_t a = 0; a < n1; ++a) {
                                const auto mu = bf1 + a;
                                for (std::size_t b = 0; b < n2; ++b) {
                                    const auto nu = bf2 + b;
                                    const double Dmunu = D(mu, nu);
                                    for (std::size_t c = 0; c < n3; ++c) {
                                        const auto lam = bf3 + c;
                                        const double Dmulambda = D(mu, lam);
                                        for (std::size_t d = 0; d < n4; ++d) {
                                            const auto sig = bf4 + d;
                                            const double v = block[
                                                ((a * n2 + b) * n3 + c) * n4 + d];
                                            const double gamma =
                                                0.5 * Dmunu * D(lam, sig)
                                                - exchange_coeff * Dmulambda * D(nu, sig);
                                            acc += gamma * v;
                                        }
                                    }
                                }
                            }
                            scatter_hessian(H_local, pert_to_dof(i),
                                             pert_to_dof(j), acc,
                                             /*libint_diagonal=*/i == j);
                        }
                    }
                }
            }
        }
    }

    for (const auto& Ht : H_tls) H += Ht;
    H = 0.5 * (H + H.transpose());
    return H;
}

// ----------------------------------------------------------------------
// UHF / UKS ERI second-derivative contraction.
// ----------------------------------------------------------------------

Eigen::MatrixXd compute_eri_hessian_contribution_uhf(
    const BasisSet& basis,
    const Molecule& mol,
    const Eigen::MatrixXd& D_alpha,
    const Eigen::MatrixXd& D_beta,
    double alpha_hf) {
    ensure_libint_initialized();

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto s2a = shell_to_atom(basis, mol);
    const std::size_t N = mol.atoms().size();
    const Eigen::Index Ndof = static_cast<Eigen::Index>(3 * N);

    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(Ndof, Ndof);

    // Total density for the J piece, per-spin densities for K.
    const Eigen::MatrixXd D_total = D_alpha + D_beta;

    libint2::Engine prototype(libint2::Operator::coulomb,
                              shells.max_nprim(), shells.max_l(), 2);
    auto engines = make_engine_pool(prototype);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> H_tls(n_threads,
                                        Eigen::MatrixXd::Zero(Ndof, Ndof));

    // Two-particle density (UHF / UKS):
    //   Γ_μνλσ = (1/2) (D_α + D_β)_μν (D_α + D_β)_λσ
    //          − (α_HF / 2) D_α_μλ D_α_νσ − (α_HF / 2) D_β_μλ D_β_νσ
    // Compared to closed-shell RHF Γ = (1/2) D_μν D_λσ − (α_HF/4) D_μλ D_νσ:
    // - J piece uses (D_α+D_β) which equals D for closed shell, so identical
    // - K piece is 2 · per-spin K (each with α_HF/2 prefactor) which sums to
    //   α_HF · K(D)/2 for closed shell — but RHF has α_HF/4. The difference
    //   (factor 2) is because UHF has separate α and β orbitals, each
    //   carrying half of D, so their squared contributions are 2 · ((D/2)² · α_HF/2)
    //   = α_HF/4 · D² — matches the RHF formula.
    const double half_alpha = 0.5 * alpha_hf;
    const int n_shells = static_cast<int>(shells.size());
    constexpr int n_pert = 12;

    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& H_local = H_tls[tid];

        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const long atom1 = s2a[s1];
        for (int s2 = 0; s2 < n_shells; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            const long atom2 = s2a[s2];
            for (int s3 = 0; s3 < n_shells; ++s3) {
                const auto bf3 = shell2bf[s3];
                const auto n3 = shells[s3].size();
                const long atom3 = s2a[s3];
                for (int s4 = 0; s4 < n_shells; ++s4) {
                    const auto bf4 = shell2bf[s4];
                    const auto n4 = shells[s4].size();
                    const long atom4 = s2a[s4];

                    engine.compute(shells[s1], shells[s2],
                                   shells[s3], shells[s4]);

                    const long centers[4] = {atom1, atom2, atom3, atom4};
                    auto pert_to_dof = [&](int p) -> int {
                        const int center = p / 3;
                        const int dim = p % 3;
                        return static_cast<int>(3 * centers[center] + dim);
                    };

                    for (int i = 0; i < n_pert; ++i) {
                        for (int j = i; j < n_pert; ++j) {
                            const std::size_t k = triu_index(i, j, n_pert);
                            const double* block = buf[k];
                            if (!block) continue;

                            double acc = 0.0;
                            for (std::size_t a = 0; a < n1; ++a) {
                                const auto mu = bf1 + a;
                                for (std::size_t b = 0; b < n2; ++b) {
                                    const auto nu = bf2 + b;
                                    const double Dmunu = D_total(mu, nu);
                                    for (std::size_t c = 0; c < n3; ++c) {
                                        const auto lam = bf3 + c;
                                        const double Damulambda = D_alpha(mu, lam);
                                        const double Dbmulambda = D_beta(mu, lam);
                                        for (std::size_t d = 0; d < n4; ++d) {
                                            const auto sig = bf4 + d;
                                            const double v = block[
                                                ((a * n2 + b) * n3 + c) * n4 + d];
                                            const double gamma =
                                                0.5 * Dmunu * D_total(lam, sig)
                                                - half_alpha * Damulambda * D_alpha(nu, sig)
                                                - half_alpha * Dbmulambda * D_beta(nu, sig);
                                            acc += gamma * v;
                                        }
                                    }
                                }
                            }
                            scatter_hessian(H_local, pert_to_dof(i),
                                             pert_to_dof(j), acc,
                                             /*libint_diagonal=*/i == j);
                        }
                    }
                }
            }
        }
    }

    for (const auto& Ht : H_tls) H += Ht;
    H = 0.5 * (H + H.transpose());
    return H;
}

// ----------------------------------------------------------------------
// Nuclear repulsion second derivative (closed form).
// ----------------------------------------------------------------------

Eigen::MatrixXd nuclear_repulsion_hessian(const Molecule& mol) {
    const auto& atoms = mol.atoms();
    const std::size_t N = atoms.size();
    const Eigen::Index Ndof = static_cast<Eigen::Index>(3 * N);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(Ndof, Ndof);

    // E_nuc = (1/2) Σ_{A≠B} Z_A Z_B / |R_A − R_B|
    // ∂²(1/r)/∂x_α∂x_β = (3 r_α r_β − r² δ_αβ) / r⁵   (with r = R_A − R_B)
    auto d2_inv_r = [](const std::array<double, 3>& r) {
        const double r2 = r[0]*r[0] + r[1]*r[1] + r[2]*r[2];
        const double r1 = std::sqrt(r2);
        const double r5 = r2 * r2 * r1;
        Eigen::Matrix3d M;
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                const double delta = (a == b) ? 1.0 : 0.0;
                M(a, b) = (3.0 * r[a] * r[b] - r2 * delta) / r5;
            }
        }
        return M;
    };

    for (std::size_t A = 0; A < N; ++A) {
        for (std::size_t B = 0; B < N; ++B) {
            if (A == B) continue;
            const std::array<double, 3> r = {
                atoms[A].xyz[0] - atoms[B].xyz[0],
                atoms[A].xyz[1] - atoms[B].xyz[1],
                atoms[A].xyz[2] - atoms[B].xyz[2],
            };
            const double ZAZB = static_cast<double>(atoms[A].Z)
                              * static_cast<double>(atoms[B].Z);
            const Eigen::Matrix3d M = d2_inv_r(r);

            // Diagonal block ∂²/∂R_A∂R_A: + Z_A Z_B M, accumulating
            // over B ≠ A. The (A, B) and (B, A) loop iterations both
            // contribute to *their own* atom's diagonal block (each
            // visit increments H(3A,3A), H(3B,3B) respectively), so
            // the diagonal contribution is correctly Σ_{B≠A} Z_A Z_B M_AB.
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    H(3*A + a, 3*A + b) += ZAZB * M(a, b);
                }
            }
            // Off-diagonal block ∂²/∂R_A∂R_B = − Z_A Z_B M. The
            // double-loop visits both (A, B) and (B, A) — each writes
            // to ONE off-diagonal block (the one named in the pair).
            // The Hessian's symmetry is preserved because M(R_A − R_B)
            // = M(−r) = M(r) (M is even in r), so H(3A, 3B) and
            // H(3B, 3A) end up with the same value.
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    H(3*A + a, 3*B + b) -= ZAZB * M(a, b);
                }
            }
        }
    }
    return H;
}

}  // namespace vibeqc
