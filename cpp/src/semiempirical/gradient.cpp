#include "vibeqc/semiempirical/gradient.hpp"
#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"
#include "vibeqc/semiempirical/core/pair_lattice.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_driver.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_multipole.hpp"
#include "vibeqc/semiempirical/methods/xtb/gfn2_aes.hpp"
#include "vibeqc/semiempirical/core/periodic_gamma.hpp"
#include <array>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "vibeqc/basis.hpp"
#include "vibeqc/gradient.hpp"        // for shell_to_atom, omp_thread_index, make_engine_pool
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"
#include "vibeqc/semiempirical/kpoints_occupations.hpp"
#include "vibeqc/init.hpp"
#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/lattice_sum.hpp"
#include "vibeqc/periodic_gradient.hpp"
#include "vibeqc/scf_mixing.hpp"      // for omp_max_threads
#include "vibeqc/thread_pool.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Repulsive gradient: dE_rep/dR_A = Σ_{B≠A} dV_rep/dR · (R_A − R_B)/R
// with V_rep(R) = A · exp(−B · R) → dV_rep/dR = −A·B · exp(−B · R)
// ---------------------------------------------------------------------------

Eigen::MatrixXd dftb0_repulsive_gradient(
    const Molecule& mol,
    const SemiempiricalParameters& params) {
    const auto& atoms = mol.atoms();
    const auto n_atoms = static_cast<Eigen::Index>(atoms.size());
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);

    for (Eigen::Index a = 0; a < n_atoms; ++a) {
        for (Eigen::Index b = a + 1; b < n_atoms; ++b) {
            Eigen::Vector3d dR;
            dR(0) = atoms[a].xyz[0] - atoms[b].xyz[0];
            dR(1) = atoms[a].xyz[1] - atoms[b].xyz[1];
            dR(2) = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = dR.norm();
            if (R < 1e-12) continue;

            double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
            if (dVdR == 0.0) continue;
            Eigen::Vector3d force = dVdR * (dR / R);

            grad.row(a) += force;
            grad.row(b) -= force;  // Newton's third law
        }
    }
    return grad;
}

// ---------------------------------------------------------------------------
// Full DFTB0 gradient
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_dftb0_gradient(
    const Molecule& mol,
    const DFTB0Result& result,
    const SemiempiricalParameters& params) {
    ensure_libint_initialized();

    // Build minimal basis internally
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    // Build shell-to-atom map from basis.shells()
    const auto shell_infos = basis.shells();
    std::vector<long> s2a(shell_infos.size());
    for (std::size_t si = 0; si < shell_infos.size(); ++si) {
        s2a[si] = static_cast<long>(shell_infos[si].atom_index);
    }
    const auto n_atoms = static_cast<Eigen::Index>(mol.atoms().size());
    const auto n_basis = result.n_basis;

    // ---- Build the effective matrix M -----------------------------------
    //
    // M_{μν} = D_{μν} · ½κ · h̄_{AB} − W_{μν}   (two-center, A≠B)
    // M_{μν} = −W_{μν}                           (on-site, A=B)
    //
    // W = 2 C_occ · diag(ε_occ) · C_occ^T (energy-weighted density)
    //
    const double kappa = params.kappa();
    const auto& atoms = mol.atoms();
    const auto& D = result.density;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_occ = result.n_occ;

    // Build energy-weighted density: W = 2 C_occ diag(ε_occ) C_occ^T
    Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n_basis, n_basis);
    for (int i = 0; i < n_occ; ++i) {
        W += eps(i) * (2.0 * C_occ.col(i) * C_occ.col(i).transpose());
    }

    // Map AOs to atoms and average on-site energies
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar_atom(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a) {
        hbar_atom[a] = params.average_on_site(atoms[a].Z);
    }
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int n_funcs = shells[static_cast<std::size_t>(s)].size();
        int bf_start = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < n_funcs; ++i) {
            ao_atom[bf_start + i] = s2a[s];
        }
    }

    // Build M matrix
    Eigen::MatrixXd M = -W;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            if (a_mu != a_nu) {
                double h_avg = 0.5 * (hbar_atom[a_mu] + hbar_atom[a_nu]);
                M(mu, nu) += D(mu, nu) * 0.5 * kappa * h_avg;
            }
        }
    }

    // ---- Compute Σ M_{μν} · dS_{μν}/dR via libint overlap gradient ------
    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_atoms), 3);

    libint2::Engine prototype(libint2::Operator::overlap,
                              shells.max_nprim(), shells.max_l(),
                              1 /* deriv_order */);
    auto engines = make_engine_pool(prototype);

    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_atoms), 3));

    const int n_shells = static_cast<int>(shells.size());
    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];

        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const long atom1 = s2a[s1];

        for (int s2 = 0; s2 < n_shells; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            const long atom2 = s2a[s2];

            engine.compute(shells[s1], shells[s2]);

            // 6 derivative buffers: d/dx1, d/dy1, d/dz1, d/dx2, d/dy2, d/dz2
            for (int icenter = 0; icenter < 2; ++icenter) {
                const long atom_i = (icenter == 0) ? atom1 : atom2;
                for (int d = 0; d < 3; ++d) {
                    const double* block = buf[icenter * 3 + d];
                    if (!block) continue;

                    double acc = 0.0;
                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            acc += M(bf1 + i, bf2 + j) * block[i * n2 + j];
                        }
                    }
                    grad_local(atom_i, d) += acc;
                }
            }
        }
    }

    for (const auto& g : grad_tls) grad += g;

    // ---- Add repulsive gradient ----
    grad += dftb0_repulsive_gradient(mol, params);

    return grad;
}


// ---------------------------------------------------------------------------
// SCC-DFTB analytic gradient (fixed-charge approximation)
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_scc_dftb_gradient(
    const Molecule& mol,
    const SCCDFTBResult& result,
    const SemiempiricalParameters& params) {
    ensure_libint_initialized();

    // Build minimal basis
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_infos = basis.shells();
    std::vector<long> s2a(shell_infos.size());
    for (std::size_t si = 0; si < shell_infos.size(); ++si) {
        s2a[si] = static_cast<long>(shell_infos[si].atom_index);
    }
    const auto n_atoms = static_cast<Eigen::Index>(mol.atoms().size());
    const auto n_basis = result.n_basis;

    const double kappa = params.kappa();
    const auto& atoms = mol.atoms();
    const auto& D = result.density;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_occ = result.n_occ;

    // Build gamma matrix internally
    Eigen::MatrixXd gamma = SemiempiricalHamiltonianBuilder::gamma_matrix(mol, params);

    // Energy-weighted density from SCC MOs
    Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n_basis, n_basis);
    for (int i = 0; i < n_occ; ++i) {
        W += eps(i) * (2.0 * C_occ.col(i) * C_occ.col(i).transpose());
    }

    // AO-to-atom map and average on-site energies
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar_atom(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a) {
        hbar_atom[a] = params.average_on_site(atoms[a].Z);
    }
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int n_funcs = shells[static_cast<std::size_t>(s)].size();
        int bf_start = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < n_funcs; ++i) {
            ao_atom[bf_start + i] = s2a[s];
        }
    }

    // Build M: DFTB0-style + SCC potential contribution.
    // The Mulliken charges depend on geometry explicitly through S:
    // ∂Δq_A/∂S_{μν} = −½ D_{μν} (δ_{μ∈A} + δ_{ν∈A}), so
    // Σ_A V_A ∂Δq_A/∂R = −½ (V_A + V_B) D_{μν} dS_{μν}/dR — the same
    // sign as the H¹ correction (Elstner et al., PRB 58, 7260 (1998),
    // Eq. 25, with our Δq > 0 = cation convention).
    // M += D * 0.5 * (kappa*h_avg − V_A − V_B)
    Eigen::VectorXd V = gamma * result.charges;
    std::vector<double> atom_V(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        atom_V[a] = V(a);

    Eigen::MatrixXd M = -W;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        double V_mu = atom_V[a_mu];
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            if (a_mu != a_nu) {
                double h_avg = 0.5 * (hbar_atom[a_mu] + hbar_atom[a_nu]);
                M(mu, nu) += D(mu, nu) * 0.5 * (kappa * h_avg - V_mu - atom_V[a_nu]);
            }
        }
    }

    // ---- Overlap derivative integral ----
    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_atoms), 3);

    libint2::Engine prototype(libint2::Operator::overlap,
                              shells.max_nprim(), shells.max_l(), 1);
    auto engines = make_engine_pool(prototype);
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_atoms), 3));

    const int n_shells = static_cast<int>(shells.size());
    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const long atom1 = s2a[s1];

        for (int s2 = 0; s2 < n_shells; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            const long atom2 = s2a[s2];
            engine.compute(shells[s1], shells[s2]);
            for (int icenter = 0; icenter < 2; ++icenter) {
                const long atom_i = (icenter == 0) ? atom1 : atom2;
                for (int d = 0; d < 3; ++d) {
                    const double* block = buf[icenter * 3 + d];
                    if (!block) continue;
                    double acc = 0.0;
                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            acc += M(bf1 + i, bf2 + j) * block[i * n2 + j];
                        }
                    }
                    grad_local(atom_i, d) += acc;
                }
            }
        }
    }
    for (const auto& g : grad_tls) grad += g;

    // ---- Repulsive gradient ----
    grad += dftb0_repulsive_gradient(mol, params);

    // ---- γ-derivative term: ½ Σ_{BC} Δq_B Δq_C · ∂γ_{BC}/∂R_A ----
    // ∂γ_{AB}/∂R = −R · γ³ (for R > 0), zero for A=B
    const auto& dq = result.charges;
    for (Eigen::Index a = 0; a < n_atoms; ++a) {
        for (Eigen::Index b = 0; b < n_atoms; ++b) {
            if (a == b) continue;
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R < 1e-12) continue;

            double g_ab = gamma(a, b);
            double dgamma_dR = -R * g_ab * g_ab * g_ab;
            // dE2/dR_A = SUM_B dq_A dq_B dgamma_AB/dR_A: the ordered
            // (a,b) loop deposits each pair's derivative into grad(a)
            // only, so no 1/2 here (the 1/2 in E2 = 1/2*SUM_{AB} is
            // cancelled by the two symmetric appearances of gamma_AB).
            // Pre-fix the 0.5 halved the SCC gamma force.
            double prefactor = dq(a) * dq(b) * dgamma_dR / R;

            grad(a, 0) += prefactor * dx;
            grad(a, 1) += prefactor * dy;
            grad(a, 2) += prefactor * dz;
        }
    }

    return grad;
}


// ---------------------------------------------------------------------------
// CP-SCC gradient "response" entry point (SE5) — now an alias.
//
// The SCC energy E = tr(D H⁰) + ½ ΣΔqγΔq + E_rep is variational in the
// density, and the Mulliken charges are an explicit function of D and S.
// At the converged SCC solution the energy is therefore stationary with
// respect to charge re-equilibration too, and the fixed-charge analytic
// gradient above is already exact (verified against central finite
// differences).  The former Z-vector machinery here existed only to
// compensate the non-variational pre-fix assembly (Σnε − E_scc with a
// wrong-sign H¹), whose spurious γ·n_val energy derivative it computed
// by finite difference.  With the assembly fixed the correction is
// identically zero, so this delegates.
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_scc_dftb_gradient_response(
    const Molecule& mol,
    const SCCDFTBResult& result,
    const SemiempiricalParameters& params) {
    return compute_scc_dftb_gradient(mol, result, params);
}


// ---------------------------------------------------------------------------
// Unrestricted DFTB0 analytic gradient
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_udftb0_gradient(
    const Molecule& mol,
    const UDFTB0Result& result,
    const SemiempiricalParameters& params) {
    ensure_libint_initialized();

    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_infos = basis.shells();
    std::vector<long> s2a(shell_infos.size());
    for (std::size_t si = 0; si < shell_infos.size(); ++si)
        s2a[si] = static_cast<long>(shell_infos[si].atom_index);
    const auto n_atoms = static_cast<Eigen::Index>(mol.atoms().size());
    const auto n_basis = result.n_basis;

    const double kappa = params.kappa();
    const auto& atoms = mol.atoms();
    const auto& Da = result.density_alpha;
    const auto& Db = result.density_beta;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_alpha = result.n_alpha;
    const int n_beta = result.n_beta;

    // Total density: D = D_alpha + D_beta
    Eigen::MatrixXd D = Da + Db;

    // Energy-weighted density: W = W_alpha + W_beta
    Eigen::MatrixXd C_occ_a = C.leftCols(n_alpha);
    Eigen::MatrixXd C_occ_b = C.leftCols(n_beta);
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n_basis, n_basis);
    for (int i = 0; i < n_alpha; ++i)
        W += eps(i) * (C_occ_a.col(i) * C_occ_a.col(i).transpose());
    for (int i = 0; i < n_beta; ++i)
        W += eps(i) * (C_occ_b.col(i) * C_occ_b.col(i).transpose());

    // AO-to-atom
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        hbar[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int nf = shells[static_cast<std::size_t>(s)].size();
        int bf0 = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < nf; ++i) ao_atom[bf0 + i] = s2a[s];
    }

    // Build M
    Eigen::MatrixXd M = -W;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            if (a_mu != a_nu) {
                double h_avg = 0.5 * (hbar[a_mu] + hbar[a_nu]);
                M(mu, nu) += D(mu, nu) * 0.5 * kappa * h_avg;
            }
        }
    }

    // Overlap derivative integral (same as closed-shell)
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);
    libint2::Engine prototype(libint2::Operator::overlap,
                              shells.max_nprim(), shells.max_l(), 1);
    auto engines = make_engine_pool(prototype);
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(n_threads, Eigen::MatrixXd::Zero(n_atoms, 3));
    const int n_shells = static_cast<int>(shells.size());
    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid]; auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];
        auto bf1 = shell2bf[s1]; auto n1 = shells[s1].size();
        long atom1 = s2a[s1];
        for (int s2 = 0; s2 < n_shells; ++s2) {
            auto bf2 = shell2bf[s2]; auto n2 = shells[s2].size();
            long atom2 = s2a[s2];
            engine.compute(shells[s1], shells[s2]);
            for (int ic = 0; ic < 2; ++ic) {
                long ai = (ic == 0) ? atom1 : atom2;
                for (int d = 0; d < 3; ++d) {
                    const double* blk = buf[ic*3 + d];
                    if (!blk) continue;
                    double acc = 0.0;
                    for (std::size_t i = 0; i < n1; ++i)
                        for (std::size_t j = 0; j < n2; ++j)
                            acc += M(bf1+i, bf2+j) * blk[i*n2 + j];
                    grad_local(ai, d) += acc;
                }
            }
        }
    }
    for (const auto& g : grad_tls) grad += g;
    grad += dftb0_repulsive_gradient(mol, params);
    return grad;
}



// ---------------------------------------------------------------------------
// Unrestricted SCC-DFTB analytic gradient
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_uscc_dftb_gradient(
    const Molecule& mol,
    const USCCDFTBResult& result,
    const SemiempiricalParameters& params) {
    ensure_libint_initialized();

    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_infos = basis.shells();
    std::vector<long> s2a(shell_infos.size());
    for (std::size_t si = 0; si < shell_infos.size(); ++si)
        s2a[si] = static_cast<long>(shell_infos[si].atom_index);
    const auto n_atoms = static_cast<Eigen::Index>(mol.atoms().size());
    const auto n_basis = result.n_basis;

    const double kappa = params.kappa();
    const auto& atoms = mol.atoms();
    const auto& Da = result.density_alpha;
    const auto& Db = result.density_beta;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_alpha = result.n_alpha;
    const int n_beta = result.n_beta;

    Eigen::MatrixXd gamma = SemiempiricalHamiltonianBuilder::gamma_matrix(mol, params);

    Eigen::MatrixXd D = Da + Db;

    Eigen::MatrixXd C_occ_a = C.leftCols(n_alpha);
    Eigen::MatrixXd C_occ_b = C.leftCols(n_beta);
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n_basis, n_basis);
    for (int i = 0; i < n_alpha; ++i)
        W += eps(i) * (C_occ_a.col(i) * C_occ_a.col(i).transpose());
    for (int i = 0; i < n_beta; ++i)
        W += eps(i) * (C_occ_b.col(i) * C_occ_b.col(i).transpose());

    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar_atom(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        hbar_atom[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int n_funcs = shells[static_cast<std::size_t>(s)].size();
        int bf_start = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < n_funcs; ++i)
            ao_atom[bf_start + i] = s2a[s];
    }

    Eigen::MatrixXd M = -W;
    Eigen::VectorXd V_uscc = gamma * result.charges;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        double V_mu = V_uscc(a_mu);
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            if (a_mu != a_nu) {
                double h_avg = 0.5 * (hbar_atom[a_mu] + hbar_atom[a_nu]);
                // −V: see compute_scc_dftb_gradient (Δq > 0 = cation).
                M(mu, nu) += D(mu, nu) * 0.5 * (kappa * h_avg - V_mu - V_uscc(a_nu));
            }
        }
    }

Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_atoms), 3);

    libint2::Engine prototype(libint2::Operator::overlap,
                              shells.max_nprim(), shells.max_l(), 1);
    auto engines = make_engine_pool(prototype);
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_atoms), 3));

    const int n_shells = static_cast<int>(shells.size());
    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const long atom1 = s2a[s1];
        for (int s2 = 0; s2 < n_shells; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            const long atom2 = s2a[s2];
            engine.compute(shells[s1], shells[s2]);
            for (int icenter = 0; icenter < 2; ++icenter) {
                const long atom_i = (icenter == 0) ? atom1 : atom2;
                for (int d = 0; d < 3; ++d) {
                    const double* block = buf[icenter * 3 + d];
                    if (!block) continue;
                    double acc = 0.0;
                    for (std::size_t i = 0; i < n1; ++i)
                        for (std::size_t j = 0; j < n2; ++j)
                            acc += M(bf1 + i, bf2 + j) * block[i * n2 + j];
                    grad_local(atom_i, d) += acc;
                }
            }
        }
    }
    for (const auto& g : grad_tls) grad += g;

    grad += dftb0_repulsive_gradient(mol, params);

    const auto& dq = result.charges;
    for (Eigen::Index a = 0; a < n_atoms; ++a) {
        for (Eigen::Index b = 0; b < n_atoms; ++b) {
            if (a == b) continue;
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R < 1e-12) continue;
            double g_ab = gamma(a, b);
            double dgamma_dR = -R * g_ab * g_ab * g_ab;
            // dE2/dR_A = SUM_B dq_A dq_B dgamma_AB/dR_A: the ordered
            // (a,b) loop deposits each pair's derivative into grad(a)
            // only, so no 1/2 here (the 1/2 in E2 = 1/2*SUM_{AB} is
            // cancelled by the two symmetric appearances of gamma_AB).
            // Pre-fix the 0.5 halved the SCC gamma force.
            double prefactor = dq(a) * dq(b) * dgamma_dR / R;
            grad(a, 0) += prefactor * dx;
            grad(a, 1) += prefactor * dy;
            grad(a, 2) += prefactor * dz;
        }
    }

    return grad;
}


// ---------------------------------------------------------------------------
// Periodic DFTB0 gradient (Gamma-point)
// ---------------------------------------------------------------------------

namespace {

struct PeriodicDFTBGammaDerivatives {
    Eigen::MatrixXd atomic_gradient;
    Eigen::Matrix3d strain_derivative = Eigen::Matrix3d::Zero();
};

PeriodicDFTBGammaDerivatives periodic_dftb_gamma_derivatives(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const Eigen::VectorXd& dq,
    const std::vector<LatticeCell>& cells,
    double cutoff_bohr,
    ShellGammaForm form) {
    const auto& atoms = system.unit_cell;
    const int n_atoms = static_cast<int>(atoms.size());

    // Differentiate exactly the gamma energy build_periodic_dftb_gamma_matrix
    // assembles:
    //   E_SCC = 1/2 dq^T [Phi_Ewald + sum_images (gamma - 1/r) + on-site] dq.
    // Elstner et al., Phys. Rev. B 58, 7260 (1998), Eqs. 19 and 23 require
    // this derivative exactly once in the SCC gradient.
    //
    // Until 2026-09-07 this differentiated a bare truncated Klopman-Ohno sum
    // by hand.  When the energy moved to the Ewald-split kernel the two
    // stopped describing the same function -- measured on the polar HF cell,
    // 15.6 % on the x component against finite differences -- so the
    // derivative now comes from the same kernel as the energy.
    std::vector<GammaSite> sites;
    std::vector<Eigen::Vector3d> positions;
    sites.reserve(atoms.size());
    positions.reserve(atoms.size());
    for (int a = 0; a < n_atoms; ++a) {
        double u = params.hubbard_u(atoms[static_cast<std::size_t>(a)].Z);
        if (u <= 0.0) u = 0.4;
        GammaSite site;
        site.atom = a;
        site.hardness = u;
        sites.push_back(site);
        positions.emplace_back(
            atoms[static_cast<std::size_t>(a)].xyz[0],
            atoms[static_cast<std::size_t>(a)].xyz[1],
            atoms[static_cast<std::size_t>(a)].xyz[2]);
    }
    std::vector<Eigen::Vector3d> translations;
    for (int axis = 0; axis < system.dim; ++axis) {
        translations.push_back(system.lattice.col(axis));
    }
    const auto records = [&positions, &cells, cutoff_bohr, n_atoms](
        const ImageRecordSink& sink) {
        const double cutoff_sq = cutoff_bohr * cutoff_bohr;
        for (const auto& cell : cells) {
            const bool zero_cell = (cell.index.array() == 0).all();
            for (int a = 0; a < n_atoms; ++a) {
                for (int b = 0; b < n_atoms; ++b) {
                    if (zero_cell && a == b) continue;
                    const Eigen::Vector3d d =
                        positions[static_cast<std::size_t>(b)] + cell.r_cart
                        - positions[static_cast<std::size_t>(a)];
                    if (d.squaredNorm() > cutoff_sq) continue;
                    ImageRecord record;
                    record.a = a;
                    record.b = b;
                    record.shift = cell.r_cart;
                    record.weight = 1.0;
                    sink(record);
                }
            }
        }
    };
    ShellGammaSpec spec;
    spec.form = form;
    spec.ko_average = KlopmanOhnoAverage::InverseHardnessMean;
    const EwaldCoulombKernel ewald(translations);

    PeriodicDFTBGammaDerivatives out{
        periodic_shell_gamma_gradient(
            sites, positions, spec, ewald, records, dq),
        periodic_shell_gamma_strain_derivative(
            sites, positions, spec, ewald, records, dq)};
    return out;
}

// Energy-weighted density W = SUM_i occ_i eps_i C_i C_i^T for a Gamma-point
// DFTB0/SCC result (issue #339). Hard-Aufbau occupations reproduce the
// historical leftCols assembly bit-for-bit, so gapped systems keep their
// exact derivative values; fractional (degenerate-frontier) occupations make
// W, and therefore every analytic derivative, an invariant of the degenerate
// eigenspace. Results that predate the occupations field fall back to hard
// Aufbau.
// `max_occupation` is the full occupancy of one spatial orbital: 2.0 for a
// closed-shell spectrum (the default), 1.0 for one spin channel of an
// unrestricted spectrum (issue #439), whose W contributions add per spin.
Eigen::MatrixXd gamma_energy_weighted_density(
    const Eigen::MatrixXd& C,
    const Eigen::VectorXd& eps,
    const Eigen::VectorXd& occupations,
    int n_occ,
    int n_basis,
    double max_occupation = 2.0) {
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n_basis, n_basis);
    // A missing occupations field (legacy result) falls back to hard Aufbau.
    bool hard_aufbau = true;
    if (occupations.size() == n_basis) {
        for (int i = 0; i < n_basis; ++i) {
            const double expected = (i < n_occ) ? max_occupation : 0.0;
            if (occupations(i) != expected) {
                hard_aufbau = false;
                break;
            }
        }
    }
    if (hard_aufbau) {
        Eigen::MatrixXd C_occ = C.leftCols(n_occ);
        for (int i = 0; i < n_occ; ++i)
            W += eps(i)
                 * (max_occupation * C_occ.col(i) * C_occ.col(i).transpose());
        return W;
    }
    for (int i = 0; i < n_basis; ++i) {
        if (occupations(i) != 0.0) {
            W += eps(i) * occupations(i)
                 * (C.col(i) * C.col(i).transpose());
        }
    }
    return W;
}

}  // namespace

Eigen::MatrixXd compute_periodic_dftb0_gradient(
    const PeriodicSystem& system,
    const PeriodicDFTB0Result& result,
    const SemiempiricalParameters& params) {
    ensure_libint_initialized();

    if (!(result.cutoff_bohr > 0.0)) {
        throw std::invalid_argument(
            "compute_periodic_dftb0_gradient: result carries no lattice-cutoff "
            "provenance; re-run the SCF with the current driver");
    }

    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = result.n_basis;
    const auto n_atoms = static_cast<Eigen::Index>(system.unit_cell.size());

    const double kappa = params.kappa();
    const auto& atoms = system.unit_cell;
    const auto& D = result.density;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_occ = result.n_occ;

    // Energy-weighted density from the SCF occupations (issue #339): the
    // degenerate-frontier equal-fractional convention makes W, and with it
    // the gradient, unique. Gapped results stay bit-identical.
    Eigen::MatrixXd W =
        gamma_energy_weighted_density(C, eps, result.occupations, n_occ, n_basis);

    // AO-to-atom
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_infos = basis.shells();
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar_atom(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        hbar_atom[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int n_funcs = shells[static_cast<std::size_t>(s)].size();
        int bf_start = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < n_funcs; ++i)
            ao_atom[bf_start + i] = shell_infos[s].atom_index;
    }

    // Elstner et al. (1998), Eq. 11: M = D*dH/dS - W. The periodic
    // Wolfsberg-Helmholtz H(g) uses dH/dS in every image block, including
    // same-unit-cell-atom pairs for g != 0. The harmless g=0 self terms
    // cancel in an atomic gradient, so one cell-independent M is sufficient.
    Eigen::MatrixXd M = -W;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            double h_avg = 0.5 * (hbar_atom[a_mu] + hbar_atom[a_nu]);
            M(mu, nu) += D(mu, nu) * 0.5 * kappa * h_avg;
        }
    }

    // Lattice sum options
    LatticeSumOptions lopt;
    lopt.cutoff_bohr = result.cutoff_bohr;

    // Build LatticeMatrixSet with -M in each cell block
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    auto cells = atom_pair_interaction_cells(system, lopt.cutoff_bohr);
    if (result.gamma_only_0) {
        cells = {cells[0]};  // match the SCF's g=0-only molecular limit
    }
    LatticeMatrixSet minus_M_set;
    minus_M_set.nbf = n_basis;
    minus_M_set.cells = cells;
    minus_M_set.blocks.resize(cells.size(), -M);
    // Mask AO pairs beyond the pair cutoff so the M(g)-against-dS(g)
    // contraction differentiates exactly the masked-S energy
    // (issue #316; pair_lattice.hpp (*)).
    mask_blocks_beyond_pair_cutoff(
        minus_M_set.blocks, cells, ao_atom, atoms, result.cutoff_bohr);

    // Use the existing overlap lattice gradient (computes -Sum_g tr(W(g) dS(g)/dR))
    Eigen::MatrixXd grad = overlap_lattice_gradient_contribution(
        basis, system, minus_M_set, lopt);

    // ---- Lattice-summed repulsive gradient ----
    for (Eigen::Index a = 0; a < n_atoms; ++a) {
        for (const auto& cell : cells) {
            const auto& g = cell.r_cart;
            for (Eigen::Index b = 0; b < n_atoms; ++b) {
                bool is_zero = (cell.index.array() == 0).all(); if (is_zero && a == b) continue;
                double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + g[0]);
                double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + g[1]);
                double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + g[2]);
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > result.cutoff_bohr) continue;
                double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                double invR = 1.0 / R;
                grad(a, 0) += dVdR * dx * invR;
                grad(a, 1) += dVdR * dy * invR;
                grad(a, 2) += dVdR * dz * invR;
            }
        }
    }

    return grad;
}


// ---------------------------------------------------------------------------
// Periodic SCC-DFTB gradient (Gamma-point)
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_periodic_scc_dftb_gradient(
    const PeriodicSystem& system,
    const PeriodicSCCDFTBResult& result,
    const SemiempiricalParameters& params) {
    if (!result.converged) {
        throw std::invalid_argument(
            "compute_periodic_scc_dftb_gradient requires a converged SCC result");
    }
    if (!(result.cutoff_bohr > 0.0)) {
        throw std::invalid_argument(
            "compute_periodic_scc_dftb_gradient: result carries no lattice-cutoff "
            "provenance; re-run the SCF with the current driver");
    }

    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = result.n_basis;
    const auto n_atoms = static_cast<Eigen::Index>(system.unit_cell.size());

    const double kappa = params.kappa();
    const auto& atoms = system.unit_cell;
    const auto& D = result.density;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_occ = result.n_occ;

    LatticeSumOptions lopt;
    lopt.cutoff_bohr = result.cutoff_bohr;
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    auto cells = atom_pair_interaction_cells(system, lopt.cutoff_bohr);

    // Match run_scc_dftb_gamma: its energy and Hamiltonian use the
    // image-summed gamma matrix, not the molecular unit-cell matrix.
    Eigen::MatrixXd gamma = build_periodic_dftb_gamma_matrix(
        system, params, lopt.cutoff_bohr, result.gamma_form);

    // Energy-weighted density from the SCF occupations (issue #339).
    Eigen::MatrixXd W =
        gamma_energy_weighted_density(C, eps, result.occupations, n_occ, n_basis);

    // AO-to-atom
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_infos = basis.shells();
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar_atom(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        hbar_atom[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int n_funcs = shells[static_cast<std::size_t>(s)].size();
        int bf_start = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < n_funcs; ++i)
            ao_atom[bf_start + i] = shell_infos[s].atom_index;
    }

    // Build M with SCC potential (−V: Δq > 0 = cation, see
    // compute_scc_dftb_gradient)
    Eigen::VectorXd V_scc = gamma * result.charges;
    Eigen::MatrixXd M = -W;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        double V_mu = V_scc(a_mu);
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            double h_avg = 0.5 * (hbar_atom[a_mu] + hbar_atom[a_nu]);
            M(mu, nu) += D(mu, nu) * 0.5
                         * (kappa * h_avg - V_mu - V_scc(a_nu));
        }
    }

    LatticeMatrixSet minus_M_set;
    minus_M_set.nbf = n_basis;
    minus_M_set.cells = cells;
    minus_M_set.blocks.resize(cells.size(), -M);
    // Mask AO pairs beyond the pair cutoff so the M(g)-against-dS(g)
    // contraction differentiates exactly the masked-S energy
    // (issue #316; pair_lattice.hpp (*)).
    mask_blocks_beyond_pair_cutoff(
        minus_M_set.blocks, cells, ao_atom, atoms, result.cutoff_bohr);

    Eigen::MatrixXd grad = overlap_lattice_gradient_contribution(
        basis, system, minus_M_set, lopt);

    // Repulsive gradient
    for (Eigen::Index a = 0; a < n_atoms; ++a) {
        for (const auto& cell : cells) {
            const auto& g = cell.r_cart;
            for (Eigen::Index b = 0; b < n_atoms; ++b) {
                bool is_zero = (cell.index.array() == 0).all(); if (is_zero && a == b) continue;
                double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + g[0]);
                double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + g[1]);
                double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + g[2]);
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > result.cutoff_bohr) continue;
                double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                double invR = 1.0 / R;
                grad(a, 0) += dVdR * dx * invR;
                grad(a, 1) += dVdR * dy * invR;
                grad(a, 2) += dVdR * dz * invR;
            }
        }
    }

    grad += periodic_dftb_gamma_derivatives(
        system, params, result.charges, cells,
        result.cutoff_bohr, result.gamma_form).atomic_gradient;

    return grad;
}


// ---------------------------------------------------------------------------
// Periodic UDFTB0 gradient (Gamma-point)
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_periodic_udftb0_gradient(
    const PeriodicSystem& system,
    const PeriodicUDFTB0Result& result,
    const SemiempiricalParameters& params) {
    ensure_libint_initialized();

    if (!(result.cutoff_bohr > 0.0)) {
        throw std::invalid_argument(
            "compute_periodic_udftb0_gradient: result carries no lattice-cutoff "
            "provenance; re-run the SCF with the current driver");
    }

    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = result.n_basis;
    const auto n_atoms = static_cast<Eigen::Index>(system.unit_cell.size());

    const double kappa = params.kappa();
    const auto& atoms = system.unit_cell;
    const auto& Da = result.density_alpha;
    const auto& Db = result.density_beta;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_alpha = result.n_alpha;
    const int n_beta = result.n_beta;

    // Total density
    Eigen::MatrixXd D = Da + Db;

    // Energy-weighted density from the SCF per-spin occupations (issue
    // #439): each channel's degenerate-frontier equal-fractional
    // convention makes W, and with it the gradient, an invariant of the
    // degenerate eigenspace.  Gapped channels stay bit-identical, and a
    // legacy result with no occupations field falls back to hard Aufbau.
    Eigen::MatrixXd W =
        gamma_energy_weighted_density(
            C, eps, result.occupations_alpha, n_alpha, n_basis, 1.0)
        + gamma_energy_weighted_density(
            C, eps, result.occupations_beta, n_beta, n_basis, 1.0);

    // AO-to-atom
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_infos = basis.shells();
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar_atom(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        hbar_atom[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int n_funcs = shells[static_cast<std::size_t>(s)].size();
        int bf_start = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < n_funcs; ++i)
            ao_atom[bf_start + i] = shell_infos[s].atom_index;
    }

    // Build M
    Eigen::MatrixXd M = -W;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            double h_avg = 0.5 * (hbar_atom[a_mu] + hbar_atom[a_nu]);
            M(mu, nu) += D(mu, nu) * 0.5 * kappa * h_avg;
        }
    }

    LatticeSumOptions lopt;
    lopt.cutoff_bohr = result.cutoff_bohr;
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    auto cells = atom_pair_interaction_cells(system, lopt.cutoff_bohr);
    if (result.gamma_only_0) {
        cells = {cells[0]};  // match the SCF's g=0-only molecular limit
    }

    LatticeMatrixSet minus_M_set;
    minus_M_set.nbf = n_basis;
    minus_M_set.cells = cells;
    minus_M_set.blocks.resize(cells.size(), -M);
    // Mask AO pairs beyond the pair cutoff so the M(g)-against-dS(g)
    // contraction differentiates exactly the masked-S energy
    // (issue #316; pair_lattice.hpp (*)).
    mask_blocks_beyond_pair_cutoff(
        minus_M_set.blocks, cells, ao_atom, atoms, result.cutoff_bohr);

    Eigen::MatrixXd grad = overlap_lattice_gradient_contribution(
        basis, system, minus_M_set, lopt);

    // Repulsive
    for (Eigen::Index a = 0; a < n_atoms; ++a) {
        for (const auto& cell : cells) {
            const auto& g = cell.r_cart;
            for (Eigen::Index b = 0; b < n_atoms; ++b) {
                bool is_zero = (cell.index.array() == 0).all(); if (is_zero && a == b) continue;
                double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + g[0]);
                double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + g[1]);
                double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + g[2]);
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > result.cutoff_bohr) continue;
                double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                double invR = 1.0 / R;
                grad(a, 0) += dVdR * dx * invR;
                grad(a, 1) += dVdR * dy * invR;
                grad(a, 2) += dVdR * dz * invR;
            }
        }
    }

    return grad;
}


// ---------------------------------------------------------------------------
// Periodic USCC-DFTB gradient (Gamma-point)
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_periodic_uscc_dftb_gradient(
    const PeriodicSystem& system,
    const PeriodicUSCCDFTBResult& result,
    const SemiempiricalParameters& params) {
    if (!result.converged) {
        throw std::invalid_argument(
            "compute_periodic_uscc_dftb_gradient requires a converged SCC result");
    }
    if (!(result.cutoff_bohr > 0.0)) {
        throw std::invalid_argument(
            "compute_periodic_uscc_dftb_gradient: result carries no lattice-cutoff "
            "provenance; re-run the SCF with the current driver");
    }

    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = result.n_basis;
    const auto n_atoms = static_cast<Eigen::Index>(system.unit_cell.size());

    const double kappa = params.kappa();
    const auto& atoms = system.unit_cell;
    const auto& Da = result.density_alpha;
    const auto& Db = result.density_beta;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_alpha = result.n_alpha;
    const int n_beta = result.n_beta;

    Eigen::MatrixXd gamma = build_periodic_dftb_gamma_matrix(
        system, params, result.cutoff_bohr);
    Eigen::MatrixXd D = Da + Db;

    // Per-spin occupation-weighted W (issue #439); see the UDFTB0 gradient.
    Eigen::MatrixXd W =
        gamma_energy_weighted_density(
            C, eps, result.occupations_alpha, n_alpha, n_basis, 1.0)
        + gamma_energy_weighted_density(
            C, eps, result.occupations_beta, n_beta, n_basis, 1.0);

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_infos = basis.shells();
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar_atom(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        hbar_atom[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int n_funcs = shells[static_cast<std::size_t>(s)].size();
        int bf_start = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < n_funcs; ++i)
            ao_atom[bf_start + i] = shell_infos[s].atom_index;
    }

    Eigen::VectorXd V_scc = gamma * result.charges;
    Eigen::MatrixXd M = -W;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        double V_mu = V_scc(a_mu);
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            double h_avg = 0.5 * (hbar_atom[a_mu] + hbar_atom[a_nu]);
            // −V: see compute_scc_dftb_gradient (Δq > 0 = cation).
            M(mu, nu) += D(mu, nu) * 0.5
                         * (kappa * h_avg - V_mu - V_scc(a_nu));
        }
    }

    LatticeSumOptions lopt;
    lopt.cutoff_bohr = result.cutoff_bohr;
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    auto cells = atom_pair_interaction_cells(system, lopt.cutoff_bohr);

    LatticeMatrixSet minus_M_set;
    minus_M_set.nbf = n_basis;
    minus_M_set.cells = cells;
    minus_M_set.blocks.resize(cells.size(), -M);
    // Mask AO pairs beyond the pair cutoff so the M(g)-against-dS(g)
    // contraction differentiates exactly the masked-S energy
    // (issue #316; pair_lattice.hpp (*)).
    mask_blocks_beyond_pair_cutoff(
        minus_M_set.blocks, cells, ao_atom, atoms, result.cutoff_bohr);

    Eigen::MatrixXd grad = overlap_lattice_gradient_contribution(
        basis, system, minus_M_set, lopt);

    for (Eigen::Index a = 0; a < n_atoms; ++a) {
        for (const auto& cell : cells) {
            const auto& g = cell.r_cart;
            for (Eigen::Index b = 0; b < n_atoms; ++b) {
                bool is_zero = (cell.index.array() == 0).all(); if (is_zero && a == b) continue;
                double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + g[0]);
                double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + g[1]);
                double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + g[2]);
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > result.cutoff_bohr) continue;
                double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                double invR = 1.0 / R;
                grad(a, 0) += dVdR * dx * invR;
                grad(a, 1) += dVdR * dy * invR;
                grad(a, 2) += dVdR * dz * invR;
            }
        }
    }

    grad += periodic_dftb_gamma_derivatives(
        system, params, result.charges, cells,
        result.cutoff_bohr, result.gamma_form).atomic_gradient;

    return grad;
}


// ---------------------------------------------------------------------------
// Periodic DFTB0 analytic stress tensor (Gamma-point)
//
// Computed as sigma = sigma_rep + sigma_elec, where:
//   sigma_rep = pair-potential virial
//   sigma_elec = atomic virial + cell-vector correction
//
// The atomic virial is computed from the electronic part of the
// atomic gradient (total gradient minus repulsive gradient).
// The cell-vector correction is the strain derivative of the
// overlap at fixed atomic positions: dS(g)/deps |_R fixed.
// ---------------------------------------------------------------------------

namespace {

std::vector<libint2::Shell> shift_shells_stress(
    const libint2::BasisSet& shells, const Eigen::Vector3d& dr) {
    std::vector<libint2::Shell> out(shells.begin(), shells.end());
    for (auto& s : out) {
        s.O[0] += dr[0]; s.O[1] += dr[1]; s.O[2] += dr[2];
    }
    return out;
}

Eigen::Matrix3d atomic_strain_derivative(
    const PeriodicSystem& system,
    const Eigen::MatrixXd& gradient) {
    Eigen::Matrix3d out = Eigen::Matrix3d::Zero();
    const auto& atoms = system.unit_cell;
    for (Eigen::Index a = 0;
         a < static_cast<Eigen::Index>(atoms.size()); ++a) {
        const Eigen::Vector3d Ra(
            atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]);
        // vibe-qc defines sigma_ij = (1/V) dE/deps_ij with
        // R' = (I + eps) R and gradient = dE/dR (not force), hence
        //   dE/deps_ij = SUM_A gradient(A,i) R_A,j.
        // The Cartesian derivative index is first and the sign is positive.
        out += gradient.row(a).transpose() * Ra.transpose();
    }
    return out;
}

Eigen::Matrix3d overlap_cell_strain_derivative(
    const libint2::BasisSet& shells,
    const std::vector<LatticeCell>& cells,
    const std::vector<Eigen::MatrixXd>& M_blocks) {
    const auto shell2bf = shells.shell2bf();
    libint2::Engine prototype(libint2::Operator::overlap,
                              shells.max_nprim(), shells.max_l(), 1);
    auto engines = make_engine_pool(prototype);
    const int n_threads = omp_max_threads();
    std::vector<Eigen::Matrix3d> cell_tls(
        n_threads, Eigen::Matrix3d::Zero());
    const int n_shells = static_cast<int>(shells.size());
    const int n_cells = static_cast<int>(cells.size());

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < n_cells; ++c) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& cell_loc = cell_tls[tid];
        const Eigen::Vector3d g = cells[c].r_cart;
        const auto shells_g = shift_shells_stress(shells, g);
        const Eigen::MatrixXd& M = M_blocks[c];

        for (int s1 = 0; s1 < n_shells; ++s1) {
            const auto bf1 = shell2bf[s1];
            const auto n1 = shells[s1].size();
            for (int s2 = 0; s2 < n_shells; ++s2) {
                const auto bf2 = shell2bf[s2];
                const auto n2 = shells_g[s2].size();
                engine.compute(shells[s1], shells_g[s2]);

                // Elstner et al. (1998), Eqs. 11 and 23, supply the M*dS
                // contraction. Under strain, the explicit image derivative is
                //   dE/deps_ij = SUM_gmunu M_munu
                //                  * dS_munu(g)/dR_ket,i * g_j.
                for (int i_cart = 0; i_cart < 3; ++i_cart) {
                    const double* block = buf[3 + i_cart];
                    if (!block) continue;
                    double acc = 0.0;
                    for (std::size_t i = 0; i < n1; ++i) {
                        for (std::size_t j = 0; j < n2; ++j) {
                            acc += M(
                                static_cast<Eigen::Index>(bf1)
                                    + static_cast<Eigen::Index>(i),
                                static_cast<Eigen::Index>(bf2)
                                    + static_cast<Eigen::Index>(j))
                                * block[i * n2 + j];
                        }
                    }
                    for (int j_cart = 0; j_cart < 3; ++j_cart)
                        cell_loc(i_cart, j_cart) += acc * g(j_cart);
                }
            }
        }
    }

    Eigen::Matrix3d out = Eigen::Matrix3d::Zero();
    for (const auto& local : cell_tls) out += local;
    return out;
}

}  // namespace

Eigen::MatrixXd compute_periodic_dftb0_stress(
    const PeriodicSystem& system,
    const PeriodicDFTB0Result& result,
    const SemiempiricalParameters& params) {
    ensure_libint_initialized();

    if (!(result.cutoff_bohr > 0.0)) {
        throw std::invalid_argument(
            "compute_periodic_dftb0_stress: result carries no lattice-cutoff "
            "provenance; re-run the SCF with the current driver");
    }

    Eigen::Matrix3d L = system.lattice;
    double V = std::abs(L.determinant());
    if (V < 1e-30) return Eigen::MatrixXd::Zero(3, 3);

    const auto n_atoms = static_cast<Eigen::Index>(system.unit_cell.size());
    const auto& atoms = system.unit_cell;

    // ---- Repulsive stress (exact pair virial) ----
    LatticeSumOptions lopt;
    lopt.cutoff_bohr = result.cutoff_bohr;
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    auto cells = atom_pair_interaction_cells(system, lopt.cutoff_bohr);
    if (result.gamma_only_0) {
        cells = {cells[0]};  // match the SCF's g=0-only molecular limit
    }

    Eigen::MatrixXd stress = Eigen::MatrixXd::Zero(3, 3);

    for (const auto& cell : cells) {
        const Eigen::Vector3d g = cell.r_cart;
        for (Eigen::Index a = 0; a < n_atoms; ++a) {
            Eigen::Vector3d Ra(atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]);
            bool is_zero = (cell.index.array() == 0).all();
            int start_b = is_zero ? static_cast<int>(a) + 1 : 0;
            for (int b = start_b; b < static_cast<int>(n_atoms); ++b) {
                Eigen::Vector3d Rb(atoms[b].xyz[0], atoms[b].xyz[1], atoms[b].xyz[2]);
                Eigen::Vector3d dR = Ra - (Rb + g);
                double R = dR.norm();
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > result.cutoff_bohr) continue;
                double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                double factor = is_zero ? 1.0 : 0.5;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        stress(i, j) += factor * dVdR * dR(i) * dR(j) / R;
            }
        }
    }

    // ---- Electronic strain derivative ----
    Eigen::MatrixXd grad_total = compute_periodic_dftb0_gradient(
        system, result, params);

    // Repulsive-only gradient
    Eigen::MatrixXd grad_rep =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_atoms), 3);
    for (const auto& cell : cells) {
        const Eigen::Vector3d g = cell.r_cart;
        for (Eigen::Index a = 0; a < n_atoms; ++a) {
            Eigen::Vector3d Ra(atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]);
            for (Eigen::Index b = 0; b < n_atoms; ++b) {
                bool is_zero = (cell.index.array() == 0).all();
                if (is_zero && a == b) continue;
                Eigen::Vector3d Rb(atoms[b].xyz[0], atoms[b].xyz[1], atoms[b].xyz[2]);
                Eigen::Vector3d dR = Ra - (Rb + g);
                double R = dR.norm();
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > result.cutoff_bohr) continue;
                double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                double factor = is_zero ? 1.0 : 1.0;
                Eigen::Vector3d contrib = factor * dVdR * (dR / R);
                grad_rep(a, 0) += contrib(0);
                grad_rep(a, 1) += contrib(1);
                grad_rep(a, 2) += contrib(2);
            }
        }
    }

    Eigen::MatrixXd grad_elec = grad_total - grad_rep;
    stress += atomic_strain_derivative(system, grad_elec);

    // 2. Cell-vector correction: sum_g M * dS(g)/dR_ket * g
    //    = strain derivative at fixed atomic positions
    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = result.n_basis;
    const auto& D = result.density;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_occ = result.n_occ;
    const double kappa = params.kappa();

    Eigen::MatrixXd Wmat =
        gamma_energy_weighted_density(C, eps, result.occupations, n_occ, n_basis);

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto& shell_infos = basis.shells();
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar_atom(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        hbar_atom[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int nf = shells[static_cast<std::size_t>(s)].size();
        int bf0 = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < nf; ++i)
            ao_atom[bf0 + i] = shell_infos[s].atom_index;
    }

    Eigen::MatrixXd Mmat = -Wmat;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            double h_avg = 0.5 * (hbar_atom[a_mu] + hbar_atom[a_nu]);
            Mmat(mu, nu) += D(mu, nu) * 0.5 * kappa * h_avg;
        }
    }

    // Per-cell masked M blocks: the strain contraction must also run
    // over exactly the masked-S interaction set (issue #316).
    std::vector<Eigen::MatrixXd> Mmat_blocks(cells.size(), Mmat);
    mask_blocks_beyond_pair_cutoff(
        Mmat_blocks, cells, ao_atom, atoms, result.cutoff_bohr);
    stress += overlap_cell_strain_derivative(shells, cells, Mmat_blocks);

    stress /= V;
    return stress;
}


// ---------------------------------------------------------------------------
// Periodic SCC-DFTB analytic stress tensor (Gamma-point)
// ---------------------------------------------------------------------------

Eigen::MatrixXd compute_periodic_scc_dftb_stress(
    const PeriodicSystem& system,
    const PeriodicSCCDFTBResult& result,
    const SemiempiricalParameters& params) {
    if (!result.converged) {
        throw std::invalid_argument(
            "compute_periodic_scc_dftb_stress requires a converged SCC result");
    }
    if (!(result.cutoff_bohr > 0.0)) {
        throw std::invalid_argument(
            "compute_periodic_scc_dftb_stress: result carries no lattice-cutoff "
            "provenance; re-run the SCF with the current driver");
    }
    ensure_libint_initialized();

    const auto n_atoms = static_cast<Eigen::Index>(system.unit_cell.size());
    const auto& atoms = system.unit_cell;
    Eigen::Matrix3d L = system.lattice;
    double V = std::abs(L.determinant());
    if (V < 1e-30) return Eigen::MatrixXd::Zero(3, 3);

    LatticeSumOptions lopt;
    lopt.cutoff_bohr = result.cutoff_bohr;
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    auto cells = atom_pair_interaction_cells(system, lopt.cutoff_bohr);

    Eigen::Matrix3d stress = Eigen::Matrix3d::Zero();

    // Repulsive pair strain derivative, counted once per physical pair.
    for (const auto& cell : cells) {
        const Eigen::Vector3d g = cell.r_cart;
        for (Eigen::Index a = 0; a < n_atoms; ++a) {
            Eigen::Vector3d Ra(atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]);
            bool is_zero = (cell.index.array() == 0).all();
            int start_b = is_zero ? static_cast<int>(a) + 1 : 0;
            for (int b = start_b; b < static_cast<int>(n_atoms); ++b) {
                Eigen::Vector3d Rb(atoms[b].xyz[0], atoms[b].xyz[1], atoms[b].xyz[2]);
                Eigen::Vector3d dR = Ra - (Rb + g);
                double R = dR.norm();
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > result.cutoff_bohr) continue;
                double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                double factor = is_zero ? 1.0 : 0.5;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        stress(i, j) += factor * dVdR * dR(i) * dR(j) / R;
            }
        }
    }

    // Split the total atomic gradient into repulsive, periodic-gamma, and
    // orbital-overlap pieces. Gamma is added below as one full pair-strain
    // tensor; leaving it in this atomic term as well would count it twice.
    Eigen::MatrixXd grad_scc = compute_periodic_scc_dftb_gradient(
        system, result, params);
    Eigen::MatrixXd grad_rep = Eigen::MatrixXd::Zero(n_atoms, 3);
    for (const auto& cell : cells) {
        const Eigen::Vector3d g = cell.r_cart;
        for (Eigen::Index a = 0; a < n_atoms; ++a) {
            Eigen::Vector3d Ra(atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]);
            for (Eigen::Index b = 0; b < n_atoms; ++b) {
                bool is_zero = (cell.index.array() == 0).all();
                if (is_zero && a == b) continue;
                Eigen::Vector3d Rb(atoms[b].xyz[0], atoms[b].xyz[1], atoms[b].xyz[2]);
                Eigen::Vector3d dR = Ra - (Rb + g);
                double R = dR.norm();
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > result.cutoff_bohr) continue;
                double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                double factor = is_zero ? 1.0 : 1.0;
                Eigen::Vector3d contrib = factor * dVdR * (dR / R);
                grad_rep(a, 0) += contrib(0);
                grad_rep(a, 1) += contrib(1);
                grad_rep(a, 2) += contrib(2);
            }
        }
    }
    const auto gamma_deriv = periodic_dftb_gamma_derivatives(
        system, params, result.charges, cells, result.cutoff_bohr,
        result.gamma_form);
    const Eigen::MatrixXd grad_orbital =
        grad_scc - grad_rep - gamma_deriv.atomic_gradient;
    stress += atomic_strain_derivative(system, grad_orbital);

    // Explicit image derivative of the overlap terms. Elstner Eq. 23 gives
    // M = D*(dH0/dS + dH1/dS) - W; vibe-qc's Delta-q convention makes
    // dH1/dS = -(V_A + V_B)/2.
    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = result.n_basis;
    const auto& D = result.density;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_occ = result.n_occ;
    const double kappa = params.kappa();

    Eigen::MatrixXd W =
        gamma_energy_weighted_density(C, eps, result.occupations, n_occ, n_basis);

    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto& shell_infos = basis.shells();
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar_atom(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        hbar_atom[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(basis.nshells()); ++s) {
        int nf = shells[static_cast<std::size_t>(s)].size();
        int bf0 = shell2bf[static_cast<std::size_t>(s)];
        for (int i = 0; i < nf; ++i)
            ao_atom[bf0 + i] = shell_infos[s].atom_index;
    }

    const Eigen::MatrixXd gamma = build_periodic_dftb_gamma_matrix(
        system, params, lopt.cutoff_bohr);
    const Eigen::VectorXd V_scc = gamma * result.charges;
    Eigen::MatrixXd M = -W;
    for (int mu = 0; mu < n_basis; ++mu) {
        const int a_mu = ao_atom[mu];
        for (int nu = 0; nu < n_basis; ++nu) {
            const int a_nu = ao_atom[nu];
            const double h_avg =
                0.5 * (hbar_atom[a_mu] + hbar_atom[a_nu]);
            M(mu, nu) += D(mu, nu) * 0.5
                         * (kappa * h_avg
                            - V_scc(a_mu) - V_scc(a_nu));
        }
    }
    // Per-cell masked M blocks (issue #316; see the DFTB0 stress).
    std::vector<Eigen::MatrixXd> M_cell_blocks(cells.size(), M);
    mask_blocks_beyond_pair_cutoff(
        M_cell_blocks, cells, ao_atom, atoms, result.cutoff_bohr);
    stress += overlap_cell_strain_derivative(shells, cells, M_cell_blocks);

    // Full strain derivative of the same periodic gamma energy used by the
    // SCF, including same-atom image pairs. This term is symmetric by
    // construction; no post-hoc triangle copy or symmetrization is needed.
    stress += gamma_deriv.strain_derivative;

    stress /= V;
    return stress;
}



// ---------------------------------------------------------------------------
// GFN2-xTB analytic gradient
//
// Full derivative of the GFN2-xTB energy functional assembled by
// run_gfn2_xtb (Bannwarth, Ehlert & Grimme, JCTC 2019,
// doi:10.1021/acs.jctc.8b01176):
//   E = Σ D·H⁰ + ½ Δq_l·γ_ll'·Δq_l' + ⅓ Σ_A Γ_A q_A³ + ½ Δq_l·V_mp
//
// Assembly (band-energy form): writing E = E_band(H_SCC) − v·(Δq+n0) + E_rest
// with the SCC Fock shift ½S(v_l + v_l'), all dv/dR feedback terms cancel and
//   dE/dR = Σ D (F + ½(v_l+v_l')) dS/dR − Σ W dS/dR       (M·dS contraction)
//         + Σ_A (∂E/∂CN'_A) dCN'_A/dR + polynomial force   (H⁰ shape terms)
//         + ½ Σ Δq Δq ∂γ_shell/∂R                          (isotropic ES)
//         + explicit AES kernel + moment-integral forces
//         + c·dX/dR                                         (AES response)
// where F = ∂H⁰/∂S and v is the converged shell potential (2nd order +
// 3rd order + AES).  The last term exists because the shipped ad-hoc AES is
// not variational w.r.t. the SCC Fock (the Fock carries the full V_mp in the
// charge channel and no dipole/quadrupole channels, while
// E_aes = ½Δq·V_mp): c = (−½V_mp, ½Kᵀ Δq) contracts the total response of
// the reduced state X = (Δq_shell, shell moments), obtained from a Z-vector
// solve in the reduced space (finite-difference Jacobians, same pattern as
// compute_scc_dftb_gradient_response).  The third-order term has no explicit
// geometry dependence — it enters only through v.
// ---------------------------------------------------------------------------

namespace {

// Stack shell moments into the 9·ns vector [ox; oy; oz; txx; txy; txz; tyy;
// tyz; tzz] used by the AES kernel matrix.
Eigen::VectorXd gfn2_stack_moments(const xtb::GFN2ShellMoments& m) {
    const int ns = m.n_shells();
    Eigen::VectorXd m9(9 * ns);
    m9.segment(0 * ns, ns) = m.ox;  m9.segment(1 * ns, ns) = m.oy;
    m9.segment(2 * ns, ns) = m.oz;  m9.segment(3 * ns, ns) = m.txx;
    m9.segment(4 * ns, ns) = m.txy; m9.segment(5 * ns, ns) = m.txz;
    m9.segment(6 * ns, ns) = m.tyy; m9.segment(7 * ns, ns) = m.tyz;
    m9.segment(8 * ns, ns) = m.tzz;
    return m9;
}

// Ad-hoc shell-resolved AES kernel as a dense matrix K (ns × 9ns):
// V_mp = K·m9 reproduces the V_mp loop in run_gfn2_xtb /
// run_gfn2_xtb_gamma (dipole: −γ³ (m·ΔR)/R³; quadrupole:
// ½γ⁵ Σ Θ·(3ΔRᵢΔRⱼ−R²δ)/R⁵ with doubled off-diagonals).  `shifts` carries
// the image translations (a single zero vector for the molecular case);
// same-atom pairs are skipped in the zero cell only, matching the drivers.
Eigen::MatrixXd gfn2_aes_kernel_matrix(
    const std::vector<GFN2ShellInfo>& shell_info,
    const Eigen::MatrixXd& gamma_shell,
    const Molecule& mol,
    const std::vector<Eigen::Vector3d>& shifts,
    double pair_cutoff = std::numeric_limits<double>::infinity()) {

    const int ns = static_cast<int>(shell_info.size());
    const auto& atoms = mol.atoms();
    Eigen::MatrixXd K = Eigen::MatrixXd::Zero(ns, 9 * ns);

    for (const auto& shift : shifts) {
        const bool zero_cell = shift.norm() < 1e-14;
        for (int si = 0; si < ns; ++si) {
            int a = shell_info[si].atom_idx;
            for (int sj = 0; sj < ns; ++sj) {
                int b = shell_info[sj].atom_idx;
                if (zero_cell && a == b) continue;
                double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + shift[0]);
                double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + shift[1]);
                double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + shift[2]);
                double R2 = dx*dx + dy*dy + dz*dz, R = std::sqrt(R2);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > pair_cutoff) continue;
                const double g = zero_cell
                    ? gamma_shell(si, sj)
                    : gfn2_shell_gamma_at_distance(
                        shell_info[si], shell_info[sj], R);
                if (g < 1e-15) continue;
                double g3 = g * g * g, g5 = g3 * g * g;
                double invR3 = 1.0 / (R2 * R);
                double invR5 = invR3 / R2;
                K(si, 0 * ns + sj) += -g3 * dx * invR3;
                K(si, 1 * ns + sj) += -g3 * dy * invR3;
                K(si, 2 * ns + sj) += -g3 * dz * invR3;
                double qxx = 3*dx*dx - R2, qxy = 3*dx*dy, qxz = 3*dx*dz;
                double qyy = 3*dy*dy - R2, qyz = 3*dy*dz, qzz = 3*dz*dz - R2;
                K(si, 3 * ns + sj) += 0.5 * g5 * qxx * invR5;
                K(si, 4 * ns + sj) += 0.5 * g5 * 2.0 * qxy * invR5;
                K(si, 5 * ns + sj) += 0.5 * g5 * 2.0 * qxz * invR5;
                K(si, 6 * ns + sj) += 0.5 * g5 * qyy * invR5;
                K(si, 7 * ns + sj) += 0.5 * g5 * 2.0 * qyz * invR5;
                K(si, 8 * ns + sj) += 0.5 * g5 * qzz * invR5;
            }
        }
    }
    return K;
}

// Explicit AES kernel force: d(½ Σ Δq_si V_mp(si))/dR at fixed charges and
// moments.  Uses ∂γ/∂R = −Rγ³ chains through γ³/γ⁵ and the derivatives of
// the ΔR tensors.
Eigen::MatrixXd gfn2_aes_kernel_force(
    const std::vector<GFN2ShellInfo>& shell_info,
    const Eigen::MatrixXd& gamma_shell,
    const Molecule& mol,
    const Eigen::VectorXd& dqs,
    const xtb::GFN2ShellMoments& m,
    const std::vector<Eigen::Vector3d>& shifts,
    double pair_cutoff = std::numeric_limits<double>::infinity(),
    Eigen::Matrix3d* strain_derivative = nullptr) {

    const int ns = static_cast<int>(shell_info.size());
    const auto& atoms = mol.atoms();
    const auto n_atoms = static_cast<Eigen::Index>(atoms.size());
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);

    for (const auto& shift : shifts) {
        const bool zero_cell = shift.norm() < 1e-14;
        for (int si = 0; si < ns; ++si) {
            int a = shell_info[si].atom_idx;
            const double coeff = 0.5 * dqs(si);
            for (int sj = 0; sj < ns; ++sj) {
                int b = shell_info[sj].atom_idx;
                if (zero_cell && a == b) continue;
                double dR[3] = {
                    atoms[a].xyz[0] - (atoms[b].xyz[0] + shift[0]),
                    atoms[a].xyz[1] - (atoms[b].xyz[1] + shift[1]),
                    atoms[a].xyz[2] - (atoms[b].xyz[2] + shift[2])};
                double R2 = dR[0]*dR[0] + dR[1]*dR[1] + dR[2]*dR[2];
                double R = std::sqrt(R2);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > pair_cutoff) continue;
                const double g = zero_cell
                    ? gamma_shell(si, sj)
                    : gfn2_shell_gamma_at_distance(
                        shell_info[si], shell_info[sj], R);
                if (g < 1e-15) continue;
                double g3 = g * g * g, g5 = g3 * g * g, g7 = g5 * g * g;
                double invR3 = 1.0 / (R2 * R);
                double invR5 = invR3 / R2;
                double invR7 = invR5 / R2;

                const double mv[3] = {m.ox(sj), m.oy(sj), m.oz(sj)};
                const double mdotr = mv[0]*dR[0] + mv[1]*dR[1] + mv[2]*dR[2];
                // Θ_sj as a full symmetric 3×3 (traceless up to roundoff).
                const double Th[3][3] = {
                    {m.txx(sj), m.txy(sj), m.txz(sj)},
                    {m.txy(sj), m.tyy(sj), m.tyz(sj)},
                    {m.txz(sj), m.tyz(sj), m.tzz(sj)}};
                const double trTh = Th[0][0] + Th[1][1] + Th[2][2];
                // Q = Σ_ij Θ_ij (3ΔRᵢΔRⱼ − R²δᵢⱼ)
                double Q = 0.0;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        Q += Th[i][j] * (3.0 * dR[i] * dR[j] - (i == j ? R2 : 0.0));

                for (int kk = 0; kk < 3; ++kk) {
                    // d(−γ³ (m·ΔR)/R³)/dΔR_k:  d(γ³) = −3γ⁵ΔR_k,
                    // d(R⁻³) = −3ΔR_k R⁻⁵.
                    double d_pd = 3.0 * g5 * dR[kk] * mdotr * invR3
                                - g3 * mv[kk] * invR3
                                + 3.0 * g3 * mdotr * dR[kk] * invR5;
                    // dQ/dΔR_k = 6(ΘΔR)_k − 2ΔR_k trΘ.
                    double ThR_k = Th[kk][0]*dR[0] + Th[kk][1]*dR[1] + Th[kk][2]*dR[2];
                    double dQ = 6.0 * ThR_k - 2.0 * dR[kk] * trTh;
                    // d(½γ⁵ Q/R⁵)/dΔR_k:  d(γ⁵) = −5γ⁷ΔR_k, d(R⁻⁵) = −5ΔR_k R⁻⁷.
                    double d_pq = 0.5 * (-5.0 * g7 * dR[kk] * Q * invR5
                                         + g5 * dQ * invR5
                                         - 5.0 * g5 * Q * dR[kk] * invR7);
                    double f = coeff * (d_pd + d_pq);
                    grad(a, kk) += f;
                    grad(b, kk) -= f;
                    if (strain_derivative != nullptr) {
                        for (int beta = 0; beta < 3; ++beta)
                            (*strain_derivative)(kk, beta) += f * dR[beta];
                    }
                }
            }
        }
    }
    return grad;
}

// Reduced-state map X(v, R): diagonalize H⁰ + ½S(v_l+v_l'), then Mulliken
// shell charges and shell multipole moments.  X = [Δq; m9] (10·ns).
Eigen::VectorXd gfn2_reduced_state(
    const Eigen::MatrixXd& H0,
    const Eigen::MatrixXd& S,
    const Eigen::VectorXd& v,
    const BasisSet& basis,
    const Molecule& mol,
    const xtb::GFN2MultipoleSet& mp_int,
    const std::vector<int>& ao_shell,
    const Eigen::VectorXd& n0_shell,
    int n_occ,
    double smearing_temperature = 0.0) {

    const int n_basis = static_cast<int>(H0.rows());
    const int ns = static_cast<int>(n0_shell.size());

    Eigen::MatrixXd H_scc = H0;
    for (int mu = 0; mu < n_basis; ++mu) {
        int sm = ao_shell[mu];
        for (int nu = 0; nu < n_basis; ++nu) {
            int sn = ao_shell[nu];
            H_scc(mu, nu) += 0.5 * S(mu, nu) * (v(sm) + v(sn));
        }
    }
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_scc, S);
    if (solver.info() != Eigen::Success)
        throw std::runtime_error("gfn2 gradient response: diag failed");
    Eigen::MatrixXd D;
    if (smearing_temperature > 0.0) {
        KPointOccupationOptions occupation_options;
        occupation_options.smearing_temperature = smearing_temperature;
        const auto occupation = compute_closed_shell_kpoint_occupations(
            {solver.eigenvalues()}, {1.0}, 2.0 * n_occ, n_occ,
            occupation_options);
        D = solver.eigenvectors()
            * occupation.occupations_per_k.front().asDiagonal()
            * solver.eigenvectors().transpose();
    } else {
        Eigen::MatrixXd C_occ = solver.eigenvectors().leftCols(n_occ);
        D = 2.0 * C_occ * C_occ.transpose();
    }
    Eigen::MatrixXd DS = D * S;

    Eigen::VectorXd X(10 * ns);
    Eigen::VectorXd dqs = Eigen::VectorXd::Zero(ns);
    for (int mu = 0; mu < n_basis; ++mu) {
        int si = ao_shell[mu];
        if (si >= 0) dqs(si) += DS(mu, mu);
    }
    dqs -= n0_shell;
    auto m = xtb::compute_shell_multipole_moments(
        D, basis, mol, mp_int, ao_shell, n0_shell, ns);
    X.head(ns) = dqs;
    X.tail(9 * ns) = gfn2_stack_moments(m);
    return X;
}

// Fock shell potential v = γΔq + V³ + K·m9 for a frozen reduced state at a
// given geometry (γ and K carry the geometry dependence of the map).
Eigen::VectorXd gfn2_fock_shell_potential(
    const std::vector<GFN2ShellInfo>& shell_info,
    const Eigen::MatrixXd& gamma_shell,
    const Eigen::MatrixXd& K,
    const Eigen::VectorXd& gam3_shell,
    const Eigen::VectorXd& dqs,
    const Eigen::VectorXd& m9,
    int /*n_atoms*/) {

    const int ns = static_cast<int>(shell_info.size());
    Eigen::VectorXd v = gamma_shell * dqs;
    // Shell-resolved third-order on-site shift (xtb thirdorder.f90 shellGam
    // branch): v³_l = -Γ_l·q_l² = -Γ_l·dq_l² with q_l = -dq_l.
    for (int si = 0; si < ns; ++si) {
        double G = gam3_shell(si);
        if (G != 0.0)
            v(si) += -G * dqs(si) * dqs(si);
    }
    v += K * m9;
    return v;
}

// AES response correction (Z-vector in the reduced SCC space).  The ad-hoc
// AES makes the SCF non-variational: c = (−½V_mp, ½KᵀΔq) ≠ 0 and the
// correction is cᵀ·dX/dR with dX/dR = (I − J·P)⁻¹·∂X/∂R|_X, where
// J = ∂X/∂v (FD over shell-potential perturbations) and P = ∂v/∂X
// (analytic: γ + third-order block, and K for the moment block).  ∂X/∂R|_X
// is evaluated by FD with the reduced state frozen: v± = γ±Δq + V³ + K±m9
// at the displaced geometry.  Without AES (c = 0) the correction vanishes
// and the assembly is exact — this mirrors compute_scc_dftb_gradient_response.
Eigen::MatrixXd gfn2_aes_response_correction(
    const Molecule& mol,
    const xtb::GFN2ParameterSet& params,
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Eigen::MatrixXd& H0,
    const xtb::GFN2MultipoleSet& mp_int,
    const std::vector<GFN2ShellInfo>& shell_info,
    const Eigen::MatrixXd& gamma_shell,
    const Eigen::MatrixXd& K,
    const std::vector<int>& ao_shell,
    const Eigen::VectorXd& n0_shell,
    const Eigen::VectorXd& gam3_shell,
    const Eigen::VectorXd& v,
    const Eigen::VectorXd& dqs,
    const Eigen::VectorXd& m9,
    const Eigen::VectorXd& V_mp,
    int n_occ) {

    const auto& atoms = mol.atoms();
    const int n_atoms = static_cast<int>(atoms.size());
    const int ns = static_cast<int>(shell_info.size());
    const int nx = 10 * ns;

    Eigen::VectorXd c(nx);
    c.head(ns) = -0.5 * V_mp;
    c.tail(9 * ns) = 0.5 * K.transpose() * dqs;
    if (c.cwiseAbs().maxCoeff() < 1e-14)
        return Eigen::MatrixXd::Zero(n_atoms, 3);

    // J = ∂X/∂v by central FD over shell-potential perturbations.
    const double h_v = 1e-4;
    Eigen::MatrixXd J(nx, ns);
    for (int si = 0; si < ns; ++si) {
        Eigen::VectorXd vp = v, vm = v;
        vp(si) += h_v;
        vm(si) -= h_v;
        Eigen::VectorXd Xp = gfn2_reduced_state(
            H0, S, vp, basis, mol, mp_int, ao_shell, n0_shell, n_occ);
        Eigen::VectorXd Xm = gfn2_reduced_state(
            H0, S, vm, basis, mol, mp_int, ao_shell, n0_shell, n_occ);
        J.col(si) = (Xp - Xm) / (2.0 * h_v);
    }

    // P = ∂v/∂X analytic.  The third-order shift is now shell-resolved
    // (xtb thirdorder.f90 shellGam branch): v³(si) = −Γ_si·dq_si² with
    // dq_si = −q_si, so ∂v³(si)/∂Δq_sj = 2·Γ_si·q_si·δ_ij — a diagonal
    // block, not the same-atom fill of the old atomic-charge form.
    Eigen::MatrixXd P(ns, nx);
    P.leftCols(ns) = gamma_shell;
    for (int si = 0; si < ns; ++si) {
        double G = gam3_shell(si);
        if (G == 0.0) continue;
        P(si, si) += 2.0 * G * (-dqs(si));
    }
    P.rightCols(9 * ns) = K;

    Eigen::MatrixXd A_resp = Eigen::MatrixXd::Identity(nx, nx) - J * P;
    Eigen::VectorXd z = A_resp.transpose().colPivHouseholderQr().solve(c);

    // ∂X/∂R at frozen reduced state, by central FD with full rebuild of the
    // geometry-dependent pieces (S, H⁰, multipole integrals, γ, K).
    const double h_R = 1e-4;
    const std::vector<Eigen::Vector3d> mol_shift = {Eigen::Vector3d::Zero()};
    Eigen::MatrixXd grad = Eigen::MatrixXd::Zero(n_atoms, 3);
    for (int a = 0; a < n_atoms; ++a) {
        for (int d = 0; d < 3; ++d) {
            Eigen::VectorXd X_pm[2];
            for (int pm = 0; pm < 2; ++pm) {
                std::vector<Atom> atoms_d(atoms.begin(), atoms.end());
                atoms_d[a].xyz[d] += (pm == 0 ? h_R : -h_R);
                Molecule mol_d(atoms_d, mol.charge(), mol.multiplicity());
                // n_primitives=0: GFN2-xTB per-element auto (H,He->3; else->4).
                // Must match the basis the converged energy used (and the
                // basis compute_gfn2_gradient builds); the previous hardcoded
                // 6 rebuilt the displaced geometry with the STO-6G tables and
                // differentiated a different surface (same bug class as
                // f2aa33aa8 fixed on the other gradient paths).
                BasisSet basis_d = SemiempiricalBasis::build(mol_d, params, 0);
                Eigen::MatrixXd S_d =
                    SemiempiricalHamiltonianBuilder::build_overlap(basis_d);
                Eigen::MatrixXd H0_d = build_gfn2_hamiltonian_zero(
                    basis_d, S_d, mol_d, params);
                xtb::GFN2MultipoleSet mp_d =
                    xtb::build_gfn2_multipole_integrals(basis_d, mol_d, true);
                Eigen::MatrixXd gam_d =
                    build_gfn2_shell_gamma(shell_info, mol_d, params);
                Eigen::MatrixXd K_d = gfn2_aes_kernel_matrix(
                    shell_info, gam_d, mol_d, mol_shift);
                Eigen::VectorXd v_d = gfn2_fock_shell_potential(
                    shell_info, gam_d, K_d, gam3_shell, dqs, m9, n_atoms);
                X_pm[pm] = gfn2_reduced_state(
                    H0_d, S_d, v_d, basis_d, mol_d, mp_d,
                    ao_shell, n0_shell, n_occ);
            }
            grad(a, d) = z.dot((X_pm[0] - X_pm[1]) / (2.0 * h_R));
        }
    }
    return grad;
}

}  // namespace

Eigen::MatrixXd compute_gfn2_gradient(
    const Molecule& mol,
    const xtb::GFN2Result& result,
    const xtb::GFN2ParameterSet& params) {

    ensure_libint_initialized();

    const auto n_atoms = static_cast<Eigen::Index>(mol.atoms().size());
    // Guard: non-converged / default-initialized result.
    if (result.n_basis == 0 || result.n_occ == 0 || result.density.size() == 0)
        return Eigen::MatrixXd::Zero(n_atoms, 3);

    // n_primitives=0: GFN2-xTB per-element auto (H,He->3; else->4) - must
    // match the basis run_gfn2_xtb built for `result`, or this gradient is
    // differentiating a different energy surface than the one converged.
    BasisSet basis = SemiempiricalBasis::build(mol, params, 0);
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_infos = basis.shells();
    std::vector<long> s2a(shell_infos.size());
    for (std::size_t si = 0; si < shell_infos.size(); ++si)
        s2a[si] = static_cast<long>(shell_infos[si].atom_index);
    const auto n_basis = result.n_basis;

    const auto& atoms = mol.atoms();
    const auto& D = result.density;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_occ = result.n_occ;
    Eigen::MatrixXd S = result.overlap.size() > 0
        ? result.overlap
        : SemiempiricalHamiltonianBuilder::build_overlap(basis);

    // Energy-weighted density
    Eigen::MatrixXd C_occ = C.leftCols(n_occ);
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n_basis, n_basis);
    for (int i = 0; i < n_occ; ++i)
        W += eps(i) * (2.0 * C_occ.col(i) * C_occ.col(i).transpose());

    // ---- Shell-resolved SCC machinery (matches run_gfn2_xtb) ----
    auto shell_info = gfn2_enumerate_shells(basis, mol, params);
    const int ns = static_cast<int>(shell_info.size());
    Eigen::MatrixXd gamma_shell = build_gfn2_shell_gamma(shell_info, mol, params);

    std::vector<int> ao_shell(n_basis, -1);
    for (int si = 0; si < ns; ++si)
        for (int i = 0; i < shell_info[si].n_funcs; ++i)
            ao_shell[shell_info[si].bf_start + i] = si;

    // GFN2 globpar l-dependent GAM3 scales: s=1.0, p=0.5, d=0.25.
    static const double gam3_l_scale_grad[4] = {1.0, 0.5, 0.25, 0.25};
    Eigen::VectorXd n0_shell = Eigen::VectorXd::Zero(ns);
    Eigen::VectorXd gam3_shell = Eigen::VectorXd::Zero(ns);
    for (int si = 0; si < ns; ++si) {
        const int Z = atoms[shell_info[si].atom_idx].Z;
        n0_shell(si) = xtb::gfn2_reference_occupation(Z, shell_info[si].l);
        const auto* e = params.element_data(Z);
        double gam3_raw = e ? e->gam3 : 0.0;
        int l = shell_info[si].l;
        gam3_shell(si) = gam3_raw * gam3_l_scale_grad[std::min(l, 3)];
    }

    // Converged shell charges Δq_l = pop_l − n0_l from the converged density.
    Eigen::MatrixXd DS = D * S;
    Eigen::VectorXd dqs = Eigen::VectorXd::Zero(ns);
    for (int mu = 0; mu < n_basis; ++mu) {
        int si = ao_shell[mu];
        if (si >= 0) dqs(si) += DS(mu, mu);
    }
    dqs -= n0_shell;

    // AES moments + kernel at the converged density.
    xtb::GFN2MultipoleSet mp_int =
        xtb::build_gfn2_multipole_integrals(basis, mol, true);
    auto moments = xtb::compute_shell_multipole_moments(
        D, basis, mol, mp_int, ao_shell, n0_shell, ns);
    moments.q = dqs;
    Eigen::VectorXd m9 = gfn2_stack_moments(moments);
    const std::vector<Eigen::Vector3d> mol_shift = {Eigen::Vector3d::Zero()};
    Eigen::MatrixXd K = gfn2_aes_kernel_matrix(
        shell_info, gamma_shell, mol, mol_shift);
    Eigen::VectorXd V_mp = K * m9;

    // Converged Fock shell potential v = γΔq + V³ + V_mp.
    Eigen::VectorXd v = gfn2_fock_shell_potential(
        shell_info, gamma_shell, K, gam3_shell, dqs, m9,
        static_cast<int>(n_atoms));

    // ---- H⁰ shape-term derivatives ----
    Eigen::VectorXd dE_dCN = Eigen::VectorXd::Zero(n_atoms);
    Eigen::MatrixXd grad_poly = Eigen::MatrixXd::Zero(n_atoms, 3);
    Eigen::Matrix3d strain_poly_unused = Eigen::Matrix3d::Zero();
    const Eigen::VectorXd molecular_cn = gfn2_h0_coordination_numbers(mol);
    Eigen::MatrixXd F = gfn2_h0_image_derivative_terms(
        basis, S, D, mol, params, Eigen::Vector3d::Zero(), molecular_cn,
        dE_dCN, grad_poly, strain_poly_unused);

    // ---- M matrix: M = -W + D∘(F - ½(v_l + v_l')) ----
    // Hamiltonian is H = H⁰ + ½S(v+v') (run_gfn2_xtb builds
    // H_scc = H0 + ½S(v+v')), and the energy-weighted density
    // gradient follows the same plus-sign convention.  BUG-022
    // flipped this to minus on the premise H = H⁰ - ½S(v+v'); the
    // central-FD gate proved the plus sign empirically correct
    // (FD residual 3e-2 Ha/bohr with minus, ~1e-9 with plus).
    // Same-atom pairs contribute nothing (dS = 0) but are kept
    // for clarity.
    Eigen::MatrixXd M = -W;
    for (int mu = 0; mu < n_basis; ++mu) {
        int sm = ao_shell[mu];
        for (int nu = 0; nu < n_basis; ++nu) {
            int sn = ao_shell[nu];
            M(mu, nu) += D(mu, nu) * (F(mu, nu) + 0.5 * (v(sm) + v(sn)));
        }
    }

    // Overlap derivative integral (same as DFTB)
    Eigen::MatrixXd grad =
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_atoms), 3);

    libint2::Engine prototype(libint2::Operator::overlap,
                              shells.max_nprim(), shells.max_l(), 1);
    auto engines = make_engine_pool(prototype);
    const int n_threads = omp_max_threads();
    std::vector<Eigen::MatrixXd> grad_tls(
        n_threads,
        Eigen::MatrixXd::Zero(static_cast<Eigen::Index>(n_atoms), 3));

    const int n_shells = static_cast<int>(shells.size());
    #pragma omp parallel for schedule(dynamic)
    for (int s1 = 0; s1 < n_shells; ++s1) {
        const auto tid = static_cast<std::size_t>(omp_thread_index());
        auto& engine = engines[tid];
        const auto& buf = engine.results();
        auto& grad_local = grad_tls[tid];
        const auto bf1 = shell2bf[s1];
        const auto n1 = shells[s1].size();
        const long atom1 = s2a[s1];

        for (int s2 = 0; s2 < n_shells; ++s2) {
            const auto bf2 = shell2bf[s2];
            const auto n2 = shells[s2].size();
            const long atom2 = s2a[s2];
            engine.compute(shells[s1], shells[s2]);
            for (int icenter = 0; icenter < 2; ++icenter) {
                const long atom_i = (icenter == 0) ? atom1 : atom2;
                for (int d = 0; d < 3; ++d) {
                    const double* block = buf[icenter * 3 + d];
                    if (!block) continue;
                    double acc = 0.0;
                    for (std::size_t i = 0; i < n1; ++i)
                        for (std::size_t j = 0; j < n2; ++j)
                            acc += M(bf1 + i, bf2 + j) * block[i * n2 + j];
                    grad_local(atom_i, d) += acc;
                }
            }
        }
    }
    for (const auto& g : grad_tls) grad += g;

    // Repulsive gradient: E_rep = Σ_{A<B} Z_A^eff Z_B^eff / R · exp(−α·R^kf)
    // is an additive term of the GFN2 total energy (Bannwarth, Ehlert &
    // Grimme, JCTC 2019, Eq. 9), so its pair force belongs in the analytic
    // gradient of the same functional.
    for (Eigen::Index a = 0; a < n_atoms; ++a) {
        for (Eigen::Index b = a + 1; b < n_atoms; ++b) {
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R < 1e-12) continue;
            double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
            if (dVdR == 0.0) continue;
            double invR = 1.0 / R;
            grad(a, 0) += dVdR * dx * invR;
            grad(a, 1) += dVdR * dy * invR;
            grad(a, 2) += dVdR * dz * invR;
            grad(b, 0) -= dVdR * dx * invR;
            grad(b, 1) -= dVdR * dy * invR;
            grad(b, 2) -= dVdR * dz * invR;
        }
    }

    // Shell-resolved isotropic ES derivative.  The energy's 2nd-order term is
    // E2 = ½ Σ_ll' Δq_l γ_ll' Δq_l' with the shell-pair Klopman-Ohno kernel
    // γ = 1/√(R² + η_ll'⁻²) (build_gfn2_shell_gamma), so ∂γ/∂R = −R·γ³.
    // The ordered (si,sj) loop deposits into grad(atom(si)) only, so no ½
    // (the ½ in E2 is cancelled by γ appearing in both orders; the halved
    // form was the 2026-07-09 defect).  On-site shell pairs are R-independent.
    for (int si = 0; si < ns; ++si) {
        const int a = shell_info[si].atom_idx;
        for (int sj = 0; sj < ns; ++sj) {
            const int b = shell_info[sj].atom_idx;
            if (a == b) continue;
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R < 1e-12) continue;
            double g_ll = gamma_shell(si, sj);
            double dgamma_dR = -R * g_ll * g_ll * g_ll;
            double prefactor = dqs(si) * dqs(sj) * dgamma_dR / R;
            grad(a, 0) += prefactor * dx;
            grad(a, 1) += prefactor * dy;
            grad(a, 2) += prefactor * dz;
        }
    }

    // H⁰ shape-term chain: CN self-energy + distance polynomial.
    grad += gfn2_cn_chain_gradient(mol, dE_dCN);
    grad += grad_poly;

    // AES kernel geometry force at fixed (Δq, shell moments).
    grad += gfn2_aes_kernel_force(
        shell_info, gamma_shell, mol, dqs, moments, mol_shift);

    // AES response correction (Z-vector; zero when the AES coupling is zero).
    // NOTE: this also carries the moment-integral Pulay variation (∂m9/∂R at
    // fixed density).  gfn2_aes_response_correction evaluates ∂X/∂R by
    // rebuilding the multipole integrals at the displaced geometry (mp_d in
    // gfn2_reduced_state), so the moment block of ∂X/∂R already includes the
    // integral Pulay term.  Adding it a second time as an explicit term
    // double-counts (verified: doing so breaks the FD agreement by the exact
    // Pulay magnitude, ~6e-4 Ha/bohr on water).
    grad += gfn2_aes_response_correction(
        mol, params, basis, S,
        build_gfn2_hamiltonian_zero(basis, S, mol, params),
        mp_int, shell_info, gamma_shell, K, ao_shell, n0_shell, gam3_shell,
        v, dqs, m9, V_mp, n_occ);

    return grad;
}


// ---------------------------------------------------------------------------
// Periodic GFN2-xTB gradient and stress (Gamma-point)
//
// The periodic energy is stationary in the reduced SCC state (shell charges
// and CAMM moments) because the AES Fock of gfn2_aes.hpp is the exact
// derivative of the AES energy with respect to the density.  The gradient is
// therefore the Hellmann-Feynman/Pulay form with no response term:
//   dE/dR = sum_g [D (F_H0 + 1/2 (v + v')) - W] dS(g)/dR          (overlap)
//         + sum_g D [cS dS + cD . dD + cQ . dQ](g)/dR + origin terms (AES Fock)
//         + H0 CN chain + distance polynomial                     (H0 shape)
//         + 1/2 dq^T dGamma/dR dq                                 (isotropic ES)
//         + explicit AES pair-kernel force + R0'(CN) chain        (AES kernel)
//         + repulsion.
// The strain derivative decomposes every term into its atomic virial
// (sum_A grad_A (x) R_A) plus the explicit image-vector part; the latter is
// what each helper's "image virial"/"strain - atomic" returns.
// ---------------------------------------------------------------------------

namespace {

struct PeriodicGFN2DerivativeTerms {
    Eigen::MatrixXd gradient;             // total (n_atoms x 3)
    Eigen::MatrixXd gradient_repulsive;   // repulsive part of `gradient`
    Eigen::Matrix3d image_strain = Eigen::Matrix3d::Zero();  // explicit
};

PeriodicGFN2DerivativeTerms periodic_gfn2_derivative_terms(
    const PeriodicSystem& system,
    const xtb::PeriodicGFN2Result& result,
    const xtb::GFN2ParameterSet& params,
    bool with_strain) {
    const xtb::PeriodicGFN2Assembly A = xtb::assemble_periodic_gfn2(
        system, params, result.cutoff_bohr, result.gamma_form);
    const auto& atoms = system.unit_cell;
    const int n_atoms = A.n_atoms;
    const int n_basis = A.n_basis;
    const int ns = A.n_shells;
    if (result.n_basis != n_basis) {
        throw std::invalid_argument(
            "periodic GFN2 derivatives: result basis does not match the system");
    }
    const auto& D = result.density;
    const auto& C = result.mo_coeffs;
    const auto& eps = result.mo_energies;
    const int n_occ = result.n_occ;
    const ImageRecordSource records = A.records();

    // Energy-weighted density from the converged occupations (fractional at
    // finite temperature: the Mermin free energy is stationary there).
    Eigen::VectorXd occ = result.occupations;
    if (occ.size() != n_basis) {
        occ = Eigen::VectorXd::Zero(n_basis);
        occ.head(n_occ).setConstant(2.0);
    }
    Eigen::MatrixXd W = Eigen::MatrixXd::Zero(n_basis, n_basis);
    for (int i = 0; i < n_basis; ++i)
        W += occ(i) * eps(i) * (C.col(i) * C.col(i).transpose());

    // Converged shell charges and CAMM moments of the returned density.
    Eigen::VectorXd dqs = result.dq_shell;
    if (dqs.size() != ns) {
        const Eigen::MatrixXd DS = D * A.S_gamma;
        dqs = Eigen::VectorXd::Zero(ns);
        for (int mu = 0; mu < n_basis; ++mu) dqs(A.ao_shell[mu]) += DS(mu, mu);
        dqs -= A.n0_shell;
    }
    Eigen::VectorXd q_atom = Eigen::VectorXd::Zero(n_atoms);
    for (int si = 0; si < ns; ++si) q_atom(A.shell_info[si].atom_idx) -= dqs(si);
    const xtb::GFN2AesMoments moments = xtb::gfn2_aes_moments(
        D, A.multipole_sums, A.ao_atom, A.positions, q_atom);
    const xtb::GFN2AesPotentials potentials = xtb::gfn2_aes_potentials(
        A.positions, moments, A.aes_params, records);

    // Isotropic shell potential v = Gamma dq + V3 (the S-channel coefficient
    // of the Fock apart from the AES side, which the multipole contraction
    // below carries).
    Eigen::VectorXd v = A.gamma_shell * dqs;
    for (int si = 0; si < ns; ++si) {
        const double G = A.gam3_shell(si);
        if (G != 0.0) v(si) += -G * dqs(si) * dqs(si);
    }

    PeriodicGFN2DerivativeTerms out;
    out.gradient = Eigen::MatrixXd::Zero(n_atoms, 3);
    out.gradient_repulsive = Eigen::MatrixXd::Zero(n_atoms, 3);

    // ---- (1) overlap contraction with the H0 shape factor and v ----
    Eigen::VectorXd dE_dCN = Eigen::VectorXd::Zero(n_atoms);
    Eigen::MatrixXd grad_poly = Eigen::MatrixXd::Zero(n_atoms, 3);
    Eigen::Matrix3d strain_poly = Eigen::Matrix3d::Zero();
    LatticeMatrixSet minus_M_set;
    minus_M_set.nbf = n_basis;
    minus_M_set.cells = A.cells;
    minus_M_set.blocks.resize(A.cells.size());
    std::vector<Eigen::MatrixXd> M_blocks;
    M_blocks.reserve(A.cells.size());
    for (std::size_t ci = 0; ci < A.cells.size(); ++ci) {
        const Eigen::MatrixXd F = gfn2_h0_image_derivative_terms(
            A.basis, A.overlap_blocks.blocks[ci], D, A.mol, params,
            A.cells[ci].r_cart, A.coordination_numbers, dE_dCN, grad_poly,
            strain_poly);
        Eigen::MatrixXd M = -W;
        for (int mu = 0; mu < n_basis; ++mu) {
            const int sm = A.ao_shell[mu];
            for (int nu = 0; nu < n_basis; ++nu) {
                const int sn = A.ao_shell[nu];
                M(mu, nu) += D(mu, nu) * (F(mu, nu) + 0.5 * (v(sm) + v(sn)));
            }
        }
        M_blocks.push_back(M);
        minus_M_set.blocks[ci] = -M;
    }
    mask_blocks_beyond_pair_cutoff(
        minus_M_set.blocks, A.cells, A.ao_atom, atoms, result.cutoff_bohr);
    mask_blocks_beyond_pair_cutoff(
        M_blocks, A.cells, A.ao_atom, atoms, result.cutoff_bohr);
    LatticeSumOptions lopt;
    lopt.cutoff_bohr = result.cutoff_bohr;
    out.gradient += overlap_lattice_gradient_contribution(
        A.basis, system, minus_M_set, lopt);
    if (with_strain) {
        out.image_strain += overlap_cell_strain_derivative(
            A.basis.libint(), A.cells, M_blocks);
    }

    // ---- (2) AES Fock through the multipole integrals and their origins ----
    const auto filler = [&](int cell_index, const Eigen::Vector3d& shift,
                            Eigen::MatrixXd& wS,
                            std::array<Eigen::MatrixXd, 3>& wD,
                            std::array<Eigen::MatrixXd, 6>& wQ) {
        xtb::gfn2_aes_fill_derivative_weights(
            cell_index, shift, D, A.ao_atom, A.positions, potentials,
            wS, wD, wQ);
    };
    const xtb::GFN2MultipoleDerivativeContraction contraction =
        xtb::contract_gfn2_multipole_lattice_derivatives(
            A.basis, atoms, A.cells, A.ao_atom, result.cutoff_bohr, filler);
    const xtb::GFN2AesOriginDerivatives origin = xtb::gfn2_aes_origin_derivatives(
        D, A.multipole_sums, A.ao_atom, A.positions, potentials);
    out.gradient += contraction.gradient + origin.gradient;
    if (with_strain) {
        out.image_strain += contraction.image_virial + origin.image_virial;
    }

    // ---- (3) explicit AES pair kernel and the R0'(CN) chain ----
    const xtb::GFN2AesKernelDerivatives kernel = xtb::gfn2_aes_kernel_derivatives(
        A.positions, moments, A.aes_params, records);
    out.gradient += kernel.gradient;
    if (with_strain) {
        out.image_strain += kernel.strain
            - atomic_strain_derivative(system, kernel.gradient);
    }
    for (int a = 0; a < n_atoms; ++a) {
        dE_dCN(a) += kernel.dE_dmrad(a) * A.aes_params.dmrad_dcn(a);
    }

    // ---- (4) H0 shape terms: CN chain (H0 self-energies + AES radii) ----
    const auto cn = gfn2_periodic_cn_chain_derivatives(system, dE_dCN);
    out.gradient += cn.atomic_gradient + grad_poly;
    if (with_strain) {
        out.image_strain += cn.strain_derivative
            - atomic_strain_derivative(system, cn.atomic_gradient);
        out.image_strain += strain_poly
            - atomic_strain_derivative(system, grad_poly);
    }

    // ---- (5) isotropic second order: lattice-summed shell gamma ----
    const EwaldCoulombKernel ewald = A.ewald();
    const ImageRecordSource gamma_records = A.gamma_records();
    const Eigen::MatrixXd es_gradient = periodic_shell_gamma_gradient(
        A.gamma_sites, A.positions, A.gamma_spec, ewald, gamma_records, dqs);
    out.gradient += es_gradient;
    if (with_strain) {
        out.image_strain += periodic_shell_gamma_strain_derivative(
            A.gamma_sites, A.positions, A.gamma_spec, ewald, gamma_records, dqs)
            - atomic_strain_derivative(system, es_gradient);
    }

    // ---- (6) lattice-summed pair repulsion ----
    for (int a = 0; a < n_atoms; ++a) {
        for (const auto& cell : A.cells) {
            const bool is_zero = (cell.index.array() == 0).all();
            for (int b = 0; b < n_atoms; ++b) {
                if (is_zero && a == b) continue;
                const double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + cell.r_cart[0]);
                const double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + cell.r_cart[1]);
                const double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + cell.r_cart[2]);
                const double R = std::sqrt(dx * dx + dy * dy + dz * dz);
                if (R < 1e-12 || R > result.cutoff_bohr) continue;
                const double dVdR = params.repulsive_derivative(
                    atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                // The energy carries 1/2 for nonzero image cells, but the
                // atomic derivative receives equal first- and second-centre
                // contributions from the symmetric +/- image list.
                out.gradient_repulsive(a, 0) += dVdR * dx / R;
                out.gradient_repulsive(a, 1) += dVdR * dy / R;
                out.gradient_repulsive(a, 2) += dVdR * dz / R;
            }
        }
    }
    out.gradient += out.gradient_repulsive;
    return out;
}

}  // namespace

Eigen::MatrixXd compute_periodic_gfn2_gradient(
    const PeriodicSystem& system,
    const xtb::PeriodicGFN2Result& result,
    const xtb::GFN2ParameterSet& params) {
    ensure_libint_initialized();

    // Fail closed on legacy/default-initialized results that carry no
    // lattice-cutoff provenance (issue #340).
    if (!(result.cutoff_bohr > 0.0)) {
        throw std::invalid_argument(
            "compute_periodic_gfn2_gradient: result carries no lattice-cutoff "
            "provenance; re-run the SCF with the current driver");
    }
    const auto n_atoms = static_cast<Eigen::Index>(system.unit_cell.size());
    if (result.n_basis == 0 || result.n_occ == 0
        || result.charges.size() == 0 || result.density.size() == 0) {
        return Eigen::MatrixXd::Zero(n_atoms, 3);
    }
    return periodic_gfn2_derivative_terms(system, result, params, false).gradient;
}

Eigen::MatrixXd compute_periodic_gfn2_stress(
    const PeriodicSystem& system,
    const xtb::PeriodicGFN2Result& result,
    const xtb::GFN2ParameterSet& params) {
    ensure_libint_initialized();

    if (!(result.cutoff_bohr > 0.0)) {
        throw std::invalid_argument(
            "compute_periodic_gfn2_stress: result carries no lattice-cutoff "
            "provenance; re-run the SCF with the current driver");
    }
    const Eigen::Matrix3d L = system.lattice;
    const double V = std::abs(L.determinant());
    if (V < 1e-30) return Eigen::MatrixXd::Zero(3, 3);
    if (result.n_basis == 0 || result.n_occ == 0
        || result.charges.size() == 0 || result.density.size() == 0) {
        return Eigen::MatrixXd::Zero(3, 3);
    }

    const auto n_atoms = static_cast<Eigen::Index>(system.unit_cell.size());
    const auto& atoms = system.unit_cell;
    const auto cells = atom_pair_interaction_cells(system, result.cutoff_bohr);

    Eigen::MatrixXd stress = Eigen::MatrixXd::Zero(3, 3);

    // ---- Repulsive stress (exact pair virial, the energy's 1/2 image factor) ----
    for (const auto& cell : cells) {
        const Eigen::Vector3d g = cell.r_cart;
        const bool is_zero = (cell.index.array() == 0).all();
        for (Eigen::Index a = 0; a < n_atoms; ++a) {
            const Eigen::Vector3d Ra(atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]);
            const int start_b = is_zero ? static_cast<int>(a) + 1 : 0;
            for (int b = start_b; b < static_cast<int>(n_atoms); ++b) {
                const Eigen::Vector3d Rb(atoms[b].xyz[0], atoms[b].xyz[1], atoms[b].xyz[2]);
                const Eigen::Vector3d dR = Ra - (Rb + g);
                const double R = dR.norm();
                if (R < 1e-12 || R > result.cutoff_bohr) continue;
                const double dVdR = params.repulsive_derivative(atoms[a].Z, atoms[b].Z, R);
                if (dVdR == 0.0) continue;
                const double factor = is_zero ? 1.0 : 0.5;
                stress += factor * dVdR / R * (dR * dR.transpose());
            }
        }
    }

    // ---- Electronic strain derivative: atomic virial + explicit image terms ----
    const PeriodicGFN2DerivativeTerms terms =
        periodic_gfn2_derivative_terms(system, result, params, true);
    const Eigen::MatrixXd grad_elec = terms.gradient - terms.gradient_repulsive;
    // With R'=(I+eps)R and gradient=dE/dR,
    // dE/deps_ij = SUM_A gradient(A,i) R_A,j (issue #310 convention).
    stress += atomic_strain_derivative(system, grad_elec);
    stress += terms.image_strain;

    stress /= V;
    return stress;
}

}  // namespace semiempirical
}  // namespace vibeqc
