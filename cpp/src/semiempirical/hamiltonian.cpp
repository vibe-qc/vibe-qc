#include "vibeqc/semiempirical/hamiltonian.hpp"

#include "vibeqc/semiempirical/core/periodic_gamma.hpp"

#include <vector>

#include "vibeqc/integrals.hpp"
#include "vibeqc/molecule.hpp"

namespace vibeqc {
namespace semiempirical {

// ---------------------------------------------------------------------------
// Overlap matrix: delegates to libint's compute_overlap.
// ---------------------------------------------------------------------------

Eigen::MatrixXd SemiempiricalHamiltonianBuilder::build_overlap(
    const BasisSet& basis) {
    return vibeqc::compute_overlap(basis);
}

// ---------------------------------------------------------------------------
// Build on-site energy list for a given atom in the basis
// ---------------------------------------------------------------------------

std::vector<double> SemiempiricalHamiltonianBuilder::on_site_energies_for_atom(
    int atom_index, const BasisSet& basis,
    const SemiempiricalParameters& params,
    int Z_atom) {
    std::vector<double> result;
    const auto& shells = basis.shells();

    for (std::size_t s = 0; s < shells.size(); ++s) {
        if (shells[s].atom_index != atom_index) continue;
        int l = shells[s].l;
        int n_orbs = shells[s].pure ? (2 * l + 1) : ((l + 1) * (l + 2) / 2);
        double e_onsite = params.on_site_energy(Z_atom, l);
        if (e_onsite == 0.0) {
            e_onsite = -0.5;
        }
        for (int i = 0; i < n_orbs; ++i) {
            result.push_back(e_onsite);
        }
    }
    return result;
}

// ---------------------------------------------------------------------------
// Build DFTB0 H⁰ from overlap and parameters
// ---------------------------------------------------------------------------

Eigen::MatrixXd SemiempiricalHamiltonianBuilder::build_hamiltonian_zero(
    const BasisSet& basis,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const SemiempiricalParameters& params) {
    const auto n_basis = static_cast<int>(basis.nbasis());
    Eigen::MatrixXd H0 = Eigen::MatrixXd::Zero(n_basis, n_basis);

    const auto& shells = basis.shells();
    const auto& atoms = mol.atoms();

    // Map each AO to its atom index and angular momentum.
    struct AoInfo {
        int atom_idx;
        int l;
    };
    std::vector<AoInfo> ao_map;
    ao_map.reserve(n_basis);
    for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
        int n_funcs = shells[s].pure
            ? (2 * shells[s].l + 1)
            : ((shells[s].l + 1) * (shells[s].l + 2) / 2);
        for (int i = 0; i < n_funcs; ++i) {
            ao_map.push_back({shells[s].atom_index, shells[s].l});
        }
    }

    const double kappa = params.kappa();

    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_map[mu].atom_idx;
        int l_mu = ao_map[mu].l;
        int Z_mu = atoms[a_mu].Z;

        for (int nu = mu; nu < n_basis; ++nu) {
            int a_nu = ao_map[nu].atom_idx;
            (void)ao_map[nu].l;  // may be unused
            int Z_nu = atoms[a_nu].Z;

            if (a_mu == a_nu) {
                if (mu == nu) {
                    double eps = params.on_site_energy(Z_mu, l_mu);
                    if (eps == 0.0) eps = -0.5;
                    H0(mu, nu) = eps;
                }
            } else {
                double h_mu = params.average_on_site(Z_mu);
                double h_nu = params.average_on_site(Z_nu);
                double h_avg = 0.5 * (h_mu + h_nu);
                double s_val = S(mu, nu);
                H0(mu, nu) = 0.5 * kappa * s_val * h_avg;
            }
        }
    }

    // Symmetrize
    for (int mu = 0; mu < n_basis; ++mu) {
        for (int nu = mu + 1; nu < n_basis; ++nu) {
            H0(nu, mu) = H0(mu, nu);
        }
    }

    return H0;
}


// ---------------------------------------------------------------------------
// Valence electron count
// ---------------------------------------------------------------------------

int SemiempiricalHamiltonianBuilder::valence_electron_count(
    const Molecule& mol,
    const SemiempiricalParameters& params) {
    int n = 0;
    for (const auto& atom : mol.atoms()) {
        n += params.valence_electrons(atom.Z);
    }
    return n - mol.charge();
}

// ---------------------------------------------------------------------------
// Mulliken charge analysis
// ---------------------------------------------------------------------------

Eigen::VectorXd SemiempiricalHamiltonianBuilder::mulliken_charges(
    const BasisSet& basis,
    const Eigen::MatrixXd& D,
    const Eigen::MatrixXd& S,
    const Molecule& mol,
    const SemiempiricalParameters& params) {
    const auto n_basis = basis.nbasis();
    const auto n_atoms = mol.atoms().size();
    const auto& shells = basis.shells();

    // Compute DS product for diagonal elements
    Eigen::MatrixXd DS = D * S;

    // Build AO-to-atom map
    std::vector<int> ao_atom(n_basis);
    const auto& libint_shells = basis.libint();
    const auto shell2bf = libint_shells.shell2bf();
    for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
        int n_funcs = libint_shells[s].size();
        int bf_start = shell2bf[s];
        for (int i = 0; i < n_funcs; ++i) {
            ao_atom[bf_start + i] = shells[s].atom_index;
        }
    }

    Eigen::VectorXd dq = Eigen::VectorXd::Zero(n_atoms);

    // Mulliken population: P_A = Σ_{μ∈A} (D S)_{μμ}
    // Charge fluctuation: Δq_A = n_val(A) − P_A
    for (Eigen::Index mu = 0; mu < static_cast<Eigen::Index>(n_basis); ++mu) {
        int a = ao_atom[mu];
        dq(a) += DS(mu, mu);
    }

    for (Eigen::Index a = 0; a < static_cast<Eigen::Index>(n_atoms); ++a) {
        int Z = mol.atoms()[a].Z;
        int n_val = params.valence_electrons(Z);
        dq(a) = static_cast<double>(n_val) - dq(a);
    }

    // Enforce total charge conservation: Σ Δq_A = q_mol. With the
    // valence-only band filling this shift is a numerical no-op
    // (tr(DS) = N_val exactly), but it keeps roundoff from leaking
    // into the SCC potential.
    double q_mol = static_cast<double>(mol.charge());
    double excess = (dq.sum() - q_mol) / static_cast<double>(n_atoms);
    dq.array() -= excess;

    return dq;
}

// ---------------------------------------------------------------------------
// gamma matrix
// ---------------------------------------------------------------------------
//
// One site per atom, hardness U, through the shared kernel of
// semiempirical/core/periodic_gamma.hpp so that the molecular, Gamma,
// k-point and SECCM SCC-DFTB routes evaluate one functional form under one
// method name (maintainer decision D1, 2026-08-28).
//
//   ShellGammaForm::KlopmanOhno (default) -- the in-house Ohno-Klopman form
//     gamma = 1/sqrt(R^2 + eta^2), eta = 0.5 (1/U_A + 1/U_B), on-site U.
//   ShellGammaForm::Elstner -- Elstner et al., Phys. Rev. B 58, 7260 (1998),
//     Eqs. 17/18: gamma = 1/R - S(tau_A, tau_B, R) with tau = 16/5 U.  This
//     is the form the published method uses and the one D1 adopts; its
//     remainder decays exponentially, which is what gives the periodic
//     lattice sum a thermodynamic limit (#425).
//
// The on-site block is U in both forms: it is a method parameter, not a
// lattice-sum question, and Elstner's equal-tau limit 5 tau/16 is exactly U.
Eigen::MatrixXd SemiempiricalHamiltonianBuilder::gamma_matrix(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    ShellGammaForm form) {
    const auto& atoms = mol.atoms();
    std::vector<GammaSite> sites;
    std::vector<Eigen::Vector3d> positions;
    sites.reserve(atoms.size());
    positions.reserve(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a) {
        double u = params.hubbard_u(atoms[a].Z);
        if (u <= 0.0) u = 0.4;  // fallback
        GammaSite site;
        site.atom = static_cast<int>(a);
        site.hardness = u;
        sites.push_back(site);
        positions.emplace_back(
            atoms[a].xyz[0], atoms[a].xyz[1], atoms[a].xyz[2]);
    }
    ShellGammaSpec spec;
    spec.form = form;
    spec.ko_average = KlopmanOhnoAverage::InverseHardnessMean;
    return build_molecular_shell_gamma(sites, positions, spec);
}

// ---------------------------------------------------------------------------
// SCC Hamiltonian correction
// ---------------------------------------------------------------------------

Eigen::MatrixXd SemiempiricalHamiltonianBuilder::build_scc_hamiltonian(
    const Eigen::MatrixXd& H0,
    const Eigen::MatrixXd& S,
    const BasisSet& basis,
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const Eigen::MatrixXd& gamma,
    const Eigen::VectorXd& delta_q) {
    const auto n_basis = static_cast<int>(basis.nbasis());
    const auto n_atoms = static_cast<Eigen::Index>(mol.atoms().size());
    const auto& shells = basis.shells();
    const auto& libint_shells = basis.libint();
    const auto shell2bf = libint_shells.shell2bf();

    // Compute electrostatic potential at each atom: V_A = Σ_C γ_{AC} Δq_C
    Eigen::VectorXd V = gamma * delta_q;

    // AO-to-atom map
    std::vector<int> ao_atom(n_basis);
    for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
        int n_funcs = libint_shells[s].size();
        int bf_start = shell2bf[s];
        for (int i = 0; i < n_funcs; ++i) {
            ao_atom[bf_start + i] = shells[s].atom_index;
        }
    }

    // H^{SCC}_{μν} = H⁰_{μν} − ½ S_{μν} (V_A + V_B) for μ∈A, ν∈B.
    // Eq. 14 of Elstner et al., Phys. Rev. B 58, 7260 (1998),
    // doi:10.1103/PhysRevB.58.7260; the minus sign absorbs our
    // Δq > 0 = cation convention (Elstner's Δq is excess electron
    // population). See build_scc_hamiltonian docs in hamiltonian.hpp.
    Eigen::MatrixXd H = H0;
    for (int mu = 0; mu < n_basis; ++mu) {
        int a_mu = ao_atom[mu];
        double V_mu = V(a_mu);
        for (int nu = 0; nu < n_basis; ++nu) {
            int a_nu = ao_atom[nu];
            double V_nu = V(a_nu);
            H(mu, nu) -= 0.5 * S(mu, nu) * (V_mu + V_nu);
        }
    }

    return H;
}

// ---------------------------------------------------------------------------
// Γ matrix (third-order Hubbard-derivative coupling, DFTB3)
// ---------------------------------------------------------------------------

Eigen::MatrixXd SemiempiricalHamiltonianBuilder::gamma3_matrix(
    const Molecule& mol,
    const SemiempiricalParameters& params) {
    const auto& atoms = mol.atoms();
    const auto n = static_cast<Eigen::Index>(atoms.size());
    Eigen::MatrixXd gm3 = Eigen::MatrixXd::Zero(n, n);
    for (Eigen::Index a = 0; a < n; ++a) {
        double Ua = params.hubbard_u(atoms[a].Z);
        double dUdQ_a = 0.0  /* hubbard_derivative: v0.15.116 compat */;
        if (Ua <= 0.0) Ua = 0.4;
        gm3(a, a) = 0.5 * dUdQ_a;
        for (Eigen::Index b = a + 1; b < n; ++b) {
            double Ub = params.hubbard_u(atoms[b].Z);
            double dUdQ_b = 0.0  /* hubbard_derivative: v0.15.116 compat */;
            if (Ub <= 0.0) Ub = 0.4;
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            double eta = 0.5 * (1.0/Ua + 1.0/Ub);
            double g3 = 1.0 / std::pow(R*R + eta*eta, 1.5);
            double dgamma_dUa = eta / (2.0 * Ua * Ua) * g3;
            double dgamma_dUb = eta / (2.0 * Ub * Ub) * g3;
            gm3(a, b) = dgamma_dUa * dUdQ_a;
            gm3(b, a) = dgamma_dUb * dUdQ_b;
        }
    }
    return gm3;
}

Eigen::VectorXd SemiempiricalHamiltonianBuilder::third_order_potential(
    const Eigen::MatrixXd& gamma3,
    const Eigen::VectorXd& delta_q) {
    const auto n = delta_q.size();
    Eigen::VectorXd V3 = Eigen::VectorXd::Zero(n);
    for (Eigen::Index a = 0; a < n; ++a) {
        V3(a) = delta_q(a) * delta_q(a) * gamma3(a, a);
        for (Eigen::Index b = 0; b < n; ++b) {
            if (a == b) continue;
            V3(a) += (2.0/3.0) * delta_q(a) * delta_q(b) * gamma3(a, b);
            V3(a) += (1.0/3.0) * delta_q(b) * delta_q(b) * gamma3(b, a);
        }
    }
    return V3;
}

Eigen::VectorXd SemiempiricalHamiltonianBuilder::scc_potential_dftb3(
    const Eigen::MatrixXd& gamma,
    const Eigen::MatrixXd& gamma3,
    const Eigen::VectorXd& delta_q) {
    return gamma * delta_q + third_order_potential(gamma3, delta_q);
}

}  // namespace semiempirical
}  // namespace vibeqc
