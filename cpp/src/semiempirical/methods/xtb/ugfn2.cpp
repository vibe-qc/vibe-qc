#include "vibeqc/semiempirical/methods/xtb/ugfn2.hpp"

#include <Eigen/Eigenvalues>
#include <stdexcept>

#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"
#include "vibeqc/semiempirical/core/hamiltonian_builders.hpp"

namespace vibeqc {
namespace semiempirical {
namespace xtb {

UGFN2Result run_ugfn2_xtb(
    const Molecule& mol,
    const GFN2ParameterSet& params,
    const XTBSccOptions& opts) {

    int mult = mol.multiplicity();
    // GFN2 is valence-only: use Z minus noble-core electrons.
    int n_val_elec = 0;
    for (const auto& atom : mol.atoms())
        n_val_elec += gfn2_valence_electrons(atom.Z);
    int n_alpha = (n_val_elec + mult - 1) / 2;
    int n_beta = (n_val_elec - mult + 1) / 2;

    for (const auto& atom : mol.atoms()) {
        if (!params.has_element(atom.Z)) {
            throw std::runtime_error("run_ugfn2_xtb: element Z="
                                     + std::to_string(atom.Z) + " not found");
        }
    }

    // n_primitives=0: GFN2-xTB per-element auto (H,He->3; else->4).
    BasisSet basis = SemiempiricalBasis::build(mol, params, 0);
    const int n_basis = static_cast<int>(basis.nbasis());
    if (n_alpha > n_basis) n_alpha = n_basis;
    if (n_beta > n_basis) n_beta = n_basis;
    const int n_atoms = static_cast<int>(mol.atoms().size());

    Eigen::MatrixXd S = SemiempiricalHamiltonianBuilder::build_overlap(basis);
    Eigen::MatrixXd H0 = build_gfn2_hamiltonian_zero(basis, S, mol, params);
    Eigen::MatrixXd gamma = build_gfn2_gamma(mol, params);

    double E_rep = 0.0;
    const auto& atoms = mol.atoms();
    for (std::size_t a = 0; a < atoms.size(); ++a)
        for (std::size_t b = a + 1; b < atoms.size(); ++b) {
            double dx = atoms[a].xyz[0] - atoms[b].xyz[0];
            double dy = atoms[a].xyz[1] - atoms[b].xyz[1];
            double dz = atoms[a].xyz[2] - atoms[b].xyz[2];
            double R = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (R > 1e-12) E_rep += params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
        }

    Eigen::VectorXd dq = Eigen::VectorXd::Zero(n_atoms);
    const double mix = std::max(0.0, std::min(1.0, opts.charge_mixing));
    const auto& shells = basis.shells();
    const auto& lib_shells = basis.libint();
    const auto shell2bf = lib_shells.shell2bf();

    UGFN2Result result;

    for (int iter = 1; iter <= opts.max_iter; ++iter) {
        Eigen::VectorXd V = gamma * dq;
        Eigen::MatrixXd H_scc = H0;
        for (int mu = 0; mu < n_basis; ++mu) {
            int a_mu = -1;
            for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                int bf0 = shell2bf[s]; int nf = lib_shells[s].size();
                if (mu >= bf0 && mu < bf0 + nf) { a_mu = shells[s].atom_index; break; }
            }
            double V_mu = V(a_mu);
            for (int nu = 0; nu < n_basis; ++nu) {
                int a_nu = -1;
                for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                    int bf0 = shell2bf[s]; int nf = lib_shells[s].size();
                    if (nu >= bf0 && nu < bf0 + nf) { a_nu = shells[s].atom_index; break; }
                }
                H_scc(mu, nu) += 0.5 * S(mu, nu) * (V_mu + V(a_nu));
            }
        }

        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(H_scc, S);
        if (solver.info() != Eigen::Success)
            throw std::runtime_error("UGFN2-xTB: diag failed");
        Eigen::VectorXd eps = solver.eigenvalues();
        Eigen::MatrixXd C = solver.eigenvectors();

        Eigen::MatrixXd C_occ_a = C.leftCols(n_alpha);
        Eigen::MatrixXd C_occ_b = C.leftCols(n_beta);
        Eigen::MatrixXd Da = C_occ_a * C_occ_a.transpose();
        Eigen::MatrixXd Db = C_occ_b * C_occ_b.transpose();
        Eigen::MatrixXd D_total = Da + Db;

        // Mulliken from total density
        Eigen::MatrixXd DS = D_total * S;
        Eigen::VectorXd dq_new = Eigen::VectorXd::Zero(n_atoms);
        std::vector<int> ao_atom(n_basis);
        for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
            int nf = lib_shells[s].size(); int bf0 = shell2bf[s];
            for (int i = 0; i < nf; ++i) ao_atom[bf0 + i] = shells[s].atom_index;
        }
        for (int mu = 0; mu < n_basis; ++mu) dq_new(ao_atom[mu]) += DS(mu, mu);
        for (int a = 0; a < n_atoms; ++a) {
            dq_new(a) = static_cast<double>(gfn2_valence_electrons(atoms[a].Z)) - dq_new(a);
        }

        double max_change = (dq_new - dq).cwiseAbs().maxCoeff();
        bool conv = (max_change < opts.conv_tol_charge);
        dq = mix * dq_new + (1.0 - mix) * dq;

        if (conv) {
            double E_elec = 0.0;
            for (int i = 0; i < n_alpha; ++i) E_elec += eps(i);
            for (int i = 0; i < n_beta; ++i) E_elec += eps(i);
            double E_scc = 0.5 * dq.dot(gamma * dq);
            result.energy = E_elec + E_rep - E_scc;
            result.e_electronic = E_elec; result.e_repulsive = E_rep;
            result.e_scc = E_scc;
            result.mo_energies = std::move(eps); result.mo_coeffs = std::move(C);
            result.density_alpha = std::move(Da); result.density_beta = std::move(Db);
            result.overlap = S; result.hamiltonian = std::move(H_scc);
            result.charges = dq;
            result.n_basis = n_basis; result.n_alpha = n_alpha;
            result.n_beta = n_beta; result.n_iter = iter;
            result.converged = true;
            return result;
        }
    }
    result.n_iter = opts.max_iter;
    return result;
}

}  // namespace xtb
}  // namespace semiempirical
}  // namespace vibeqc
