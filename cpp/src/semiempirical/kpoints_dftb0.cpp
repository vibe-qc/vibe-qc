#include "vibeqc/semiempirical/kpoints_dftb0.hpp"

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/core/pair_lattice.hpp"

namespace vibeqc {
namespace semiempirical {

// Build Gamma-point H⁰ blocks and return H⁰(g) + S(g) as LatticeMatrixSet
DFTB0LatticeBlocks build_h0_s_per_cell(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    double cutoff) {
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)):
    // the same per-(pair, g) interaction set the Gamma drivers sum, so
    // the Bloch sums fold exactly onto matched Gamma supercells.
    const auto cells = atom_pair_interaction_cells(system, cutoff);
    auto S_lattice = compute_overlap_lattice_explicit(basis, system, cells);
    const int n_basis = static_cast<int>(basis.nbasis());
    const auto& atoms = system.unit_cell;
    const double kappa = params.kappa();

    const auto& shells = basis.shells();
    const auto& lib_shells = basis.libint();
    const auto shell2bf = lib_shells.shell2bf();

    // AO-to-atom + avg on-site
    std::vector<int> ao_atom(n_basis);
    std::vector<double> hbar(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a)
        hbar[a] = params.average_on_site(atoms[a].Z);
    for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
        int nf = lib_shells[s].size();
        for (int i = 0; i < nf; ++i)
            ao_atom[shell2bf[s] + i] = shells[s].atom_index;
    }
    mask_blocks_beyond_pair_cutoff(
        S_lattice.blocks, cells, ao_atom, atoms, cutoff);

    std::vector<Eigen::MatrixXd> H_blocks, S_blocks;
    H_blocks.reserve(cells.size());
    S_blocks.reserve(cells.size());

    for (std::size_t ci = 0; ci < cells.size(); ++ci) {
        const auto& S_g = S_lattice.blocks[ci];
        S_blocks.push_back(S_g);

        Eigen::MatrixXd H_g = Eigen::MatrixXd::Zero(n_basis, n_basis);
        bool is_zero = (cells[ci].index.array() == 0).all();

        for (int mu = 0; mu < n_basis; ++mu) {
            int a_mu = ao_atom[mu];
            for (int nu = 0; nu < n_basis; ++nu) {
                int a_nu = ao_atom[nu];
                if (S_g(mu, nu) == 0.0) continue;
                if (!is_zero || a_mu != a_nu) {
                    double h_avg = 0.5 * (hbar[a_mu] + hbar[a_nu]);
                    H_g(mu, nu) = 0.5 * kappa * S_g(mu, nu) * h_avg;
                }
            }
        }

        if (is_zero) {
            int ao_idx = 0;
            for (int s = 0; s < static_cast<int>(shells.size()); ++s) {
                int a = shells[s].atom_index;
                int l = shells[s].l;
                double eps = params.on_site_energy(atoms[a].Z, l);
                int nf = lib_shells[s].size();
                for (int i = 0; i < nf; ++i, ++ao_idx)
                    H_g(ao_idx, ao_idx) = eps;
            }
        }

        H_blocks.push_back(H_g);
    }
    DFTB0LatticeBlocks result;
    result.cells = std::move(S_lattice.cells);
    result.h0 = std::move(H_blocks);
    result.overlap = std::move(S_blocks);
    return result;
}

int validate_dftb_kpoint_request(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    double cutoff_bohr,
    const char* caller) {
    const std::string prefix = std::string(caller) + ": ";
    if (system.unit_cell.empty()) {
        throw std::invalid_argument(prefix + "empty unit cell");
    }
    if (system.multiplicity != 1) {
        throw std::invalid_argument(prefix + "closed-shell only");
    }
    if (!std::isfinite(cutoff_bohr) || cutoff_bohr < 0.0) {
        throw std::invalid_argument(
            prefix + "cutoff_bohr must be finite and >= 0");
    }
    if (kmesh.kpoints.empty()) {
        throw std::invalid_argument(prefix + "k-point mesh is empty");
    }
    if (!kmesh.weights.empty() &&
        kmesh.weights.size() != kmesh.kpoints.size()) {
        throw std::invalid_argument(
            prefix + "k-point weights must match k-point count");
    }
    for (const auto& kpoint : kmesh.kpoints) {
        if (!kpoint.allFinite()) {
            throw std::invalid_argument(prefix + "k-points must be finite");
        }
    }
    if (!kmesh.weights.empty()) {
        double weight_sum = 0.0;
        double weight_compensation = 0.0;
        for (double weight : kmesh.weights) {
            if (!std::isfinite(weight) || weight < 0.0) {
                throw std::invalid_argument(
                    prefix + "k-point weights must be finite and nonnegative");
            }
            const double corrected = weight - weight_compensation;
            const double updated = weight_sum + corrected;
            weight_compensation =
                (updated - weight_sum) - corrected;
            weight_sum = updated;
        }
        const double weight_tolerance =
            256.0 * std::numeric_limits<double>::epsilon() *
            std::max(1.0, static_cast<double>(kmesh.weights.size()));
        if (std::abs(weight_sum - 1.0) > weight_tolerance) {
            throw std::invalid_argument(
                prefix + "k-point weights must sum to 1");
        }
    }

    int n_valence = -system.charge;
    for (const auto& atom : system.unit_cell) {
        if (!params.has_element(atom.Z)) {
            throw std::invalid_argument(
                prefix + "element Z=" + std::to_string(atom.Z) +
                " is not in the parameter set");
        }
        n_valence += params.valence_electrons(atom.Z);
    }
    if (n_valence < 0 || n_valence % 2 != 0) {
        throw std::invalid_argument(
            prefix + "even nonnegative valence electron count required");
    }
    return n_valence;
}

// ---------------------------------------------------------------------------
// k-point DFTB0
// ---------------------------------------------------------------------------

namespace {

KPointOccupationResult compute_bandpath_occupations(
    const std::vector<Eigen::VectorXd>& eps_per_k,
    int n_valence_electrons,
    int n_occ) {
    KPointOccupationResult result;
    result.occupations_per_k.reserve(eps_per_k.size());
    double homo = -std::numeric_limits<double>::infinity();
    double lumo = std::numeric_limits<double>::infinity();
    for (const auto& energies : eps_per_k) {
        const auto one_point = compute_closed_shell_kpoint_occupations(
            {energies}, {1.0}, n_valence_electrons, n_occ);
        result.occupations_per_k.push_back(
            one_point.occupations_per_k.front());
        if (n_occ > 0) {
            homo = std::max(homo, energies(n_occ - 1));
        }
        if (n_occ < energies.size()) {
            lumo = std::min(lumo, energies(n_occ));
        }
    }
    if (std::isfinite(homo + lumo)) {
        result.fermi_level = 0.5 * (homo + lumo);
    }
    return result;
}

KPointDFTB0Result run_dftb0_kpoints_impl(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    double cutoff_bohr,
    const KPointOccupationOptions& occupation_options,
    bool independent_bandpath_occupations) {

    validate_kpoint_occupation_options(
        occupation_options, "run_dftb0_kpoints");
    const int n_val_e = validate_dftb_kpoint_request(
        system, params, kmesh, cutoff_bohr, "run_dftb0_kpoints");
    // Pair-distance image selection (issue #316; pair_lattice.hpp (*)).
    const auto cells_for_guard =
        atom_pair_interaction_cells(system, cutoff_bohr);
    require_nonzero_lattice_image(
        cells_for_guard, cutoff_bohr, "run_dftb0_kpoints");

    Molecule mol = system.unit_cell_molecule();
    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    // Cap at n_basis — a valence-minimal basis holds at most n_basis
    // doubly-occupied MOs per k-point; uncapped counts read past the
    // eigensolution (UB; see run_dftb0).
    int n_occ = n_val_e / 2;
    if (n_occ > n_basis) n_occ = n_basis;

    const auto lattice =
        build_h0_s_per_cell(basis, system, params, cutoff_bohr);
    const auto& cells = lattice.cells;

    KPointDFTB0Result result;
    result.n_basis = n_basis;
    result.n_occ = n_occ;
    result.n_kpoints = static_cast<int>(kmesh.size());
    result.smearing_temperature = occupation_options.smearing_temperature;

    for (std::size_t ik = 0; ik < kmesh.size(); ++ik) {
        const auto& k = kmesh.kpoints[ik];

        // Build H(k) and S(k) as complex matrices
        Eigen::MatrixXcd Hk = Eigen::MatrixXcd::Zero(n_basis, n_basis);
        Eigen::MatrixXcd Sk = Eigen::MatrixXcd::Zero(n_basis, n_basis);

        for (std::size_t ci = 0; ci < cells.size(); ++ci) {
            double phase = k.dot(cells[ci].r_cart);
            std::complex<double> eikr(std::cos(phase), std::sin(phase));
            Hk += eikr * lattice.h0[ci].cast<std::complex<double>>();
            Sk += eikr * lattice.overlap[ci].cast<std::complex<double>>();
        }

        Hk = 0.5 * (Hk + Hk.adjoint().eval());
        Sk = 0.5 * (Sk + Sk.adjoint().eval());
        const auto bands = diagonalize_bloch(Hk, Sk);
        const Eigen::VectorXd& eps = bands.energies;
        result.eps_per_k.push_back(eps);

        for (int i = 0; i < n_basis; ++i)
            result.band_energies.push_back(eps(i));
    }

    auto occupation = independent_bandpath_occupations
                          ? compute_bandpath_occupations(
                                result.eps_per_k, n_val_e, n_occ)
                          : compute_closed_shell_kpoint_occupations(
                                result.eps_per_k,
                                kmesh.weights,
                                n_val_e,
                                n_occ,
                                occupation_options);
    // Issue #434. DFTB0 is non-self-consistent: this single fill is the
    // accepted spectrum and the band energy below is built straight from it,
    // so an unresolvable cut here is a wrong answer with no later iterate to
    // correct it. The independent_bandpath branch fills each k separately
    // with no global particle constraint, so it cannot race and records
    // nothing -- the guard is a no-op there.
    reject_unresolved_frontier_cut(
        occupation, occupation_options, "run_dftb0_kpoints");
    double E_elec = 0.0;
    for (std::size_t ik = 0; ik < kmesh.size(); ++ik) {
        const double wk = kmesh.weights.empty()
                              ? 1.0 / static_cast<double>(kmesh.size())
                              : kmesh.weights[ik];
        for (int band = 0; band < n_basis; ++band) {
            E_elec += wk * occupation.occupations_per_k[ik](band) *
                      result.eps_per_k[ik](band);
        }
    }

    // Repulsive energy (same as Gamma-point)
    const auto& atoms = system.unit_cell;
    double E_rep = 0.0;
    for (const auto& cell : cells) {
        bool is_zero = (cell.index.array() == 0).all();
        for (int a = 0; a < static_cast<int>(atoms.size()); ++a) {
            int start_b = is_zero ? a + 1 : 0;
            for (int b = start_b; b < static_cast<int>(atoms.size()); ++b) {
                double dx = atoms[a].xyz[0] - (atoms[b].xyz[0] + cell.r_cart[0]);
                double dy = atoms[a].xyz[1] - (atoms[b].xyz[1] + cell.r_cart[1]);
                double dz = atoms[a].xyz[2] - (atoms[b].xyz[2] + cell.r_cart[2]);
                double R = std::sqrt(dx*dx + dy*dy + dz*dz);
                if (R < 1e-12) continue;
                // Pair-distance cutoff (issue #316; pair_lattice.hpp (*)).
                if (R > cutoff_bohr) continue;
                double factor = is_zero ? 1.0 : 0.5;
                E_rep += factor * params.repulsive_energy(atoms[a].Z, atoms[b].Z, R);
            }
        }
    }

    result.e_electronic = E_elec;
    result.e_repulsive = E_rep;
    result.energy = E_elec + E_rep;
    result.fermi_level = occupation.fermi_level;
    result.entropy = occupation.entropy;
    result.free_energy =
        result.energy - occupation_options.smearing_temperature * result.entropy;
    result.occupations_per_k = std::move(occupation.occupations_per_k);
    // Measured band edges + gaps from the occupations actually used
    // (issue #426; convention documented on KPointBandEdges).
    store_kpoint_band_edges(
        result,
        compute_kpoint_band_edges(result.eps_per_k, result.occupations_per_k));
    return result;
}

}  // namespace

KPointDFTB0Result run_dftb0_kpoints(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const BlochKMesh& kmesh,
    double cutoff_bohr,
    const KPointOccupationOptions& occupation_options) {
    return run_dftb0_kpoints_impl(
        system,
        params,
        kmesh,
        cutoff_bohr,
        occupation_options,
        false);
}

// ---------------------------------------------------------------------------
// Band-structure path
// ---------------------------------------------------------------------------

KPointDFTB0Result run_dftb0_bandpath(
    const PeriodicSystem& system,
    const SemiempiricalParameters& params,
    const std::vector<Eigen::Vector3d>& kpath,
    double cutoff_bohr) {

    BlochKMesh kmesh;
    kmesh.kpoints = kpath;
    kmesh.weights.assign(kpath.size(), 1.0 / kpath.size());
    return run_dftb0_kpoints_impl(
        system,
        params,
        kmesh,
        cutoff_bohr,
        {},
        true);
}

}  // namespace semiempirical
}  // namespace vibeqc
