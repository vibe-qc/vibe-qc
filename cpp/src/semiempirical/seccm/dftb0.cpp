#include "vibeqc/semiempirical/seccm/dftb0.hpp"

#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/periodic_gradient.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"

#include "seccm_common.h"

namespace vibeqc {
namespace semiempirical {
namespace seccm {

using detail::ShellKey;

inline void validate_inputs(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    double hermiticity_tolerance,
    double gap_tolerance) {
    // One, two, and three cyclic dimensions are all supported. Every
    // stage below this point is dimension-generic, and is the same code
    // SCC-DFTB-SECCM already runs in 2-D and 3-D:
    // detail::assemble_weighted_dftb_h0 takes system.dim from
    // topology.translations.size(), the gradient builds the same
    // LatticeMatrixSet and calls the same
    // overlap_lattice_gradient_contribution, and validate_common_inputs
    // already enforces 1 <= D <= 3 with linear independence.
    //
    // DFTB0 carries no SCC, no gamma matrix and no Madelung embedding,
    // so its only long-range object is the exponentially decaying
    // overlap: the Wigner-Seitz truncation is the sole error and it
    // decays exponentially in every dimension (measured against the
    // matched-mesh Bloch reference at 4.2e-8, 2.6e-10 and 2.9e-12
    // Ha/cell for R = 3, 4, 5, the same to two significant figures in
    // 1-D, 2-D and 3-D).
    //
    // The even-replica fail-closed gate that SCC-DFTB-SECCM carries
    // deliberately does NOT apply here: that pathology is an SCC
    // charge-map failure, and this charge-free route shows no
    // divergence at even R even at 95.5% fractional record ownership
    // (maintainer decision D2, 2026-08-28).
    detail::validate_common_inputs(
        mol,
        [&params](int Z) { return params.has_element(Z); },
        topology,
        group_order,
        geometry_tolerance,
        hermiticity_tolerance,
        gap_tolerance,
        "DFTB0-SECCM",
        [](int) {
            // Scope is defined by the explicit repulsive-pair table
            // below, not by a hardcoded element list. The shipped
            // vibeqc-inhouse-dftb-screening-v1 set carries explicit
            // repulsives for H, C, N, O, F, P, S and Cl and for every
            // one of their 36 unordered pairs, so the pair predicate is
            // complete and self-describing; an element missing from the
            // parameter set is already rejected by the has_element
            // check above.
        },
        [&params](int Z1, int Z2) {
            if (params.repulsive_pair(Z1, Z2) == nullptr) {
                throw std::invalid_argument(
                    "DFTB0-SECCM requires an explicit repulsive pair for "
                    "every element pair in the cell; the shipped "
                    "vibeqc-inhouse-dftb-screening-v1 set covers "
                    "H, C, N, O, F, P, S, Cl");
            }
        });
}

DFTB0SECCMResult run_dftb0_seccm(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    double hermiticity_tolerance,
    double gap_tolerance,
    bool compute_gradient) {
    validate_inputs(
        mol,
        params,
        topology,
        group_order,
        geometry_tolerance,
        hermiticity_tolerance,
        gap_tolerance);

    const int n_valence =
        SemiempiricalHamiltonianBuilder::valence_electron_count(mol, params);
    if (n_valence <= 0 || n_valence % 2 != 0) {
        throw std::invalid_argument(
            "DFTB0-SECCM requires a positive even valence-electron count");
    }

    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    const int n_basis = static_cast<int>(basis.nbasis());
    const int n_occ = n_valence / 2;
    if (n_valence > 2 * n_basis || n_occ >= n_basis) {
        throw std::invalid_argument(
            "DFTB0-SECCM occupancy does not fit the finite-cluster AO space "
            "or leaves no LUMO");
    }

    const std::pair<Eigen::MatrixXd, Eigen::MatrixXd> assembled =
        detail::assemble_weighted_dftb_h0(mol, params, topology, basis, n_basis);
    Eigen::MatrixXd overlap = assembled.first;
    Eigen::MatrixXd hamiltonian = assembled.second;
    if (!overlap.allFinite() || !hamiltonian.allFinite()) {
        throw std::runtime_error(
            "DFTB0-SECCM matrix assembly produced non-finite values");
    }
    if (detail::max_abs(overlap - overlap.transpose()) > hermiticity_tolerance
        || detail::max_abs(hamiltonian - hamiltonian.transpose())
            > hermiticity_tolerance) {
        throw std::runtime_error(
            "DFTB0-SECCM reverse-image assembly is not Hermitian");
    }

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> overlap_solver(overlap);
    Eigen::MatrixXd ortho;
    if (overlap_solver.info() != Eigen::Success
        || overlap_solver.eigenvalues().minCoeff() <= hermiticity_tolerance) {
        // WS-weighted cyclic overlap with non-positive modes: screen the
        // non-positive subspace (Peintinger-Bredow 2014) instead of hard
        // failing; the kept subspace must still cover the occupied
        // manifold, otherwise the adapter fails closed as before.
        ortho = detail::canonical_orthogonalizer(
            overlap, hermiticity_tolerance, n_occ, "DFTB0-SECCM");
    }
    Eigen::VectorXd energies;
    Eigen::MatrixXd coefficients;
    if (ortho.size() > 0) {
        const detail::CanonicalEigensolution canonical =
            detail::solve_canonical_orthogonalized(
                hamiltonian, ortho, "DFTB0-SECCM");
        energies = canonical.eigenvalues;
        coefficients = canonical.coefficients;
    } else {
        Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
            hamiltonian, overlap);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "DFTB0-SECCM generalized eigensolver failed");
        }
        energies = solver.eigenvalues();
        coefficients = solver.eigenvectors();
    }
    const double gap = energies(n_occ) - energies(n_occ - 1);
    if (!std::isfinite(gap) || gap <= gap_tolerance) {
        throw std::runtime_error(
            "DFTB0-SECCM T3a requires a positive finite-torus HOMO-LUMO gap");
    }

    double cyclic_electronic = 0.0;
    for (int orbital = 0; orbital < n_occ; ++orbital) {
        cyclic_electronic += 2.0 * energies(orbital);
    }
    const auto& atoms = mol.atoms();
    double cyclic_repulsive = 0.0;
    int n_records = 0;
    for (int central = 0; central < static_cast<int>(atoms.size()); ++central) {
        for (const auto& image : topology.cells[central]) {
            cyclic_repulsive += 0.5 * image.weight
                * params.repulsive_energy(
                    atoms[central].Z,
                    atoms[image.origin].Z,
                    image.disp.norm());
            ++n_records;
        }
    }
    const double normalization = static_cast<double>(group_order);

    DFTB0SECCMResult result;
    result.electronic_energy = cyclic_electronic / normalization;
    result.repulsive_energy = cyclic_repulsive / normalization;
    result.energy = result.electronic_energy + result.repulsive_energy;
    result.total_cyclic_energy = cyclic_electronic + cyclic_repulsive;
    result.cyclic_electronic_energy = cyclic_electronic;
    result.cyclic_repulsive_energy = cyclic_repulsive;
    result.homo_lumo_gap = gap;
    result.mo_energies = std::move(energies);
    result.mo_coeffs = std::move(coefficients);
    const Eigen::MatrixXd occupied = result.mo_coeffs.leftCols(n_occ);
    result.density = 2.0 * occupied * occupied.transpose();
    result.overlap = std::move(overlap);
    result.hamiltonian = std::move(hamiltonian);
    result.n_basis = n_basis;
    result.n_occ = n_occ;
    result.group_order = group_order;
    result.n_records = n_records;
    if (compute_gradient) {
        result.gradient = compute_dftb0_seccm_gradient(
            mol,
            params,
            topology,
            result,
            group_order,
            geometry_tolerance);
        result.has_gradient = true;
    }
    return result;
}

Eigen::MatrixXd compute_dftb0_seccm_gradient(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    const DFTB0SECCMResult& result,
    int group_order,
    double geometry_tolerance) {
    validate_inputs(
        mol,
        params,
        topology,
        group_order,
        geometry_tolerance,
        1.0e-10,
        1.0e-8);
    if (result.group_order != group_order || result.n_basis <= 0
        || result.n_occ <= 0 || result.n_occ >= result.n_basis
        || result.density.rows() != result.n_basis
        || result.density.cols() != result.n_basis
        || result.mo_coeffs.rows() != result.n_basis
        || result.mo_coeffs.cols() != result.n_basis
        || result.mo_energies.size() != result.n_basis) {
        throw std::invalid_argument(
            "DFTB0-SECCM gradient requires a matching finite-cluster result");
    }
    if (!result.density.allFinite() || !result.mo_coeffs.allFinite()
        || !result.mo_energies.allFinite()) {
        throw std::invalid_argument(
            "DFTB0-SECCM gradient result matrices must be finite");
    }

    BasisSet basis = SemiempiricalBasis::build(mol, params, 6);
    if (static_cast<int>(basis.nbasis()) != result.n_basis) {
        throw std::invalid_argument(
            "DFTB0-SECCM gradient basis does not match the energy result");
    }
    const auto& shells = basis.libint();
    const auto shell2bf = shells.shell2bf();
    const auto shell_info = basis.shells();
    std::vector<int> ao_atom(static_cast<std::size_t>(result.n_basis));
    for (std::size_t shell = 0; shell < shell_info.size(); ++shell) {
        for (std::size_t local = 0; local < shells[shell].size(); ++local) {
            ao_atom[shell2bf[shell] + local] = shell_info[shell].atom_index;
        }
    }

    const Eigen::MatrixXd occupied = result.mo_coeffs.leftCols(result.n_occ);
    Eigen::MatrixXd energy_weighted = Eigen::MatrixXd::Zero(
        result.n_basis, result.n_basis);
    for (int orbital = 0; orbital < result.n_occ; ++orbital) {
        energy_weighted += result.mo_energies(orbital)
            * (2.0 * occupied.col(orbital) * occupied.col(orbital).transpose());
    }

    const auto& atoms = mol.atoms();
    Eigen::MatrixXd effective = -energy_weighted;
    for (int mu = 0; mu < result.n_basis; ++mu) {
        const int central = ao_atom[static_cast<std::size_t>(mu)];
        for (int nu = 0; nu < result.n_basis; ++nu) {
            const int origin = ao_atom[static_cast<std::size_t>(nu)];
            if (central == origin) continue;
            const double average = 0.5
                * (params.average_on_site(atoms[central].Z)
                   + params.average_on_site(atoms[origin].Z));
            effective(mu, nu) += result.density(mu, nu)
                * 0.5 * params.kappa() * average;
        }
    }

    std::vector<ShellKey> shell_keys{{0, 0, 0}};
    for (const auto& cell : topology.cells) {
        for (const auto& image : cell) {
            if (std::find(
                    shell_keys.begin(),
                    shell_keys.end(),
                    image.image_shell_label)
                == shell_keys.end()) {
                shell_keys.push_back(image.image_shell_label);
            }
        }
    }
    LatticeMatrixSet weighted_effective;
    weighted_effective.nbf = result.n_basis;
    weighted_effective.cells.reserve(shell_keys.size());
    weighted_effective.blocks.assign(
        shell_keys.size(),
        Eigen::MatrixXd::Zero(result.n_basis, result.n_basis));
    std::map<ShellKey, std::size_t> block_by_label;
    for (std::size_t index = 0; index < shell_keys.size(); ++index) {
        LatticeCell cell;
        cell.index = Eigen::Vector3i(
            shell_keys[index][0],
            shell_keys[index][1],
            shell_keys[index][2]);
        cell.r_cart = detail::shell_translation(
            shell_keys[index], topology.translations);
        weighted_effective.cells.push_back(cell);
        block_by_label.emplace(shell_keys[index], index);
    }
    for (int mu = 0; mu < result.n_basis; ++mu) {
        const int central = ao_atom[static_cast<std::size_t>(mu)];
        for (int nu = 0; nu < result.n_basis; ++nu) {
            const int origin = ao_atom[static_cast<std::size_t>(nu)];
            if (central == origin) continue;
            for (const auto& image : topology.cells[central]) {
                if (image.origin != origin) continue;
                const auto block = block_by_label.at(image.image_shell_label);
                weighted_effective.blocks[block](mu, nu) =
                    -image.weight * effective(mu, nu);
            }
        }
    }

    PeriodicSystem system;
    system.dim = static_cast<int>(topology.translations.size());
    system.unit_cell = atoms;
    system.charge = mol.charge();
    system.multiplicity = mol.multiplicity();
    for (std::size_t axis = 0; axis < topology.translations.size(); ++axis) {
        system.lattice.col(static_cast<Eigen::Index>(axis)) =
            topology.translations[axis];
    }
    LatticeSumOptions unused_options;
    Eigen::MatrixXd gradient = overlap_lattice_gradient_contribution(
        basis, system, weighted_effective, unused_options);

    for (int central = 0; central < static_cast<int>(atoms.size()); ++central) {
        for (const auto& image : topology.cells[central]) {
            const double distance = image.disp.norm();
            const double derivative = params.repulsive_derivative(
                atoms[central].Z, atoms[image.origin].Z, distance);
            if (derivative == 0.0) continue;
            const Eigen::Vector3d contribution =
                0.5 * image.weight * derivative * image.disp / distance;
            gradient.row(central) -= contribution.transpose();
            gradient.row(image.origin) += contribution.transpose();
        }
    }
    gradient /= static_cast<double>(group_order);
    if (!gradient.allFinite()) {
        throw std::runtime_error(
            "DFTB0-SECCM gradient produced non-finite values");
    }
    return gradient;
}

}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
