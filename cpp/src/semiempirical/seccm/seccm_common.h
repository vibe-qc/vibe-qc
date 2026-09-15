// Shared internals for SECCM (semiempirical cyclic-cluster) adapters.
//
// Hamiltonian-independent WS-record validation plus the DFTB-family
// supercell assembly (weighted S / H0 / gamma). Extracted verbatim from
// the validated DFTB0-SECCM adapter (seccm/dftb0.cpp, T3a) so that the
// SCC-DFTB SECCM adapter and later method adapters consume the exact same
// topology contract and assembly arithmetic. This header is private to
// cpp/src/semiempirical/seccm/ — it is not public API.

#pragma once

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "vibeqc/semiempirical/core/periodic_gamma.hpp"

#include "vibeqc/lattice_integrals.hpp"
#include "vibeqc/molecule.hpp"
#include "vibeqc/periodic.hpp"
#include "vibeqc/semiempirical/basis.hpp"
#include "vibeqc/semiempirical/hamiltonian.hpp"
#include "vibeqc/semiempirical/parameters.hpp"
#include "vibeqc/semiempirical/seccm/topology.hpp"

namespace vibeqc {
namespace semiempirical {
namespace seccm {
namespace detail {

using RecordKey = std::tuple<int, int, int, int, int>;
using ShellKey = std::array<int, 3>;

inline RecordKey record_key(int central, const WSImage& image) {
    return {
        central,
        image.origin,
        image.image_shell_label[0],
        image.image_shell_label[1],
        image.image_shell_label[2],
    };
}

inline RecordKey reverse_key(int central, const WSImage& image) {
    return {
        image.origin,
        central,
        -image.image_shell_label[0],
        -image.image_shell_label[1],
        -image.image_shell_label[2],
    };
}

inline Eigen::Vector3d shell_translation(
    const ShellKey& label,
    const std::vector<Eigen::Vector3d>& translations) {
    Eigen::Vector3d result = Eigen::Vector3d::Zero();
    for (std::size_t axis = 0; axis < translations.size(); ++axis) {
        result += static_cast<double>(label[axis]) * translations[axis];
    }
    return result;
}

inline double max_abs(const Eigen::MatrixXd& matrix) {
    return matrix.size() == 0 ? 0.0 : matrix.cwiseAbs().maxCoeff();
}

// Common SECCM input validation: neutral closed-shell molecule, one to
// three independent cyclic translations, a full directed-ownership record
// set, and exact reverse images. ``element_check`` applies the route's
// element-scope restriction; ``pair_check`` applies its pairwise-parameter
// requirement. Error strings carry ``route_name`` so each adapter keeps its
// own public identity.
inline void validate_common_inputs(
    const Molecule& mol,
    const std::function<bool(int)>& has_element,
    const WSTopology& topology,
    int group_order,
    double geometry_tolerance,
    double hermiticity_tolerance,
    double gap_tolerance,
    const std::string& route_name,
    const std::function<void(int)>& element_check,
    const std::function<void(int, int)>& pair_check,
    bool allow_charged = false) {
    const int natoms = static_cast<int>(mol.atoms().size());
    if (natoms < 1) {
        throw std::invalid_argument(route_name + " requires at least one atom");
    }
    if (mol.multiplicity() != 1) {
        throw std::invalid_argument(
            route_name + " supports only closed-shell inputs");
    }
    if (mol.charge() != 0 && !allow_charged) {
        throw std::invalid_argument(
            route_name + " supports only neutral inputs");
    }
    if (group_order < 1) {
        throw std::invalid_argument(
            route_name + " requires a positive finite-group order");
    }
    if (topology.translations.empty() || topology.translations.size() > 3) {
        throw std::invalid_argument(
            route_name + " requires one to three cyclic translations");
    }
    if (static_cast<int>(topology.cells.size()) != natoms) {
        throw std::invalid_argument(
            route_name + " topology atom count does not match the molecule");
    }
    if (!std::isfinite(geometry_tolerance) || geometry_tolerance <= 0.0
        || !std::isfinite(hermiticity_tolerance)
        || hermiticity_tolerance <= 0.0
        || !std::isfinite(gap_tolerance) || gap_tolerance <= 0.0) {
        throw std::invalid_argument(
            route_name + " tolerances must be positive and finite");
    }

    Eigen::MatrixXd translation_matrix(
        static_cast<Eigen::Index>(topology.translations.size()), 3);
    for (std::size_t axis = 0; axis < topology.translations.size(); ++axis) {
        if (!topology.translations[axis].allFinite()) {
            throw std::invalid_argument(
                route_name + " cyclic translations must be finite");
        }
        translation_matrix.row(static_cast<Eigen::Index>(axis)) =
            topology.translations[axis].transpose();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(translation_matrix);
    if (svd.rank() != static_cast<Eigen::Index>(topology.translations.size())) {
        throw std::invalid_argument(
            route_name + " cyclic translations must be linearly independent");
    }

    for (const auto& atom : mol.atoms()) {
        element_check(atom.Z);
        if (!has_element(atom.Z)) {
            throw std::invalid_argument(
                route_name + " parameter set is missing an input element");
        }
        for (double coordinate : atom.xyz) {
            if (!std::isfinite(coordinate)) {
                throw std::invalid_argument(
                    route_name + " atomic coordinates must be finite");
            }
        }
    }
    for (const auto& left : mol.atoms()) {
        for (const auto& right : mol.atoms()) {
            pair_check(left.Z, right.Z);
        }
    }

    std::map<RecordKey, const WSImage*> records;
    std::vector<std::vector<double>> ownership(
        static_cast<std::size_t>(natoms),
        std::vector<double>(static_cast<std::size_t>(natoms), 0.0));
    for (int central = 0; central < natoms; ++central) {
        for (const auto& image : topology.cells[central]) {
            if (image.origin < 0 || image.origin >= natoms) {
                throw std::invalid_argument(
                    route_name + " record has an out-of-range origin");
            }
            if (image.origin == central) {
                throw std::invalid_argument(
                    route_name + " does not accept self-image records");
            }
            if (!std::isfinite(image.weight) || image.weight <= 0.0
                || image.ownership_multiplicity < 1
                || std::abs(
                       image.weight
                       - 1.0 / static_cast<double>(image.ownership_multiplicity))
                    > 1.0e-12) {
                throw std::invalid_argument(
                    route_name
                    + " record has inconsistent fractional ownership");
            }
            for (std::size_t axis = 0; axis < 3; ++axis) {
                const int label = image.image_shell_label[axis];
                if (axis >= topology.translations.size() && label != 0) {
                    throw std::invalid_argument(
                        route_name + " record has an invalid image-shell label");
                }
            }
            if (!image.disp.allFinite() || image.disp.norm() <= 1.0e-12) {
                throw std::invalid_argument(
                    route_name
                    + " record displacement must be finite and nonzero");
            }
            const auto& central_atom = mol.atoms()[central];
            const auto& origin_atom = mol.atoms()[image.origin];
            Eigen::Vector3d expected;
            for (int axis = 0; axis < 3; ++axis) {
                expected[axis] = origin_atom.xyz[axis] - central_atom.xyz[axis];
            }
            expected += shell_translation(
                image.image_shell_label, topology.translations);
            if ((expected - image.disp).norm() > geometry_tolerance) {
                throw std::invalid_argument(
                    route_name + " record displacement is stale for the molecule");
            }
            if (!records.emplace(record_key(central, image), &image).second) {
                throw std::invalid_argument(
                    route_name + " topology contains a duplicate discrete record");
            }
            ownership[central][image.origin] += image.weight;
        }
    }
    for (int central = 0; central < natoms; ++central) {
        for (int origin = 0; origin < natoms; ++origin) {
            const double expected = central == origin ? 0.0 : 1.0;
            if (std::abs(ownership[central][origin] - expected) > 1.0e-12) {
                throw std::invalid_argument(
                    route_name
                    + " directed ownership does not cover each distinct "
                      "origin exactly once");
            }
        }
    }
    for (int central = 0; central < natoms; ++central) {
        for (const auto& image : topology.cells[central]) {
            const auto reverse = records.find(reverse_key(central, image));
            if (reverse == records.end()) {
                throw std::invalid_argument(
                    route_name + " topology is missing an exact reverse image");
            }
            if (reverse->second->ownership_multiplicity
                    != image.ownership_multiplicity
                || std::abs(reverse->second->weight - image.weight) > 1.0e-12
                || (reverse->second->disp + image.disp).norm()
                    > geometry_tolerance) {
                throw std::invalid_argument(
                    route_name
                    + " reverse images disagree on geometry or ownership");
            }
        }
    }
}

inline std::vector<ShellKey> collect_shell_keys(const WSTopology& topology) {
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
    return shell_keys;
}

// True when every WS record is the zero-translation image of its origin with
// full weight: the cyclic cluster is then exactly the ordinary molecule and a
// molecular-limit adapter may evaluate the method's molecular driver on the
// same atom set.
inline bool is_trivial_molecular_records(const WSTopology& topology) {
    for (const auto& cell : topology.cells) {
        for (const auto& image : cell) {
            if (image.image_shell_label != ShellKey{0, 0, 0}
                || std::abs(image.weight - 1.0) > 1.0e-12) {
                return false;
            }
        }
    }
    return true;
}

inline std::vector<LatticeCell> shell_lattice_cells(
    const std::vector<ShellKey>& shell_keys,
    const std::vector<Eigen::Vector3d>& translations) {
    std::vector<LatticeCell> lattice_cells;
    lattice_cells.reserve(shell_keys.size());
    for (const auto& label : shell_keys) {
        LatticeCell cell;
        cell.index = Eigen::Vector3i(label[0], label[1], label[2]);
        cell.r_cart = shell_translation(label, translations);
        lattice_cells.push_back(cell);
    }
    return lattice_cells;
}

struct AOInfo {
    int atom;
    int angular_momentum;
};

inline std::vector<AOInfo> ao_info_for_basis(const BasisSet& basis) {
    const auto shells = basis.shells();
    const auto& libint_shells = basis.libint();
    const auto shell2bf = libint_shells.shell2bf();
    std::vector<AOInfo> ao_info(static_cast<std::size_t>(basis.nbasis()));
    for (std::size_t shell = 0; shell < shells.size(); ++shell) {
        for (std::size_t local = 0; local < libint_shells[shell].size(); ++local) {
            ao_info[shell2bf[shell] + local] = {
                shells[shell].atom_index,
                shells[shell].l,
            };
        }
    }
    return ao_info;
}

// Canonical orthogonalization of the cyclic overlap matrix. Peintinger and
// Bredow, J. Comput. Chem. 35 (2014), eq. 5 and the "critical eigenvalues
// of the overlap matrix" discussion: the WS-weighted cyclic overlap S_CCM
// is a known-finite failure mode of the C-point approximation for small
// clusters and cells whose atoms sit on Wigner-Seitz boundaries. The
// prescribed remedy is to screen the non-positive subspace (canonical
// orthogonalization) or enlarge the cluster; when the screening cannot
// represent the occupied manifold the calculation must still fail closed.
//
// Contract:
//   * empty result - every eigenvalue of `overlap` is above `threshold`;
//     the caller keeps the legacy generalized-solve path, preserving
//     bit-for-bit parity for every cell that already passed the
//     positive-definiteness gate.
//   * (n, k) matrix X with k < n - `overlap` has eigenvalues at or below
//     `threshold`; X = U_kept Lambda_kept^{-1/2} maps the generalized
//     problem H C = S C E onto the ordinary problem (X^T H X) C' = C' E
//     with C = X C'. Modes with eigenvalues <= threshold are screened,
//     exactly the paper's prescription.
//   * throws - the kept subspace is empty or smaller than the occupied
//     manifold (n_kept < n_occ), so no physical solve is possible.
inline Eigen::MatrixXd canonical_orthogonalizer(
    const Eigen::MatrixXd& overlap,
    double threshold,
    int n_occ,
    const std::string& route_name) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(overlap);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            route_name + " overlap eigensolve failed");
    }
    const Eigen::VectorXd& eigenvalues = solver.eigenvalues();
    if (eigenvalues.minCoeff() > threshold) {
        return Eigen::MatrixXd();
    }
    int n_kept = 0;
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) > threshold) ++n_kept;
    }
    if (n_kept < n_occ) {
        throw std::runtime_error(
            route_name + " overlap matrix is not positive definite");
    }
    Eigen::MatrixXd ortho(overlap.rows(), n_kept);
    int column = 0;
    for (Eigen::Index i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues(i) <= threshold) continue;
        ortho.col(column) = solver.eigenvectors().col(i)
            / std::sqrt(eigenvalues(i));
        ++column;
    }
    return ortho;
}

// One canonical-orthogonalized solve (ordinary eigenproblem in the
// X-transformed basis, coefficients back-rotated to the original basis).
struct CanonicalEigensolution {
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd coefficients;  // columns in the original AO basis
};

// Screening can leave exactly enough orbitals for the occupied manifold,
// but no LUMO. AO dimension is not the returned spectrum dimension (#151).
// Keep this separate from canonical_orthogonalizer's occupied-space guard:
// a missing frontier is unavailable, not a zero gap that smearing can waive.
inline double finite_torus_homo_lumo_gap(
    const Eigen::VectorXd& energies, int n_occ) {
    if (n_occ <= 0 || n_occ >= energies.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return energies(n_occ) - energies(n_occ - 1);
}

inline CanonicalEigensolution solve_canonical_orthogonalized(
    const Eigen::MatrixXd& hamiltonian,
    const Eigen::MatrixXd& ortho,
    const std::string& route_name) {
    const Eigen::MatrixXd transformed =
        ortho.transpose() * hamiltonian * ortho;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(transformed);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            route_name + " canonical-orthogonalized eigensolve failed");
    }
    return {solver.eigenvalues(), ortho * solver.eigenvectors()};
}

// DFTB-family supercell assembly: WS-weighted overlap and H0 matrices for
// the whole cluster, using explicit shell-label overlap blocks. Verbatim
// arithmetic from the validated DFTB0-SECCM adapter.
inline std::pair<Eigen::MatrixXd, Eigen::MatrixXd> assemble_weighted_dftb_h0(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    const BasisSet& basis,
    int n_basis) {
    const std::vector<ShellKey> shell_keys = collect_shell_keys(topology);
    const std::vector<LatticeCell> lattice_cells =
        shell_lattice_cells(shell_keys, topology.translations);

    PeriodicSystem system;
    system.dim = static_cast<int>(topology.translations.size());
    system.unit_cell = mol.atoms();
    system.charge = mol.charge();
    system.multiplicity = mol.multiplicity();
    for (std::size_t axis = 0; axis < topology.translations.size(); ++axis) {
        system.lattice.col(static_cast<Eigen::Index>(axis)) =
            topology.translations[axis];
    }
    const auto overlap_blocks =
        compute_overlap_lattice_explicit(basis, system, lattice_cells);
    std::map<ShellKey, std::size_t> block_by_label;
    for (std::size_t index = 0; index < shell_keys.size(); ++index) {
        block_by_label.emplace(shell_keys[index], index);
    }

    const std::vector<AOInfo> ao_info = ao_info_for_basis(basis);

    Eigen::MatrixXd overlap = Eigen::MatrixXd::Zero(n_basis, n_basis);
    Eigen::MatrixXd hamiltonian = Eigen::MatrixXd::Zero(n_basis, n_basis);
    const Eigen::MatrixXd& home_overlap =
        overlap_blocks.blocks[block_by_label.at({0, 0, 0})];
    const auto& atoms = mol.atoms();
    for (int mu = 0; mu < n_basis; ++mu) {
        const int central = ao_info[static_cast<std::size_t>(mu)].atom;
        for (int nu = 0; nu < n_basis; ++nu) {
            const int origin = ao_info[static_cast<std::size_t>(nu)].atom;
            if (central == origin) {
                overlap(mu, nu) = home_overlap(mu, nu);
                if (mu == nu) {
                    double onsite = params.on_site_energy(
                        atoms[central].Z,
                        ao_info[static_cast<std::size_t>(mu)].angular_momentum);
                    hamiltonian(mu, nu) = onsite == 0.0 ? -0.5 : onsite;
                }
                continue;
            }
            for (const auto& image : topology.cells[central]) {
                if (image.origin != origin) continue;
                const auto block_index =
                    block_by_label.at(image.image_shell_label);
                const double s = overlap_blocks.blocks[block_index](mu, nu);
                overlap(mu, nu) += image.weight * s;
                const double average = 0.5
                    * (params.average_on_site(atoms[central].Z)
                       + params.average_on_site(atoms[origin].Z));
                hamiltonian(mu, nu) +=
                    image.weight * 0.5 * params.kappa() * s * average;
            }
        }
    }
    return {overlap, hamiltonian};
}

inline std::vector<Eigen::Vector3d> seccm_atom_coords(const Molecule& mol) {
    std::vector<Eigen::Vector3d> coords;
    coords.reserve(mol.atoms().size());
    for (const auto& atom : mol.atoms()) {
        coords.emplace_back(atom.xyz[0], atom.xyz[1], atom.xyz[2]);
    }
    return coords;
}

inline ImageRecordSource seccm_image_records(
    const WSTopology& topology,
    const std::vector<Eigen::Vector3d>& atom_coords) {
    return [&topology, &atom_coords](const ImageRecordSink& sink) {
        for (std::size_t central = 0; central < topology.cells.size();
             ++central) {
            const int a = static_cast<int>(central);
            for (const auto& image : topology.cells[central]) {
                ImageRecord record;
                record.a = a;
                record.b = image.origin;
                record.shift = image.disp
                    - (atom_coords[static_cast<std::size_t>(image.origin)]
                       - atom_coords[static_cast<std::size_t>(a)]);
                record.weight = image.weight;
                sink(record);
            }
        }
    };
}

// Ownership weight of the (bra atom a, ket atom b, cell) block, in exactly
// the convention the supercell overlap assembly above uses: a same-atom pair
// takes the home block at unit weight, and every other pair takes each of its
// Wigner-Seitz records at that record's fractional weight.  This is the
// GFN2PairWeight the multipole lattice sums consume, so the AES integrals see
// the same image inventory as S and H0 rather than the home-cell molecule
// (issue #348).
// Ewald width conventions, unchanged from the kernels this replaces: the
// shared Madelung machinery's volume-adaptive 3-D width, the validated
// 0.85 sqrt(pi/A) slab width (indo::_madkonst_2d), and the wire 4/L
// (ewald_1d.h wire_madkonst_1d).
inline double seccm_ewald_alpha(const WSTopology& topology) {
    constexpr double kPi = 3.14159265358979323846;
    switch (topology.translations.size()) {
        case 3: {
            Eigen::Matrix3d lattice;
            lattice.col(0) = topology.translations[0];
            lattice.col(1) = topology.translations[1];
            lattice.col(2) = topology.translations[2];
            return std::sqrt(kPi) / std::cbrt(std::abs(lattice.determinant()));
        }
        case 2: {
            const double area = topology.translations[0]
                .cross(topology.translations[1]).norm();
            return 0.85 * std::sqrt(kPi) / std::sqrt(area);
        }
        case 1:
            return 4.0 / topology.translations[0].norm();
        default:
            throw std::runtime_error(
                "SECCM ewald_gamma requires a 1-D, 2-D, or 3-D "
                "cyclic topology");
    }
}

// One charge site per atom, hardness U: the SCC-DFTB site list of the
// shared kernel.
inline std::vector<GammaSite> dftb_gamma_sites(
    const Molecule& mol, const SemiempiricalParameters& params) {
    std::vector<GammaSite> sites;
    const auto& atoms = mol.atoms();
    sites.reserve(atoms.size());
    for (std::size_t a = 0; a < atoms.size(); ++a) {
        double u = params.hubbard_u(atoms[a].Z);
        if (u <= 0.0) u = 0.4;
        GammaSite site;
        site.atom = static_cast<int>(a);
        site.hardness = u;
        sites.push_back(site);
    }
    return sites;
}

// Self-consistent periodic SCC-DFTB gamma for a cyclic SECCM topology:
//
//   Gamma_AB = Phi_Ewald(R_B - R_A)
//            + sum_{WS records} w [gamma_AB(r) - 1/r]
//            + delta_{AB} U_A
//
// the same arithmetic as the GFN2-SECCM kernel and as the Gamma-periodic
// SCC-DFTB driver, differing only in the record weights.  This is the
// embedded route: the Coulomb tail rides the converged Ewald series
// instead of being added afterwards as a Madelung correction, and the
// remainder is what the Wigner-Seitz inventory truncates.
//
// Whether that truncation has a thermodynamic limit is a property of
// `form`, not of the boundary (#211, #425): the Klopman-Ohno remainder
// falls off as R^-3 with a pair-dependent amplitude, the Elstner remainder
// exponentially.
inline Eigen::MatrixXd ewald_dftb_gamma(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    ShellGammaForm form,
    double alpha = 0.0) {
    const std::vector<Eigen::Vector3d> coords = seccm_atom_coords(mol);
    ShellGammaSpec spec;
    spec.form = form;
    spec.ko_average = KlopmanOhnoAverage::InverseHardnessMean;
    const EwaldCoulombKernel ewald(
        topology.translations,
        alpha > 0.0 ? alpha : seccm_ewald_alpha(topology));
    return build_periodic_shell_gamma(
        dftb_gamma_sites(mol, params), coords, spec, ewald,
        seccm_image_records(topology, coords));
}

inline Eigen::MatrixXd weighted_dftb_gamma_klopman_ohno(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology);

// Unembedded DFTB-family WS-weighted supercell gamma: every other atom
// contributes through its single nearest image inside the central atom's WS
// cell, fractionally weighted. gamma(a, a) = U_a (on-site Hubbard).  No
// long-range tail at all -- see ewald_dftb_gamma for the embedded route.
inline Eigen::MatrixXd weighted_dftb_gamma(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology,
    ShellGammaForm form = ShellGammaForm::KlopmanOhno) {
    if (form == ShellGammaForm::Elstner) {
        const std::vector<Eigen::Vector3d> coords = seccm_atom_coords(mol);
        ShellGammaSpec spec;
        spec.form = form;
        spec.ko_average = KlopmanOhnoAverage::InverseHardnessMean;
        Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(
            static_cast<Eigen::Index>(coords.size()),
            static_cast<Eigen::Index>(coords.size()));
        const auto sites = dftb_gamma_sites(mol, params);
        for (std::size_t a = 0; a < sites.size(); ++a) {
            gamma(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(a)) =
                shell_gamma_onsite(spec, sites[a].hardness, sites[a].hardness);
        }
        for (std::size_t central = 0; central < sites.size(); ++central) {
            for (const auto& image : topology.cells[central]) {
                const double r = image.disp.norm();
                if (r < 1.0e-12) continue;
                gamma(static_cast<Eigen::Index>(central),
                      static_cast<Eigen::Index>(image.origin)) +=
                    image.weight
                    * shell_gamma_pair(
                        spec, sites[central].hardness,
                        sites[static_cast<std::size_t>(image.origin)].hardness,
                        r);
            }
        }
        return gamma;
    }
    return weighted_dftb_gamma_klopman_ohno(mol, params, topology);
}

inline Eigen::MatrixXd weighted_dftb_gamma_klopman_ohno(
    const Molecule& mol,
    const SemiempiricalParameters& params,
    const WSTopology& topology) {
    const auto& atoms = mol.atoms();
    const int n = static_cast<int>(atoms.size());
    Eigen::MatrixXd gamma = Eigen::MatrixXd::Zero(n, n);
    for (int a = 0; a < n; ++a) {
        double Ua = params.hubbard_u(atoms[a].Z);
        if (Ua <= 0.0) Ua = 0.4;
        gamma(a, a) = Ua;
    }
    for (int central = 0; central < n; ++central) {
        double Ua = params.hubbard_u(atoms[central].Z);
        if (Ua <= 0.0) Ua = 0.4;
        for (const auto& image : topology.cells[central]) {
            const int origin = image.origin;
            double Ub = params.hubbard_u(atoms[origin].Z);
            if (Ub <= 0.0) Ub = 0.4;
            const double eta = 0.5 * (1.0 / Ua + 1.0 / Ub);
            const double R = image.disp.norm();
            gamma(central, origin) +=
                image.weight / std::sqrt(R * R + eta * eta);
        }
    }
    return gamma;
}

}  // namespace detail
}  // namespace seccm
}  // namespace semiempirical
}  // namespace vibeqc
