// Phase 14 — Effective Core Potential (ECP) integral interface,
// wrapping libecpint (https://github.com/robashaw/libecpint).
//
// Phase 14a ships build infrastructure + a callable wrapper that
// reports library version and confirms libecpint links cleanly into
// vibeqc_core.
//
// Phase 14b (this file) ships ECP matrix elements via the built-in
// libecpint XML library (Stuttgart-Köln "ecpNNmdf" small-core ECPs
// + LANL2DZ). The interface produces a real symmetric V_ECP matrix
// in the (spherical) AO basis ready to add into Hcore. Cartesian →
// spherical transform uses libint's ``solidharmonics::tform_cols``
// / ``tform_rows`` per-shell-pair primitives, since vibe-qc forces
// ``set_pure(true)`` on every libint::Shell.
//
// SCF wiring (14c), CRYSTAL-format ECP-block parser (14d), and
// validation against published reference values (14e) follow.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "basis.hpp"
#include "lattice_sum.hpp"
#include "molecule.hpp"
#include "periodic.hpp"

namespace vibeqc {

// ---- Inline ECP primitive data (Phase 14g) -----------------------------
struct ECPPrimitiveBlock {
    int n_primitive;
    std::vector<double> exponents;
    std::vector<double> coefficients;
    std::vector<int> ams;
    std::vector<int> ns;
};


// Returns the linked libecpint version. Used as a smoke-test that the
// dependency is wired in correctly.
std::string libecpint_version();

// Diagnostic of the linked radial quadrature, independent of ECP assembly.
std::pair<double, bool> libecpint_gaussian_quadrature(
    int npoints, double exponent, double center, double tolerance);

// Per-atom ECP placement: the atom's atomic number (used as a key into
// libecpint's XML library) and its Cartesian position (bohr).
struct ECPCenter {
    int Z;
    std::array<double, 3> xyz;
};

// Look up the number of core electrons replaced by each ECP atom in a
// given library. Returns a map ``{Z -> ncore}`` covering every charge
// in ``charges`` that appears in the library. Used by the SCF drivers
// to compute the effective nuclear-attraction integrals
// ``V_n(Z_eff)`` and effective nuclear repulsion when ECPs are
// present (otherwise we double-count the core's nuclear contribution
// once via libint's bare V_n and again via the ECP).
std::map<int, int> ecp_core_electrons(
    const std::vector<int>& charges,
    const std::string& library_name,
    const std::string& share_dir = "");

// Compute the bare nuclear-attraction matrix
//   V_{μν} = Σ_A −q_A · ⟨χ_μ | 1/|r − R_A| | χ_ν⟩
// with caller-supplied charges ``q_A``. Used by SCF drivers to apply
// effective charges ``Z_A − ncore_A`` when ECPs replace core
// electrons. Reduces to ``compute_nuclear(basis, mol)`` exactly when
// ``charges = {a.Z for a in mol.atoms()}``.
Eigen::MatrixXd compute_nuclear_with_charges(
    const BasisSet& basis,
    const std::vector<std::array<double, 3>>& positions,
    const std::vector<double>& charges);

// One-electron quantities for an ECP-aware SCF.
// ``V`` is the bare nuclear-attraction matrix using effective charges
// ``Z_eff = Z − ncore`` on every ECP-bearing atom; ``V_ecp`` is the
// libecpint ECP matrix. Either is the full all-electron value when
// ``ecp_centers`` is empty (``V_ecp`` is then the zero matrix).
// ``E_nuc`` is the effective nuclear repulsion ``Σ_{A<B} Z_eff_A
// Z_eff_B / R_AB``.
// ``total_ncore`` is the sum of replaced-core electrons across every
// ECP centre — the SCF subtracts it from ``mol.n_electrons()`` to get
// the valence electron count that fills the orbital basis. Zero when
// ``ecp_centers`` is empty.
struct ECPHcore {
    Eigen::MatrixXd V;
    Eigen::MatrixXd V_ecp;
    double E_nuc;
    int total_ncore = 0;
};

// Select and validate the mutually-exclusive molecular ECP input routes used
// by RHF/UHF/RKS/UKS before they build any one-electron integrals.  Inline
// primitive metadata must never fall through to the XML/all-electron path:
// doing so would silently discard a requested operator.  Likewise, XML and
// inline inputs in the same option object are ambiguous and fail closed.
enum class MolecularECPInput {
    NONE,
    XML_LIBRARY,
    INLINE_PRIMITIVES,
};

MolecularECPInput validate_molecular_ecp_dispatch(
    const std::vector<ECPCenter>& xml_centers,
    const std::string& xml_library,
    const std::vector<ECPPrimitiveBlock>& primitive_blocks,
    const std::vector<std::array<double, 3>>& primitive_centers,
    const std::vector<double>& effective_charges,
    int total_ncore,
    const std::string& route);

ECPHcore compute_ecp_one_electron(
    const BasisSet& basis,
    const Molecule& mol,
    const std::vector<ECPCenter>& ecp_centers,
    const std::string& ecp_library);

ECPHcore compute_ecp_one_electron_from_primitives(
    const BasisSet& basis,
    const Molecule& mol,
    const std::vector<std::array<double, 3>>& ecp_centers,
    const std::vector<ECPPrimitiveBlock>& primitives,
    const std::vector<double>& effective_charges,
    int total_ncore);

// Compute the AO-basis ECP matrix
//   V_ECP_{μν} = ⟨χ_μ | V_ECP | χ_ν⟩
// using libecpint's built-in ECP library (XML files installed under
// ``share_dir`` — defaults to the vendored
// ``third_party/libecpint/install/share/libecpint``).
//
// Parameters
// ----------
// basis
//     The vibe-qc :class:`BasisSet`. ``set_pure(true)`` is assumed
//     (vibe-qc forces this); the integrals are returned in the
//     spherical basis.
// ecp_centers
//     One entry per atom that carries an ECP. Atoms not in this list
//     use the bare (full-electron) one-electron Hamiltonian
//     unchanged.
// library_name
//     libecpint XML library name. Common choices:
//       - "ecp10mdf" (Z = 11–18, Stuttgart-Köln 10-electron core)
//       - "ecp28mdf" (Z = 29–36)
//       - "ecp46mdf" (Z = 47–54)
//       - "ecp60mdf" (Z = 57–71 lanthanides)
//       - "ecp78mdf" (Z = 79–86)
//       - "lanl2dz"  (Hay-Wadt LANL2DZ)
//     The XML file ``library_name + ".xml"`` must exist under
//     ``share_dir``.
// share_dir
//     Path to libecpint's XML directory. When empty, defaults to the
//     vendored path baked at build time
//     (LIBECPINT_VENDORED_SHARE_DIR).
//
// Returns
// -------
// Real symmetric ``(n_bf, n_bf)`` matrix in the spherical AO basis.
Eigen::MatrixXd compute_ecp_matrix(
    const BasisSet& basis,
    const std::vector<ECPCenter>& ecp_centers,
    const std::string& library_name = "ecp10mdf",
    const std::string& share_dir = "");

// Per-atom effective-charges vector ``Z_eff_A = Z_A − n_core_A``,
// matching the SCF's nuclear-attraction / nuclear-repulsion convention
// when ECPs replace core electrons (compute_ecp_one_electron). ECPs
// are matched to atoms by (Z, position) — atoms not covered by any ECP
// center keep their bare ``Z`` (i.e. the helper produces the
// all-electron charge vector when ``ecp_centers`` is empty).
std::vector<double> ecp_effective_charges(
    const Molecule& mol,
    const std::vector<ECPCenter>& ecp_centers,
    const std::string& library_name,
    const std::string& share_dir = "");

// Analytic nuclear-coordinate gradient of E_ECP = tr(D · V_ECP).
// Differentiates V_ECP via libecpint's first-derivative API
// (``init(1)`` + ``compute_first_derivs``); each (n_cart × n_cart)
// derivative block is transformed Cartesian → spherical (matching the
// V_ECP build) and contracted with the density. Returns an
// ``(n_atoms, 3)`` matrix in Hartree / bohr. Empty ``ecp_centers``
// returns a zero matrix.
//
// libecpint's atom order is determined by first appearance in the
// gaussian basis (clustered by shell coordinates). vibe-qc's basis
// construction emits shells per atom in ``mol.atoms()`` order, so the
// derivative index maps 1:1 to ``mol.atoms()`` indices.
Eigen::MatrixXd compute_ecp_gradient_contribution(
    const BasisSet& basis,
    const Molecule& mol,
    const std::vector<ECPCenter>& ecp_centers,
    const Eigen::MatrixXd& D,
    const std::string& library_name = "ecp10mdf",
    const std::string& share_dir = "");

// Inline-primitive counterpart of ``compute_ecp_gradient_contribution``
// (#574). libecpint's derivative machinery does not depend on how the ECP
// basis was supplied: ``init(deriv)`` assigns ECP -> atom ids purely by
// centre position (api.cpp ``ECPIntegrator::init``, 1e-4 bohr tolerance)
// and ``compute_first_derivs`` reads only those ids plus the primitive
// data. ``set_ecp_basis`` therefore supports first derivatives exactly as
// ``set_ecp_basis_from_library`` does.
//
// ``ecp_centers`` / ``primitives`` are per-ECP-centre and must be the same
// length; they are the ``ecp_primitive_centers`` / ``ecp_primitive_blocks``
// the SCF ran with. Returns an ``(n_atoms, 3)`` matrix in Hartree / bohr;
// empty ``primitives`` returns a zero matrix.
Eigen::MatrixXd compute_ecp_gradient_contribution_from_primitives(
    const BasisSet& basis,
    const Molecule& mol,
    const std::vector<std::array<double, 3>>& ecp_centers,
    const std::vector<ECPPrimitiveBlock>& primitives,
    const Eigen::MatrixXd& D);

// Compute V_ECP matrix from inline primitive data (Phase 14g).
Eigen::MatrixXd compute_ecp_matrix_from_primitives(
    const BasisSet& basis,
    const std::vector<double>& center_xyz,
    const std::vector<ECPPrimitiveBlock>& primitives);

// Lattice-summed V_ECP for periodic SCF.
LatticeMatrixSet compute_ecp_lattice_from_primitives(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const std::vector<std::array<double, 3>>& home_ecp_centers,
    const std::vector<ECPPrimitiveBlock>& home_primitives);

// Fixed-density derivative of exactly the same AO/projector image domain.
// Each translated shell and ECP projector moves with its owning home atom.
Eigen::MatrixXd ecp_lattice_gradient_contribution_from_primitives(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeMatrixSet& density,
    const LatticeSumOptions& opts,
    const std::vector<std::array<double, 3>>& home_ecp_centers,
    const std::vector<ECPPrimitiveBlock>& home_primitives);

}  // namespace vibeqc
