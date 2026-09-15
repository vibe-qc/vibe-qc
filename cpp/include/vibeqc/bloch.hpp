// k-space machinery for periodic calculations.
//
// Bloch sum. For a real-space periodic matrix M(g) stored as a
// LatticeMatrixSet, the k-space matrix is
//
//   M(k) = Σ_g e^{i k · g} M(g)
//
// in Cartesian (k, g) units of bohr⁻¹ and bohr. This works unchanged in any
// dimensionality — components of k along vacuum directions are zero and
// their contribution drops out.
//
// k-point mesh. Monkhorst–Pack with optional symmetry-reduction to the
// irreducible Brillouin zone via spglib.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <complex>
#include <vector>

#include "lattice_sum.hpp"
#include "periodic.hpp"

namespace vibeqc {

using ComplexMatrix = Eigen::Matrix<std::complex<double>,
                                    Eigen::Dynamic, Eigen::Dynamic>;
using ComplexVector = Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1>;

// Single-k Bloch sum of a real-space lattice matrix. ``k_cart`` in bohr⁻¹.
ComplexMatrix bloch_sum(const LatticeMatrixSet& real_space,
                        const Eigen::Vector3d& k_cart);

// k-point mesh. Lists of Cartesian k-vectors and integration weights.
//
// ``ir_mapping`` (if non-empty) has one entry per full-mesh grid point and
// maps it to its representative index in ``kpoints``. When symmetry is not
// used, ir_mapping is left empty and weights are uniform.
struct BlochKMesh {
    std::array<int, 3> mesh = {1, 1, 1};
    std::array<int, 3> is_shift = {0, 0, 0};

    std::vector<Eigen::Vector3d> kpoints;  // Cartesian, bohr⁻¹
    std::vector<double> weights;           // sum to 1
    std::vector<int> ir_mapping;           // empty if full mesh

    std::size_t size() const noexcept { return kpoints.size(); }
};

// Monkhorst–Pack mesh over the first ``dim`` reciprocal-lattice axes of
// ``system.reciprocal_lattice()``. Non-periodic directions contribute a
// single k-point at 0 regardless of mesh[i].
//
// If ``use_symmetry`` is true, spglib is used to reduce the mesh to the
// irreducible Brillouin zone (requires system.symmetry to be populated;
// throws otherwise). Otherwise the full mesh is returned with equal
// weights.
//
// ``is_shift[i] ∈ {0, 1}`` — 1 shifts the grid by half a step along axis i,
// putting the points at (m + ½)/N instead of m/N.
//
// The half-step offset is the EVEN-mesh case, not the odd one. Monkhorst &
// Pack, Phys. Rev. B 13, 5188 (1976), Eq. (3) places the classical points at
// u_r = (2r − q − 1)/(2q), r = 1…q: for odd q that set contains 0, so the
// classical mesh is already Γ-centred and equals is_shift = 0 here; for even
// q it does not, and the classical mesh equals is_shift = 1. vibe-qc's own
// default is Γ-centred at every N (is_shift = {0, 0, 0}), which is a
// different convention from ASE/GPAW at even N — see
// tests/test_kmesh_convention_provenance.py.
//
// The exact integer addressing behind the mesh (doubled modular addresses,
// crystal-momentum conservation, the reciprocal-lattice wrap) lives in
// kmesh_address.hpp; this function builds its unreduced grid through it.
BlochKMesh monkhorst_pack(const PeriodicSystem& system,
                          std::array<int, 3> mesh,
                          std::array<int, 3> is_shift = {0, 0, 0},
                          bool use_symmetry = false);

// Diagonalize a single-k (generally non-Hermitian-by-construction but in
// exact arithmetic Hermitian) Fock or core matrix in the non-orthogonal
// basis with metric S(k). Returns band energies (real, ascending) and
// complex eigenvector matrix C with F·C = S·C·diag(ε). Standard approach:
// S = U·diag(s)·U†, X = U·diag(s^{-½}); diagonalize X†·F·X.
//
// Throws if any eigenvalue of S is below ``lindep_threshold`` (default
// 1e-9); this typically signals a too-small cell for the basis.
struct BandDiag {
    Eigen::VectorXd energies;   // size nbf
    ComplexMatrix coefficients; // nbf × nbf, columns = MOs in AO basis
};
BandDiag diagonalize_bloch(const ComplexMatrix& F_k,
                           const ComplexMatrix& S_k,
                           double lindep_threshold = 1.0e-9);

}  // namespace vibeqc
