#pragma once
// Native C++ Gaussian multipole-moment integral engine.
//
// POLIPO (POst-LIbint POlymoment) is a self-contained alternative to
// the vendored libint for computing Cartesian and spherical multipole
// moments of Gaussian shell pairs.  It eliminates the libint HRR-gate
// patch (removing `l1>0 && l2>0` gate for `sphemultipole`) and
// supports L up to 12 (beyond libint's current L≤5 limit).
//
// Features:
//   - Recurrence-based Cartesian moment evaluation (McMurchie-Davidson
//     Hermite expansion for the 1/r^L kernel, or direct Rys quadrature
//     for the Gaussian product).
//   - Real solid-harmonic conversion to spherical moments via the
//     Cartesian→spherical pseudoinverse (same convention as libint).
//   - Shell-pair recurrence for O(L^3) scaling instead of O(L^4).
//   - OpenMP parallelisation over shell pairs.
//   - Selectable via multipole_engine="polipo" runtime parameter.
//
// References
// ----------
// Pisani, Dovesi, and Roetti (1988), Ch. II.4c - periodic bipolar expansion.
// Helgaker, Jørgensen & Olsen, *Molecular Electronic-Structure Theory*
//   (2000), Ch. 9 — Cartesian Gaussian integrals + Hermite recurrence.
// Pisani-Dovesi (1980), Sec. 4 - adjoined diffuse s-Gaussian convention;
// standard Gaussian-product and binomial moment-shift identities.
//
// Architecture
// ------------
// This file declares the types and entry points.  The implementation
// lives in `polipo_moments.cpp`.  The Python binding surface mirrors
// the existing `compute_multipole_moments_lattice` function so POLIPO
// can be dropped in as a libint replacement with no Python-side changes.

#include <cstddef>
#include <vector>
#include <array>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>

namespace vibeqc {

// ---------------------------------------------------------------------------
//  Shell descriptor
// ---------------------------------------------------------------------------

/// One contracted Gaussian shell.
struct PolipoShellInfo {
    int l;                          ///< angular momentum (0=s, 1=p, …)
    bool pure;                      ///< true = spherical harmonic, false = Cartesian
    std::array<double, 3> origin;   ///< shell centre (bohr)
    std::vector<double> exponents;  ///< primitive exponents
    std::vector<double> coeffs;     ///< contraction coefficients (unnormalised)
    int bf_offset;                  ///< first basis function index
    int n_bf;                       ///< number of basis functions (2*l+1 for pure)
};

// ---------------------------------------------------------------------------
//  Lattice cell descriptor
// ---------------------------------------------------------------------------

/// One lattice cell for periodic moment computation.
struct PolipoCellInfo {
    std::array<double, 3> r_cart;   ///< cell vector (bohr)
    std::array<int, 3> index;       ///< integer lattice index
};

// ---------------------------------------------------------------------------
//  Multipole result set
// ---------------------------------------------------------------------------

/// Per-cell, per-component shell-pair multipole moments.
/// Mirrors the layout of ``LatticeMultipoleSet`` in the libint path.
struct PolipoMultipoleSet {
    int nbf;                        ///< number of basis functions
    int L_max;                      ///< maximum multipole order
    bool spherical;                 ///< true = real solid harmonics, false = Cartesian
    std::vector<PolipoCellInfo> cells;
    std::array<double, 3> origin;   ///< expansion origin (bohr)
    // blocks[c][comp] = (nbf, nbf) column-major moment matrix
    std::vector<std::vector<Eigen::MatrixXd>> blocks;
};

// ---------------------------------------------------------------------------
//  Computation options
// ---------------------------------------------------------------------------

struct PolipoOptions {
    double cutoff_bohr = 5.0;       ///< lattice sum cutoff
    int L_max = 4;                  ///< maximum multipole order (0-12)
    bool spherical = false;         ///< true -> spherical harmonics
    bool use_ryz = false;           ///< true -> Rys quadrature (default: Hermite)
    int num_threads = 0;            ///< 0 = use OpenMP default
};

// ---------------------------------------------------------------------------
//  Entry point — compute shell-pair multipole moments on a lattice
// ---------------------------------------------------------------------------

/// Compute Cartesian or spherical multipole moments for all shell pairs
/// on a periodic lattice.
///
/// Parameters
/// ----------
/// shells : vector of PolipoShellInfo
///     Basis set shell descriptors.
/// cells : vector of PolipoCellInfo
///     Lattice cells within the cutoff.
/// opts : PolipoOptions
///     Computation options (L_max, spherical flag, etc.).
///
/// Returns
/// -------
/// PolipoMultipoleSet with per-cell, per-component moment matrices.
PolipoMultipoleSet compute_polipo_moments_lattice(
    const std::vector<PolipoShellInfo>& shells,
    const std::vector<PolipoCellInfo>& cells,
    const PolipoOptions& opts);

// ---------------------------------------------------------------------------
//  Low-level — single shell-pair Cartesian moment
// ---------------------------------------------------------------------------

/// Compute Cartesian multipole moments of a single shell-pair product
/// distribution expanded around *origin*.
///
/// The integral is:
///   M_{ijk}^{ab} = ∫ φ_a(r) φ_b(r - R_cell) (x - ox)^i (y - oy)^j (z - oz)^k dr
///
/// where i+j+k ≤ L_max.
///
/// Parameters
/// ----------
/// sh_a, sh_b : shell descriptors (must have at least one primitive).
/// R_cell : lattice translation vector for shell b (bohr).
/// origin : expansion centre (bohr).
/// L_max : maximum multipole order.
///
/// Returns
/// -------
/// Moments as (n_comp, n_bf_a, n_bf_b) array where n_comp = (L_max+1)(L_max+2)(L_max+3)/6
/// and the Cartesian ordering matches libint's convention.
Eigen::Tensor<double, 3> compute_polipo_shell_pair_moments(
    const PolipoShellInfo& sh_a,
    const PolipoShellInfo& sh_b,
    const std::array<double, 3>& R_cell,
    const std::array<double, 3>& origin,
    int L_max);

} // namespace vibeqc
