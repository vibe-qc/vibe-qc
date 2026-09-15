// Reciprocal-space Poisson solver for periodic Coulomb kernels.
//
// Given a charge density ρ(r) sampled on a uniform 3D grid inside a
// (possibly non-orthogonal) unit cell with lattice vectors a_1, a_2,
// a_3, compute the potential
//
//     V(r) = Σ_{R} ∫  K(|r − r' + R|) ρ(r') dr'
//
// on the same grid by FFT-based convolution. The sum runs over all
// direct-lattice vectors R; V(r) is the true periodic lattice sum, not
// a truncated real-space approximation.
//
// Two kernels are supplied:
//
//   * **erf-screened Coulomb** — K(r) = erf(ω r) / r. This is the
//     *long-range* piece of the Ewald-decomposed 1/r operator and is
//     what Phase 12e-c-3 uses for the long-range Hartree J. Its
//     reciprocal-space form is
//
//         Ṽ(G) = (4π / G²) · exp(-G² / (4 ω²)) · ρ̃(G)   for G ≠ 0
//
//     with Ṽ(G=0) fixed to zero — a neutral periodic cell has no
//     well-defined absolute potential, only differences matter. Users
//     who need a specific gauge can shift V by a constant after the
//     call.
//
//   * **unscreened Coulomb** — K(r) = 1/r. The usual 4π/G² kernel
//     *without* the erf filter. Provided mainly for validation (at
//     ω → ∞ the erf-screened result must agree with this).
//
// The implementation works in fractional grid coordinates and applies
// the full reciprocal-lattice metric to G·G, so orthorhombic,
// monoclinic, and fully triclinic cells use the same FFT kernel. This
// is the substrate for native FFTDF on primitive solid cells.
//
// The FFTW "many" interface would let us batch multiple ρ grids into
// a single plan; we don't use it yet. FFTW plans are cached per grid
// shape inside the implementation so back-to-back calls with the same
// shape (the common pattern during SCF iterations) reuse plans.

#pragma once

#include <Eigen/Dense>
#include <array>
#include <cstddef>

namespace vibeqc {

// A 3D grid stored in row-major (x fastest, then y, then z) order so
// the underlying std::vector<double> layout matches FFTW's natural
// row-major convention. Dimensions n_x × n_y × n_z; total n_x*n_y*n_z
// samples.
struct ScalarField3D {
    std::size_t nx = 0, ny = 0, nz = 0;
    std::vector<double> data;           // size = nx*ny*nz

    void resize(std::size_t x, std::size_t y, std::size_t z) {
        nx = x; ny = y; nz = z;
        data.assign(nx * ny * nz, 0.0);
    }
    double& operator()(std::size_t ix, std::size_t iy, std::size_t iz) {
        return data[(ix * ny + iy) * nz + iz];
    }
    double operator()(std::size_t ix, std::size_t iy, std::size_t iz) const {
        return data[(ix * ny + iy) * nz + iz];
    }
    std::size_t size() const { return data.size(); }
};

// Solve the periodic Poisson equation with the erf-screened Coulomb
// kernel. ``rho`` is the input charge density sampled on a uniform
// grid spanning one unit cell; the returned field lives on the same
// grid and is the potential V(r) in Hartree (if ρ is in e/bohr³ and
// the lattice is in bohr).
//
// Parameters
// ----------
// rho
//     Input density. Passed by value — the input is not modified.
// lattice_bohr
//     3×3 matrix whose columns are a_1, a_2, a_3.
// omega
//     Ewald splitting parameter (1 / bohr). ω → 0 recovers the
//     unscreened kernel (see also :func:`solve_poisson_coulomb`); ω
//     large means very short-ranged, with the long-range V vanishing.
//
// Returned field is zero-mean (the G=0 component is dropped). If the
// caller needs a specific reference potential, add the constant back.
//
// Throws ``std::invalid_argument`` if ``lattice_bohr`` is singular.
ScalarField3D solve_poisson_erf_screened(
    const ScalarField3D& rho,
    const Eigen::Matrix3d& lattice_bohr,
    double omega);

// Unscreened periodic Poisson (K(r) = 1/r). Same conventions as
// :func:`solve_poisson_erf_screened`. Offered for validation: the
// ω → ∞ limit of the erf-screened result must converge to this.
ScalarField3D solve_poisson_coulomb(
    const ScalarField3D& rho,
    const Eigen::Matrix3d& lattice_bohr);

// Return the cell volume in bohr³ given the lattice matrix. Useful
// for integral normalisation in tests.
double cell_volume(const Eigen::Matrix3d& lattice_bohr);

// Evaluate the Hartree energy ½ ∫∫ ρ(r) K(|r-r'|) ρ(r') dr dr' for a
// density on the grid and either kernel:
//
//     E_H = ½ · V_cell / N_grid · Σ_g ρ(r_g) V(r_g)
//
// Convenience for tests — can also be computed by hand via
//     rho.dot(V) * (cell_volume / n_grid_points) * 0.5
//
// in the caller's code.
double hartree_energy(const ScalarField3D& rho,
                       const ScalarField3D& V,
                       double cell_volume_bohr3);

// FFTW3 library version string, e.g. "3.3.10-sse2-avx". Reads the
// linked FFTW's compile-time version macro at module load time, so
// the value reflects the build that was linked (not whatever FFTW
// happens to be on LD_LIBRARY_PATH).
//
// Surfaced through ``vibeqc.banner.library_versions()`` as the
// "fftw3" key, matching the libint / libxc / spglib / libecpint
// pattern. The FFT-Poisson route (and the upcoming GAPW route)
// both depend on FFTW3, so the banner needs to record which build
// is in play (CLAUDE.md § 6).
std::string fftw3_version();

}  // namespace vibeqc
