// Periodic-Γ COSX (RIJCOSX) JKBuilder.
//
// Wraps the molecular COSX K-build for Γ-point periodic SCF by
// using a periodic Becke grid (grid points folded into the unit
// cell) while keeping the same DF-J and seminumerical K kernels.
// The K-build sums exchange over the truncated image-cell list
// (``direct_lattice_cells(system, opts.cutoff_bohr)``) — see the
// ``image_cells`` parameter of ``compute_cosx_k`` (cosx.hpp) for
// the lattice-summed form (M3a).
//
// The Γ-point approximation assumes the density matrix is cell-
// diagonal (P(g ≠ 0) ≈ 0), valid for large unit cells (molecular
// crystals, surfaces with vacuum).  Full multi-k COSX with Bloch
// phases over lattice translations is deferred to a future phase.

#pragma once

#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "basis.hpp"
#include "grid.hpp"
#include "jk_builder.hpp"
#include "lattice_sum.hpp"
#include "periodic.hpp"

namespace vibeqc {

// Γ-only periodic COSX JKBuilder.  The DF Coulomb piece uses the
// molecular DensityFitting (valid because the density is Γ-point).
// The COSX K piece uses a periodic Becke grid built from
// ``build_grid_periodic`` (grid points in the unit cell with
// periodic boundary conditions).
//
// ``aux_basis`` is the auxiliary basis for DF-J (same as molecular
// RIJCOSX).  ``cosx_grid_opts`` defaults to the standard COSX tier;
// pass a finer grid for production.
std::unique_ptr<JKBuilder> make_periodic_gamma_cosx_jk_builder(
    const BasisSet& basis,
    const BasisSet& aux_basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const GridOptions& cosx_grid_opts = GridOptions());

// Γ-only periodic RIJCOSX JKBuilder for tight cells.  Uses the same
// periodic COSX K-build as ``make_periodic_gamma_cosx_jk_builder``,
// but replaces the molecular DensityFitting J-build with a periodic-
// correct GDF J build from a pre-constructed Lpq tensor (e.g. from
// Python ``build_lpq_compcell`` or ``build_lpq_native_fft``).
//
// ``lpq`` is a list of (n_orb, n_orb) symmetric matrices, one per
// kept auxiliary function.  The tensor is built once in Python and
// passed across the pybind11 boundary.  Shape: n_kept × n_orb × n_orb.
//
// The COSX-K piece is identical to the molecular-limit builder:
// periodic Becke grid, with the exchange summed over the truncated
// image-cell list (M3a single-lattice-sum model — adds the home-bra
// → image-ket exchange class; Γ-folded GDF exchange parity is still
// open, see handovers/HANDOVER_RIJCOSX_M3A.md). No exxdiv='ewald' Madelung
// shift is applied — COSX-K is entirely real-space (direct-truncated
// lattice sum) and does not have a G=0 mode that needs correcting.
std::unique_ptr<JKBuilder> make_periodic_tight_cosx_jk_builder(
    const BasisSet& basis,
    const std::vector<Eigen::MatrixXd>& lpq,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const GridOptions& cosx_grid_opts = GridOptions());

}  // namespace vibeqc
