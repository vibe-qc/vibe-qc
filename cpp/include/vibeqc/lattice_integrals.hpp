// One-electron AO integrals in the periodic (lattice-summed) basis.
//
// Convention. Let χ_μ be an AO in the reference unit cell and let χ_νg be
// the same basis function translated by lattice vector g. Then:
//
//   S_μν(g) = ⟨ χ_μ(r) | χ_ν(r − g) ⟩
//   T_μν(g) = ⟨ χ_μ(r) | −½ ∇² | χ_ν(r − g) ⟩
//   V_μν(g) = Σ_{A, h} −Z_A ⟨ χ_μ(r) | 1/|r − (R_A + h)| | χ_ν(r − g) ⟩
//
// where the A index runs over unit-cell atoms and h runs over lattice
// vectors within a (separately specified) nuclear cutoff. The returned
// LatticeMatrixSet contains one nbf × nbf block per lattice cell g.
//
// These functions are the periodic generalisations of the molecular
// compute_overlap / compute_kinetic / compute_nuclear in integrals.hpp;
// for a system with a single cell (dim=0-equivalent) the g=0 block
// reproduces the molecular result exactly.

#pragma once

#include "basis.hpp"
#include "ewald.hpp"
#include "grid.hpp"
#include "lattice_sum.hpp"
#include "periodic.hpp"

namespace vibeqc {

// The lattice cells the cutoff-driven one-electron builders below actually
// enumerate: the translations reaching any (bra, ket) shell pair whose
// PHYSICAL separation |O_mu - O_nu - g| is within ``cutoff_bohr``, rather
// than those with |g| <= cutoff_bohr. See lattice_pair_cells.hpp for why
// the latter is not translation invariant.
//
// A superset of ``direct_lattice_cells(system, cutoff_bohr)`` and, because
// that function stable-sorts by |r|, an exact extension of it: the plain
// ball's cells come first, in the same order. Callers that must align a
// one-electron matrix with a cell list built elsewhere (the symmetry-reduced
// path, cell-resolved consumers) should build that list from here.
std::vector<LatticeCell> pair_complete_lattice_cells(
    const BasisSet& basis,
    const PeriodicSystem& system,
    double cutoff_bohr);

// Common physical ERI enclosure for native and Python consumers. Pair and
// interaction supports are distinct; consumers align cells by integer key.
std::vector<LatticeCell> physical_eri_lattice_cells(
    const BasisSet& basis, const PeriodicSystem& system,
    double pair_cutoff_bohr, double interaction_cutoff_bohr);

// Exact source labels selected by the shared physical nuclear selector for
// each supplied AO midpoint. Provenance only; no integral or alternate domain.
std::vector<std::vector<std::array<int, 4>>> physical_nuclear_source_indices(
    const PeriodicSystem& system, const Eigen::MatrixXd& midpoints,
    double cutoff_bohr);

LatticeMatrixSet compute_overlap_lattice(const BasisSet& basis,
                                         const PeriodicSystem& system,
                                         const LatticeSumOptions& opts);

LatticeMatrixSet compute_kinetic_lattice(const BasisSet& basis,
                                         const PeriodicSystem& system,
                                         const LatticeSumOptions& opts);

LatticeMatrixSet compute_nuclear_lattice(const BasisSet& basis,
                                         const PeriodicSystem& system,
                                         const LatticeSumOptions& opts);

LatticeMatrixSet compute_nuclear_lattice_with_charges(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const std::vector<double>& effective_charges);

// ---------------------------------------------------------------------------
// Symmetry-reduced (explicit-cell) one-electron integral lattice sums
// ---------------------------------------------------------------------------

// ``pair_cutoff`` bounds the PHYSICAL separation |O_mu - O_nu - g| of each
// term, exactly as the cutoff-driven builders above do. Pass the caller's
// ``cutoff_bohr`` to reproduce ``compute_{overlap,kinetic}_lattice`` on a
// subset of its cells (what the symmetry-reduced path needs); leave it at
// the 0.0 default to compute every pair of every supplied cell.
LatticeMatrixSet compute_overlap_lattice_explicit(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    double pair_cutoff = 0.0);

LatticeMatrixSet compute_kinetic_lattice_explicit(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const std::vector<LatticeCell>& cells,
    double pair_cutoff = 0.0);

LatticeMatrixSet compute_nuclear_lattice_explicit(
    const BasisSet& basis,
    const PeriodicSystem& system,
    const LatticeSumOptions& opts,
    const std::vector<LatticeCell>& cells);

// ---------------------------------------------------------------------------
// Short-range nuclear attraction (Ewald building block)
//
//   V^{erfc}_μν(g; ω) = Σ_{A, h} −Z_A ⟨ χ_μ | erfc(ω·r_A,h) / r_A,h | χ_ν^g ⟩
//
// where ω is the Ewald screening parameter (= α in the Ewald formulation).
// In the limit ω → 0 this reduces to the standard 1/r nuclear attraction;
// in the limit ω → ∞ it vanishes. The full nuclear attraction decomposes as
//
//   V^{nuc}(g) = V^{erfc}(g; ω) + V^{erf}(g; ω)
//
// for any choice of ω. The short-range erfc part converges exponentially
// as a real-space lattice sum over the {A, h} nuclear images, which makes
// it the ideal "short-range" piece in the Ewald partitioning of V(g).
//
// The complementary erf piece is smooth and Gaussian-tail bounded: it is
// evaluated in reciprocal space as part of the full Gaussian-charge Ewald
// scheme — that work lands in Phase 12e-c together with the ERI Ewald,
// because both pieces share the Gaussian-AO-pair Fourier-transform
// machinery.
//
// Consumers today: used by the Phase 12e-c Ewald dispatch of
// compute_nuclear_lattice. Exposed as a public entry point so tests can
// cover the short-range component in isolation.
//
// ``source_tol`` > 0 (pair-complete 3-D sums only) gives each AO product its
// own nuclear source radius, no larger than ``nuclear_cutoff_bohr``, outside
// which the summed erfc attraction is bounded by ``source_tol`` Hartree per
// matrix element. ``0`` (default) keeps every image within the cutoff.
LatticeMatrixSet compute_nuclear_erfc_lattice(const BasisSet& basis,
                                              const PeriodicSystem& system,
                                              double omega,
                                              const LatticeSumOptions& opts,
                                              double source_tol = 0.0);

// ---------------------------------------------------------------------------
// Full 3D Ewald-summed nuclear-attraction lattice sum (Phase 12e-c)
//
//   V^{Ewald}_μν(g) = ∫ χ_μ(r) v^{Ewald}_nuc(r) χ_ν(r − g) dr
//
// with v^{Ewald}_nuc the Ewald-summed nuclear Coulomb potential:
//
//   v^{Ewald}_nuc(r) = Σ_{A, g} [−Z_A erfc(α |r − R_A − g|) / |r − R_A − g|]
//                    + (4π / V) Σ_{G ≠ 0} [−ρ̃(G)] · e^{i G·r} · e^{−G²/(4α²)} / G²
//                    + jellium background
//
// The matrix element is evaluated by numerical integration on a molecular
// Becke grid — the same grid vibeqc uses for DFT. Grid accuracy sets the
// floor on the integral error (~1e−6 Ha in practice with a medium grid).
//
// Works for any dim == 3. For 1D / 2D systems the caller should stay with
// ``compute_nuclear_lattice`` (DIRECT_TRUNCATED), since the slab / line
// Ewald variants land in later sub-phases.
//
// The ``grid`` must have been built on ``system.unit_cell_molecule()`` to
// get the atomic partition right; callers that already build a DFT grid
// for the SCF should pass it in here to avoid re-building.
LatticeMatrixSet compute_nuclear_lattice_ewald(const BasisSet& basis,
                                               const PeriodicSystem& system,
                                               const Grid& grid,
                                               const LatticeSumOptions& opts,
                                               const EwaldOptions& ewald_opts = {});

// ---------------------------------------------------------------------------
// Lattice-summed SAP (superposition-of-atomic-potentials) guess potential
//
//   V^{SAP}_μν(g) = ∫ χ_μ(r) v^{SAP}(r) χ_ν(r − g) dr
//
// with v^{SAP} the lattice sum of the per-atom erf-screened SAP potentials,
//
//   v^{SAP}(r) = −Σ_{A, h} Σ_i c_i^A · erf(ω_i^A |r − R_A − h|) / |r − R_A − h|,
//   ω_i^A = √α_i^A,
//
// the periodic generalisation of the molecular SAP Fock's V_SAP term (see
// ``sap_expansions`` in guess.hpp for the fitted (α_i, c_i) data). The sum is
// Ewald split at η (``ewald_opts``): each SAP term's erfc complement and the
// erfc(η) partner are libint-analytic real-space sums over the nuclear images,
// and the smooth remainder Σ_A Q_A erf(η r)/r with Q_A = Σ_i c_i^A is summed
// in reciprocal space against the analytic AO-pair Fourier transform of each
// block. V_SAP(g) is therefore exact for the chosen cutoffs and does not
// change when an atom moves by a lattice vector (#483). The G = 0 background
// term shifts F_SAP = T + V_SAP by c·S and leaves the guess density unchanged.
//
// ``grid`` (built on ``system.unit_cell_molecule()``) is used only for a basis
// with a shell outside the pair-FT kernel's contract (Cartesian l > 0, or
// l > 6): there the long-range term falls back to quadrature on it, accurate
// to the grid floor but not translation invariant.
//
// Works for dim == 3. Elements without SAP data in the chosen basis are
// rejected (a clear error) rather than silently degraded — the bare-nuclear
// fallback the molecular path uses needs the full nuclear Ewald and is a
// separate follow-up.
LatticeMatrixSet compute_vsap_lattice(const BasisSet& basis,
                                      const PeriodicSystem& system,
                                      const Grid& grid,
                                      const std::string& sap_basis_name,
                                      const LatticeSumOptions& opts,
                                      const EwaldOptions& ewald_opts = {});

}  // namespace vibeqc
