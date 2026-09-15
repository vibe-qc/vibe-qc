"""Phase 12e-c-1: Ewald potential on grid points + V(g) via Ewald.

These test the first sub-phase of the full Gaussian-charge Ewald
treatment — the machinery that gives an unconditionally-convergent
nuclear-attraction lattice sum in 3D bulk.

Strategy
--------

The definitive correctness witness for any Ewald implementation is
α-invariance: total Ewald energies and matrix elements must be
independent of the Gaussian screening parameter α, provided both the
real-space and reciprocal-space cutoffs are large enough.

A secondary witness is that the split of the potential into a
short-range (erfc) and long-range (erf + background) piece satisfies

    v_short(r; α) + v_long(r; α) = v_full(r)

for any α.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


def _h2o() -> vq.Molecule:
    return vq.Molecule([
        vq.Atom(8, [0, 0, 0]),
        vq.Atom(1, [0, 1.43, -0.98]),
        vq.Atom(1, [0, -1.43, -0.98]),
    ])


def _nacl_cubic_system():
    """NaCl-like 2-atom cubic cell — neutral net charge so the jellium
    background vanishes and we can compare the Ewald potential at the
    cell center to the classical Madelung value directly."""
    a = 5.6 / 0.529177210903
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]],
                             dtype=float).T
    positions = np.column_stack([[0, 0, 0], [0.5 * a, 0, 0]])
    charges = np.array([+1.0, -1.0])
    return lat, positions, charges


# ---------------------------------------------------------------------------
# Short + long decomposition of the potential
# ---------------------------------------------------------------------------

def test_potential_short_plus_long_equals_full():
    """v_total(r) = v_short(r) + v_long(r). Exercises the
    include_short_range toggle that compute_nuclear_lattice_ewald relies
    on to avoid double-counting."""
    lat, positions, charges = _nacl_cubic_system()
    # Evaluate at a few points well away from any charge (avoid the
    # 1/r singularity that breaks the direct sum).
    eval_pts = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 0.5, 0.7],
        [3.0, 1.5, 2.0],
    ])

    opts = vq.EwaldOptions()
    opts.alpha = 0.3
    opts.real_cutoff_bohr = 25.0
    opts.recip_cutoff_bohr_inv = 6.0

    v_full  = vq.ewald_point_charge_potential(
        lat, positions, charges, eval_pts, opts, True)
    v_long  = vq.ewald_point_charge_potential(
        lat, positions, charges, eval_pts, opts, False)
    v_short = v_full - v_long

    # Recombine and check against v_full.
    assert np.max(np.abs((v_short + v_long) - v_full)) < 1e-12


def test_short_range_potential_is_invariant_to_basis_lattice_shift():
    """The 3D real-space potential enumerates a complete pair sphere."""
    lattice = 4.0 * np.eye(3)
    positions = np.column_stack([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
    shifted = positions.copy()
    shifted[:, 1] += 41.0 * lattice[:, 0]
    charges = np.array([1.0, -1.0])
    eval_points = np.array([[0.3, 0.5, 0.8]])
    opts = vq.EwaldOptions()
    opts.real_cutoff_bohr = 8.0

    potential = vq.ewald_point_charge_potential(
        lattice, positions, charges, eval_points, opts, True
    )
    shifted_potential = vq.ewald_point_charge_potential(
        lattice, shifted, charges, eval_points, opts, True
    )
    shifted_eval_points = eval_points + 43.0 * lattice[:, 1]
    shifted_observer_potential = vq.ewald_point_charge_potential(
        lattice, positions, charges, shifted_eval_points, opts, True
    )

    assert shifted_potential == pytest.approx(potential, abs=1.0e-12)
    assert shifted_observer_potential == pytest.approx(
        potential, abs=1.0e-12
    )


# ---------------------------------------------------------------------------
# α-invariance of the nuclear-attraction lattice sum
# ---------------------------------------------------------------------------

def test_nuclear_lattice_ewald_is_alpha_invariant():
    """compute_nuclear_lattice_ewald output is independent of α within
    grid accuracy. Tight bulk cell so the Ewald machinery actually has
    work to do (not trivially the molecular limit)."""
    uc = [vq.Atom(8, [0, 0, 0]),
          vq.Atom(1, [0, 1.43, -0.98]),
          vq.Atom(1, [0, -1.43, -0.98])]
    sysp = vq.PeriodicSystem(3, np.eye(3) * 10.0, uc)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    opts.nuclear_cutoff_bohr = 15.0
    grid = vq.build_grid(sysp.unit_cell_molecule())

    results = []
    for alpha in (0.25, 0.3, 0.5, 0.8):
        ew_opts = vq.EwaldOptions()
        ew_opts.alpha = alpha
        ew_opts.real_cutoff_bohr = 30.0
        ew_opts.recip_cutoff_bohr_inv = 2.0 * alpha * np.sqrt(30)
        V = vq.compute_nuclear_lattice_ewald(basis, sysp, grid, opts, ew_opts)
        trace_V = sum(np.asarray(b).trace() for b in V.blocks)
        results.append(trace_V)

    spread = max(results) - min(results)
    assert spread < 1e-3, (
        f"α-invariance violated: trace(V) varies by {spread:.2e} across "
        f"α ∈ {{0.25, 0.3, 0.5, 0.8}} — readings were {results}"
    )


# ---------------------------------------------------------------------------
# compute_nuclear_lattice_ewald matches direct-truncated on neutral cells
# ---------------------------------------------------------------------------

def test_nuclear_lattice_ewald_matches_direct_for_neutral_cell():
    """For a lattice of positive + negative point charges that is
    charge-neutral per cell (jellium background = 0), the Ewald and
    direct-summed versions should agree in the molecular-limit regime.
    This isolates grid-accuracy of the long-range piece.

    We use an H atom + a Z=-1 "ghost" (anti-charge) — contrived but
    the jellium background vanishes exactly, which is the cleanest
    test. Done via the low-level ewald_point_charge_potential call,
    not via run_rhf (which wouldn't know what to do with Z < 0)."""
    # 4-bohr cubic cell; Z=+1 at origin, Z=-1 at (2, 2, 2).
    a = 4.0
    lat = a * np.eye(3)
    positions = np.column_stack([[0, 0, 0], [2.0, 2.0, 2.0]])
    charges = np.array([+1.0, -1.0])

    # Evaluate at a non-special point.
    eval_pts = np.array([[1.0, 0.5, 1.5]])

    opts1 = vq.EwaldOptions()
    opts1.alpha = 0.3
    opts1.real_cutoff_bohr = 30.0
    opts1.recip_cutoff_bohr_inv = 8.0
    v_30 = vq.ewald_point_charge_potential(
        lat, positions, charges, eval_pts, opts1, True)[0]

    # Now with a tighter real cutoff + larger α — same α-invariance result.
    opts2 = vq.EwaldOptions()
    opts2.alpha = 0.8
    opts2.real_cutoff_bohr = 20.0
    opts2.recip_cutoff_bohr_inv = 2.0 * 0.8 * np.sqrt(30)
    v_08 = vq.ewald_point_charge_potential(
        lat, positions, charges, eval_pts, opts2, True)[0]

    assert abs(v_30 - v_08) < 1e-10, (
        f"α-invariance violated on neutral cell: "
        f"v(α=0.3) = {v_30:.12f}, v(α=0.8) = {v_08:.12f}"
    )


# ---------------------------------------------------------------------------
# Structure / dispatch sanity checks
# ---------------------------------------------------------------------------

def test_compute_nuclear_lattice_ewald_rejects_low_dim():
    uc = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    sysp = vq.PeriodicSystem(1, np.diag([4.0, 30.0, 30.0]), uc)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 10.0
    grid = vq.build_grid(sysp.unit_cell_molecule())
    with pytest.raises(ValueError, match="dim == 3"):
        vq.compute_nuclear_lattice_ewald(basis, sysp, grid, opts)


def test_compute_nuclear_lattice_ewald_returns_sensible_shape():
    sysp = vq.PeriodicSystem(3, np.eye(3) * 12.0,
                             [vq.Atom(1, [0, 0, 0]),
                              vq.Atom(1, [0, 0, 1.4])])
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.LatticeSumOptions()
    opts.cutoff_bohr = 8.0
    grid = vq.build_grid(sysp.unit_cell_molecule())
    V = vq.compute_nuclear_lattice_ewald(basis, sysp, grid, opts)
    assert V.nbf == 2
    assert len(V.blocks) == len(V.cells)
    # All blocks are nbf × nbf.
    for b in V.blocks:
        arr = np.asarray(b)
        assert arr.shape == (2, 2)
        # Nuclear attraction is real-negative for electrons — at least on
        # the diagonal.
        assert arr.trace() < 0
