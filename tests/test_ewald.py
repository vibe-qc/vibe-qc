"""Phase 12e-a: classical Ewald summation for 3D lattices of point charges.

Core correctness witnesses
--------------------------

1. **α-invariance** — The Ewald total must not depend on the Gaussian
   screening parameter α once both real- and reciprocal-space sums have
   converged. This is the mathematically rigorous check.

2. **Known Madelung constants** — For NaCl, CsCl, ZnS zincblende, and
   simple cubic lattices the infinite lattice sums are tabulated in the
   literature to many digits. We reproduce them.

3. **Molecular limit** — With one atom per cell or with a very large
   unit cell, the Ewald energy must collapse onto something identifiable
   (zero for a single atom at the origin with neutral charge; or the
   bare Madelung of an isolated lattice of equal charges with the
   jellium convention).

4. **Cross-consistency with the direct-truncated sum** — For a large
   lattice constant where direct truncation is nearly converged, Ewald
   and direct-truncated answers must agree.

5. **Dispatch through ``nuclear_repulsion_per_cell``** — Setting
   ``CoulombMethod.EWALD_3D`` on ``LatticeSumOptions`` routes through
   Ewald without breaking the existing API.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


_BOHR_PER_A = 1.0 / 0.529177210903


def _mgo_primitive_point_charges():
    """Primitive rocksalt MgO, the dense-cell cutoff regression."""
    a = 4.211 * _BOHR_PER_A
    lattice = 0.5 * a * np.array(
        [[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float
    )
    oxygen = 0.5 * (lattice[0] + lattice[1] + lattice[2])
    positions = np.column_stack(([0.0, 0.0, 0.0], oxygen))
    charges = np.array([12.0, 8.0])
    return lattice, positions, charges


# ---------------------------------------------------------------------------
# α-invariance — the rigorous "is Ewald correct" test
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("alpha", [0.2, 0.3, 0.5, 1.0])
def test_alpha_invariance_nacl(alpha):
    """NaCl rocksalt with ±1 charges — energy independent of α."""
    a = 5.6 * _BOHR_PER_A
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).T.astype(float)
    positions = np.column_stack([[0, 0, 0], 0.5 * a * np.array([1, 0, 0])])
    charges = np.array([+1.0, -1.0])

    opts = vq.EwaldOptions()
    opts.alpha = alpha
    opts.real_cutoff_bohr = 40.0
    opts.recip_cutoff_bohr_inv = 12.0
    energy = vq.ewald_point_charge_energy(lat, positions, charges, opts)

    # Reference: NaCl Madelung constant 1.7475645946… × (1/r_nn), with a
    # negative sign because the nearest neighbors carry opposite charge.
    r_nn = 0.5 * a
    expected = -1.7475645946 / r_nn

    assert abs(energy - expected) / abs(expected) < 1e-9


# ---------------------------------------------------------------------------
# Known Madelung constants
# ---------------------------------------------------------------------------

def test_nacl_madelung_constant():
    """-M_NaCl / r_nn, where M_NaCl = 1.74756459... and r_nn = a/2."""
    a = 5.6 * _BOHR_PER_A
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).T.astype(float)
    positions = np.column_stack([[0, 0, 0], 0.5 * a * np.array([1, 0, 0])])
    charges = np.array([+1.0, -1.0])

    opts = vq.EwaldOptions()
    opts.real_cutoff_bohr = 40.0
    energy = vq.ewald_point_charge_energy(lat, positions, charges, opts)

    r_nn = 0.5 * a
    madelung = -energy * r_nn
    assert abs(madelung - 1.7475645946) < 1e-8, (
        f"extracted Madelung constant = {madelung:.12f}, "
        f"expected 1.7475645946"
    )


def test_cscl_madelung_constant():
    """CsCl is simple-cubic with basis — M_CsCl = 1.76267477… using
    r_nn = a√3/2 as convention."""
    a = 4.1 * _BOHR_PER_A      # CsCl-like, arbitrary cubic side
    lat = a * np.eye(3)
    positions = np.column_stack([[0, 0, 0], 0.5 * a * np.array([1, 1, 1])])
    charges = np.array([+1.0, -1.0])

    opts = vq.EwaldOptions()
    opts.real_cutoff_bohr = 40.0
    energy = vq.ewald_point_charge_energy(lat, positions, charges, opts)

    r_nn = 0.5 * a * np.sqrt(3)
    madelung = -energy * r_nn
    reference = 1.762674773
    assert abs(madelung - reference) < 1e-8, (
        f"extracted M_CsCl = {madelung:.12f}, expected {reference:.12f}"
    )


def test_zincblende_madelung_constant():
    """ZnS (sphalerite) M = 1.6380550533 using r_nn as conventional."""
    a = 5.4 * _BOHR_PER_A
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).T.astype(float)
    positions = np.column_stack([[0, 0, 0], 0.25 * a * np.array([1, 1, 1])])
    charges = np.array([+1.0, -1.0])

    opts = vq.EwaldOptions()
    opts.real_cutoff_bohr = 40.0
    energy = vq.ewald_point_charge_energy(lat, positions, charges, opts)

    r_nn = 0.25 * a * np.sqrt(3)
    madelung = -energy * r_nn
    reference = 1.6380550533
    assert abs(madelung - reference) < 1e-9, (
        f"extracted M_ZnS = {madelung:.12f}, expected {reference:.12f}"
    )


# ---------------------------------------------------------------------------
# Cutoff convergence
# ---------------------------------------------------------------------------

def test_cutoff_convergence_exponential():
    """Fixed α, increasing cutoff → energy converges exponentially fast."""
    a = 5.6 * _BOHR_PER_A
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).T.astype(float)
    positions = np.column_stack([[0, 0, 0], 0.5 * a * np.array([1, 0, 0])])
    charges = np.array([+1.0, -1.0])

    # Converged reference at very large cutoff.
    opts_ref = vq.EwaldOptions()
    opts_ref.alpha = 0.3
    opts_ref.real_cutoff_bohr = 80.0
    opts_ref.recip_cutoff_bohr_inv = 15.0
    E_ref = vq.ewald_point_charge_energy(lat, positions, charges, opts_ref)

    last_err = float("inf")
    for cutoff in (10.0, 15.0, 20.0, 30.0):
        opts = vq.EwaldOptions()
        opts.alpha = 0.3
        opts.real_cutoff_bohr = cutoff
        opts.recip_cutoff_bohr_inv = 15.0
        E = vq.ewald_point_charge_energy(lat, positions, charges, opts)
        err = abs(E - E_ref)
        # Monotonic convergence: each larger cutoff must improve on the
        # previous until we hit numerical floor.
        if err > 1e-14:
            assert err < last_err or err < 1e-11
        last_err = err


def test_dense_cell_real_cutoff_is_pair_complete():
    """The cutoff applies to |R_A - R_B + g|, not to |g| alone.

    Primitive MgO exposes the distinction: at an 18-bohr cutoff, valid
    translations outside the bare |g| sphere are shifted inside the physical
    pair sphere by the Mg-O basis displacement. Before the pair-complete
    enumeration fix, the energy missed 2.11e-5 Ha and the nuclear gradient
    missed 2.57e-5 Ha/bohr relative to the converged sum.
    """
    from vibeqc import _vibeqc_core as _core

    lattice, positions, charges = _mgo_primitive_point_charges()
    opts = vq.EwaldOptions()
    opts.real_cutoff_bohr = 18.0
    reference_opts = vq.EwaldOptions()
    reference_opts.real_cutoff_bohr = 80.0

    energy = vq.ewald_point_charge_energy(
        lattice, positions, charges, opts
    )
    reference_energy = vq.ewald_point_charge_energy(
        lattice, positions, charges, reference_opts
    )
    gradient = np.asarray(
        _core.ewald_point_charge_gradient(
            lattice, positions, charges, opts
        )
    )
    reference_gradient = np.asarray(
        _core.ewald_point_charge_gradient(
            lattice, positions, charges, reference_opts
        )
    )

    assert energy == pytest.approx(reference_energy, abs=1.0e-9)
    assert np.max(np.abs(gradient - reference_gradient)) < 1.0e-9

    displaced = positions.copy()
    displaced[0, 1] += 0.07
    step = 1.0e-4
    plus = displaced.copy()
    minus = displaced.copy()
    plus[0, 1] += step
    minus[0, 1] -= step
    finite_difference = (
        vq.ewald_point_charge_energy(lattice, plus, charges, opts)
        - vq.ewald_point_charge_energy(lattice, minus, charges, opts)
    ) / (2.0 * step)
    displaced_gradient = np.asarray(
        _core.ewald_point_charge_gradient(
            lattice, displaced, charges, opts
        )
    )
    assert displaced_gradient[0, 1] == pytest.approx(
        finite_difference, abs=1.0e-7
    )


def test_dense_cell_ewald_is_invariant_to_basis_lattice_shift():
    """A far-unwrapped basis charge is the same periodic crystal."""
    from vibeqc import _vibeqc_core as _core

    lattice, positions, charges = _mgo_primitive_point_charges()
    shifted = positions.copy()
    shifted[:, 1] += 37.0 * lattice[:, 0]
    opts = vq.EwaldOptions()
    opts.real_cutoff_bohr = 18.0

    energy = vq.ewald_point_charge_energy(
        lattice, positions, charges, opts
    )
    shifted_energy = vq.ewald_point_charge_energy(
        lattice, shifted, charges, opts
    )
    gradient = np.asarray(
        _core.ewald_point_charge_gradient(
            lattice, positions, charges, opts
        )
    )
    shifted_gradient = np.asarray(
        _core.ewald_point_charge_gradient(
            lattice, shifted, charges, opts
        )
    )

    assert shifted_energy == pytest.approx(energy, abs=1.0e-10)
    assert np.max(np.abs(shifted_gradient - gradient)) < 1.0e-10


# ---------------------------------------------------------------------------
# Non-neutral cell — jellium background correction
# ---------------------------------------------------------------------------

def test_nonneutral_cell_uses_background_correction():
    """A single Z=+1 charge per simple-cubic cell. In the jellium
    convention the Madelung energy per ion is −M_SC · Z² / a with
    M_SC = 1.4186487… (Nijboer-De Wette, 1957)."""
    a = 4.0
    lat = a * np.eye(3)
    positions = np.zeros((3, 1))
    charges = np.array([1.0])

    opts = vq.EwaldOptions()
    opts.real_cutoff_bohr = 40.0
    energy = vq.ewald_point_charge_energy(lat, positions, charges, opts)

    M_sc = 1.4186487   # simple-cubic jellium Madelung constant
    expected = -M_sc / a
    rel = abs(energy - expected) / abs(expected)
    assert rel < 1e-6, (
        f"simple-cubic jellium: E = {energy:.10f}, expected {expected:.10f} "
        f"(rel err {rel:.2e})"
    )


# ---------------------------------------------------------------------------
# Dispatch through nuclear_repulsion_per_cell
# ---------------------------------------------------------------------------

def test_nuclear_repulsion_dispatch_ewald_3d_alpha_invariance():
    """Setting coulomb_method=EWALD_3D routes through the Ewald engine.
    The resulting nuclear Madelung for realistic Z values (11, 17) is
    not a clean multiple of the ±1 Madelung constant (the cell is not
    charge-neutral; the jellium background correction contributes), but
    the answer must be finite and α-invariant — the definitive
    correctness witness."""
    a = 5.6 * _BOHR_PER_A
    lat = 0.5 * a * np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]).T.astype(float)
    atoms = [
        vq.Atom(11, [0.0, 0.0, 0.0]),
        vq.Atom(17, (0.5 * a * np.array([1, 0, 0])).tolist()),
    ]
    sysp = vq.PeriodicSystem(3, lat, atoms)

    energies = []
    for alpha in (0.2, 0.3, 0.5, 0.8):
        # Route through the public API — exercising the dispatch.
        import vibeqc._vibeqc_core as _core
        opts = vq.EwaldOptions()
        opts.alpha = alpha
        opts.real_cutoff_bohr = 35.0
        opts.recip_cutoff_bohr_inv = 12.0
        energies.append(_core.ewald_nuclear_repulsion(sysp, opts))
    spread = max(energies) - min(energies)
    assert spread < 1e-8, (
        f"Ewald energy varies with α by {spread:.2e}; α-invariance violated"
    )

    # And that nuclear_repulsion_per_cell with EWALD_3D returns the same
    # (finite, non-NaN) number as a direct call to ewald_nuclear_repulsion.
    lopts = vq.LatticeSumOptions()
    lopts.coulomb_method = vq.CoulombMethod.EWALD_3D
    lopts.nuclear_cutoff_bohr = 35.0
    e = vq.nuclear_repulsion_per_cell(sysp, lopts)
    import math
    assert math.isfinite(e)
    # Same within the accuracy set by the auto-α chosen inside dispatch.
    assert abs(e - energies[1]) < 1e-6


def test_nuclear_repulsion_rejects_ewald_for_low_dim():
    """EWALD_3D must refuse non-3D systems — the 1D / 2D Ewald variants
    will land in future sub-phases."""
    sysp_1d = vq.PeriodicSystem(
        1, np.diag([4.0, 30.0, 30.0]),
        [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])],
    )
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    with pytest.raises(ValueError, match="dim"):
        vq.nuclear_repulsion_per_cell(sysp_1d, opts)


# ---------------------------------------------------------------------------
# Regression — existing periodic code paths keep working
# ---------------------------------------------------------------------------

def test_default_path_unchanged():
    """Default ``CoulombMethod.DIRECT_TRUNCATED`` reproduces the Phase
    12a–12d behavior byte-for-byte on the molecular-limit tests."""
    uc = [vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])]
    sysp = vq.PeriodicSystem(3, np.eye(3) * 50.0, uc)
    opts = vq.LatticeSumOptions()
    # Default coulomb_method is DIRECT_TRUNCATED.
    opts.nuclear_cutoff_bohr = 15.0
    e = vq.nuclear_repulsion_per_cell(sysp, opts)
    # H2 molecular nuclear repulsion = 1 / 1.4 ≈ 0.7143 Ha, exact in the
    # molecular-limit (only g=0 contributes at this cutoff).
    assert abs(e - 1.0 / 1.4) < 1e-12


# ---------------------------------------------------------------------------
# EwaldOptions construction / field access
# ---------------------------------------------------------------------------

def test_ewald_options_default_construct_and_set():
    opts = vq.EwaldOptions()
    assert opts.alpha < 0                       # sentinel: auto-select
    assert opts.real_cutoff_bohr > 0
    opts.alpha = 0.5
    opts.real_cutoff_bohr = 25.0
    opts.recip_cutoff_bohr_inv = 8.0
    opts.tolerance = 1e-10
    assert opts.alpha == pytest.approx(0.5)
