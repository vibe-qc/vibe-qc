"""Tests for the finite-difference Hessian on the GAPW / GPW route
(v0.12 R3).

Pins :func:`vibeqc.compute_hessian_gpw` and
:func:`vibeqc.compute_vibrational_frequencies` on H2 / STO-3G. The
Hessian is built by central differences on the *analytic* GPW
gradient, so each test triples as a consistency check on the
analytic-forces machinery from v0.12 R2.

Test surface:

1. ``compute_hessian_gpw`` on H2 returns a ``(6, 6)`` symmetric
   matrix.
2. Translational symmetry of a 2-atom system: the upper-left and
   lower-right ``3×3`` blocks coincide (both atoms feel the same
   restoring force from a translation of the other).
3. Harmonic frequencies on H2 separate into three near-zero
   translations, two near-zero librations, and one stretch in the
   experimental H2 wavenumber band (3000–5500 cm⁻¹ for STO-3G).
4. The stretch mode is real (positive wavenumber), not imaginary.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import vibeqc
from vibeqc.periodic_gapw_grid import GAPWExperimentalWarning
from vibeqc.periodic_gapw_hessian import (
    compute_hessian_gpw,
    compute_vibrational_frequencies,
)
from vibeqc.periodic_gapw_j import run_periodic_rhf_gpw


warnings.simplefilter("ignore", GAPWExperimentalWarning)
pytestmark = pytest.mark.filterwarnings(
    "ignore::vibeqc.periodic_gapw_grid.GAPWExperimentalWarning"
)


# ---------- Helpers -------------------------------------------------------


def _h2_periodic_system(d_bohr: float = 1.4, L_bohr: float = 12.0):
    """H2 aligned with the x axis at bond length ``d_bohr`` in a
    cubic ``L_bohr`` box. Same convention as
    tests/test_periodic_gapw_gradient.py.
    """
    p1 = [L_bohr / 2 - 0.5 * d_bohr, L_bohr / 2, L_bohr / 2]
    p2 = [L_bohr / 2 + 0.5 * d_bohr, L_bohr / 2, L_bohr / 2]
    sys = vibeqc.PeriodicSystem()
    sys.dim = 3
    sys.lattice = np.eye(3, dtype=np.float64) * L_bohr
    sys.unit_cell = [
        vibeqc.Atom(1, p1),
        vibeqc.Atom(1, p2),
    ]
    return sys


def _h2_central_scf(d_bohr: float = 1.4, L_bohr: float = 12.0,
                    cutoff_ha: float = 200.0):
    sys = _h2_periodic_system(d_bohr=d_bohr, L_bohr=L_bohr)
    mol = vibeqc.Molecule(list(sys.unit_cell), 0, 1)
    basis = vibeqc.BasisSet(mol, "sto-3g")
    res = run_periodic_rhf_gpw(
        sys, basis, cutoff_ha=cutoff_ha, functional=None, quiet=True,
    )
    assert res.converged
    return sys, basis, res


# Module-level fixtures: building the Hessian is the expensive bit
# (12 SCFs + 12 analytic gradients), so reuse it across tests via a
# session-scoped fixture.


@pytest.fixture(scope="module")
def h2_hessian_bundle():
    sys, basis, res = _h2_central_scf(d_bohr=1.4, L_bohr=12.0)
    H = compute_hessian_gpw(
        sys, basis, res,
        basis_name="sto-3g",
        fd_step_bohr=0.02,
        functional=None,
    )
    return sys, basis, res, H


# ---------- 1. Shape + symmetry ------------------------------------------


def test_gpw_hessian_h2_shape_and_symmetry(h2_hessian_bundle):
    """H2 → 2 atoms × 3 axes = 6 DOFs, square + symmetric to a few
    µHa/bohr² (FD truncation + analytic-FD noise floor)."""
    _, _, _, H = h2_hessian_bundle
    assert H.shape == (6, 6)

    # Symmetric: the post-symmetrisation step guarantees this within
    # float round-off, but pin it explicitly.
    asym = np.max(np.abs(H - H.T))
    assert asym < 1e-10, (
        f"Hessian not symmetric: max |H - H.T| = {asym:.3e}"
    )


# ---------- 2. Translational symmetry of the 2-atom blocks ---------------


def test_gpw_hessian_h2_translational_block_symmetry(h2_hessian_bundle):
    """For a 2-atom homonuclear system the on-site blocks H[0:3,0:3]
    and H[3:6,3:6] both encode the curvature of E w.r.t. moving one
    atom; by exchange symmetry of identical nuclei they agree (up to
    FD noise from the analytic-gradient FD piece)."""
    _, _, _, H = h2_hessian_bundle
    block_a = H[0:3, 0:3]
    block_b = H[3:6, 3:6]
    diff = np.max(np.abs(block_a - block_b))
    # Loose because each block carries finite-step error from the
    # outer Hessian FD (~5e-4 Ha/bohr² typical) and inner gradient FD.
    assert diff < 5e-3, (
        f"H[0:3,0:3] vs H[3:6,3:6] mismatch: {diff:.3e} Ha/bohr²\n"
        f"block_a =\n{block_a}\nblock_b =\n{block_b}"
    )


# ---------- 3. Frequencies — three zero modes + one stretch --------------


def test_gpw_hessian_h2_vibrational_frequencies(h2_hessian_bundle):
    """Diagonalising the mass-weighted Hessian on H2 yields six
    modes:

    * three acoustic translations (~0 cm⁻¹) — the translational
      invariance of the energy w.r.t. uniform shifts in a vacuum-
      padded periodic box,
    * two soft bend / libration modes perpendicular to the bond —
      these are *not* exactly zero in a periodic vacuum-padded box
      because the FFT grid + Ewald gauge break the in-vacuum
      rotation symmetry weakly (small but nonzero curvature),
    * one stretch at the H2 frequency.

    STO-3G overestimates the harmonic stretch (~5000 cm⁻¹ vs
    experimental 4400 cm⁻¹), so we accept the wide 3000–5500 cm⁻¹
    band.
    """
    _, _, _, H = h2_hessian_bundle
    freqs, _ = compute_vibrational_frequencies(H, masses_amu=[1.008, 1.008])
    assert freqs.shape == (6,)

    # Ascending order from eigh; the stretch is the largest (most
    # positive) wavenumber.
    stretch = freqs[-1]

    # Three acoustic translations sit at ~0 cm⁻¹.
    translations = freqs[:3]
    assert np.max(np.abs(translations)) < 50.0, (
        f"Acoustic-translation modes are not near zero: "
        f"{translations} cm⁻¹"
    )

    # Two bend modes are softer than the stretch but nonzero in
    # the periodic vacuum-padded cell. Just demand they sit well
    # below the stretch.
    bends = freqs[3:5]
    assert np.max(np.abs(bends)) < stretch * 0.5, (
        f"Bend modes {bends} cm⁻¹ are not soft compared to the "
        f"stretch {stretch:.1f} cm⁻¹."
    )

    # The H2 stretch in the experimental / STO-3G band.
    assert 3000.0 < stretch < 5500.0, (
        f"H2 stretch frequency {stretch:.1f} cm⁻¹ out of band "
        f"(3000-5500 cm⁻¹). Full spectrum: {freqs}"
    )


# ---------- 4. Stretch is real (no imaginary) ----------------------------


def test_gpw_hessian_h2_stretch_is_real(h2_hessian_bundle):
    """The H2 stretch mode must be a positive (real) wavenumber. A
    negative (imaginary) stretch would mean we built the Hessian at
    a maximum / saddle rather than a minimum, or that the FD step is
    so small the noise floor dominates."""
    _, _, _, H = h2_hessian_bundle
    freqs, _ = compute_vibrational_frequencies(H, masses_amu=[1.008, 1.008])
    stretch = freqs[-1]
    assert stretch > 0.0, (
        f"H2 stretch came back negative (imaginary): {stretch:.1f} cm⁻¹"
    )
