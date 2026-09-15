"""BIPOLE-SR-PAD-OVERCONSERVATIVE regression pins (registry 2026-08-05).

The default SR ket-image pad (``bipole_sr_image_extent``) was measured
~2x over-radius against its own convergence: home-cell J/K contractions
on the four rocksalt/STO-3G probe cases are BIT-IDENTICAL beyond an
absolute 16-26 bohr traversal ball, while the ε=1e-6 default padded to
37-40 bohr (investigation evidence: pad_kernel_probe.py ladders,
2026-08-05). Root cause, two stacked slack terms in the derivation:

1. γ_bra = γ_ket = γ_min conflated a PRIMITIVE exponent with a PAIR
   (product-Gaussian) exponent — a pair exponent is the sum of its two
   primitive exponents, so the slowest pair decay is 2·γ_min per side:
   ``1/m_ω <= 1/γ_min + 1/ω²``, not ``2/γ_min + 1/ω²``.
2. The tail radius inverted the loose bound ``erfc(x) <= exp(-x²)``
   (``x = √ln(1/ε) = 3.717`` at ε=1e-6) instead of the exact inverse
   ``x = erfc⁻¹(ε) = 3.459``.

This file pins the corrected derivation (still rigorous — the pad must
cover every measured convergence radius with margin, no constants tuned
to the probe set) and the padded-cell-count traversal proxy that the
Fock-build cost scales with (cost ~ n_out x n_pad²).

Measured bit-stability radii (absolute ball, cutoff 6, home J/K
contraction, sr_range_screening on; 2026-08-05 investigation
EVIDENCE.md — the CONVERGENCE FLOOR the derived pad must stay above):

    LiH/STO-3G 26 bohr | LiF/STO-3G 26 | MgO/STO-3G 20 | NaCl/STO-3G 26
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import erfcinv

import vibeqc as vq
from vibeqc._vibeqc_core import direct_lattice_cells
from vibeqc.bipole_ext_el_pole import crystal_default_ewald_alpha
from vibeqc.pbc_bipole_common import bipole_sr_image_extent

ANG2BOHR = 1.0 / 0.529177210903

# (a [Angstrom], Z1, Z2, measured bit-stable absolute radius at cutoff 6)
_ROCKSALT = {
    "lih": (4.0840, 3, 1, 26.0),
    "lif": (4.0100, 3, 9, 26.0),
    "mgo": (4.2170, 12, 8, 20.0),
    "nacl": (5.6400, 11, 17, 26.0),
}


def _case(name):
    a_ang, z1, z2, stable_abs = _ROCKSALT[name]
    a = a_ang * ANG2BOHR
    lattice = (a / 2.0) * np.array(
        [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]]
    )
    system = vq.PeriodicSystem(
        3, lattice, [vq.Atom(z1, [0, 0, 0]), vq.Atom(z2, [a / 2, a / 2, a / 2])]
    )
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    omega = float(crystal_default_ewald_alpha(abs(np.linalg.det(lattice))))
    return system, basis, omega, stable_abs


def _gamma_min(basis) -> float:
    return min(
        float(np.asarray(s.exponents, dtype=float).min())
        for s in basis.shells()
    )


def _max_atom(system) -> float:
    return max(
        float(np.linalg.norm(np.asarray(a.xyz)))
        for a in system.unit_cell_molecule().atoms
    )


@pytest.mark.parametrize("name", sorted(_ROCKSALT))
def test_pad_is_pair_exponent_exact_erfc_inverse(name):
    """The pad equals the corrected closed form.

    erfc(√m_ω R) <= ε at R = erfc⁻¹(ε)·√(1/m_ω) with the slowest pair
    decay 1/m_ω = 1/γ_min + 1/ω² (pair exponents >= 2·γ_min each), plus
    the max atom-centroid offset. Pre-fix the helper returned
    √(ln(1/ε)·(2/γ_min + 1/ω²)) + max_atom — strictly larger.
    """
    system, basis, omega, _ = _case(name)
    eps = 1e-6
    gmin = _gamma_min(basis)
    expected = float(erfcinv(eps)) * float(
        np.sqrt(1.0 / gmin + 1.0 / omega**2)
    ) + _max_atom(system)
    got = bipole_sr_image_extent(basis, system, omega, precision=eps)
    assert got == pytest.approx(expected, rel=1e-12)
    # And it genuinely dropped below the pre-fix global-worst-case value.
    pre_fix = float(
        np.sqrt(np.log(1.0 / eps) * (2.0 / gmin + 1.0 / omega**2))
    ) + _max_atom(system)
    assert got < pre_fix - 3.0  # bohr; ~4-8 bohr on the rocksalt probes


@pytest.mark.parametrize("name", sorted(_ROCKSALT))
def test_pad_covers_measured_convergence_with_margin(name):
    """Rigor floor: the derived default ball must contain every radius
    at which the probe ladders measured bit-identical home J/K.

    The measured radii are VALIDATION data, not inputs: the pad is the
    closed-form erfc-tail bound above; this test only checks it never
    dips below the physics it must cover.
    """
    system, basis, omega, stable_abs = _case(name)
    cutoff = 6.0  # the probe-ladder cutoff the radii were measured at
    extent = cutoff + bipole_sr_image_extent(
        basis, system, omega, precision=1e-6
    )
    assert extent >= stable_abs


def test_padded_cell_count_traversal_proxy_lih():
    """Timing-independent traversal proxy (cost ~ n_out x n_pad²).

    LiH/STO-3G at the 8-bohr witness cutoff: the ε=1e-6 default ball
    held 2243 cells pre-fix; the corrected bound yields 1157 (~3.8x
    less traversal). Pin an envelope, not the exact count, so unrelated
    lattice-enumeration changes do not false-fail; the pre-fix count
    sits far outside it.
    """
    system, basis, omega, _ = _case("lih")
    extent = 8.0 + bipole_sr_image_extent(basis, system, omega, precision=1e-6)
    n_pad = len(list(direct_lattice_cells(system, extent)))
    assert n_pad <= 1300  # pre-fix: 2243
    # ...while still covering the measured 26-bohr convergence ball.
    assert extent >= 26.0
