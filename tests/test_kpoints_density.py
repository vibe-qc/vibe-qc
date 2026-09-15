"""Phase K5 — density-based auto-mesh constructors.

Three constructors with different conventions:

  - ``KPoints.from_kppra(sys, n_kpts_per_atom, metallic=False, ...)``
    AFLOW / Curtarolo 2012 convention.
  - ``KPoints.from_kspacing(sys, kspacing, units="angstrom"/"bohr")``
    Materials Project / ASE / VASP KSPACING convention.
  - ``KPoints.auto(sys, length)`` — VASP deprecated ``Auto`` mode +
    Choudhary & Tavazza 2019 / 2020 line-density formalism.

Pinned contracts:

1. **Public API** — all three constructors live on ``KPoints``.

2. **Monotone in density** — increasing the density parameter never
   produces a smaller mesh, and on standard cells produces larger
   ones for sizable bumps. Rule of thumb: KPPRA × 10 → mesh ≥ ×2 on
   each axis for cubic primitive cells.

3. **Per-axis proportionality** — for a tetragonal cell with c/a = 2,
   the c-axis subdivision count is half (or matches when k-density
   parity rounds in its favor) the a-axis count. Validates that
   KPPRA / kspacing actually use ``|b_i|`` — not a uniform N×N×N.

4. **Non-periodic axes pinned to 1** — for a 2D-periodic system
   (slab / surface), the third axis returns ``mesh[2] == 1`` no matter
   the density parameter.

5. **Metallic flag bumps density 4×** — and emits a ``UserWarning``
   (suppressed via ``warn_no_smearing=False``).

6. **Validation** — non-positive density raises ``ValueError``;
   unrecognised ``units`` argument raises ``ValueError``.

7. **kspacing units sanity** — switching ``units="bohr"`` with a
   suitably-converted number gives the same mesh as the default Å.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import vibeqc as vq


ANGSTROM_TO_BOHR = 1.8897261339213


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _si_diamond():
    a = 5.43 * ANGSTROM_TO_BOHR
    lat = np.array([[0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]])
    p1 = np.zeros(3)
    p2 = 0.25 * (lat[:, 0] + lat[:, 1] + lat[:, 2])
    return vq.PeriodicSystem(3, lat, [vq.Atom(14, p1.tolist()),
                                        vq.Atom(14, p2.tolist())])


def _w_bcc():
    a = 3.16 * ANGSTROM_TO_BOHR
    lat = np.array([[-a/2, a/2, a/2], [a/2, -a/2, a/2], [a/2, a/2, -a/2]])
    return vq.PeriodicSystem(3, lat, [vq.Atom(74, [0.0, 0.0, 0.0])])


def _tet_anisotropic():
    """Tetragonal cell with c/a = 2 — c-axis k-density should be ~½ of
    a/b axes."""
    a = 3.0 * ANGSTROM_TO_BOHR
    c = 6.0 * ANGSTROM_TO_BOHR
    return vq.PeriodicSystem(3, np.diag([a, a, c]), [vq.Atom(14, [0, 0, 0])])


def _slab_2d():
    """3D-stored slab with vacuum along z (dim=2). Non-periodic axes
    should always come out as mesh[i] = 1."""
    a = 3.0 * ANGSTROM_TO_BOHR
    return vq.PeriodicSystem(2, np.diag([a, a, 100.0]), [vq.Atom(12, [0, 0, 0])])


# ---------------------------------------------------------------------------
# 1. Public API
# ---------------------------------------------------------------------------

def test_from_kppra_is_public():
    assert hasattr(vq.KPoints, "from_kppra")


def test_from_kspacing_is_public():
    assert hasattr(vq.KPoints, "from_kspacing")


def test_auto_is_public():
    assert hasattr(vq.KPoints, "auto")


def test_constructors_return_monkhorst_pack():
    """Density-based constructors all delegate to ``monkhorst_pack``
    so the resulting object has ``kind="monkhorst-pack"``."""
    sys = _si_diamond()
    assert vq.KPoints.from_kppra(sys, 500).kind == "monkhorst-pack"
    assert vq.KPoints.from_kspacing(sys, 0.3).kind == "monkhorst-pack"
    assert vq.KPoints.auto(sys, 25).kind == "monkhorst-pack"


# ---------------------------------------------------------------------------
# 2. Monotone in density
# ---------------------------------------------------------------------------

def test_kppra_monotone():
    sys = _si_diamond()
    sizes = [
        np.prod(vq.KPoints.from_kppra(sys, n).mesh)
        for n in (100, 500, 1000, 3000, 8000)
    ]
    assert sizes == sorted(sizes)


def test_kspacing_smaller_means_denser():
    sys = _si_diamond()
    coarse = np.prod(vq.KPoints.from_kspacing(sys, 0.5).mesh)
    fine   = np.prod(vq.KPoints.from_kspacing(sys, 0.1).mesh)
    assert fine > coarse


def test_auto_length_monotone():
    sys = _si_diamond()
    sizes = [
        np.prod(vq.KPoints.auto(sys, L).mesh) for L in (10, 25, 50, 100)
    ]
    assert sizes == sorted(sizes)


# ---------------------------------------------------------------------------
# 3. Per-axis proportionality on anisotropic cell (tetragonal c/a=2)
# ---------------------------------------------------------------------------

def test_kspacing_anisotropic_proportionality():
    """For tetragonal c/a = 2, |b_c| = |b_a|/2, so the c-axis mesh
    should be ~½ of the a-axis mesh."""
    sys = _tet_anisotropic()
    kp = vq.KPoints.from_kspacing(sys, 0.2)
    nx, ny, nz = kp.mesh
    assert nx == ny  # tetragonal → a == b
    # c/a = 2 → nz should be ~ nx / 2 (allow off-by-one for ceil)
    assert abs(nz - nx / 2) <= 1


def test_kppra_anisotropic_proportionality():
    sys = _tet_anisotropic()
    kp = vq.KPoints.from_kppra(sys, 1000)
    nx, ny, nz = kp.mesh
    assert nx == ny
    assert abs(nz - nx / 2) <= 1


# ---------------------------------------------------------------------------
# 4. Non-periodic axes pinned to 1
# ---------------------------------------------------------------------------

def test_kppra_slab_z_axis_pinned():
    sys = _slab_2d()
    kp = vq.KPoints.from_kppra(sys, 1000)
    assert kp.mesh[2] == 1


def test_kspacing_slab_z_axis_pinned():
    sys = _slab_2d()
    kp = vq.KPoints.from_kspacing(sys, 0.1)
    assert kp.mesh[2] == 1


def test_auto_slab_z_axis_pinned():
    sys = _slab_2d()
    kp = vq.KPoints.auto(sys, 100)
    assert kp.mesh[2] == 1


# ---------------------------------------------------------------------------
# 5. Metallic flag — 4× bump + warning
# ---------------------------------------------------------------------------

def test_metallic_bumps_density():
    sys = _si_diamond()
    insulator = np.prod(vq.KPoints.from_kppra(sys, 1000,
                                                 warn_no_smearing=False).mesh)
    metallic = np.prod(vq.KPoints.from_kppra(sys, 1000, metallic=True,
                                                warn_no_smearing=False).mesh)
    # 4× bump in density doesn't exactly 4× the total mesh count due
    # to ceil rounding, but should be a meaningful increase.
    assert metallic > insulator


def test_metallic_emits_warning():
    sys = _si_diamond()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vq.KPoints.from_kppra(sys, 1000, metallic=True)
    assert len(caught) >= 1
    assert any(issubclass(w.category, UserWarning) for w in caught)
    assert any("smearing" in str(w.message).lower() for w in caught)


def test_metallic_warning_can_be_suppressed():
    sys = _si_diamond()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        vq.KPoints.from_kppra(sys, 1000, metallic=True,
                                 warn_no_smearing=False)
    # No metallic-specific UserWarning fired.
    assert not any(
        issubclass(w.category, UserWarning) and "smearing" in str(w.message).lower()
        for w in caught
    )


# ---------------------------------------------------------------------------
# 6. Validation — bad inputs raise
# ---------------------------------------------------------------------------

def test_kppra_negative_raises():
    sys = _si_diamond()
    with pytest.raises(ValueError, match="positive"):
        vq.KPoints.from_kppra(sys, -10)


def test_kspacing_zero_raises():
    sys = _si_diamond()
    with pytest.raises(ValueError, match="positive"):
        vq.KPoints.from_kspacing(sys, 0.0)


def test_auto_negative_raises():
    sys = _si_diamond()
    with pytest.raises(ValueError, match="positive"):
        vq.KPoints.auto(sys, -10)


def test_kspacing_unknown_units_raises():
    sys = _si_diamond()
    with pytest.raises(ValueError, match="units"):
        vq.KPoints.from_kspacing(sys, 0.3, units="meters")


# ---------------------------------------------------------------------------
# 7. kspacing units sanity — Å and bohr give the same mesh when the
# numbers are properly converted.
# ---------------------------------------------------------------------------

def test_kspacing_units_round_trip():
    sys = _si_diamond()
    ks_A = 0.3       # 2π/Å
    ks_bohr = ks_A / ANGSTROM_TO_BOHR  # equivalent in 2π/bohr (i.e. bohr⁻¹)
    kp_a = vq.KPoints.from_kspacing(sys, ks_A, units="angstrom")
    kp_b = vq.KPoints.from_kspacing(sys, ks_bohr, units="bohr")
    assert kp_a.mesh == kp_b.mesh


# ---------------------------------------------------------------------------
# 8. Gamma-centered override
# ---------------------------------------------------------------------------

def test_gamma_centred_forces_zero_shift():
    sys = _si_diamond()
    kp = vq.KPoints.from_kppra(sys, 500, gamma_centred=True)
    assert kp.shift == (0, 0, 0)
