"""Dense-ionic EWALD_3D fail-closed guard (CLAUDE.md §7).

The Γ-only EWALD_3D driver returns a ~2 Ha-wrong energy when periodic
images of the basis overlap, because its molecular-limit density convention
(D(g≠0)=0) drops the cross-cell density-matrix blocks. Rather than hand back
a wrong number, the driver now *fails closed* and points the user at the
correct routes (GDF / BIPOLE / GPW). The user-facing EWALD_3D route
(``jk_method='fft_poisson'``) is retired entirely.

The criterion is the largest cross-cell AO overlap |S_μν(L≠0)| — a
**basis-dependent** quantity, not a geometric one. It catches the
basis-overlap collapse mode measured 2026-06-13 (LDA / STO-3G):

* **LiH** rocksalt — Li's 2s/2p are extraordinarily diffuse (α=0.048), so
  even at the experimental lattice the periodic images overlap strongly
  (max cross-cell overlap 0.41): EWALD_3D −35.9655 vs GDF −33.9940,
  |Δ| = 1.97 Ha → the driver must RAISE. (2026-06-13 numbers; after
  b3f74aa9 paired the XC density set with the periodic Becke grid the
  same runs give −31.2656 vs −32.1402, |Δ| = 0.87 Ha: still Ha-scale
  wrong, the guard's rationale is unchanged.)
* **MgO** rocksalt (overlap 0.19) and **NaCl** rocksalt (0.12) sit below this
  overlap-specific guard. That does not make the molecular-limit EWALD bridge
  a PySCF-GDF absolute-energy parity route for condensed cells; the regression
  suite must use the public GDF routes for that contract. The guard must NOT
  fire solely from geometry — LiH and MgO have near-identical nearest-image
  *contact* (3.86 vs 3.98 bohr), so a geometric criterion would wrongly refuse
  MgO.

Run:
    .venv/bin/pytest tests/test_periodic_dense_ionic_energy_parity.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import vibeqc as vq
from vibeqc._vibeqc_core import CoulombMethod, LatticeSumOptions
from vibeqc.periodic_rhf_ewald import (
    _DENSE_IONIC_MAX_XCELL_OVERLAP,
    _max_cross_cell_ao_overlap,
)


def _rocksalt_conventional(z_cation: int, z_anion: int, a_ang: float):
    """Conventional cubic rocksalt cell (8 atoms, 4 formula units)."""
    a = a_ang / 0.529177210903
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(z_cation, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(z_anion, [fx * a, fy * a, fz * a]))
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _lih_rocksalt_conventional():
    return _rocksalt_conventional(3, 1, 4.084)


def _common_opts():
    """Shared SCF options for both EWALD_3D and GDF runs."""
    opts = vq.PeriodicKSOptions()
    opts.functional = "lda"
    opts.max_iter = 30
    opts.use_diis = True
    opts.damping = 0.7
    opts.initial_guess = vq.InitialGuess.SAD
    opts.conv_tol_energy = 1e-7
    return opts


def _ewald_opts():
    opts = _common_opts()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    return opts


def _dense_ewald_lat_opts():
    """The lattice-sum options the dense-cell driver path resolves to
    (EWALD_3D gauge, cutoff floored to 18 bohr) — for evaluating the guard
    criterion directly without an SCF."""
    lo = LatticeSumOptions()
    lo.coulomb_method = CoulombMethod.EWALD_3D
    lo.cutoff_bohr = 18.0
    lo.nuclear_cutoff_bohr = 25.0
    return lo


# ---------------------------------------------------------------------------
# Fail-closed on the buggy (image-overlapping) cell
# ---------------------------------------------------------------------------


def test_lih_rocksalt_ewald3d_fails_closed():
    """Γ-only EWALD_3D must RAISE on LiH rocksalt — Li's diffuse STO-3G
    valence makes the periodic images overlap, so the molecular-limit energy
    is ~2 Ha wrong (CLAUDE.md §7)."""
    system, basis = _lih_rocksalt_conventional()
    with pytest.raises(ValueError, match="overlap"):
        vq.run_rks_periodic_gamma_ewald3d(
            system, basis, _ewald_opts(), spacing_bohr=0.5, progress=False
        )


def test_dense_ionic_error_names_correct_routes():
    """The refusal must point users at the supported routes."""
    system, basis = _lih_rocksalt_conventional()
    with pytest.raises(ValueError) as exc:
        vq.run_rks_periodic_gamma_ewald3d(
            system, basis, _ewald_opts(), spacing_bohr=0.5, progress=False
        )
    msg = str(exc.value)
    for route in ("gdf", "bipole", "gpw"):
        assert route in msg, f"refusal should name jk_method={route!r}: {msg}"


# ---------------------------------------------------------------------------
# The criterion is basis-aware and surgical (does not over-fire)
# ---------------------------------------------------------------------------


def test_overlap_criterion_separates_lih_from_compact_crystals():
    """The guard is a basis-aware cross-cell-overlap criterion: LiH (diffuse
    Li valence) sits above the threshold, while the compact MgO and NaCl
    rocksalts sit below it — even though MgO's nearest-image *contact*
    (3.98 bohr) is essentially the same as LiH's (3.86 bohr). This pins the
    overlap-collapse guard only; condensed-cell PySCF-GDF parity is owned by
    the public GDF routes."""
    lo = _dense_ewald_lat_opts()

    lih_sys, lih_bas = _lih_rocksalt_conventional()
    lih_overlap = _max_cross_cell_ao_overlap(lih_sys, lih_bas, lo)
    assert lih_overlap > _DENSE_IONIC_MAX_XCELL_OVERLAP, (
        f"LiH cross-cell overlap {lih_overlap:.3f} should exceed the guard "
        f"threshold {_DENSE_IONIC_MAX_XCELL_OVERLAP} (it is the buggy case)."
    )

    for name, (zc, za, a) in {
        "MgO": (12, 8, 4.21),
        "NaCl": (11, 17, 5.640),
    }.items():
        system, basis = _rocksalt_conventional(zc, za, a)
        overlap = _max_cross_cell_ao_overlap(system, basis, lo)
        assert overlap < _DENSE_IONIC_MAX_XCELL_OVERLAP, (
            f"{name} cross-cell overlap {overlap:.3f} should be below the "
            f"guard threshold {_DENSE_IONIC_MAX_XCELL_OVERLAP}; the "
            "overlap-specific guard must not refuse it."
        )


# ---------------------------------------------------------------------------
# Live PASS guards
# ---------------------------------------------------------------------------


def test_lih_rocksalt_gdf_energy_in_physical_range():
    """GDF on LiH rocksalt gives a physical energy — it is the correct
    production route for the image-overlapping regime (live PASS guard)."""
    system, basis = _lih_rocksalt_conventional()
    r = vq.run_rks_periodic_gamma_gdf(system, basis, _common_opts(), progress=False)
    assert -50.0 < r.energy < -10.0, (
        f"LiH rocksalt GDF converged to {r.energy:.2f} Ha, "
        "outside the physical range (-50, -10) Ha."
    )


def test_fft_poisson_user_route_is_retired():
    """The user-facing Γ-only EWALD_3D route is retired: requesting
    ``jk_method='fft_poisson'`` raises and names the supported routes
    (instant — no SCF)."""
    from vibeqc.periodic_jk_method import PeriodicJKMethod, validate_jk_method

    with pytest.raises(ValueError) as exc:
        validate_jk_method(
            PeriodicJKMethod.FFT_POISSON,
            lattice=np.eye(3) * 10.0,
            basis_name="sto-3g",
        )
    msg = str(exc.value).lower()
    assert "retired" in msg
    for route in ("gdf", "bipole", "gpw"):
        assert route in msg, f"retirement error should name {route!r}: {msg}"


def test_dense_ionic_bypass_runs_but_is_unreliable():
    """The internal ``allow_dense_ionic`` escape hatch still runs the driver
    (for mechanics testing), and the energy it returns must never be
    trusted: the Γ-only EWALD_3D route keeps the molecular-limit density
    convention D(g≠0)=0 in its J build, which drops exactly the cross-cell
    density blocks this image-overlapping cell needs (the documented
    collapse). The specific wrong value is convention-dependent, so it is
    not pinned here: −35.9655 Ha as measured 2026-06-13 (home-cell-only XC
    density), −31.2656 Ha since b3f74aa9 paired the XC density set with
    the periodic Becke grid (Γ-torus set; the J build is unchanged). The
    GDF reference on this cell moved −33.9940 → −32.1402 Ha under the same
    fix, and the bypass stays Ha-scale away from it (|Δ| = 0.87 Ha,
    2026-07-11). The value is accelerator-independent (EDIIS_DIIS default,
    plain DIIS, and pre-cb712e21 defaults all converge to the identical
    −31.265579 Ha in 7 iterations), so re-pinning a window around any one
    number just re-arms this trap for the next convention fix. Pin the
    contract instead: it runs, converges, is bounded, and disagrees with
    GDF."""
    system, basis = _lih_rocksalt_conventional()
    r = vq.run_rks_periodic_gamma_ewald3d(
        system,
        basis,
        _ewald_opts(),
        spacing_bohr=0.5,
        allow_dense_ionic=True,
        progress=False,
    )
    assert r.converged, (
        "bypassed EWALD_3D on LiH should still converge (mechanics guard); "
        f"got converged=False after {r.n_iter} iterations."
    )
    assert -45.0 < r.energy < -25.0, (
        f"bypassed EWALD_3D on LiH gave {r.energy:.4f} Ha, outside the "
        "(-45, -25) Ha sanity window spanning the known wrong stationary "
        "points (-35.97 pre-b3f74aa9, -31.27 after)."
    )
    # GDF reference for this cell: -32.1402 Ha (2026-07-11, post-b3f74aa9;
    # was -33.9940 before it). The live GDF guard above re-checks the
    # route's physical range; this constant only anchors the disagreement.
    e_gdf_ref = -32.1402
    assert abs(r.energy - e_gdf_ref) > 0.5, (
        f"bypassed EWALD_3D on LiH gave {r.energy:.4f} Ha, within 0.5 Ha "
        f"of the GDF reference {e_gdf_ref} Ha. The molecular-limit bypass "
        "agreeing with GDF on an image-overlapping cell would mean the "
        "documented collapse mode is gone -- re-evaluate the fail-closed "
        "guard rather than trusting this route."
    )
