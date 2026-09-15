"""Tests for ``vq.eigs_preflight`` — the standalone EIGS-equivalent.

Mirrors CRYSTAL's EIGS keyword (manual p. 103, 398): build the
overlap matrix at every requested k-point, diagonalise, classify
severity. **No SCF runs.** The user can validate basis-set
suitability at a controlled point in the workflow before committing
to expensive SCF cycles.

Key behaviours:

* **Clean basis at Γ** (H₂/STO-3G in 30 bohr box) → severity = ok.
* **Zone-boundary linear dependence** — LiH/STO-3G is well-conditioned
  at Γ but loses positive-definiteness at the X-points (b/2, 0, 0)
  and (0, b/2, 0). This is the textbook k-point-dependent failure
  mode (Searle, Bernasconi, Harrison ARCHER eCSE04-16, 2017).
* **No SCF side effects** — calling ``eigs_preflight`` doesn't
  populate any caches that affect a subsequent SCF.
* **Reporting** — formatted output is stable and contains the
  per-k summary table + actionable recommendations on
  warn/error/critical.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _h2_in_box(L: float = 30.0):
    sysp = vq.PeriodicSystem(
        dim=3, lattice=L * np.eye(3),
        unit_cell=[
            vq.Atom(1, [L / 2, L / 2, L / 2 - 0.7]),
            vq.Atom(1, [L / 2, L / 2, L / 2 + 0.7]),
        ],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis


def _lih_conventional():
    a = 4.084 / 0.529177210903   # bohr
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                        (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0),
                        (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))
    sysp = vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    return sysp, basis, a


# ---------------------------------------------------------------------------
# Clean cases
# ---------------------------------------------------------------------------

def test_eigs_preflight_h2_clean_at_gamma():
    sysp, basis = _h2_in_box()
    rep = vq.eigs_preflight(sysp, basis)
    assert rep.severity == "ok"
    assert rep.is_safe_to_run_scf()
    assert rep.n_basis == 2
    assert rep.n_kpoints == 1
    assert rep.worst_min_eigenvalue > 0
    assert np.isfinite(rep.worst_condition_number)


def test_eigs_preflight_default_kpoint_is_gamma():
    """When no k-points are passed, the preflight only checks Γ."""
    sysp, basis = _h2_in_box()
    rep = vq.eigs_preflight(sysp, basis)
    assert rep.n_kpoints == 1
    np.testing.assert_allclose(rep.k_points_cart, [[0, 0, 0]])


def test_eigs_preflight_runs_no_scf_side_effects():
    """Calling eigs_preflight should leave no state that would
    short-circuit a subsequent SCF — no cached densities, no
    modified basis."""
    sysp, basis = _h2_in_box()
    rep = vq.eigs_preflight(sysp, basis)
    assert rep.severity == "ok"
    # Subsequent SCF should run normally and produce the standard
    # H2 energy (matches the v0.7 fix value).
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.max_iter = 5
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged
    # v0.10.0: ω auto-derived from nuclear_cutoff_bohr (49f8ae91)
    # for jellium cancellation. Energy reference re-baselined at
    # nuclear_cutoff_bohr=25 → ω ≈ 0.21, in a 30-bohr box on the
    # 75³ FFT grid (spacing 0.4). Was -1.115774 Ha at the pre-override
    # ω = 0.5, then -1.1114 under the pre-restoration gauge.
    # v0.12.0 re-pin: the 86c39851 EWALD_3D gauge restoration recovers
    # the molecular limit — isolated H2/STO-3G RHF at this geometry is
    # -1.1167143 Ha (verified with vq.run_rhf), and the 30-bohr-box
    # driver now reproduces it to sub-mHa.
    assert r.energy == pytest.approx(-1.1167, abs=1e-3)


# ---------------------------------------------------------------------------
# Multi-k & zone-boundary failure
# ---------------------------------------------------------------------------

def test_eigs_preflight_lih_critical_at_zone_boundary():
    """The textbook CRYSTAL/Searle observation: LiH/STO-3G's S(Γ) is
    already non-PSD with the bare libint lattice sum. At zone-boundary
    k-points the situation is at least as bad."""
    sysp, basis, a = _lih_conventional()
    b = 2 * np.pi / a
    # Γ + 3 zone-boundary k-points.
    k_test = np.array([
        [0,    0,    0],
        [b/2,  0,    0],
        [0,    b/2,  0],
        [b/2,  b/2,  b/2],
    ])
    rep = vq.eigs_preflight(sysp, basis, k_test)
    assert rep.n_kpoints == 4
    # At least the X-points must be critical.
    severities = [r.severity for r in rep.per_k_reports]
    assert "critical" in severities
    # Worst severity is critical → not safe to run SCF.
    assert rep.severity == "critical"
    assert not rep.is_safe_to_run_scf()


def test_eigs_preflight_per_k_report_carries_full_info():
    """Each per-k report is a full LinearDependenceReport — same
    fields as the in-SCF preflight."""
    sysp, basis, a = _lih_conventional()
    rep = vq.eigs_preflight(sysp, basis, [[2 * np.pi / a / 2, 0, 0]])
    assert rep.n_kpoints == 1
    sub = rep.per_k_reports[0]
    assert sub.severity == "critical"
    assert sub.n_negative >= 1
    assert sub.min_eigenvalue < 0
    assert sub.max_eigenvalue > 0


def test_eigs_preflight_with_filtered_basis_passes_at_lih():
    """With ``vq.make_basis(..., exp_to_discard=0.1)``, the LiH
    overlap is PSD — eigs_preflight clears at Γ."""
    sysp, _, _ = _lih_conventional()
    basis_f = vq.make_basis(sysp.unit_cell_molecule(), "sto-3g",
                             exp_to_discard=0.1)
    rep = vq.eigs_preflight(sysp, basis_f)
    assert rep.severity in ("ok", "warn")   # was 'critical' unfiltered
    assert rep.is_safe_to_run_scf()
    vq.clear_filtered_basis_cache()


# ---------------------------------------------------------------------------
# Reporting / formatting
# ---------------------------------------------------------------------------

def test_format_eigs_report_clean_case():
    sysp, basis = _h2_in_box()
    rep = vq.eigs_preflight(sysp, basis)
    text = vq.format_eigs_report(rep)
    assert "ok" in text or "well-conditioned" in text
    assert "Safe to run SCF" in text


def test_format_eigs_report_critical_case_recommends_pob_and_make_basis():
    sysp, basis, a = _lih_conventional()
    b = 2 * np.pi / a
    rep = vq.eigs_preflight(sysp, basis, [[b/2, 0, 0]])
    assert rep.severity == "critical"
    text = vq.format_eigs_report(rep)
    assert "pob-tzvp" in text or "pob-tzvp-rev2" in text
    assert "make_basis" in text
    assert "exp_to_discard" in text
    assert "TOLINTEG" in text   # the CRYSTAL screening-vs-basis distinction


def test_format_eigs_report_per_k_eigenvalues_optional():
    sysp, basis, a = _lih_conventional()
    rep = vq.eigs_preflight(sysp, basis, [[0, 0, 0], [np.pi / a, 0, 0]])
    text_compact = vq.format_eigs_report(rep, show_per_k_eigenvalues=False)
    text_verbose = vq.format_eigs_report(rep, show_per_k_eigenvalues=True)
    assert len(text_verbose) > len(text_compact)
    assert "lowest" in text_verbose
    assert "lowest" not in text_compact


# ---------------------------------------------------------------------------
# Threshold customisation
# ---------------------------------------------------------------------------

def test_eigs_preflight_tighter_warn_threshold_promotes_severity():
    """A near-zero eigenvalue that's "ok" at the default warn=1e-6
    becomes "warn" / "error" at a tighter threshold."""
    sysp, basis = _h2_in_box()
    # Default thresholds: H2/STO-3G at Γ is OK.
    rep_default = vq.eigs_preflight(sysp, basis)
    assert rep_default.severity == "ok"

    # Aggressive: warn at 1.0, error at 0.5 — picks up the smallest
    # eigenvalue (~0.34).
    rep_strict = vq.eigs_preflight(
        sysp, basis,
        warn_threshold=1.0, error_threshold=0.5,
    )
    assert rep_strict.severity in ("warn", "error")


def test_eigs_preflight_threshold_propagates_to_subreports():
    sysp, basis = _h2_in_box()
    rep = vq.eigs_preflight(
        sysp, basis,
        warn_threshold=2e-6, error_threshold=2e-9,
    )
    for sub in rep.per_k_reports:
        assert sub.warn_threshold == 2e-6
        assert sub.error_threshold == 2e-9
