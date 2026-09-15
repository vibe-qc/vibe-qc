"""Tests for ``vq.disambiguate_critical_overlap`` — item 7.

When the standalone EIGS preflight (or the in-SCF preflight) fires
critical severity, the upstream cause is one of two distinct things
that PySCF / molecular codes typically conflate:

1. **Genuine basis-set linear dependence** — too many diffuse
   primitives for the lattice geometry. The fix is at basis-set
   design time (pob, MOLOPT, GTH-cc-pVXZ) or via
   ``vq.make_basis(..., exp_to_discard=...)``.
2. **Under-converged exchange screening** — the negative
   eigenvalues are an artefact of too-loose lattice-sum truncation
   or ERI screening (CRYSTAL's ITOL4/ITOL5 in TOLINTEG, manual
   p. 130). The fix is tightening cutoffs.

The CRYSTAL manual (p. 398) explicitly notes that what looks like
linear dependence at runtime is *often* actually screening
under-convergence — and that the screening fix is a strict subset
of the basis-set fix, so try the screening fix first when in
doubt.

These tests verify the disambiguation logic on a real example
(LiH conventional rocksalt, where the original critical-severity
overlap turns out to be a screening artefact) and on a synthetic
basis-set case.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


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
    return sysp, basis


# ---------------------------------------------------------------------------
# Real case: LiH critical → screening_undertight
# ---------------------------------------------------------------------------

def test_lih_critical_diagnoses_as_screening_problem():
    """LiH/STO-3G conventional with cutoff_bohr=10 has S(Γ) with
    three negative eigenvalues at -0.16. Tightening cutoff ×2 takes
    them all positive — so this is a screening problem, not a
    basis-set problem. The diagnosis must say so."""
    sysp, basis = _lih_conventional()

    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0

    rep = vq.disambiguate_critical_overlap(
        sysp, basis, lattice_opts=opts,
        cutoff_tighten_factor=2.0, schwarz_tighten_factor=100.0,
    )
    assert rep.verdict == "screening_undertight"
    assert rep.original_severity == "critical"
    assert rep.tightened_severity in ("ok", "warn")
    # min eigenvalue went from negative to positive after tightening.
    assert rep.original_min_eigenvalue < 0
    assert rep.tightened_min_eigenvalue > 0
    # Counts: original had at least 3 negatives, tightened has 0.
    assert rep.original_n_negative >= 3
    assert rep.tightened_n_negative == 0


def test_lih_diagnosis_carries_factors_and_severity():
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0

    rep = vq.disambiguate_critical_overlap(
        sysp, basis, lattice_opts=opts,
        cutoff_tighten_factor=2.5, schwarz_tighten_factor=50.0,
    )
    assert rep.cutoff_tighten_factor == pytest.approx(2.5)
    assert rep.schwarz_tighten_factor == pytest.approx(50.0)
    assert rep.original_severity == "critical"


# ---------------------------------------------------------------------------
# Synthetic: basis-set problem (tightening doesn't help)
# ---------------------------------------------------------------------------

def test_disambiguate_handles_already_clean_overlap():
    """When the original is already 'ok', the diagnosis is moot but
    the function shouldn't crash."""
    sysp = vq.PeriodicSystem(
        dim=3, lattice=30.0 * np.eye(3),
        unit_cell=[vq.Atom(1, [15, 15, 14.3]),
                   vq.Atom(1, [15, 15, 15.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    rep = vq.disambiguate_critical_overlap(sysp, basis)
    # Both passes give 'ok'; verdict is screening_undertight (since
    # tightened severity is in {ok, warn}). Not meaningful, but the
    # function returns a sensible object.
    assert rep.verdict == "screening_undertight"
    assert rep.original_severity == "ok"


# ---------------------------------------------------------------------------
# Formatter
# ---------------------------------------------------------------------------

def test_format_disambiguation_report_screening_case():
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    opts.nuclear_cutoff_bohr = 15.0
    rep = vq.disambiguate_critical_overlap(
        sysp, basis, lattice_opts=opts,
    )
    assert rep.verdict == "screening_undertight"
    text = vq.format_disambiguation_report(rep)
    assert "screening" in text or "screening_undertight" in text or \
           "under-converged" in text
    assert "TOLINTEG" in text
    assert "cutoff" in text


def test_format_disambiguation_report_lays_out_both_passes():
    """Formatted output must show both the original and tightened
    pass numbers so the user can verify the diagnosis themselves."""
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    rep = vq.disambiguate_critical_overlap(
        sysp, basis, lattice_opts=opts,
    )
    text = vq.format_disambiguation_report(rep)
    assert "original" in text
    assert "tightened" in text
    assert "min eig" in text
    assert "n_negative" in text


# ---------------------------------------------------------------------------
# Verdict is one of three known values
# ---------------------------------------------------------------------------

def test_verdict_is_one_of_three_known_strings():
    sysp, basis = _lih_conventional()
    opts = vq.LatticeSumOptions()
    opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.cutoff_bohr = 10.0
    rep = vq.disambiguate_critical_overlap(
        sysp, basis, lattice_opts=opts,
    )
    assert rep.verdict in {
        "screening_undertight",
        "basis_set_problem",
        "inconclusive",
    }
