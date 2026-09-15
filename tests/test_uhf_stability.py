"""UHF internal-stability regression: E(UHF) <= E(ROHF) is a variational identity.

Registry defect UHF-ABOVE-ROHF-VARIATIONAL-INVERSION
(agentic-loop/bug-claims.md, judgements-2026-08-05-backlog): the ROHF
determinant lies inside the UHF variational space, so a UHF solution
reported as converged must never sit above the ROHF energy on the same
system/basis/state. On O2 (charge 0, multiplicity 5, cc-pVDZ) the default
molecular UHF landed on an internally UNSTABLE excited SCF solution at
-149.1039798772 Ha — +59.15 mHa ABOVE the ROHF energy -149.1631321384 Ha
— while reporting clean convergence (10 iterations, S^2 = 6.0463).

Measured during the fix investigation (~/.claude-fixer-runs/uhf-above-rohf/
EVIDENCE.md): the internal (real UHF->UHF) orbital-rotation Hessian at that
solution has lowest eigenvalue -0.0869 (doubly degenerate), i.e. it is a
saddle point of the SCF energy surface, not a local minimum (stability
criterion per Lehtola, Molecules 25, 1218 (2020), Section 10; original
condition Seeger & Pople, J. Chem. Phys. 66, 3045 (1977)). Reconverging
after a line-searched rotation along the unstable mode reaches the true
UHF minimum at -149.1799299198 Ha (16.80 mHa BELOW ROHF, S^2 = 6.0222).

Geometry adapted from qc-input-library deck 00949 (o2-triplet/rohf/def2-svp;
R(O-O) = 1.208 Angstrom), multiplicity 3 -> 5 and basis cc-pVDZ to match
the registry evidence (no m=5 O2 deck exists in the input library).
"""

from __future__ import annotations

import numpy as np
import pytest

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    ROHFOptions,
    UHFOptions,
    UKSOptions,
    run_rohf,
    run_uhf,
)

ANGSTROM_TO_BOHR = 1.0 / 0.529177210903

# Internally derived reference values for O2 m=5 / cc-pVDZ at
# R(O-O) = 1.208 A (this defect's investigation, 2026-08-05; see the
# module docstring). NOT external-program numbers: the point pinned here
# is the variational ordering and which SCF solution the driver lands on.
E_UHF_MIN = -149.1799299198  # true UHF minimum (internally stable)
E_UHF_EXCITED = -149.1039798772  # the unstable saddle the old driver found
E_ROHF = -149.1631321384


def _o2_quintet() -> Molecule:
    return Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(8, [0.0, 0.0, 1.208 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=5,
    )


def _cr2_singlet(distance: float) -> Molecule:
    return Molecule(
        [
            Atom(24, [0.0, 0.0, -distance / 2]),
            Atom(24, [0.0, 0.0, distance / 2]),
        ],
        multiplicity=1,
    )


def test_cr2_stability_solver_finds_global_lowest_sector():
    """Issue 138: the R=2.5 saddle's global mode is not diag.argmin's irrep."""
    mol = _cr2_singlet(2.5)
    opts = UHFOptions()
    opts.stability_max_retries = 0
    result = run_uhf(mol, BasisSet(mol, "sto-3g"), opts)

    assert result.converged
    assert result.stability_analysis_converged
    assert result.internal_instability
    assert result.n_stability_restarts == 0
    assert result.stability_eigenvalue == pytest.approx(
        -0.3438948984, abs=1e-6
    )


def test_o2_quintet_uhf_below_rohf():
    """The reproducer: default-option UHF must not sit above ROHF."""
    mol = _o2_quintet()

    uopts = UHFOptions()
    uopts.max_iter = 200
    ures = run_uhf(mol, "cc-pvdz", uopts)
    assert ures.converged

    ropts = ROHFOptions()
    ropts.max_iter = 200
    rres = run_rohf(mol, BasisSet(mol, "cc-pvdz"), ropts)
    assert rres.converged
    assert rres.energy == pytest.approx(E_ROHF, abs=1e-6)

    # The variational identity this defect violated: +59.15 mHa inverted
    # pre-fix. Small tolerance for the identity itself; the stronger
    # assertion below pins WHICH solution was found.
    assert ures.energy <= rres.energy + 1e-8, (
        f"E(UHF) = {ures.energy:.10f} Ha lies ABOVE E(ROHF) = "
        f"{rres.energy:.10f} Ha (+{(ures.energy - rres.energy) * 1e3:.2f} "
        "mHa): UHF landed on an excited SCF solution"
    )

    # The driver must land on the internally stable UHF minimum, not just
    # any solution below ROHF.
    assert ures.energy == pytest.approx(E_UHF_MIN, abs=1e-6)
    assert ures.s_squared == pytest.approx(6.0222, abs=5e-3)


def test_o2_quintet_stability_metadata():
    """The stability check reports its verdict on the result object.

    With the Hund-split SAD guess (2026-08-08), the O2 m=5 SCF lands
    directly in the correct basin — no corrective restart is needed.
    The stability verdict confirms the solution is internally stable."""
    mol = _o2_quintet()
    opts = UHFOptions()
    opts.max_iter = 200
    res = run_uhf(mol, "cc-pvdz", opts)
    assert res.converged

    assert res.stability_checked
    assert res.stability_analysis_converged
    # With the improved guess the solution is stable from the start.
    assert res.n_stability_restarts == 0
    assert res.stability_eigenvalue >= -opts.stability_tol


def test_o2_quintet_stability_check_opt_out():
    """stability_check=False returns the same energy as the default path.

    The Hund-split SAD guess (2026-08-08) places the SCF in the correct
    basin from the first iteration, so the stability escape is never
    triggered.  With stability_check=False the same minimum is reached."""
    mol = _o2_quintet()
    opts = UHFOptions()
    opts.max_iter = 200
    opts.stability_check = False
    res = run_uhf(mol, "cc-pvdz", opts)
    assert res.converged
    assert not res.stability_checked
    assert res.n_stability_restarts == 0
    # The correct UHF minimum, same as with stability ON.
    assert res.energy == pytest.approx(E_UHF_MIN, abs=1e-6)


def test_oh_doublet_stable_solution_unchanged():
    """Control: a stable UHF solution passes the check with no restart.

    OH radical / def2-svp converges to an internally stable minimum; the
    stability pass must not perturb the converged energy.
    """
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=2,
    )
    opts = UHFOptions()
    opts.max_iter = 200
    res = run_uhf(mol, "def2-svp", opts)
    assert res.converged
    assert res.stability_checked
    assert res.stability_analysis_converged
    assert res.n_stability_restarts == 0
    assert res.stability_eigenvalue >= -opts.stability_tol

    opts_off = UHFOptions()
    opts_off.max_iter = 200
    opts_off.stability_check = False
    res_off = run_uhf(mol, "def2-svp", opts_off)
    assert res.energy == pytest.approx(res_off.energy, abs=1e-9)


def test_uhf_stability_budget_exhaustion_is_no_verdict():
    """Issue #398: an exhausted Davidson solve must fail closed."""
    mol = Molecule(
        [
            Atom(8, [0.0, 0.0, 0.0]),
            Atom(1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR]),
        ],
        charge=0,
        multiplicity=2,
    )
    opts = UHFOptions()
    opts.max_iter = 200
    opts.stability_davidson_max_iter = 0
    res = run_uhf(mol, "def2-svp", opts)

    assert res.converged
    assert res.stability_checked
    assert not res.stability_analysis_converged
    assert not res.internal_instability


# ---------------------------------------------------------------------------
# BUG 88 — FeCl3 UHF/cc-pVDZ root-selection failure (FIXED 2026-08-08)
#
# The old proportional-spin-split SAD guess landed FeCl3 UHF/cc-pVDZ on an
# excited SCF saddle 105.87 mHa above the ground state.  The Hund-split
# SAD guess (per-atom occ_alpha / occ_beta, Hund's rule) places the SCF
# in the correct basin from iteration 1, matching ORCA to 2.5e-9 Ha.
# These tests pin (a) the old behaviour is gone and (b) the new behaviour
# is internally stable without corrective restarts.
# ---------------------------------------------------------------------------

# FeCl3 UHF/cc-pVDZ reference energies:
#   ORCA 6.1.1:                     -2641.1344915755 Ha  (S^2 = 8.767533)
#   vibe-qc Hund-split SAD (fixed): -2641.1344915780 Ha  (S^2 = 8.767533)
#   vibe-qc old proportional (bug): -2641.0286223698 Ha  (+105.87 mHa)
FECL3_CCPVDZ_UHF_GROUND = -2641.1344915780   # new Hund-split guess result
FECL3_CCPVDZ_UHF_EXCITED = -2641.0286223698   # old proportional-split (BUG 88)


def _fecl3_sextet():
    """Trigonal-planar FeCl3, high-spin d5 Fe(III) — the BUG 88 / BUG 118 system."""
    return Molecule(
        [
            Atom(26, [0.0, 0.0, 0.0]),
            Atom(17, [2.13 * ANGSTROM_TO_BOHR, 0.0, 0.0]),
            Atom(17, [-1.065 * ANGSTROM_TO_BOHR, 1.84463411 * ANGSTROM_TO_BOHR, 0.0]),
            Atom(17, [-1.065 * ANGSTROM_TO_BOHR, -1.84463411 * ANGSTROM_TO_BOHR, 0.0]),
        ],
        charge=0,
        multiplicity=6,
    )


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_bug88_fecl3_uhf_ccpvdz_reaches_ground_state():
    """BUG 88 / issue #398: FeCl3 reaches and certifies its ground state.

    The Hund-split SAD guess places the SCF in the correct basin from the
    start.  The post-SCF Davidson solve must also reach its sign-certifying
    residual with the bounded default budget, without a stability restart.
    """
    mol = _fecl3_sextet()
    opts = UHFOptions()
    opts.max_iter = 300
    res = run_uhf(mol, "cc-pvdz", opts)
    assert res.converged
    assert res.stability_checked
    assert res.stability_analysis_converged
    # Internally stable from the start — no restarts needed.
    assert res.n_stability_restarts == 0
    assert not res.internal_instability
    assert res.stability_eigenvalue >= -opts.stability_tol
    # Energy matches ORCA 6.1.1 to machine precision.
    orca_ref = -2641.1344915755
    assert abs(res.energy - orca_ref) < 1e-6, (
        f"delta from ORCA = {(res.energy - orca_ref)*1000:.3f} mHa"
    )
    # S^2 should match ORCA's 8.767533 (BUG 77).
    assert res.s_squared == pytest.approx(8.767533, abs=1e-3)
    assert res.s_squared_ideal == pytest.approx(8.75, abs=1e-4)


@pytest.mark.slow
@pytest.mark.timeout(600)
def test_bug88_fecl3_uhf_ccpvdz_stability_off_same_result():
    """BUG 88 (FIXED): with stability_check=False, FeCl3 UHF/cc-pVDZ
    reaches the same ground state — the guess quality, not the stability
    escape, now determines the basin."""
    mol = _fecl3_sextet()
    opts = UHFOptions()
    opts.max_iter = 300
    opts.stability_check = False
    res = run_uhf(mol, "cc-pvdz", opts)
    assert res.converged
    assert not res.stability_checked
    # The ground state, NOT the old excited saddle.
    assert res.energy < FECL3_CCPVDZ_UHF_EXCITED + 1e-3, (
        f"energy {res.energy:.10f} is near the old excited saddle "
        f"({FECL3_CCPVDZ_UHF_EXCITED:.10f}) — the Hund-split guess "
        f"should have landed in the ground-state basin"
    )
    orca_ref = -2641.1344915755
    assert abs(res.energy - orca_ref) < 1e-6


# ---------------------------------------------------------------------------
# GitLab #566 -- three states, not two.
#
# ``internal_instability`` is the fail-closed conjunction
# ``stability_checked && converged && lambda < -tol``.  False therefore means
# EITHER certified-stable OR no verdict at all, and two consumers keyed on it
# alone: the multi-guess filter (which could rank an unexamined seed above a
# verified one on energy) and the loud instability warning (which stayed
# silent for a run with no verdict).
# ---------------------------------------------------------------------------


class _FakeUHF:
    """Minimal stand-in carrying only what the multi-guess filter reads."""

    def __init__(self, energy, *, checked, stab_converged, unstable=False):
        self.energy = float(energy)
        self.converged = True
        self.s_squared = 0.75
        self.stability_checked = checked
        self.stability_analysis_converged = stab_converged
        self.internal_instability = unstable
        self.stability_eigenvalue = -1.0e-3 if unstable else 1.0e-3


def _oh_doublet():
    return Molecule(
        [Atom(8, [0.0, 0.0, 0.0]),
         Atom(1, [0.0, 0.0, 0.97 * ANGSTROM_TO_BOHR])],
        charge=0,
        multiplicity=2,
    )


def test_multi_guess_does_not_promote_a_no_verdict_seed(monkeypatch):
    """A lower-energy seed with no stability verdict must not be selected.

    Pre-fix the filter skipped only ``internal_instability`` seeds, so an
    alternate whose stability Davidson exhausted its budget -- carrying the
    same ``internal_instability == False`` a certified-stable result carries --
    won the comparison on energy alone and was returned as the answer.
    """
    import vibeqc

    primary = _FakeUHF(-75.0, checked=True, stab_converged=True)
    # Lower in energy, but the analysis ran and produced no verdict.
    alternate = _FakeUHF(-75.5, checked=True, stab_converged=False)

    calls = {"n": 0}

    def fake_uhf(molecule, basis, options):
        calls["n"] += 1
        return primary if calls["n"] == 1 else alternate

    monkeypatch.setattr(vibeqc, "_run_uhf_cxx", fake_uhf)

    opts = UHFOptions()
    opts.multi_guess_seeds = [1, 2]
    out = vibeqc.run_uhf(_oh_doublet(), "sto-3g", opts)

    assert calls["n"] > 1, "the multi-guess loop did not run"
    assert out.energy == primary.energy, (
        "a no-verdict alternate was promoted over a verdict-carrying seed "
        f"({out.energy} vs {primary.energy})"
    )


def test_multi_guess_still_promotes_when_stability_was_never_checked(
    monkeypatch,
):
    """The guard must not disable multi-guess for runs without the analysis.

    ``stability_analysis_converged`` is also False when the analysis was never
    requested, so keying on it alone would have set aside every seed and
    silently disabled multi-guess selection. ``stability_checked`` is what
    separates "examined, no verdict" from "never examined".
    """
    import vibeqc

    primary = _FakeUHF(-75.0, checked=False, stab_converged=False)
    alternate = _FakeUHF(-75.5, checked=False, stab_converged=False)

    calls = {"n": 0}

    def fake_uhf(molecule, basis, options):
        calls["n"] += 1
        return primary if calls["n"] == 1 else alternate

    monkeypatch.setattr(vibeqc, "_run_uhf_cxx", fake_uhf)

    opts = UHFOptions()
    opts.multi_guess_seeds = [1]
    out = vibeqc.run_uhf(_oh_doublet(), "sto-3g", opts)

    assert out.energy == alternate.energy, (
        "the #566 guard disabled multi-guess for a run that never ran the "
        "stability analysis -- stability_checked must gate it"
    )


def test_multi_guess_still_rejects_a_genuinely_unstable_seed(monkeypatch):
    """The original fail-closed behaviour is unchanged."""
    import vibeqc

    primary = _FakeUHF(-75.0, checked=True, stab_converged=True)
    alternate = _FakeUHF(-75.5, checked=True, stab_converged=True, unstable=True)

    calls = {"n": 0}

    def fake_uhf(molecule, basis, options):
        calls["n"] += 1
        return primary if calls["n"] == 1 else alternate

    monkeypatch.setattr(vibeqc, "_run_uhf_cxx", fake_uhf)

    opts = UHFOptions()
    opts.multi_guess_seeds = [1]
    out = vibeqc.run_uhf(_oh_doublet(), "sto-3g", opts)

    assert out.energy == primary.energy


@pytest.mark.parametrize(
    ("options_factory", "kernel_name", "runner_name"),
    [
        (UHFOptions, "_run_uhf_cxx", "run_uhf"),
        (UKSOptions, "_run_uks_cxx", "run_uks"),
    ],
)
def test_no_verdict_seed_does_not_hide_later_certified_same_state(
    monkeypatch,
    options_factory,
    kernel_name,
    runner_name,
):
    """Rejected evidence must not poison fingerprint deduplication (#566).

    Two seeds can converge to the same SCF state while only the later
    stability solve obtains a verdict.  The no-verdict state is unranked; it
    therefore cannot make the certified copy look like an already-ranked
    duplicate.
    """
    import vibeqc

    primary = _FakeUHF(-75.0, checked=True, stab_converged=True)
    no_verdict = _FakeUHF(-75.5, checked=True, stab_converged=False)
    certified = _FakeUHF(-75.5, checked=True, stab_converged=True)
    outcomes = iter((primary, no_verdict, certified))

    monkeypatch.setattr(
        vibeqc,
        kernel_name,
        lambda molecule, basis, options: next(outcomes),
    )

    opts = options_factory()
    opts.multi_guess_seeds = [1, 2]
    out = getattr(vibeqc, runner_name)(_oh_doublet(), "sto-3g", opts)

    assert out is certified


def test_no_verdict_run_warns_that_stability_is_unverified(tmp_path, monkeypatch):
    """A budget-exhausted stability solve must say so at the warning surface.

    Pre-fix only the .out carried an UNVERIFIED line; the loud warning was
    gated on ``internal_instability`` alone, so a consumer reading warnings saw
    a no-verdict run as indistinguishable from a certified-stable one.
    """
    import vibeqc as vq

    monkeypatch.chdir(tmp_path)
    opts = UHFOptions()
    opts.max_iter = 200
    opts.stability_davidson_max_iter = 0   # force the no-verdict state
    vq.run_job(
        _oh_doublet(), basis="def2-svp", method="uhf",
        uhf_options=opts, output="noverdict",
    )
    out = (tmp_path / "noverdict.out").read_text()

    assert "STABILITY UNVERIFIED" in out, (
        "a no-verdict stability analysis produced no warning in the .out"
    )
    # It must not claim instability either -- there is no verdict.
    assert "SCF INTERNAL INSTABILITY" not in out
