"""Tests for the pre-flight linear-dependence diagnostic.

The diagnostic inspects the AO overlap matrix and returns a
structured report. Two test regimes:

1. **Well-behaved bases** — STO-3G / 6-31G on standard geometries —
   return ``severity == "ok"`` with min eigenvalue well above the
   warn threshold.

2. **Pathological bases** — large diffuse bases on unusually close
   atoms — return ``severity == "warn"`` or ``"error"`` with the
   offending basis functions identified by their atom, shell
   angular momentum, and minimum primitive exponent.

Also checks that:

- ``raise_if_severe`` raises :class:`LinearDependenceError` for
  severe reports, carrying the report on ``.report`` for inspection.
- The offender list is ordered by contribution weight (largest first).
- The formatter output is stable and self-describing.
"""

from __future__ import annotations

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Well-behaved cases
# ---------------------------------------------------------------------------

def test_h2o_sto3g_is_well_conditioned():
    """H2O at a normal geometry with STO-3G is the baseline
    well-conditioned case. Min eigenvalue should be ~O(1e-1)."""
    mol = vq.Molecule([
        vq.Atom(8, [0, 0, 0]),
        vq.Atom(1, [0, 1.5, -1.2]),
        vq.Atom(1, [0, -1.5, -1.2]),
    ])
    basis = vq.BasisSet(mol, "sto-3g")
    report = vq.check_linear_dependence(basis)
    assert report.severity == "ok"
    assert report.n_below_warn == 0
    assert report.n_below_error == 0
    assert report.min_eigenvalue > 1e-3
    assert report.n_basis == basis.nbasis
    # Eigenvalue array is sorted ascending.
    assert np.all(np.diff(report.eigenvalues) >= -1e-14)


def test_well_conditioned_has_empty_offender_list():
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    basis = vq.BasisSet(mol, "sto-3g")
    report = vq.check_linear_dependence(basis)
    assert report.severity == "ok"
    assert report.offenders == []


def test_condition_number_is_max_over_min():
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    basis = vq.BasisSet(mol, "sto-3g")
    report = vq.check_linear_dependence(basis)
    assert report.condition_number == pytest.approx(
        report.max_eigenvalue / report.min_eigenvalue, rel=1e-12
    )


# ---------------------------------------------------------------------------
# Pathological case: tight H2 with aug-cc-pVTZ
# ---------------------------------------------------------------------------

def _tight_h2_augmented():
    """Two H atoms at 0.5 bohr (way too close) with aug-cc-pVTZ —
    reliably triggers near-linear-dependence via the doubled diffuse
    s/p functions."""
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0.5, 0, 0])])
    basis = vq.BasisSet(mol, "aug-cc-pvtz")
    return mol, basis


def test_pathological_case_flagged_as_warn_or_error():
    _, basis = _tight_h2_augmented()
    report = vq.check_linear_dependence(basis)
    assert report.severity in {"warn", "error"}, (
        f"expected near-linear-dependence flag, got severity={report.severity}, "
        f"min_eig={report.min_eigenvalue:.3e}"
    )
    assert report.min_eigenvalue < vq.DEFAULT_WARN_THRESHOLD
    assert report.n_below_warn >= 1


def test_offenders_identify_diffuse_functions():
    """The near-null space is spanned by the most-diffuse pair of
    s-functions on the two too-close H atoms. Verify the top
    offender is one of the diffuse functions."""
    _, basis = _tight_h2_augmented()
    report = vq.check_linear_dependence(basis)
    assert len(report.offenders) > 0
    top = report.offenders[0]
    # Diffuse means small exponent; 'aug' adds α ~ 0.025-0.1 on H
    assert top.min_exponent < 0.5, (
        f"top offender unexpectedly tight: a_min = {top.min_exponent}"
    )
    # Must identify an atom in the molecule (0 or 1).
    assert 0 <= top.atom_index <= 1
    # Weight should be non-trivial (at least a few percent).
    assert top.weight > 0.05


def test_offenders_ordered_by_weight():
    _, basis = _tight_h2_augmented()
    report = vq.check_linear_dependence(basis)
    assert len(report.offenders) >= 2
    weights = [o.weight for o in report.offenders]
    for i in range(len(weights) - 1):
        assert weights[i] >= weights[i + 1], (
            f"offenders not sorted by weight: {weights}"
        )


# ---------------------------------------------------------------------------
# Threshold overrides
# ---------------------------------------------------------------------------

def test_tight_thresholds_promote_warn_to_error():
    """Raising error_threshold above the smallest eigenvalue must
    reclassify the same basis from warn → error."""
    _, basis = _tight_h2_augmented()
    report_default = vq.check_linear_dependence(basis)
    # Tighten: make anything below 1e-3 an error (aggressive).
    report_tight = vq.check_linear_dependence(
        basis, warn_threshold=1e-2, error_threshold=1e-3,
    )
    # Default thresholds (1e-6 / 1e-8): likely warn.
    # Aggressive thresholds: definitely error (min eig ~ 1e-7 < 1e-3).
    assert report_tight.severity == "error"
    # Same eigenvalues come back regardless of thresholds.
    assert np.allclose(report_default.eigenvalues, report_tight.eigenvalues)


def test_threshold_sanity_check():
    """error_threshold must be strictly less than warn_threshold."""
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0])], charge=-1)
    basis = vq.BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError):
        vq.check_linear_dependence(basis, warn_threshold=1e-8, error_threshold=1e-6)


# ---------------------------------------------------------------------------
# raise_if_severe
# ---------------------------------------------------------------------------

def test_raise_if_severe_passes_on_ok():
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    basis = vq.BasisSet(mol, "sto-3g")
    report = vq.check_linear_dependence(basis)
    # Should not raise.
    vq.raise_if_severe(report)


def test_raise_if_severe_raises_on_error():
    """With a sufficiently aggressive error threshold, the tight-H2
    case crosses into error territory and raises."""
    _, basis = _tight_h2_augmented()
    report = vq.check_linear_dependence(
        basis, warn_threshold=1e-2, error_threshold=1e-5,
    )
    with pytest.raises(vq.LinearDependenceError) as exc_info:
        vq.raise_if_severe(report)
    # The exception carries the full report for inspection.
    assert exc_info.value.report is report
    assert "Linear dependence" in str(exc_info.value)


def test_raise_if_severe_allow_warn_false_raises_on_warn():
    """By default warn does not raise; allow_warn=False promotes
    warn to error behavior."""
    _, basis = _tight_h2_augmented()
    report = vq.check_linear_dependence(basis)
    if report.severity == "warn":
        # Default: no raise.
        vq.raise_if_severe(report)
        # Strict mode: raises.
        with pytest.raises(vq.LinearDependenceError):
            vq.raise_if_severe(report, allow_warn=False)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def test_format_report_ok_case():
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    basis = vq.BasisSet(mol, "sto-3g")
    report = vq.check_linear_dependence(basis)
    text = vq.format_linear_dependence_report(report)
    assert "well-conditioned" in text
    assert "basis dimension" in text
    assert "condition number" in text
    # No offender table on ok reports.
    assert "Top" not in text


def test_format_report_warn_case_shows_offenders():
    _, basis = _tight_h2_augmented()
    report = vq.check_linear_dependence(basis)
    text = vq.format_linear_dependence_report(report)
    # Some severity-specific language must appear.
    assert report.severity in text or "near-linearly-dependent" in text or \
           "linearly dependent" in text
    if report.offenders:
        assert "a_min" in text
        assert "weight" in text


def test_format_report_handles_infinite_condition_number():
    """Robust formatting if the min eigenvalue goes to 0 (or below)."""
    # Build a fake report by hand to exercise the inf code path.
    rep = vq.LinearDependenceReport(
        n_basis=2,
        eigenvalues=np.array([0.0, 1.0]),
        min_eigenvalue=0.0,
        max_eigenvalue=1.0,
        condition_number=float("inf"),
        warn_threshold=1e-6,
        error_threshold=1e-8,
        n_below_warn=1,
        n_below_error=1,
    )
    text = vq.format_linear_dependence_report(rep)
    assert "+inf" in text


# ---------------------------------------------------------------------------
# v0.7: check_overlap_matrix on a precomputed S + critical-severity tier
# ---------------------------------------------------------------------------

def test_check_overlap_matrix_clean_identity():
    """Identity matrix is the cleanest possible overlap — severity ok."""
    rep = vq.check_overlap_matrix(np.eye(5), label="identity")
    assert rep.severity == "ok"
    assert rep.n_basis == 5
    assert rep.min_eigenvalue == pytest.approx(1.0)
    assert rep.max_eigenvalue == pytest.approx(1.0)
    assert rep.condition_number == pytest.approx(1.0)
    assert rep.n_negative == 0
    assert rep.label == "identity"


def test_check_overlap_matrix_near_null_eigenvalue():
    """A single 1e-9 eigenvalue trips error severity."""
    S = np.diag([1e-9, 1.0, 1.0])
    rep = vq.check_overlap_matrix(S, label="near-null")
    assert rep.severity == "error"
    assert rep.n_below_error == 1
    assert rep.n_negative == 0


def test_check_overlap_matrix_negative_eigenvalue_is_critical():
    """Negative eigenvalues escalate severity to 'critical' regardless
    of how many warn/error eigenvalues there are. This is the periodic-
    SCF failure mode (LiH conventional cell at v0.6.x)."""
    # Three negative eigenvalues, like LiH/STO-3G/Γ.
    S = np.diag([-0.16, -0.16, -0.16, 1.0, 1.0])
    rep = vq.check_overlap_matrix(S, label="LiH-like")
    assert rep.severity == "critical"
    assert rep.n_negative == 3
    assert rep.min_eigenvalue == pytest.approx(-0.16)
    assert not np.isfinite(rep.condition_number)   # +inf when min < 0


def test_critical_severity_supersedes_error():
    """If a matrix has BOTH a negative eigenvalue AND a near-null one,
    severity is 'critical' (the negative is the deeper bug)."""
    S = np.diag([-0.5, 1e-9, 1.0])
    rep = vq.check_overlap_matrix(S, label="mixed")
    assert rep.severity == "critical"
    assert rep.n_negative == 1
    assert rep.n_below_error == 2   # both -0.5 and 1e-9 are below 1e-8


def test_negative_threshold_is_tunable():
    """A -1e-12 eigenvalue is numerical noise (PSD up to eps); a -0.16
    is real. The threshold lets the caller distinguish."""
    S_noise = np.diag([-1e-12, 1.0, 1.0])
    rep_default = vq.check_overlap_matrix(S_noise, label="noise")
    # Default threshold = -1e-6; -1e-12 is NOT below it.
    assert rep_default.severity != "critical"
    rep_tight = vq.check_overlap_matrix(
        S_noise, label="noise-strict", negative_threshold=-1e-15,
    )
    # Strict threshold catches the noise eigenvalue.
    assert rep_tight.severity == "critical"


def test_raise_if_severe_critical_raises_by_default():
    rep = vq.check_overlap_matrix(np.diag([-0.16, 1.0]))
    assert rep.severity == "critical"
    with pytest.raises(vq.LinearDependenceError) as exc_info:
        vq.raise_if_severe(rep)
    assert "positive-definiteness" in str(exc_info.value)
    assert exc_info.value.report is rep


def test_raise_if_severe_critical_can_be_overridden():
    """allow_critical=True lets the caller proceed with canonical orth
    on a non-PSD S — useful for users who know their basis has only
    numerical-noise negatives."""
    rep = vq.check_overlap_matrix(np.diag([-1e-9, 1.0]),
                                   negative_threshold=-1e-12)
    assert rep.severity == "critical"
    # Default raises; allow_critical=True does not.
    with pytest.raises(vq.LinearDependenceError):
        vq.raise_if_severe(rep)
    vq.raise_if_severe(rep, allow_critical=True)


def test_check_overlap_matrix_complex_hermitian():
    """The Bloch-summed S(k) is generally Hermitian complex — eigvalsh
    must still extract real eigenvalues correctly."""
    H = np.array(
        [[1.0 + 0j, 0.1 + 0.2j], [0.1 - 0.2j, 1.0 + 0j]],
        dtype=complex,
    )
    rep = vq.check_overlap_matrix(H, label="S(k)")
    assert rep.severity == "ok"
    # Both eigenvalues real and ~1 ± |0.1+0.2j|.
    assert rep.min_eigenvalue > 0


def test_format_critical_report_explains_action():
    rep = vq.check_overlap_matrix(np.diag([-0.16, 1.0]), label="bad")
    text = vq.format_linear_dependence_report(rep)
    assert "CRITICAL" in text or "lost positive-definiteness" in text
    # Action block should mention the upstream-bug failure modes.
    assert "lattice cutoff" in text or "image-cell" in text


# ---------------------------------------------------------------------------
# v0.7: scf_preflight_overlap_check banner integration
# ---------------------------------------------------------------------------

class _CapturingLogger:
    """Minimal ProgressLogger stand-in that captures the messages
    the preflight emits, so we can assert on them."""
    def __init__(self):
        self.info_lines = []
        self.raw_lines = []
        self.warning_lines = []

    def info(self, msg):
        self.info_lines.append(str(msg))

    def write_raw(self, msg):
        self.raw_lines.append(str(msg))

    def warning(self, msg):
        self.warning_lines.append(str(msg))


def test_preflight_clean_emits_one_line():
    plog = _CapturingLogger()
    vq.scf_preflight_overlap_check(np.eye(3), plog=plog, label="S(test)")
    # Exactly one info line, no full report.
    assert len(plog.info_lines) == 1
    assert "overlap [S(test)]" in plog.info_lines[0]
    assert "severity=ok" in plog.info_lines[0]
    assert plog.raw_lines == []


def test_preflight_critical_emits_full_report_and_raises():
    plog = _CapturingLogger()
    with pytest.raises(vq.LinearDependenceError):
        vq.scf_preflight_overlap_check(
            np.diag([-0.16, -0.16, 1.0]),
            plog=plog, label="S(LiH-like)",
        )
    # Banner line tagged CRITICAL.
    assert any("[CRITICAL]" in line for line in plog.info_lines)
    # Full multi-line report dumped.
    assert any("lost positive-definiteness" in raw for raw in plog.raw_lines)


def test_preflight_can_suppress_raise():
    plog = _CapturingLogger()
    rep = vq.scf_preflight_overlap_check(
        np.diag([-0.16, 1.0]),
        plog=plog, label="bad",
        raise_on_severe=False,
    )
    # Did NOT raise, but DID emit the full report.
    assert rep.severity == "critical"
    assert any("[CRITICAL]" in line for line in plog.info_lines)
    assert any("lost positive-definiteness" in raw for raw in plog.raw_lines)


def test_preflight_warn_emits_full_report_but_does_not_raise():
    plog = _CapturingLogger()
    # warn-but-not-error: 1e-7 is below default warn (1e-6) but above
    # default error (1e-8).
    rep = vq.scf_preflight_overlap_check(
        np.diag([1e-7, 1.0, 1.0]), plog=plog, label="diffuse",
    )
    assert rep.severity == "warn"
    assert any("[WARN]" in line for line in plog.info_lines)
    assert plog.raw_lines    # full report dumped


def test_preflight_critical_strict_allow_critical_skips_raise():
    plog = _CapturingLogger()
    # Even though severity is critical, allow_critical=True lets it pass.
    vq.scf_preflight_overlap_check(
        np.diag([-0.16, 1.0]), plog=plog, label="bad",
        allow_critical=True,
    )
    # Banner line still emitted.
    assert any("[CRITICAL]" in line for line in plog.info_lines)


# ---------------------------------------------------------------------------
# Integration: the LiH dense-ionic case at v0.7 raises before SCF runs
# ---------------------------------------------------------------------------

def test_periodic_lih_conventional_raises_critical_before_scf():
    """LiH conventional cubic (a = 4.084 Å, RKS-LDA, EWALD_3D, Γ-only)
    has S(Γ) with three eigenvalues at ~-0.16 — non-PSD. The v0.7
    preflight must catch this and refuse to run SCF, rather than let
    the silent canonical-orthogonalisation truncation produce a
    plausible-but-wrong total energy.

    NOTE — by default ``auto_optimize_truncation=True`` (the v0.7
    default) intercepts this failure BEFORE the preflight runs, by
    automatically tightening lattice cutoffs until the overlap is
    PSD again. This test exercises the *abort path* specifically by
    disabling the auto-optimisation, which is what users get when
    they explicitly opt out for hand-tuning.
    """
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

    opts = vq.PeriodicKSOptions()
    opts.functional = "lda"
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.max_iter = 1
    opts.use_diis = False
    opts.conv_tol_energy = 1e-3

    # allow_dense_ionic: bypass the dense-ionic fail-closed guard so the
    # linear-dependence preflight (the abort path under test) is what runs
    # and raises here, not the earlier dense-ionic guard.
    with pytest.raises(vq.LinearDependenceError) as exc_info:
        vq.run_rks_periodic_gamma_ewald3d(
            sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
            auto_optimize_truncation=False,
            allow_dense_ionic=True,
        )
    rep = exc_info.value.report
    assert rep.severity == "critical"


def test_periodic_lih_default_auto_optimizes_past_critical():
    """Companion to the abort-path test: by DEFAULT (v0.7+),
    ``auto_optimize_truncation=True`` is on and the optimiser
    automatically grows cutoff_bohr / tightens schwarz_threshold
    until the overlap is PSD. Verified at the lat_opts level — we
    check the optimisation kicks in without running the (long)
    full SCF.
    """
    a = 4.084 / 0.529177210903
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                        (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0),
                        (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))
    sysp = vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")

    opts_lat = vq.LatticeSumOptions()
    opts_lat.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts_lat.cutoff_bohr = 10.0
    opts_lat.nuclear_cutoff_bohr = 15.0

    # Driving the optimiser directly is what auto_optimize_truncation
    # would do inside the SCF driver. Verify the rescue.
    rep = vq.optimize_truncation(sysp, basis, lattice_opts=opts_lat)
    assert rep.converged
    assert rep.final_n_negative == 0
    assert rep.final_severity in ("ok", "warn")
    # Default-ON behaviour grew the cutoff (and / or tightened schwarz).
    assert (
        rep.optimized_lattice_opts.cutoff_bohr > opts_lat.cutoff_bohr
        or rep.optimized_lattice_opts.schwarz_threshold < opts_lat.schwarz_threshold
    )


def test_periodic_h2_in_big_box_passes_preflight():
    """H2 in a 30 bohr box has a clean diagonal S(Γ) with both
    eigenvalues O(1). The preflight should pass through silently
    and SCF should run."""
    sysp = vq.PeriodicSystem(
        dim=3, lattice=30.0 * np.eye(3),
        unit_cell=[vq.Atom(1, [15, 15, 14.3]),
                   vq.Atom(1, [15, 15, 15.7])],
    )
    basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
    opts = vq.PeriodicRHFOptions()
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.max_iter = 5

    # No exception means the preflight was happy with S.
    r = vq.run_rhf_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.4,
    )
    assert r.converged
    # Sanity-check: H2 in a 30-bohr box reduces to the isolated molecular
    # limit (no inter-cell interaction), and the EWALD_3D total is
    # ω-invariant (F2), so the energy is the molecular H2/STO-3G value at
    # R = 1.4 bohr regardless of the ω = 0.5 split. (Corrected from a stale
    # -1.1114 reference that predated the F2 ω-invariance fix; the driver
    # has reproduced the molecular limit -1.116714 since.)
    assert r.energy == pytest.approx(-1.116714, abs=1e-3)
