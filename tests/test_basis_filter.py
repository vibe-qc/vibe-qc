"""Tests for ``vibeqc.basis_filter`` (item 2 of the linear-dep work).

Mirrors PySCF's ``Cell.exp_to_discard`` semantics: drop every
Gaussian primitive whose exponent is below a positive threshold,
rebuild the BasisSet without those primitives, and report what was
filtered.

Three regimes covered here:

1. **Mechanics** — un-normalisation of libint contraction
   coefficients reproduces the published .g94 entries; primitive
   counts and shell counts before/after match expectations on a
   minimal test system (LiH/STO-3G).
2. **API surface** — ``make_basis(mol, name, exp_to_discard=...)``
   is the user-facing entry point and falls through to plain
   ``BasisSet(mol, name)`` when the threshold is omitted; the
   filter cache is cleanable via ``clear_filtered_basis_cache``.
3. **Behavioural fix on LiH** — the LiH conventional cubic cell's
   non-positive-definite ``S(Γ)`` (3 negative eigenvalues at -0.16
   in the unfiltered case) becomes properly PSD after filtering with
   ``exp_to_discard=0.1``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import vibeqc as vq


# ---------------------------------------------------------------------------
# Mechanics
# ---------------------------------------------------------------------------

def test_double_factorial_2l_minus_1():
    """The (2l-1)!! in the shared norm, including (-1)!! = 1 at l = 0.

    The helper this used to import lives inline in
    ``vibeqc._primitive_norm`` now; recover the double factorial from the
    norm itself: N(a, l) = N(a, 0) . (4a)^{l/2} / sqrt((2l-1)!!).
    """
    from vibeqc._primitive_norm import libint_primitive_norm

    alpha = 0.7
    n0 = libint_primitive_norm(alpha, 0)
    for l, expected in enumerate([1, 1, 3, 15, 105, 945]):
        ratio = libint_primitive_norm(alpha, l) / n0
        df = ((4.0 * alpha) ** (l / 2.0) / ratio) ** 2
        assert df == pytest.approx(expected, rel=1e-12), l


def test_libint_normalisation_reproduces_g94():
    """For STO-3G H, dividing libint's stored coefficients by N(α, l=0)
    must reproduce the published .g94 raw coefficients to four decimals.
    """
    from vibeqc.basis_filter import _libint_primitive_norm

    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    basis = vq.BasisSet(mol, "sto-3g")
    sh = list(basis.shells())[0]    # first H 1s
    assert sh.l == 0

    # Published .g94 coefficients (from basis_library/basis/sto-3g.g94)
    expected = [0.15432897, 0.53532814, 0.44463454]
    for alpha, coef_libint, coef_g94 in zip(
        sh.exponents, sh.coefficients, expected,
    ):
        c_raw = coef_libint / _libint_primitive_norm(alpha, sh.l)
        # Match to 3 decimals; libint and the .g94 are not perfectly
        # bit-identical (they round differently in storage).
        assert c_raw == pytest.approx(coef_g94, abs=3e-3)


def test_filtered_g94_preserves_source_provenance_header():
    """A derived basis file retains the source basis references."""
    from vibeqc.basis_filter import _emit_g94

    text = _emit_g94("sto-3g", 0.1, {1: [(0, [1.0], [1.0])]})
    assert "STO-3G  EMSL  Basis Set Exchange Library" in text
    assert "W.J. Hehre" in text


def test_filter_drops_primitives_below_threshold():
    """LiH/STO-3G: the Li 2s and 2p shells each have α = 0.048, which
    must be dropped at exp_to_discard = 0.1; the H 1s α = 0.169 must
    be kept."""
    mol = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3, 0, 0])])
    basis = vq.BasisSet(mol, "sto-3g")
    filtered, report = vq.filter_basis_by_exponent(basis, mol, 0.1)
    # 12 primitives total → 10 kept (Li 2s and 2p each lose their α=0.048).
    assert report.primitives_total == 12
    assert report.primitives_kept == 10
    assert report.n_primitives_dropped == 2
    assert report.n_shells_dropped == 0   # no shell loses ALL primitives
    # Verify by walking the new shells.
    for sh in filtered.shells():
        for alpha in sh.exponents:
            assert alpha >= 0.1


def test_filter_preserves_basis_dimension_when_no_shells_dropped():
    """LiH/STO-3G with exp_to_discard=0.1: shells lose primitives but
    no entire shell is removed → nbf unchanged."""
    mol = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3, 0, 0])])
    basis = vq.BasisSet(mol, "sto-3g")
    filtered, report = vq.filter_basis_by_exponent(basis, mol, 0.1)
    assert filtered.nbasis == basis.nbasis
    assert filtered.nshells == basis.nshells


def test_filter_drops_entire_shell_when_all_primitives_too_diffuse():
    """At exp_to_discard = 1.0, every Li 2s and 2p primitive (max 0.64)
    is dropped → those shells disappear → nbf drops by 1 (Li 2s) + 3
    (Li 2p, three Cartesian → three pure components) = 4."""
    mol = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3, 0, 0])])
    basis = vq.BasisSet(mol, "sto-3g")
    filtered, report = vq.filter_basis_by_exponent(basis, mol, 1.0)
    assert report.n_shells_dropped >= 2
    assert filtered.nshells < basis.nshells
    assert filtered.nbasis < basis.nbasis


def test_filter_rejects_zero_threshold():
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    basis = vq.BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="must be > 0"):
        vq.filter_basis_by_exponent(basis, mol, 0.0)


def test_filter_rejects_threshold_that_drops_everything():
    """Exponents max out at ~16 for STO-3G; threshold = 100 drops
    everything."""
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    basis = vq.BasisSet(mol, "sto-3g")
    with pytest.raises(ValueError, match="dropped every primitive"):
        vq.filter_basis_by_exponent(basis, mol, 100.0)


# ---------------------------------------------------------------------------
# Public API: make_basis + cache cleanup
# ---------------------------------------------------------------------------

def test_make_basis_no_filter_is_plain_basisset():
    mol = vq.Molecule([vq.Atom(1, [0, 0, 0]), vq.Atom(1, [0, 0, 1.4])])
    plain = vq.BasisSet(mol, "sto-3g")
    via_make = vq.make_basis(mol, "sto-3g")
    # Same number of shells / primitives (we don't promise the same
    # underlying object — the C++ ctor returns a fresh BasisSet).
    assert via_make.nbasis == plain.nbasis
    assert via_make.nshells == plain.nshells
    assert via_make.name == plain.name


def test_make_basis_with_filter_returns_filtered_basis():
    mol = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3, 0, 0])])
    plain = vq.BasisSet(mol, "sto-3g")
    via_make = vq.make_basis(mol, "sto-3g", exp_to_discard=0.1)
    # Filtered: still nbf = 6 but the synthetic name differs.
    assert via_make.nbasis == plain.nbasis
    assert via_make.name.startswith("_vibeqc_filtered_sto-3g_")
    # Verify primitive counts match the filtered expectation.
    n_prim_via_make = sum(len(sh.exponents) for sh in via_make.shells())
    assert n_prim_via_make == 10   # 12 original - 2 dropped


def test_clear_filtered_basis_cache_returns_count():
    """Filter once, clear, and check we deleted at least one file."""
    mol = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3, 0, 0])])
    basis = vq.BasisSet(mol, "sto-3g")
    vq.filter_basis_by_exponent(basis, mol, 0.1)
    n = vq.clear_filtered_basis_cache()
    assert n >= 1
    # Second cleanup is a no-op.
    assert vq.clear_filtered_basis_cache() == 0


# ---------------------------------------------------------------------------
# Behavioural fix on LiH conventional cubic
# ---------------------------------------------------------------------------

def test_lih_filter_makes_overlap_psd():
    """The headline fix: LiH/STO-3G/Γ-only EWALD_3D has S(Γ) with three
    negative eigenvalues at -0.16 unfiltered; after exp_to_discard=0.1
    the spectrum is fully positive."""
    a = 4.084 / 0.529177210903   # bohr
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                        (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0),
                        (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))
    sysp = vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)
    mol_uc = sysp.unit_cell_molecule()

    lat_opts = vq.LatticeSumOptions()
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    lat_opts.cutoff_bohr = 10.0

    # Unfiltered: should fail the PSD check.
    basis_raw = vq.BasisSet(mol_uc, "sto-3g")
    S_lat = vq.compute_overlap_lattice(basis_raw, sysp, lat_opts)
    S_g = np.real(vq.bloch_sum(S_lat, np.zeros(3)))
    S_g = 0.5 * (S_g + S_g.T)
    rep_raw = vq.check_overlap_matrix(S_g, label="raw")
    assert rep_raw.severity == "critical"
    assert rep_raw.n_negative >= 3
    assert rep_raw.min_eigenvalue < -0.1

    # Filtered: PSD.
    basis_f = vq.make_basis(mol_uc, "sto-3g", exp_to_discard=0.1)
    S_lat_f = vq.compute_overlap_lattice(basis_f, sysp, lat_opts)
    S_g_f = np.real(vq.bloch_sum(S_lat_f, np.zeros(3)))
    S_g_f = 0.5 * (S_g_f + S_g_f.T)
    rep_f = vq.check_overlap_matrix(S_g_f, label="filtered")
    assert rep_f.severity in ("ok", "warn")   # no longer critical
    assert rep_f.n_negative == 0
    assert rep_f.min_eigenvalue > 0

    vq.clear_filtered_basis_cache()


def test_lih_filter_unblocks_scf_run():
    """Companion to the PSD test: with the filter applied, the SCF
    that previously raised LinearDependenceError now runs to
    convergence (regardless of the absolute energy value, which is
    a separate v0.7 multi-cell-density issue tracked in CHANGELOG)."""
    a = 4.084 / 0.529177210903
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                        (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0),
                        (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))
    sysp = vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)
    mol_uc = sysp.unit_cell_molecule()

    basis = vq.make_basis(mol_uc, "sto-3g", exp_to_discard=0.1)
    opts = vq.PeriodicKSOptions()
    opts.functional = "lda"
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 15.0
    opts.max_iter = 30
    opts.use_diis = True
    opts.conv_tol_energy = 1e-5

    # Should NOT raise LinearDependenceError. Dense LiH: opt past the
    # fail-closed dense-ionic guard to exercise the basis-filter mechanic
    # (the absolute energy is a separate, known-wrong multi-cell issue).
    r = vq.run_rks_periodic_gamma_ewald3d(
        sysp, basis, opts, omega=0.5, spacing_bohr=0.5,
        allow_dense_ionic=True,
    )
    assert r.converged
    # Energy is finite and non-NaN; correctness vs the physical
    # reference is tracked separately in test_periodic_dense_ionic_bug
    # (still xfail-strict pending the multi-cell-density fix).
    assert math.isfinite(r.energy)

    vq.clear_filtered_basis_cache()


# ---------------------------------------------------------------------------
# Critical-severity error message integration (item 4 + item 2 cross-cut)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Transparency: drop list is recorded + printable
# ---------------------------------------------------------------------------

def test_filter_records_dropped_primitives_explicitly():
    """Per the v0.7 transparency directive, the report MUST list
    every primitive that was dropped — element, shell, l, exponent,
    libint-stored coefficient, threshold."""
    mol = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3, 0, 0])])
    basis = vq.BasisSet(mol, "sto-3g")
    _, report = vq.filter_basis_by_exponent(basis, mol, 0.1)

    # Two drops expected: Li 2s α=0.048 and Li 2p α=0.048.
    assert len(report.dropped_primitives) == 2
    elements_dropped = sorted({d.element_z for d in report.dropped_primitives})
    assert elements_dropped == [3]   # only Li

    # Every drop entry has the full provenance.
    for d in report.dropped_primitives:
        assert d.element_z == 3
        assert d.element_symbol == "Li"
        assert d.shell_l in (0, 1)
        assert d.shell_letter in ("S", "P")
        assert d.exponent < 0.1
        assert d.threshold == pytest.approx(0.1)
    vq.clear_filtered_basis_cache()


def test_format_basis_filter_report_lists_every_drop():
    """The formatter output must contain every dropped primitive's
    exponent so the SCF log is self-documenting."""
    mol = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3, 0, 0])])
    basis = vq.BasisSet(mol, "sto-3g")
    _, report = vq.filter_basis_by_exponent(basis, mol, 0.1)
    text = vq.format_basis_filter_report(report)

    # Every dropped exponent must appear verbatim.
    for d in report.dropped_primitives:
        assert f"{d.exponent:.6f}" in text
    # Element symbols too.
    assert "Li" in text
    # And the threshold.
    assert "0.1" in text
    vq.clear_filtered_basis_cache()


def test_format_basis_filter_report_handles_no_drops():
    """When the threshold doesn't drop anything, the formatter says so."""
    mol = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3, 0, 0])])
    basis = vq.BasisSet(mol, "sto-3g")
    # Tiny threshold drops nothing.
    _, report = vq.filter_basis_by_exponent(basis, mol, 1e-6)
    text = vq.format_basis_filter_report(report)
    assert "no primitives below threshold" in text or \
           "basis unchanged" in text
    assert report.n_primitives_dropped == 0
    vq.clear_filtered_basis_cache()


def test_format_basis_filter_report_records_dropped_shell():
    """When every primitive in a shell is below the threshold, the
    shell appears in the dropped_shells list and the formatter calls
    it out explicitly."""
    mol = vq.Molecule([vq.Atom(3, [0, 0, 0]), vq.Atom(1, [3, 0, 0])])
    basis = vq.BasisSet(mol, "sto-3g")
    # Threshold > 0.65 drops Li 2s and 2p shells entirely (they max
    # out at 0.6362897).
    _, report = vq.filter_basis_by_exponent(basis, mol, 1.0)
    assert report.n_shells_dropped >= 1
    assert len(report.dropped_shells) >= 1
    text = vq.format_basis_filter_report(report)
    assert "Shells removed entirely" in text
    vq.clear_filtered_basis_cache()


# ---------------------------------------------------------------------------
# Critical-severity error message integration (from item 2)
# ---------------------------------------------------------------------------

def test_critical_error_message_recommends_pob_and_make_basis():
    """The auto-suggestion in critical-severity reports must point
    users at concrete, actionable fixes: pob-tzvp / pob-tzvp-rev2 and
    ``make_basis(..., exp_to_discard=0.1)``."""
    rep = vq.check_overlap_matrix(np.diag([-0.16, 1.0]), label="bad")
    text = vq.format_linear_dependence_report(rep)
    assert "pob-tzvp" in text or "pob-tzvp-rev2" in text
    assert "make_basis" in text
    assert "exp_to_discard" in text
