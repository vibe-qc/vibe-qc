"""Smoke tests for the CP2K subprocess runner (gapw chat, M1f).

The runner lives at ``examples/regression/core/runner_cp2k.py`` and
provides the CP2K parity oracle the v0.10.x GAPW route needs
(PySCF.pbc and CRYSTAL14 are GTO-only and can't validate plane-wave
Hartree). The full end-to-end run requires a CP2K install; what we
test here is everything that doesn't:

* the executable / data-file discovery degrades cleanly when CP2K
  isn't present (``is_available()`` returns False, the runner emits
  a ``CodeRow`` with ``status='unavailable'`` and a useful note),
* the GPW input deck is well-formed and contains the keywords the
  M2 driver will be diffed against (``METHOD GPW``, ``XC_FUNCTIONAL
  PBE``, ``BASIS_SET DZVP-MOLOPT-SR-GTH``, ``POTENTIAL GTH-PBE-q4``,
  cell vectors, scaled coordinates),
* the output parser recovers energy / convergence / n_iter / version
  from synthetic CP2K-shaped strings (we shape these by hand from
  CP2K's published printout conventions, so we don't need a real
  .out file in the test data),
* the method whitelist refuses UHF / UKS / MP2 with a clear note
  (M3+ scope),
* the GTH valence map covers the M1 demo systems.

Mirrors the test patterns in ``test_runner_crystal.py`` /
``test_runner_pyscf.py`` where they exist; otherwise follows the
generic vibeqc test conventions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The regression-suite package isn't installed as ``vibeqc.*``;
# import it relatively the same way run_suite.py does.
REGRESSION_ROOT = Path(__file__).parent.parent / "examples" / "regression"
sys.path.insert(0, str(REGRESSION_ROOT.parent.parent))

from examples.regression.core import runner_cp2k  # noqa: E402
from examples.regression.core.spec import (        # noqa: E402
    AtomFrac, MethodSpec, PeriodicSpec,
)


# ---------- Fixtures: tiny Si and MgO specs --------------------------------


@pytest.fixture
def si_diamond_spec() -> PeriodicSpec:
    """Conventional 2-atom Si diamond cell (5.43 Å cubic)."""
    return PeriodicSpec(
        id="si-diamond",
        family="diamond",
        lattice_ang=(
            (5.43, 0.0, 0.0),
            (0.0, 5.43, 0.0),
            (0.0, 0.0, 5.43),
        ),
        space_group="Fd-3m",
        atoms=(
            AtomFrac(symbol="Si", z=14, frac=(0.0, 0.0, 0.0)),
            AtomFrac(symbol="Si", z=14, frac=(0.25, 0.25, 0.25)),
        ),
    )


@pytest.fixture
def mgo_rocksalt_spec() -> PeriodicSpec:
    """8-atom conventional MgO rocksalt cell (4.21 Å cubic)."""
    return PeriodicSpec(
        id="mgo-rocksalt",
        family="rocksalt",
        lattice_ang=(
            (4.21, 0.0, 0.0),
            (0.0, 4.21, 0.0),
            (0.0, 0.0, 4.21),
        ),
        space_group="Fm-3m",
        atoms=(
            AtomFrac(symbol="Mg", z=12, frac=(0.0, 0.0, 0.0)),
            AtomFrac(symbol="Mg", z=12, frac=(0.5, 0.5, 0.0)),
            AtomFrac(symbol="Mg", z=12, frac=(0.5, 0.0, 0.5)),
            AtomFrac(symbol="Mg", z=12, frac=(0.0, 0.5, 0.5)),
            AtomFrac(symbol="O", z=8, frac=(0.5, 0.0, 0.0)),
            AtomFrac(symbol="O", z=8, frac=(0.0, 0.5, 0.0)),
            AtomFrac(symbol="O", z=8, frac=(0.0, 0.0, 0.5)),
            AtomFrac(symbol="O", z=8, frac=(0.5, 0.5, 0.5)),
        ),
    )


@pytest.fixture
def rks_pbe_method() -> MethodSpec:
    return MethodSpec(id="rks-pbe", scf="rks", xc="pbe")


# ---------- Deck builder ---------------------------------------------------


def test_deck_contains_gpw_method_and_pbe_xc(
    si_diamond_spec, rks_pbe_method,
):
    deck = runner_cp2k.build_cp2k_input(
        spec=si_diamond_spec, basis_kw="DZVP-MOLOPT-SR-GTH",
        func_kw="PBE", pseudo_suffix="PBE",
        kmesh=(1, 1, 1),
        conv_tol_energy=1e-7, max_iter=40,
        pw_cutoff_ry=300.0, rel_cutoff_ry=60.0,
        basis_file="/path/to/BASIS_MOLOPT",
        potential_file="/path/to/GTH_POTENTIALS",
    )
    assert "METHOD GPW" in deck
    assert "&XC_FUNCTIONAL PBE" in deck
    assert "BASIS_SET DZVP-MOLOPT-SR-GTH" in deck
    assert "POTENTIAL GTH-PBE-q4" in deck
    assert "RUN_TYPE ENERGY" in deck
    assert "PERIODIC XYZ" in deck
    # MGRID block must appear with the requested PW cutoff.
    assert "CUTOFF 300.0" in deck
    assert "REL_CUTOFF 60.0" in deck
    # SCF convergence threshold matches the request.
    assert "EPS_SCF 1.00e-07" in deck
    # SCALED coords + cell vectors.
    assert "SCALED" in deck
    assert "0.2500000000 0.2500000000 0.2500000000" in deck


def test_deck_omits_kpoints_block_at_gamma(
    si_diamond_spec, rks_pbe_method,
):
    deck = runner_cp2k.build_cp2k_input(
        spec=si_diamond_spec, basis_kw="DZVP-MOLOPT-SR-GTH",
        func_kw="PBE", pseudo_suffix="PBE",
        kmesh=(1, 1, 1),
        conv_tol_energy=1e-7, max_iter=40,
        pw_cutoff_ry=300.0, rel_cutoff_ry=60.0,
        basis_file="BASIS_MOLOPT", potential_file="GTH_POTENTIALS",
    )
    assert "&KPOINTS" not in deck


def test_deck_emits_kpoints_block_for_dense_mesh(
    si_diamond_spec, rks_pbe_method,
):
    deck = runner_cp2k.build_cp2k_input(
        spec=si_diamond_spec, basis_kw="DZVP-MOLOPT-SR-GTH",
        func_kw="PBE", pseudo_suffix="PBE",
        kmesh=(4, 4, 4),
        conv_tol_energy=1e-7, max_iter=40,
        pw_cutoff_ry=300.0, rel_cutoff_ry=60.0,
        basis_file="BASIS_MOLOPT", potential_file="GTH_POTENTIALS",
    )
    assert "&KPOINTS" in deck
    assert "SCHEME MONKHORST-PACK 4 4 4" in deck
    assert "&END KPOINTS" in deck


def test_deck_hf_path_swaps_hfx_block(
    si_diamond_spec,
):
    rhf = MethodSpec(id="rhf", scf="rhf", xc=None)
    deck = runner_cp2k.build_cp2k_input(
        spec=si_diamond_spec, basis_kw="DZVP-MOLOPT-SR-GTH",
        func_kw="HF", pseudo_suffix="PADE",  # LDA pseudo as a sane default
        kmesh=(1, 1, 1),
        conv_tol_energy=1e-7, max_iter=40,
        pw_cutoff_ry=300.0, rel_cutoff_ry=60.0,
        basis_file="BASIS_MOLOPT", potential_file="GTH_POTENTIALS",
    )
    assert "&HF" in deck
    assert "FRACTION 1.0" in deck
    assert "XC_FUNCTIONAL NONE" in deck


def test_deck_per_element_kind_blocks(mgo_rocksalt_spec, rks_pbe_method):
    deck = runner_cp2k.build_cp2k_input(
        spec=mgo_rocksalt_spec, basis_kw="DZVP-MOLOPT-SR-GTH",
        func_kw="PBE", pseudo_suffix="PBE",
        kmesh=(1, 1, 1),
        conv_tol_energy=1e-7, max_iter=40,
        pw_cutoff_ry=300.0, rel_cutoff_ry=60.0,
        basis_file="BASIS_MOLOPT", potential_file="GTH_POTENTIALS",
    )
    # One &KIND block per element (Mg + O), not per atom.
    assert deck.count("&KIND Mg") == 1
    assert deck.count("&KIND O") == 1
    # Valence counts come from the GTH map.
    assert "POTENTIAL GTH-PBE-q10" in deck  # Mg semicore
    assert "POTENTIAL GTH-PBE-q6" in deck   # O


# ---------- GTH valence map -----------------------------------------------


@pytest.mark.parametrize("symbol, expected", [
    ("H", 1), ("C", 4), ("N", 5), ("O", 6), ("F", 7),
    ("Na", 9), ("Mg", 10), ("Si", 4), ("Cl", 7),
])
def test_gth_valence_lookup(symbol, expected):
    assert runner_cp2k._n_valence_electrons(symbol) == expected


def test_gth_valence_unknown_element_raises():
    with pytest.raises(KeyError, match="GTH-pseudopotential valence"):
        runner_cp2k._n_valence_electrons("Uuo")


# ---------- Method whitelist ----------------------------------------------


@pytest.mark.parametrize("scf, xc, expected", [
    ("rks", "lda", "LDA"),
    ("rks", "pbe", "PBE"),
    ("rks", "blyp", "BLYP"),
    # CP2K's B3LYP preset is the VWN5 flavor — exactly vibe-qc's bare
    # "b3lyp" (ORCA definition); "b3lyp5" is the explicit spelling.
    ("rks", "b3lyp", "B3LYP"),
    ("rks", "b3lyp5", "B3LYP"),
    ("rhf", None, "HF"),
])
def test_method_whitelist_accepts_m1f_set(scf, xc, expected):
    method = MethodSpec(id="m", scf=scf, xc=xc)
    assert runner_cp2k._cp2k_functional(method) == expected


@pytest.mark.parametrize("scf, xc, post", [
    ("uhf", None, None),     # open-shell — M3+
    ("uks", "pbe", None),    # open-shell DFT — M3+
    ("rks", "wb97x", None),  # XC not on whitelist
    ("rks", "b3lypg", None),   # Gaussian-flavor B3LYP — CP2K preset is VWN5
    ("rks", "b3lyp/g", None),  # (same, ORCA spelling)
    ("rhf", None, "mp2"),    # post-HF — out of scope
])
def test_method_whitelist_refuses_out_of_scope(scf, xc, post):
    method = MethodSpec(id="m", scf=scf, xc=xc, post=post)
    assert runner_cp2k._cp2k_functional(method) is runner_cp2k._UNSUPPORTED


# ---------- Output parser --------------------------------------------------


def test_parser_extracts_total_energy():
    fake_out = (
        " CP2K| version string:                                   CP2K version 9.1\n"
        " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]:        -7.6037215834312\n"
    )
    result = runner_cp2k.parse_cp2k_out(fake_out)
    assert result.energy_ha == pytest.approx(-7.6037215834312)
    assert result.version == "9.1"


def test_parser_last_energy_wins():
    """CP2K prints one energy per converged SCF + a final summary line;
    the last value is the canonical converged result, same convention
    as the CRYSTAL parser."""
    fake_out = (
        "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -10.5\n"
        "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -11.2\n"
        "ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -11.873\n"
    )
    result = runner_cp2k.parse_cp2k_out(fake_out)
    assert result.energy_ha == pytest.approx(-11.873)


def test_parser_detects_scf_converged():
    fake_out = "  *** SCF run converged in    17 steps ***\n"
    result = runner_cp2k.parse_cp2k_out(fake_out)
    assert result.converged is True
    assert result.n_iter == 17


def test_parser_detects_scf_not_converged():
    fake_out = "  *** SCF run NOT converged ***\n"
    result = runner_cp2k.parse_cp2k_out(fake_out)
    assert result.converged is False


def test_parser_captures_first_error_line():
    fake_out = (
        " ENERGY| Total FORCE_EVAL ( QS ) energy [a.u.]: -5.0\n"
        " *** ERROR in qs_environment: basis set not in library ***\n"
        " *** Fatal error follows ***\n"
    )
    result = runner_cp2k.parse_cp2k_out(fake_out)
    # First matching error line wins.
    assert "ERROR in qs_environment" in result.error_line


def test_parser_returns_none_on_empty():
    result = runner_cp2k.parse_cp2k_out("")
    assert result.energy_ha is None
    assert result.version is None
    assert result.converged is None
    assert result.n_iter is None
    assert result.error_line == ""


# ---------- Availability gating --------------------------------------------


def test_is_available_returns_bool():
    """The is_available() probe never raises; it returns a bool that
    test code can branch on (skip parity assertions on CP2K-less
    machines)."""
    assert isinstance(runner_cp2k.is_available(), bool)


def test_run_periodic_case_unavailable_when_cp2k_missing(
    si_diamond_spec, rks_pbe_method, tmp_path, monkeypatch,
):
    """If no CP2K binary is on PATH, the runner emits a clean
    ``status='unavailable'`` row with a useful note rather than
    raising."""
    # Force the executable lookup to fail by clearing the
    # environment override AND making `which` return None for every
    # cp2k name. We can't easily monkeypatch shutil.which globally
    # without affecting other tests, so we patch the function the
    # runner actually calls.
    monkeypatch.setattr(runner_cp2k, "_cp2k_executable", lambda: None)
    log_path = tmp_path / "log.txt"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    row = runner_cp2k.run_periodic_case(
        run_id="test", target="dev",
        spec=si_diamond_spec, basis_name="dzvp-molopt-sr-gth",
        method=rks_pbe_method, kmesh=(1, 1, 1),
        log_path=log_path, workdir=workdir,
    )
    assert row.code == "cp2k"
    assert row.status == "unavailable"
    assert "cp2k executable not on PATH" in row.note
    assert row.energy_ha is None


def test_run_periodic_case_unavailable_on_unsupported_method(
    si_diamond_spec, tmp_path, monkeypatch,
):
    """Force cp2k 'available' so we get past the executable check,
    then submit a UHF method that the M1f whitelist refuses."""
    monkeypatch.setattr(runner_cp2k, "_cp2k_executable",
                         lambda: "/fake/cp2k")
    log_path = tmp_path / "log.txt"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    uhf_method = MethodSpec(id="uhf", scf="uhf", xc=None)
    row = runner_cp2k.run_periodic_case(
        run_id="test", target="dev",
        spec=si_diamond_spec, basis_name="dzvp-molopt-sr-gth",
        method=uhf_method, kmesh=(1, 1, 1),
        log_path=log_path, workdir=workdir,
    )
    assert row.status == "unavailable"
    assert "unsupported method" in row.note
    assert "M3+ scope" in row.note


def test_run_periodic_case_unavailable_on_unknown_basis(
    si_diamond_spec, rks_pbe_method, tmp_path, monkeypatch,
):
    monkeypatch.setattr(runner_cp2k, "_cp2k_executable",
                         lambda: "/fake/cp2k")
    log_path = tmp_path / "log.txt"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    row = runner_cp2k.run_periodic_case(
        run_id="test", target="dev",
        spec=si_diamond_spec, basis_name="def2-svp",  # not a GTH basis
        method=rks_pbe_method, kmesh=(1, 1, 1),
        log_path=log_path, workdir=workdir,
    )
    assert row.status == "unavailable"
    assert "not in GTH-pseudo set" in row.note


# ---------- Hard rules (CLAUDE.md §10) -------------------------------------


def test_runner_does_not_import_cp2k():
    """Lock in CLAUDE.md § 10: vibe-qc must NEVER ``import cp2k``
    (or any cp2k Python module). The runner can only spawn cp2k as
    a subprocess. Verify by reading the source file."""
    src = (REGRESSION_ROOT / "core" / "runner_cp2k.py").read_text()
    # No bare or attribute import of cp2k. We use a simple line
    # scan to avoid false positives on docstring mentions.
    for lineno, line in enumerate(src.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if stripped.startswith('"') or stripped.startswith("'"):
            continue
        assert not stripped.startswith("import cp2k"), (
            f"runner_cp2k.py:{lineno}: forbidden 'import cp2k' "
            f"(CLAUDE.md §10)"
        )
        assert not stripped.startswith("from cp2k"), (
            f"runner_cp2k.py:{lineno}: forbidden 'from cp2k' "
            f"(CLAUDE.md §10)"
        )
