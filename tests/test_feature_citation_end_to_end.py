"""End-to-end feature-citation surface — newly-wired routes reach disk.

Companion to ``tests/test_pob_citation_end_to_end.py``. The 2026-06
citation-database expansion added routing categories (``acceleration``,
``properties``, ``drivers`` gradient/hessian/tddft, ``scf_guess``) and
``assemble()`` flags (``uses_gradient``, ``uses_hessian``, ``uses_soscf``,
``uses_trah``, ``uses_tddft``, ``acceleration=[...]``, ``properties=[...]``,
…). The route tables *and* the ``assemble()`` layer were already unit-tested
in ``tests/test_citations.py`` — but several **already-shipped** features did
not pass the flags from their runners, so the citation never reached a job's
``.out`` / ``.bibtex`` / ``.references``.

This file pins the **runner wiring**: when a user actually exercises the
feature through the public driver (``run_job`` / ``run_tddft_*``), the
feature's defining-paper citation must surface in at least one of the
citation files the driver writes. Each test asserts on a citation entry that
is *unique to that feature's route*, so a green test proves the wiring fired
— not some incidental pull.

Channel-agnostic, exactly like the pob-* pin: whether the entry arrives via
``.bibtex``, ``.references``, or the in-``.out`` references block is not
asserted — the contract is "the user sees it", per CLAUDE.md § 8.

Per CLAUDE.md § 8 (citation discipline) + AGENTS.md § 8.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vibeqc as vq
from vibeqc.dispersion_d4 import dftd4_available
from vibeqc.output.citations.registry import load_default_database

# Loaded once: the test fingerprints are the *current* BibTeX keys of the
# pinned entries, so a rename in database.toml is followed automatically
# while a dropped entry / route fails loudly.
_DB = load_default_database()


def _bibtex_keys(*entry_keys: str) -> list[str]:
    """Resolve database entry keys → their BibTeX keys (which appear
    verbatim as ``@<kind>{<bibtex_key>,`` in every emitted ``.bibtex``)."""
    out: list[str] = []
    for key in entry_keys:
        entry = _DB.entries().get(key)
        assert entry is not None, (
            f"citation entry {key!r} is gone from database.toml — the "
            f"end-to-end pin is stale; update the expected fingerprint."
        )
        out.append(entry.bibtex_key)
    return out


def _h2() -> vq.Molecule:
    """Smallest closed-shell fixture — routes through RHF. 1.4 bohr."""
    return vq.Molecule([
        vq.Atom(1, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.4]),
    ])


def _surface(stem: Path) -> str:
    """Concatenate every citation-surface file ``run_job`` may write."""
    present = [
        stem.with_suffix(ext)
        for ext in (".references", ".bibtex", ".out")
        if stem.with_suffix(ext).is_file()
    ]
    assert present, f"no citation-surface files written for {stem.name!r}"
    return "\n".join(p.read_text(errors="replace") for p in present)


def _assert_cited(surface: str, entry_keys: list[str], *, feature: str) -> None:
    for entry_key, bibtex_key in zip(entry_keys, _bibtex_keys(*entry_keys)):
        assert bibtex_key in surface, (
            f"{feature}: citation {entry_key!r} (BibTeX key {bibtex_key!r}) "
            f"did not reach the emitted citation surface. The route exists in "
            f"database.toml and assemble() fires it — the runner is not "
            f"passing the flag when the feature actually runs."
        )


# --------------------------------------------------------------------------- #
# Acceleration — RI-J / RIJCOSX (density_fit / cosx flags)                     #
# --------------------------------------------------------------------------- #


def test_rij_coulomb_fitting_cites_eichkorn(tmp_path: Path) -> None:
    """``density_fit=True`` (RI-J Coulomb fitting) ⇒ Whitten/Dunlap/Eichkorn
    via ``acceleration=["rij"]``."""
    opts = vq.RHFOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-jk"
    stem = tmp_path / "h2_rij"
    vq.run_job(_h2(), basis="def2-svp", method="rhf",
               output=str(stem), rhf_options=opts, verbose=0)
    _assert_cited(
        _surface(stem),
        ["whitten_1973", "eichkorn_rij_1995"],
        feature="RI-J",
    )


def test_rijcosx_exchange_cites_neese(tmp_path: Path) -> None:
    """``cosx=True`` on top of density fitting (RIJCOSX chain-of-spheres
    exchange) ⇒ Neese 2009 via ``acceleration=["rijcosx"]``."""
    opts = vq.RHFOptions()
    opts.density_fit = True
    opts.aux_basis = "def2-svp-jk"
    opts.cosx = True
    stem = tmp_path / "h2_rijcosx"
    vq.run_job(_h2(), basis="def2-svp", method="rhf",
               output=str(stem), rhf_options=opts, verbose=0)
    _assert_cited(_surface(stem), ["neese_rijcosx_2009"], feature="RIJCOSX")


# --------------------------------------------------------------------------- #
# Drivers — analytic gradient + Hessian / vibrational analysis                #
# --------------------------------------------------------------------------- #


def test_geometry_optimization_cites_analytic_gradient(tmp_path: Path) -> None:
    """A geometry optimisation evaluates the analytic atomic gradient at
    every step ⇒ Pulay 1969 + Hellmann-Feynman via ``uses_gradient=True``."""
    stem = tmp_path / "h2_opt"
    vq.run_job(_h2(), basis="sto-3g", method="rhf",
               output=str(stem), optimize=True, verbose=0)
    _assert_cited(
        _surface(stem),
        ["pulay_forces_1969", "feynman_forces_1939"],
        feature="analytic gradient",
    )


def test_hessian_run_cites_vibrational_analysis(tmp_path: Path) -> None:
    """``hessian=True`` runs a finite-difference Hessian of the analytic
    gradient ⇒ Wilson-Decius-Cross via ``uses_hessian=True`` *and* the
    gradient papers (the Hessian consumes analytic forces)."""
    stem = tmp_path / "h2_hess"
    vq.run_job(_h2(), basis="sto-3g", method="rhf",
               output=str(stem), hessian=True, verbose=0)
    surface = _surface(stem)
    _assert_cited(surface, ["wilson_decius_cross_1955"], feature="Hessian")
    _assert_cited(surface, ["pulay_forces_1969"], feature="Hessian-gradient")


# --------------------------------------------------------------------------- #
# Properties — Mulliken / Löwdin / Mayer population analysis                   #
# --------------------------------------------------------------------------- #


def test_population_analysis_cites_mulliken_lowdin_mayer(tmp_path: Path) -> None:
    """The default population dump computes + surfaces Mulliken / Löwdin
    charges and Mayer bond orders ⇒ all three defining papers via
    ``properties=[...]``."""
    stem = tmp_path / "h2_pop"
    # write_population_file defaults True; no special options needed.
    vq.run_job(_h2(), basis="sto-3g", method="rhf",
               output=str(stem), verbose=0)
    _assert_cited(
        _surface(stem),
        ["mulliken_1955", "lowdin_1950", "mayer_bond_order_1983"],
        feature="population analysis",
    )


def test_failed_closed_npa_is_not_cited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A requested population dump does not imply successful NPA."""
    import vibeqc.output.formats.population as population_module

    real_compute = population_module.compute_population_summary

    def _failed_npa(*args, **kwargs):
        summary = real_compute(*args, **kwargs)
        summary.npa_atoms = []
        summary.errors["npa"] = "unsupported: occupancy-weighted NAOs unavailable"
        return summary

    monkeypatch.setattr(
        population_module,
        "compute_population_summary",
        _failed_npa,
    )
    stem = tmp_path / "h2_npa_failed"
    vq.run_job(
        _h2(),
        basis="sto-3g",
        method="rhf",
        output=str(stem),
        verbose=0,
    )
    surface = _surface(stem)
    assert _bibtex_keys("reed_weinhold_npa_1985")[0] not in surface
    _assert_cited(surface, ["mulliken_1955"], feature="population fallback")


def test_successful_npa_is_cited(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NPA route fires when non-empty NPA rows actually land."""
    import vibeqc.output.formats.population as population_module

    real_compute = population_module.compute_population_summary

    def _successful_npa(*args, **kwargs):
        summary = real_compute(*args, **kwargs)
        summary.npa_atoms = [
            (0, "H", 1.0, 0.0),
            (1, "H", 1.0, 0.0),
        ]
        summary.errors.pop("npa", None)
        return summary

    monkeypatch.setattr(
        population_module,
        "compute_population_summary",
        _successful_npa,
    )
    stem = tmp_path / "h2_npa_success"
    vq.run_job(
        _h2(),
        basis="sto-3g",
        method="rhf",
        output=str(stem),
        verbose=0,
    )
    _assert_cited(
        _surface(stem),
        ["reed_weinhold_npa_1985"],
        feature="successful NPA",
    )


def test_hirshfeld_charges_cite_hirshfeld_1977(tmp_path: Path) -> None:
    """When the best-effort Hirshfeld column actually computes (the default
    on a clean closed-shell run), the .out Atomic-charges table surfaces it
    ⇒ Hirshfeld 1977 via ``properties=[..., "hirshfeld"]``. The runner gates
    this on the column having really been built (Becke-Lebedev grid + SAD
    promolecule), not merely on the route existing — see the failure-path
    companion below."""
    stem = tmp_path / "h2_hirshfeld"
    vq.run_job(_h2(), basis="sto-3g", method="rhf",
               output=str(stem), verbose=0)
    _assert_cited(_surface(stem), ["hirshfeld_1977"], feature="Hirshfeld charges")


def test_hirshfeld_build_failure_is_not_cited(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Precision gate (CLAUDE.md § 8: no over-claim). If the Hirshfeld
    grid / SAD-promolecule build fails, the column never reaches the .out
    Atomic-charges table — so Hirshfeld 1977 must NOT be cited, even though
    the route exists and ``assemble()`` would happily fire it. The required
    basis-space Mulliken charges still cite, proving the run is otherwise
    healthy and we suppressed *only* the analysis that didn't compute."""
    def _boom(*_args, **_kwargs):
        raise RuntimeError("forced Hirshfeld grid failure (test)")

    # Patch the symbol _format_properties_block resolves at call time, so the
    # best-effort Hirshfeld try/except trips exactly as a real grid/SAD build
    # failure would — Mulliken/Löwdin are untouched.
    monkeypatch.setattr(
        "vibeqc.output.formats.scf_log.hirshfeld_charges", _boom,
    )
    stem = tmp_path / "h2_hirshfeld_fail"
    vq.run_job(_h2(), basis="sto-3g", method="rhf",
               output=str(stem), verbose=0)
    surface = _surface(stem)
    _assert_cited(
        surface, ["mulliken_1955"],
        feature="Mulliken (Hirshfeld-failure control)",
    )
    (hirsh_key,) = _bibtex_keys("hirshfeld_1977")
    assert hirsh_key not in surface, (
        "Hirshfeld 1977 was cited even though the Hirshfeld build failed — "
        "the citation over-claims a best-effort analysis that never reached "
        "the .out Atomic-charges table."
    )


# --------------------------------------------------------------------------- #
# ECP detection -- implicit versus explicit SCF options                       #
# --------------------------------------------------------------------------- #


def test_implicit_rhf_options_emit_the_same_ecp_citations(tmp_path: Path) -> None:
    """GitLab #657: the short ``run_job`` form must retain ECP attribution.

    ``run_job`` attaches LANL2DZ's ECP metadata to the effective options it
    sends to the SCF. Citation detection must inspect those effective options,
    not only an options object supplied by the caller.
    """
    import tomllib

    molecule = vq.Molecule([
        vq.Atom(16, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 1.815, 1.425]),
        vq.Atom(1, [0.0, -1.815, 1.425]),
    ])
    common = {
        "basis": "lanl2dz",
        "method": "rhf",
        "write_molden_file": False,
        "write_xyz_file": False,
        "write_population_file": False,
        "output_qvf": False,
        "progress": False,
        "verbose": 0,
    }
    implicit_stem = tmp_path / "h2s_implicit_options"
    explicit_stem = tmp_path / "h2s_explicit_options"
    vq.run_job(molecule, output=implicit_stem, **common)
    vq.run_job(
        molecule,
        output=explicit_stem,
        rhf_options=vq.RHFOptions(),
        **common,
    )

    expected_entries = (
        "dolg_cao_pseudopotentials_2011",
        "shaw_gilbert_libecpint",
    )
    expected_entry_set = set(expected_entries)
    bibtex_keys = dict(zip(expected_entries, _bibtex_keys(*expected_entries)))

    def _ecp_entries_by_artifact(stem: Path) -> dict[str, set[str]]:
        manifest = tomllib.loads(
            stem.with_suffix(".system").read_text(encoding="utf-8")
        )
        system_keys = {
            entry["key"] for entry in manifest["citations"].get("entries", [])
        }
        bibtex = stem.with_suffix(".bibtex").read_text(encoding="utf-8")
        references = stem.with_suffix(".references").read_text(encoding="utf-8")
        return {
            ".system": expected_entry_set & system_keys,
            ".bibtex": {
                entry
                for entry, key in bibtex_keys.items()
                if f"{{{key}," in bibtex
            },
            ".references": {
                entry
                for entry, key in bibtex_keys.items()
                if f"({key})" in references
            },
        }

    expected_by_artifact = {
        suffix: expected_entry_set
        for suffix in (".system", ".bibtex", ".references")
    }
    implicit = _ecp_entries_by_artifact(implicit_stem)
    explicit = _ecp_entries_by_artifact(explicit_stem)
    assert implicit == explicit == expected_by_artifact


# --------------------------------------------------------------------------- #
# SCF accelerator -- cited from the executed SCF, not the caller's struct     #
# --------------------------------------------------------------------------- #

_ACCELERATOR_KEYS = frozenset({
    "pulay_diis_1980",
    "pulay_diis_1982",
    "kudin_ediis_2002",
    "garza_scuseria_2012",
    "hu_yang_adiis_2010",
    "kollmar_kdiis_1997",
})


def _accelerator_keys(stem: Path) -> set[str]:
    import tomllib

    manifest = tomllib.loads(stem.with_suffix(".system").read_text("utf-8"))
    return _ACCELERATOR_KEYS & {
        entry["key"] for entry in manifest["citations"].get("entries", [])
    }


def _accelerator_common() -> dict:
    return {
        "basis": "sto-3g",
        "method": "rhf",
        "write_molden_file": False,
        "write_xyz_file": False,
        "write_population_file": False,
        "output_qvf": False,
        "progress": False,
        "verbose": 0,
    }


def test_implicit_and_explicit_rhf_options_emit_the_same_accelerator_citations(
    tmp_path: Path,
) -> None:
    """GitLab #682: ``run_job`` materialises ``RHFOptions()`` when the caller
    passes none, so the implicit and the explicit-``RHFOptions()`` run
    dispatch the same SCF. The accelerator citation used to be read from the
    caller's struct (absent => plain DIIS; present => its EDIIS+DIIS default),
    so two numerically identical runs recorded different reference sets. It
    is now read from the dispatched options and the executed trace, so the
    two sets are equal."""
    molecule = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.8]),
        vq.Atom(1, [1.7, 0.0, -0.5]),
    ])
    implicit_stem = tmp_path / "h2o_implicit_accelerator"
    explicit_stem = tmp_path / "h2o_explicit_accelerator"
    r_implicit = vq.run_job(molecule, output=implicit_stem, **_accelerator_common())
    r_explicit = vq.run_job(
        molecule,
        output=explicit_stem,
        rhf_options=vq.RHFOptions(),
        **_accelerator_common(),
    )
    assert r_implicit.energy == r_explicit.energy
    implicit = _accelerator_keys(implicit_stem)
    explicit = _accelerator_keys(explicit_stem)
    assert implicit == explicit
    assert {"pulay_diis_1980", "pulay_diis_1982"} <= implicit


def test_accelerator_citations_follow_the_extrapolation_steps_that_ran(
    tmp_path: Path,
) -> None:
    """GitLab #682: the trace records which extrapolation reached the Fock
    matrix each iteration (``SCFIteration.accelerator_step``), and the
    EDIIS+DIIS hybrid is credited only when an EDIIS branch was taken.

    Kudin, Scuseria and Cancès 2002 define EDIIS; Garza and Scuseria 2012
    recommend the EDIIS+DIIS hybrid with the commutator-norm switch. A hybrid
    whose switch metric never exceeded the threshold executed Pulay DIIS on
    every cycle and cites Pulay 1980 / 1982 alone."""
    molecule = vq.Molecule([
        vq.Atom(8, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.0, 0.0, 1.8]),
        vq.Atom(1, [1.7, 0.0, -0.5]),
    ])
    pulay = {"pulay_diis_1980", "pulay_diis_1982"}
    hybrid = pulay | {"kudin_ediis_2002", "garza_scuseria_2012"}

    def run(tag, **fields):
        opts = vq.RHFOptions()
        for name, value in fields.items():
            setattr(opts, name, value)
        stem = tmp_path / f"h2o_{tag}"
        result = vq.run_job(
            molecule, output=stem, rhf_options=opts, **_accelerator_common()
        )
        steps = {int(e.accelerator_step) for e in result.scf_trace}
        return _accelerator_keys(stem), steps

    # Plain DIIS: only Pulay steps, only Pulay citations.
    keys, steps = run("diis", scf_accelerator=vq.SCFAccelerator.DIIS)
    assert keys == pulay
    assert steps <= {0, 1} and 1 in steps

    # Hybrid configured, switch metric never above the threshold: every
    # applied step is DIIS, so the hybrid is not credited.
    keys, steps = run(
        "hybrid_never_ediis",
        scf_accelerator=vq.SCFAccelerator.EDIIS_DIIS,
        ediis_diis_switch_threshold=1.0e12,
    )
    assert keys == pulay
    assert steps <= {0, 1} and 1 in steps

    # Hybrid configured, switch metric always above the threshold: EDIIS
    # branches ran, so Kudin 2002 and Garza-Scuseria 2012 are credited.
    keys, steps = run(
        "hybrid_always_ediis",
        scf_accelerator=vq.SCFAccelerator.EDIIS_DIIS,
        ediis_diis_switch_threshold=0.0,
    )
    assert keys == hybrid
    assert 3 in steps and steps <= {0, 3}

    # Pure EDIIS: Kudin 2002 without the hybrid paper.
    keys, steps = run("ediis", scf_accelerator=vq.SCFAccelerator.EDIIS)
    assert keys == pulay | {"kudin_ediis_2002"}
    assert 3 in steps and steps <= {0, 3}


def test_detect_scf_accelerator_prefers_dispatched_options_and_executed_steps() -> None:
    """Unit contract of ``runner._detect_scf_accelerator`` (#682)."""
    from types import SimpleNamespace

    from vibeqc.runner import _detect_scf_accelerator

    def trace(*steps):
        return SimpleNamespace(
            scf_trace=[SimpleNamespace(accelerator_step=s) for s in steps]
        )

    configured = SimpleNamespace(scf_accelerator=vq.SCFAccelerator.EDIIS_DIIS)
    diis_only = SimpleNamespace(scf_accelerator=vq.SCFAccelerator.DIIS)

    # No caller struct, but the runner dispatched one: the dispatched struct
    # decides (pre-fix this returned None => default "diis").
    assert (
        _detect_scf_accelerator("rhf", None, None, None, None,
                                options_used=configured, result=None)
        == "ediis_diis"
    )
    # Dispatched struct beats the caller's struct (convergence retry).
    assert (
        _detect_scf_accelerator("rhf", configured, None, None, None,
                                options_used=diis_only, result=None)
        == "diis"
    )
    # Executed steps refine the hybrid: no EDIIS branch taken => plain DIIS.
    assert (
        _detect_scf_accelerator("rhf", None, None, None, None,
                                options_used=configured, result=trace(0, 1, 1))
        == "diis"
    )
    assert (
        _detect_scf_accelerator("rhf", None, None, None, None,
                                options_used=configured, result=trace(0, 3, 1))
        == "ediis_diis"
    )
    # A trace without any applied extrapolation reports the configuration.
    assert (
        _detect_scf_accelerator("rhf", None, None, None, None,
                                options_used=configured, result=trace(0, 0))
        == "ediis_diis"
    )
    # Non-switching accelerators are reported as configured.
    kdiis = SimpleNamespace(scf_accelerator=vq.SCFAccelerator.KDIIS)
    assert (
        _detect_scf_accelerator("rhf", None, None, None, None,
                                options_used=kdiis, result=trace(0, 2))
        == "kdiis"
    )
    # Post-SCF paths with no options anywhere stay None.
    assert _detect_scf_accelerator("rhf", None, None, None, None) is None


# --------------------------------------------------------------------------- #
# Methods — SOSCF / TRAH second-order convergers                              #
# --------------------------------------------------------------------------- #


def test_soscf_threshold_cites_second_order_scf(tmp_path: Path) -> None:
    """An armed ``soscf_threshold`` ⇒ Bacskay 1981 + Neese 2000 via
    ``uses_soscf=True``."""
    opts = vq.RHFOptions()
    opts.soscf_threshold = 1.0
    stem = tmp_path / "h2_soscf"
    vq.run_job(_h2(), basis="sto-3g", method="rhf",
               output=str(stem), rhf_options=opts, verbose=0)
    _assert_cited(
        _surface(stem),
        ["bacskay_qcscf_1981", "neese_soscf_2000"],
        feature="SOSCF",
    )


def test_trah_threshold_cites_helmich_paris(tmp_path: Path) -> None:
    """An armed ``trah_threshold`` ⇒ Helmich-Paris 2021 via
    ``uses_trah=True``."""
    opts = vq.RHFOptions()
    opts.trah_threshold = 1.0
    stem = tmp_path / "h2_trah"
    vq.run_job(_h2(), basis="sto-3g", method="rhf",
               output=str(stem), rhf_options=opts, verbose=0)
    _assert_cited(_surface(stem), ["helmich_paris_trah_2021"], feature="TRAH")


# --------------------------------------------------------------------------- #
# SCF initial guess — explicit SAD                                            #
# --------------------------------------------------------------------------- #


def test_explicit_sad_guess_cites_van_lenthe(tmp_path: Path) -> None:
    """An explicit ``initial_guess=SAD`` ⇒ van Lenthe 2006 via
    ``scf_guess="sad"``."""
    opts = vq.RHFOptions()
    opts.initial_guess = vq.InitialGuess.SAD
    stem = tmp_path / "h2_sad"
    vq.run_job(_h2(), basis="sto-3g", method="rhf",
               output=str(stem), rhf_options=opts, verbose=0)
    _assert_cited(_surface(stem), ["vanlenthe_sad_2006"], feature="SAD guess")


def test_materialized_default_patom_guess_cites_van_lenthe(
    tmp_path: Path,
) -> None:
    """A runner-created default options object still contributes its guess."""
    stem = tmp_path / "h2_default_patom"
    vq.run_job(
        _h2(),
        basis="sto-3g",
        method="rhf",
        output=str(stem),
        verbose=0,
    )
    _assert_cited(
        _surface(stem),
        ["vanlenthe_sad_2006"],
        feature="materialized default PATOM guess",
    )


def test_explicit_sap_guess_cites_both_source_papers(tmp_path: Path) -> None:
    """The public SAP selector emits both papers routed by ``scf_guess``."""
    stem = tmp_path / "h2_sap"
    vq.run_job(
        _h2(),
        basis="sto-3g",
        method="rhf",
        initial_guess="sap",
        output=str(stem),
        verbose=0,
    )
    _assert_cited(
        _surface(stem),
        ["lehtola_sap_2019", "lehtola_visscher_engel_sap_2020"],
        feature="SAP guess",
    )


def test_closed_shell_auto_to_patom_cites_executed_guess(tmp_path: Path) -> None:
    """Molecular closed-shell AUTO resolves to PATOM and cites that method."""
    stem = tmp_path / "h2_auto_patom"
    vq.run_job(
        _h2(),
        basis="sto-3g",
        method="rhf",
        initial_guess="auto",
        output=str(stem),
        verbose=0,
    )
    _assert_cited(
        _surface(stem),
        ["vanlenthe_sad_2006"],
        feature="AUTO-resolved PATOM guess",
    )

    assert "initial_guess" in stem.with_suffix(".out").read_text()
    assert "AUTO -> PATOM" in stem.with_suffix(".out").read_text()


def test_open_shell_auto_to_sad_cites_sad_not_sap(tmp_path: Path) -> None:
    """Open-shell AUTO resolves to SAD and must not be labelled as SAP."""
    radical = vq.Molecule(
        [vq.Atom(1, [0.0, 0.0, -0.7]), vq.Atom(1, [0.0, 0.0, 0.7])],
        charge=1,
        multiplicity=2,
    )
    stem = tmp_path / "h2plus_auto_sad"
    result = vq.run_job(
        radical,
        basis="sto-3g",
        method="uhf",
        initial_guess="auto",
        output=str(stem),
        verbose=0,
    )
    assert result.converged
    surface = _surface(stem)
    _assert_cited(surface, ["vanlenthe_sad_2006"], feature="AUTO-resolved SAD")
    for sap_key in _bibtex_keys(
        "lehtola_sap_2019", "lehtola_visscher_engel_sap_2020"
    ):
        assert sap_key not in surface


# --------------------------------------------------------------------------- #
# TDDFT — run_tddft_tda(output=...) writes the formalism papers              #
# --------------------------------------------------------------------------- #


def test_tddft_tda_emits_runge_gross_casida_hirata(tmp_path: Path) -> None:
    """``run_tddft_tda(..., output=stem)`` ⇒ Runge-Gross 1984 + Casida 1995
    (``uses_tddft=True``) plus Hirata-Head-Gordon 1999 for the Tamm-Dancoff
    variant (``tddft_variant="tda"``)."""
    mol = _h2()
    basis = vq.BasisSet(mol, "sto-3g")
    scf_opts = vq.RHFOptions()
    scf_opts.conv_tol_energy = 1e-10
    res = vq.run_rhf(mol, basis, scf_opts)
    n_occ = mol.n_electrons() // 2

    stem = tmp_path / "h2_tddft_tda"
    from vibeqc.tddft import run_tddft_tda

    run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ,
        n_states=1, output=str(stem),
    )
    _assert_cited(
        _surface(stem),
        ["runge_gross_1984", "casida_tddft_1995", "hirata_headgordon_tda_1999"],
        feature="TDDFT-TDA",
    )


def test_tddft_casida_emits_runge_gross_without_tda_paper(tmp_path: Path) -> None:
    """Full Casida (``run_tddft_casida``) ⇒ Runge-Gross + Casida, but *not*
    the TDA-only Hirata paper (the variant flag stays off)."""
    mol = _h2()
    basis = vq.BasisSet(mol, "sto-3g")
    scf_opts = vq.RHFOptions()
    scf_opts.conv_tol_energy = 1e-10
    res = vq.run_rhf(mol, basis, scf_opts)
    n_occ = mol.n_electrons() // 2

    stem = tmp_path / "h2_tddft_casida"
    from vibeqc.tddft import run_tddft_casida

    run_tddft_casida(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ,
        n_states=1, output=str(stem),
    )
    surface = _surface(stem)
    _assert_cited(
        surface,
        ["runge_gross_1984", "casida_tddft_1995"],
        feature="TDDFT-Casida",
    )
    (hirata_key,) = _bibtex_keys("hirata_headgordon_tda_1999")
    assert hirata_key not in surface, (
        "full Casida must not pull the TDA-only Hirata-Head-Gordon paper — "
        "tddft_variant should be None for run_tddft_casida."
    )


def test_tddft_hybrid_kernel_emits_bauernschmitt(tmp_path: Path) -> None:
    """A hybrid-functional TDA run (response kernel α·HF-exchange + f_xc)
    ⇒ Bauernschmitt-Ahlrichs 1996 via ``tddft_hybrid_kernel=True``; a
    pure-functional run must not pull it."""
    import numpy as np

    from vibeqc.tddft import run_tddft_tda

    mol = _h2()
    basis = vq.BasisSet(mol, "sto-3g")
    n_occ = mol.n_electrons() // 2

    opts = vq.RKSOptions()
    opts.functional = "pbe0"
    opts.conv_tol_energy = 1e-10
    res = vq.run_rks(mol, basis, opts)

    stem = tmp_path / "h2_tddft_tda_pbe0"
    run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ,
        n_states=1, functional="pbe0",
        density_ao=np.asarray(res.density), output=str(stem),
    )
    _assert_cited(
        _surface(stem),
        ["bauernschmitt_ahlrichs_1996"],
        feature="TDDFT-hybrid-kernel",
    )

    opts = vq.RKSOptions()
    opts.functional = "lda"
    opts.conv_tol_energy = 1e-10
    res = vq.run_rks(mol, basis, opts)

    stem = tmp_path / "h2_tddft_tda_lda"
    run_tddft_tda(
        mol, basis, res.mo_energies, res.mo_coeffs, n_occ,
        n_states=1, functional="lda",
        density_ao=np.asarray(res.density), output=str(stem),
    )
    (bauernschmitt_key,) = _bibtex_keys("bauernschmitt_ahlrichs_1996")
    assert bauernschmitt_key not in _surface(stem), (
        "pure-functional TDA (c_x = 0) must not pull the hybrid-kernel "
        "Bauernschmitt-Ahlrichs paper — tddft_hybrid_kernel should be "
        "False when the exchange admixture is zero."
    )


# --------------------------------------------------------------------------- #
# Periodic runner — default SAD guess routes through the same scf_guess flag  #
# --------------------------------------------------------------------------- #


def test_bipole_sr_range_screening_cites_charge_pair_bound(tmp_path: Path) -> None:
    """The padded default cites the charge-pair bound; the historical opt-out does not."""
    import numpy as np

    from vibeqc import PeriodicSystem
    from vibeqc.periodic_runner import run_periodic_job

    def _run(stem: Path, **kw) -> str:
        cell = PeriodicSystem(
            3,
            np.eye(3) * 6.0,
            unit_cell=[
                vq.Atom(1, [0.0, 0.0, 0.0]),
                vq.Atom(1, [0.0, 0.0, 1.4]),
            ],
        )
        basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
        run_periodic_job(
            cell, basis, method="RHF", output=stem,
            jk_method="bipole", bipole_cutoff_bohr=6.0,
            max_iter=8, conv_tol_energy=1e-4,
            write_xyz_file=False, write_poscar_file=False,
            write_cif_file=False, write_xsf_structure_file=False,
            write_molden_file=False, write_population_file=False,
            write_density=False, output_qvf=False,
            citations=True, verbose=0, **kw,
        )
        return _surface(stem)

    on = _run(tmp_path / "h2_bipole_srr_default")
    _assert_cited(on, ["sun_range_separated_exchange_2023"], feature="SR range screening")
    (bib_key,) = _bibtex_keys("sun_range_separated_exchange_2023")
    off = _run(tmp_path / "h2_bipole_legacy", sr_image_precision=None)
    assert bib_key not in off, (
        "Charge-pair citation fired on the historical unpadded BIPOLE route"
    )


def test_periodic_default_sad_guess_cites_van_lenthe(tmp_path: Path) -> None:
    """``run_periodic_job`` defaults to ``initial_guess="SAD"`` ⇒ van Lenthe
    2006 via ``scf_guess="sad"``, alongside the always-on spglib periodic
    citation. Mirrors the fast H₂-in-a-6-bohr-box fixture used elsewhere."""
    import numpy as np

    from vibeqc import PeriodicSystem
    from vibeqc.periodic_runner import run_periodic_job

    lattice = np.eye(3) * 6.0
    cell = PeriodicSystem(
        3,
        lattice,
        unit_cell=[
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.4]),
        ],
    )
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "h2_periodic_sad"
    run_periodic_job(
        cell, basis, method="RHF", output=stem,
        max_iter=30, conv_tol_energy=1e-6,
        write_xyz_file=False, write_poscar_file=False, write_cif_file=False,
        write_xsf_structure_file=False, write_molden_file=False,
        write_population_file=False, write_density=False, output_qvf=False,
        citations=True, verbose=0,
    )
    _assert_cited(_surface(stem), ["vanlenthe_sad_2006"], feature="periodic SAD")


def test_periodic_auto_reports_and_cites_resolved_sad(tmp_path: Path) -> None:
    """Periodic AUTO provenance follows the concrete unified-engine choice."""
    import numpy as np

    from vibeqc import PeriodicSystem
    from vibeqc.periodic_runner import run_periodic_job

    cell = PeriodicSystem(
        3,
        np.eye(3) * 6.0,
        unit_cell=[
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.4]),
        ],
    )
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "h2_periodic_auto"
    run_periodic_job(
        cell,
        basis,
        method="RHF",
        initial_guess=" auto ",
        output=stem,
        max_iter=30,
        conv_tol_energy=1e-6,
        write_xyz_file=False,
        write_poscar_file=False,
        write_cif_file=False,
        write_xsf_structure_file=False,
        write_molden_file=False,
        write_population_file=False,
        write_density=False,
        output_qvf=False,
        citations=True,
        verbose=0,
    )

    surface = _surface(stem)
    assert "initial_guess       = AUTO -> SAD" in surface
    _assert_cited(surface, ["vanlenthe_sad_2006"], feature="periodic AUTO->SAD")
    for sap_key in _bibtex_keys(
        "lehtola_sap_2019", "lehtola_visscher_engel_sap_2020"
    ):
        assert sap_key not in surface


def test_periodic_explicit_sap_guess_cites_both_source_papers(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The public periodic runner builds SAP and emits its two papers."""
    import numpy as np

    import vibeqc.guess as guess_module
    from vibeqc import PeriodicSystem
    from vibeqc.periodic_runner import run_periodic_job

    calls = 0
    original = guess_module.compute_vsap_lattice

    def counted_vsap(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(guess_module, "compute_vsap_lattice", counted_vsap)

    cell = PeriodicSystem(
        3,
        np.eye(3) * 10.0,
        unit_cell=[vq.Atom(2, [0.0, 0.0, 0.0])],
    )
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "he_periodic_sap"
    result = run_periodic_job(
        cell,
        basis,
        method="RHF",
        initial_guess="SAP",
        output=stem,
        max_iter=30,
        conv_tol_energy=1e-6,
        write_xyz_file=False,
        write_poscar_file=False,
        write_cif_file=False,
        write_xsf_structure_file=False,
        write_molden_file=False,
        write_population_file=False,
        write_density=False,
        output_qvf=False,
        citations=True,
        verbose=0,
    )
    assert result.converged
    assert calls == 1
    _assert_cited(
        _surface(stem),
        ["lehtola_sap_2019", "lehtola_visscher_engel_sap_2020"],
        feature="periodic SAP",
    )


# --------------------------------------------------------------------------- #
# k-point construction conventions (numerics=[...] from KPoints provenance)    #
# --------------------------------------------------------------------------- #
#
# Until 2026-07-10 no production caller ever passed `numerics=` to assemble(),
# so every [routes.numerics] row was unreachable from a real job while the unit
# tests in test_citations.py -- which call assemble(numerics=[...]) directly --
# kept passing. These pin the *runner wiring*, which is the part that was
# missing.


def _h2_cell():
    import numpy as np

    from vibeqc import PeriodicSystem

    return PeriodicSystem(
        3,
        np.eye(3) * 6.0,
        unit_cell=[vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])],
    )


def _run_periodic(cell, stem, **kw):
    from vibeqc.periodic_runner import run_periodic_job

    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
    kw.setdefault("output_qvf", False)
    kw.setdefault("method", "RHF")
    run_periodic_job(
        cell, basis, output=stem,
        max_iter=30, conv_tol_energy=1e-6,
        write_xyz_file=False, write_poscar_file=False, write_cif_file=False,
        write_xsf_structure_file=False, write_molden_file=False,
        write_population_file=False, write_density=False,
        citations=True, verbose=0, **kw,
    )


def test_kppra_mesh_cites_the_aflow_density_convention(tmp_path: Path) -> None:
    """A mesh chosen by ``KPoints.from_kppra`` cites AFLOW (Curtarolo 2012).

    The mesh itself is Monkhorst-Pack; what is cited is the KPPRA density
    convention that picked it, carried on ``KPoints.citation_numerics``."""
    cell = _h2_cell()
    stem = tmp_path / "h2_kppra"
    _run_periodic(cell, stem, kpoints=vq.KPoints.from_kppra(cell, 4))
    _assert_cited(_surface(stem), ["curtarolo_aflow_2012"], feature="KPPRA mesh")


def test_plain_monkhorst_pack_does_not_cite_kppra(tmp_path: Path) -> None:
    """Negative control. A hand-specified mesh follows no published density
    convention; over-citing is as wrong as under-citing, and without this the
    test above would pass even if the runner cited AFLOW unconditionally."""
    cell = _h2_cell()
    stem = tmp_path / "h2_plain_mp"
    _run_periodic(cell, stem, kpoints=vq.KPoints.monkhorst_pack(cell, [2, 2, 2]))
    (aflow,) = _bibtex_keys("curtarolo_aflow_2012")
    assert aflow not in _surface(stem), (
        "a plain Monkhorst-Pack mesh cited the AFLOW KPPRA convention it never "
        "used -- the runner is passing the numerics key unconditionally."
    )


def test_seekpath_band_path_cites_hpkot(tmp_path: Path) -> None:
    """An attached band structure cites the HPKOT path convention (Hinuma 2017).

    The key rides the BandStructure's ``kpath``, not the SCF mesh: the two are
    different k-point sets and only the path follows a published convention."""
    pytest.importorskip("seekpath")
    from vibeqc.bands import band_structure_hcore

    cell = _h2_cell()
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
    kpath = vq.KPoints.band_path(cell).to_kpath()
    bs = band_structure_hcore(cell, basis, kpath)
    stem = tmp_path / "h2_bands"
    _run_periodic(cell, stem, band_structure=bs)
    _assert_cited(_surface(stem), ["hinuma_seekpath_2017"], feature="HPKOT band path")


def test_periodic_system_manifest_carries_assembled_citations(
    tmp_path: Path,
) -> None:
    """Regression: ``run_periodic_job`` assembled its citations and wrote
    ``.bibtex`` / ``.references`` / the in-``.out`` block, but never fed
    the rows to ``ManifestUpdater.set_citations`` — so every periodic
    ``.system`` ended with ``[citations] count = 0`` (qc-input-library
    artifact 01706-c-diamond-rhf-sto3g-k222-qvf, v0.15.28). The
    ``.system`` manifest is the internal-provenance destination for the
    *full* assembled list (CLAUDE.md § 8), so the count must match the
    entries and cover the routes this run fired."""
    import tomllib

    import numpy as np

    from vibeqc import PeriodicSystem
    from vibeqc.periodic_runner import run_periodic_job

    lattice = np.eye(3) * 6.0
    cell = PeriodicSystem(
        3,
        lattice,
        unit_cell=[
            vq.Atom(1, [0.0, 0.0, 0.0]),
            vq.Atom(1, [0.0, 0.0, 1.4]),
        ],
    )
    basis = vq.BasisSet(cell.unit_cell_molecule(), "sto-3g")
    stem = tmp_path / "h2_periodic_manifest_citations"
    run_periodic_job(
        cell, basis, method="RHF", output=stem,
        max_iter=30, conv_tol_energy=1e-6,
        write_xyz_file=False, write_poscar_file=False, write_cif_file=False,
        write_xsf_structure_file=False, write_molden_file=False,
        write_population_file=False, write_density=False, output_qvf=False,
        citations=True, verbose=0,
    )
    manifest = tomllib.loads(
        stem.with_suffix(".system").read_text(encoding="utf-8")
    )
    cites = manifest["citations"]
    entries = cites.get("entries", [])
    assert cites["count"] == len(entries) > 0, (
        "periodic .system [citations] is disconnected from the assembled "
        "citations — the runner did not call ManifestUpdater.set_citations."
    )
    keys = {e["key"] for e in entries}
    # Spot-check routes this run certainly fired: the periodic flag
    # (spglib, entry togo_shinohara_tanaka_2024) and the default SAD guess.
    assert "vanlenthe_sad_2006" in keys
    assert "togo_shinohara_tanaka_2024" in keys


# --------------------------------------------------------------------------- #
# COOP/COHP bonding analysis (properties=["coop", "cohp"])                     #
# --------------------------------------------------------------------------- #
#
# Until 2026-07-11 the periodic runner computed COOP/COHP for the QVF
# dos.coop / dos.cohp sections but never passed properties=[...] to
# assemble(), so the defining papers (Hughbanks-Hoffmann 1983, Dronskowski
# 1993, Deringer 2011, LOBSTER) were unreachable from any real periodic job
# while the routes.properties unit tests kept passing. These pin the runner
# wiring.

_COHP_ENTRY_KEYS = [
    "hughbanks_hoffmann_coop_1983",
    "dronskowski_cohp_1993",
    "deringer_cohp_2011",
    "maintz_lobster3_2016",
]


def test_periodic_coop_cohp_cites_the_defining_papers(tmp_path: Path) -> None:
    """``coop_cohp=True`` with QVF output runs the COOP/COHP analysis ⇒ the
    COOP (Hughbanks-Hoffmann) + COHP (Dronskowski-Bloechl) + projection
    (Deringer/LOBSTER) papers reach the citation surface *and* the full
    assembled list in the ``.system`` manifest."""
    import tomllib

    cell = _h2_cell()
    stem = tmp_path / "h2_coop_cohp"
    _run_periodic(
        cell, stem,
        output_qvf=True, coop_cohp=True, dos_kmesh=[2, 2, 2],
    )
    _assert_cited(_surface(stem), _COHP_ENTRY_KEYS, feature="COOP/COHP")
    # The analysis must actually have produced its QVF sections -- otherwise
    # the citation pin above would hold even for a silently-failed analysis.
    assert stem.with_suffix(".qvf").is_file()
    manifest = tomllib.loads(
        stem.with_suffix(".system").read_text(encoding="utf-8")
    )
    keys = {e["key"] for e in manifest["citations"].get("entries", [])}
    for entry_key in _COHP_ENTRY_KEYS:
        assert entry_key in keys, (
            f"COOP/COHP: {entry_key!r} reached the user-facing surface but "
            f"not the .system [citations] manifest."
        )


def test_periodic_coop_cohp_without_qvf_is_refused_and_not_cited(
    tmp_path: Path,
) -> None:
    """Negative control. ``coop_cohp=True`` with ``output_qvf=False`` cannot
    run the analysis (the COOP/COHP block lives inside the QVF/DOS section and
    has no other sink). Until IID 195 the runner silently dropped the request
    and this test pinned that the papers were at least not cited for an
    analysis that never ran. The runner now fails closed before SCF, so the
    over-claim is unreachable: the refusal is the control, and no citation
    surface may exist for the stem."""
    cell = _h2_cell()
    stem = tmp_path / "h2_coop_no_qvf"
    with pytest.raises(ValueError, match="coop_cohp=True requires output_qvf=True"):
        _run_periodic(cell, stem, output_qvf=False, coop_cohp=True)
    assert not stem.with_suffix(".bibtex").exists(), (
        "COOP/COHP: a citation surface was written although the run was "
        "refused before SCF -- the analysis never ran, so nothing may cite it."
    )


# --------------------------------------------------------------------------- #
# Dispersion — D4 method paper + per-parametrization damping-fit paper        #
# --------------------------------------------------------------------------- #

_needs_dftd4 = pytest.mark.skipif(
    not dftd4_available(), reason="optional dftd4 backend not installed"
)


@_needs_dftd4
def test_hf_d4_cites_all_three_d4_papers(tmp_path: Path) -> None:
    """``dispersion="d4"`` ⇒ the full Caldeweyher 2017+2019+2020 set
    upstream dftd4 asks users to cite (maintainer decision 2026-06-11)
    via ``dispersion="d4"`` in ``assemble()``. Also pins the runner gap
    fixed in 2026-06: ``run_job`` used to leave ``_dispersion_key``
    unset on the D4 path, so a D4 run cited nothing for its dispersion
    correction."""
    stem = tmp_path / "h2_hf_d4"
    vq.run_job(_h2(), basis="sto-3g", method="rhf",
               dispersion="d4", output=str(stem), verbose=0)
    _assert_cited(
        _surface(stem),
        ["caldeweyher_d4_2017", "caldeweyher_d4_2019", "caldeweyher_d4_2020"],
        feature="HF-D4",
    )


@_needs_dftd4
def test_r2scan_d4_cites_the_damping_fit_paper(tmp_path: Path) -> None:
    """r²SCAN-D4 damping was fit in Ehlert 2021, not the 2019 D4 method
    paper ⇒ the fit paper must surface alongside the three-paper method
    set, via ``dispersion_params="r2scan"`` firing
    ``routes.dispersion_params["d4:r2scan"]`` (CLAUDE.md § 8: a run cites
    the paper its damping parameters come from)."""
    stem = tmp_path / "h2_r2scan_d4"
    vq.run_job(_h2(), basis="sto-3g", method="rks", functional="r2scan",
               dispersion="d4", output=str(stem), verbose=0)
    _assert_cited(
        _surface(stem),
        ["caldeweyher_d4_2017", "caldeweyher_d4_2019",
         "caldeweyher_d4_2020", "ehlert_r2scan_d4_2021"],
        feature="r2SCAN-D4 damping fit",
    )


# --------------------------------------------------------------------------- #
# AICCM front door M0 -- the wired AICCM routes cite what they execute          #
# (handovers/HANDOVER_AICCM_STANDARD_METHOD.md). T2 placement per D-8: nothing #
# here imports vibeqc.periodic.ccm; the routes are reached through            #
# run_periodic_job and the warning classes through their public vibeqc names. #
# --------------------------------------------------------------------------- #

_CCM_LINEAGE = ["bredow_geudtner_jug_ccm_2001", "peintinger_ccm_2014"]


def _run_aiccm(cell, stem: Path, warning, **kw) -> str:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", warning)
        _run_periodic(cell, stem, **kw)
    return _surface(stem)


def test_real_gamma_job_cites_the_ccm_lineage_and_the_gdf_stack(tmp_path: Path) -> None:
    """``method="aiccm", variant="real-gamma"`` (M1 front door; the legacy
    ``jk_method="real-gamma"`` maps onto the same member) is the real-Γ
    producer of the neutral Γ-CCM Hamiltonian ⇒ ``routes.methods["real-gamma"]``
    (both CCM papers) plus the GDF stack its fold cderi executes (``uses_gdf``
    ⇒ Sun-Berkelbach 2017). Before M0 the runner passed ``method="RHF"`` (a
    silent routes.methods miss) and ``uses_gdf=False``, so the references
    block read as a plain periodic RHF. A single point must not cite the
    gradient paper."""
    stem = tmp_path / "h2_real_gamma"
    surface = _run_aiccm(
        _h2_cell(), stem, vq.AICCM2026DevAExperimentalWarning,
        method="aiccm", variant="real-gamma", aiccm_lattice_extension=(2, 1, 1),
    )
    _assert_auto_hcore_artifacts(stem, surface)
    _assert_cited(surface, _CCM_LINEAGE, feature="real-gamma CCM lineage")
    _assert_cited(surface, ["sun_berkelbach_gdf_2017"], feature="real-gamma GDF stack")
    (pulay,) = _bibtex_keys("pulay_forces_1969")
    assert pulay not in surface, (
        "a real-gamma single point cited the analytic-gradient paper it never "
        "used -- uses_gradient is being passed unconditionally."
    )


def test_four_center_job_cites_the_ccm_lineage(tmp_path: Path) -> None:
    """The public literal-WSSC arm surfaces its two defining CCM papers."""
    stem = tmp_path / "h2_four_center"
    surface = _run_aiccm(
        _h2_cell(),
        stem,
        vq.AICCM2026DevAExperimentalWarning,
        method="aiccm",
        variant="four-center",
        aiccm_lattice_extension=(1, 1, 1),
    )
    _assert_auto_hcore_artifacts(stem, surface)
    _assert_cited(surface, _CCM_LINEAGE, feature="four-center CCM lineage")


def test_neutral_bloch_job_cites_the_ccm_lineage_and_the_gdf_stack(
    tmp_path: Path,
) -> None:
    """``method="aiccm", variant="neutral-bloch"`` is the OTHER producer of the
    same neutral Γ-CCM Hamiltonian (ruling R1) ⇒ the pre-positioned
    ``routes.methods["neutral-bloch"]`` row fires with both CCM papers, plus
    the GDF stack it literally executes. A single point owes no gradient
    paper. The plain-GDF negative control below is the other half: the same
    multi-k drivers on the same mesh must NOT cite the CCM lineage, or the
    row would be decoration rather than provenance."""
    stem = tmp_path / "h2_neutral_bloch"
    surface = _run_aiccm(
        _h2_cell(), stem, vq.AICCM2026DevAExperimentalWarning,
        method="aiccm", variant="neutral-bloch",
        aiccm_lattice_extension=(2, 1, 1),
    )
    _assert_cited(surface, _CCM_LINEAGE, feature="neutral-bloch CCM lineage")
    _assert_cited(
        surface, ["sun_berkelbach_gdf_2017"], feature="neutral-bloch GDF stack"
    )
    (pulay,) = _bibtex_keys("pulay_forces_1969")
    assert pulay not in surface


def test_plain_gdf_job_does_not_cite_the_ccm_lineage(tmp_path: Path) -> None:
    """The negative control for both neutral producers: ``jk_method="gdf"`` on
    the same Gamma-centred mesh executes the same drivers and must cite the
    GDF stack only."""
    stem = tmp_path / "h2_plain_gdf"
    _run_periodic(_h2_cell(), stem, jk_method="gdf", kpoints=(2, 1, 1))
    surface = _surface(stem)
    _assert_cited(surface, ["sun_berkelbach_gdf_2017"], feature="plain GDF")
    for key in _bibtex_keys(*_CCM_LINEAGE):
        assert key not in surface, (
            "a plain unit-cell GDF job cited the CCM lineage: the "
            "routes.methods rows are firing on the wrong selector."
        )


def test_chi_job_cites_the_ccm_lineage(tmp_path: Path) -> None:
    """``method="aiccm", variant="chi"`` (M1 front door; legacy
    ``jk_method="aiccm2026dev-b"``) passes the bare χ key ⇒ both CCM papers
    reach the citation surface (the database.toml claim at the aiccm2026dev-b
    row, unpinned until M0)."""
    stem = tmp_path / "h2_chi"
    surface = _run_aiccm(
        _h2_cell(), stem, vq.AICCM2026DevBExperimentalWarning,
        method="aiccm", variant="chi", aiccm_backend="four_center",
        aiccm_lattice_extension=(1, 1, 1),
    )
    _assert_cited(surface, _CCM_LINEAGE, feature="chi CCM lineage")


@pytest.mark.parametrize("method", ["rohf", "roks"])
def test_ro_sap_guess_cites_executed_construction(tmp_path, method):
    stem = tmp_path / method
    kw = {"functional": "lda"} if method == "roks" else {}
    result = vq.run_job(
        _h2(), basis="sto-3g", method=method, initial_guess="sap",
        output=stem, verbose=0, **kw,
    )
    assert result.guess_selection.effective == vq.InitialGuess.SAP
    _assert_cited(_surface(stem),
                  ["lehtola_sap_2019", "lehtola_visscher_engel_sap_2020"],
                  feature="RO SAP guess")

def _assert_auto_hcore_artifacts(stem: Path, surface: str) -> None:
    import tomllib

    assert "initial_guess       = AUTO -> HCORE" in surface
    manifest = tomllib.loads(stem.with_suffix(".system").read_text())
    assert manifest["run"]["initial_guess_requested"] == "AUTO"
    assert manifest["run"]["initial_guess"] == "HCORE"
    assert manifest["run"]["initial_guess_transport"] == "HCORE"
    for key in _bibtex_keys("vanlenthe_sad_2006", "lehtola_sap_2019"):
        assert key not in surface
