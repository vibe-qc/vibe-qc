"""``vibeqc.output.citations`` — DB loader, assembly, writers.

Pins the contract documented in
``docs/design_output_module.md § Citation database``:

  1. The bundled ``database.toml`` loads cleanly via
     :func:`load_default_database` — no missing entries referenced
     from any route.
  2. ``schema_version`` matches ``CitationDatabase.SCHEMA_VERSION``.
  3. Every functional / basis / dispersion model exercised in the test
     suite resolves through ``assemble(plan, ...)`` to at least one
     citation (no silent gaps in routing for the v0.8.0 surface).
  4. The vibe-qc software citation is always the first entry.
  5. ``write_bibtex`` and ``write_references`` produce non-empty files
     containing the assembled bibtex_keys / titles.
  6. Template substitution: ``{{VIBEQC_VERSION}}`` resolves to the
     running package version in the software citation.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from vibeqc.output import OutputPlan
from vibeqc.output.citations import (
    Citation,
    CitationDatabase,
    DatabaseError,
    assemble,
    citation_manifest_rows,
    format_bibtex,
    format_references,
    format_references_block,
    load_database,
    load_default_database,
    write_bibtex,
    write_references,
)
from vibeqc.semiempirical import SemiempiricalRoutePlan

# ---------------------------------------------------------------------------
# Database load + structural integrity
# ---------------------------------------------------------------------------


def test_default_database_loads_without_errors() -> None:
    db = load_default_database()
    assert isinstance(db, CitationDatabase)
    # The bundled database has at least the v0.8.0-on-main coverage.
    entries = db.entries()
    assert "vibeqc_software" in entries
    assert "libint_valeev" in entries
    assert "libxc_2018" in entries
    assert "luise_skala_2025" in entries
    assert "poschel_skala_cp2k_2026" in entries
    assert "sun_pyscf_2020" in entries
    assert "peintinger_pob_tzvp_2013" in entries
    assert "pbe_1996" in entries
    assert "becke_1993" in entries
    assert "pulay_diis_1980" in entries


def test_database_schema_version_matches_loader() -> None:
    db = load_default_database()
    # If schema_version drifts in the TOML, this test fails fast — the
    # loader's strict schema-version check enforces the contract.
    assert db.SCHEMA_VERSION == "1"


def test_ccm_long_range_reference_metadata_and_scope() -> None:
    """#444 corrects attribution without crediting inactive GFN2 operators."""
    db = load_default_database()
    entries = db.entries()
    reference = entries["janetzko_ccm_long_range_2002"]
    assert reference.doi == "10.1063/1.1473802"
    assert reference.pages == "8994-9004"
    assert reference.year == 2002
    assert "finite Evjen-weighted" in entries[
        "bredow_geudtner_jug_ccm_2001"
    ].notes
    # The pre-existing selector credits an actually resolved CCM Madelung
    # operator; this test does not claim that a rejected 3-D gamma run ran it.
    base = SemiempiricalRoutePlan.from_request("gfn2", boundary="seccm")
    active_plan = base.with_seccm_runtime(
        periodic_dimension=3, electrostatics_family="madelung"
    )
    active = db.assemble(**active_plan.citation_assemble_kwargs).citations
    assert "janetzko_ccm_long_range_2002" in {c.key for c in active}
    for dimension, family in ((3, "none"), (1, "ewald_gamma"), (2, "ewald_gamma")):
        plan = base.with_seccm_runtime(
            periodic_dimension=dimension, electrostatics_family=family
        )
        inactive = db.assemble(**plan.citation_assemble_kwargs).citations
        assert "janetzko_ccm_long_range_2002" not in {c.key for c in inactive}


def test_load_database_rejects_dangling_route_reference(
    tmp_path: Path,
) -> None:
    bad = tmp_path / "broken.toml"
    bad.write_text(
        'schema_version = "1"\n'
        "[entries.foo]\n"
        'kind = "article"\n'
        'bibtex_key = "foo_2020"\n'
        'authors = ["A, B"]\n'
        'title = "T"\n'
        "\n"
        "[routes.basis_sets]\n"
        '"bar" = ["nonexistent_entry"]\n',
        encoding="utf-8",
    )
    with pytest.raises(DatabaseError, match="missing entry"):
        load_database(bad)


# ---------------------------------------------------------------------------
# Assembly: software always first, expected entries fire
# ---------------------------------------------------------------------------


def _plan(
    tmp_path: Path, *, basis: str, functional: str | None = None, method: str = "rks"
) -> OutputPlan:
    return OutputPlan.from_run_job_kwargs(
        output=tmp_path / "job",
        method=method,
        basis=basis,
        functional=functional,
    )


def test_software_citation_is_first(tmp_path: Path) -> None:
    result = assemble(_plan(tmp_path, basis="sto-3g"))
    assert len(result) > 0
    assert result.citations[0].key == "vibeqc_software"


@pytest.mark.parametrize("numerics", ["bipole_erfc_panel", "bipole_erfc_bloch", "bipole_finite_panel", "bipole_finite_jk"])
def test_native_erfc_numerics_have_range_separation_sources(tmp_path: Path, numerics: str) -> None:
    result = assemble(_plan(tmp_path, basis="sto-3g", method="rhf"), numerics=[numerics])
    by_key = {c.key: c for c in result.citations}
    for key, doi in {
        "sun_range_separated_exchange_2023": "10.1063/5.0155815",
        "mcmurchie_davidson_1978": "10.1016/0021-9991(78)90092-X",
    }.items():
        assert by_key[key].doi == doi
        assert by_key[key].print is True
    assert by_key["libint_valeev"].print is True


@pytest.mark.parametrize("numerics", ["bipole_ewald_gram", "bipole_finite_panel", "bipole_finite_jk"])
def test_native_ewald_gram_has_its_fourier_and_finite_k_sources(tmp_path: Path, numerics: str) -> None:
    result = assemble(_plan(tmp_path, basis="sto-3g", method="rhf"),
                      numerics=[numerics])
    by_key = {c.key: c for c in result.citations}
    for key, doi in {
        "sun_berkelbach_gdf_2017": "10.1063/1.4998644",
        "mcmurchie_davidson_1978": "10.1016/0021-9991(78)90092-X",
        "sundararaman_arias_exx_2013": "10.1103/PhysRevB.87.165122",
    }.items():
        assert by_key[key].doi == doi
        assert by_key[key].print is True


@pytest.mark.parametrize("numerics", ["periodic_ao_bloch_transport", "periodic_orbital_sewing"])
def test_native_ao_bloch_transport_has_its_symmetry_source(tmp_path: Path, numerics: str) -> None:
    plan = _plan(tmp_path, basis="sto-3g", method="rhf")
    selected = assemble(plan, numerics=[numerics])
    citations = {c.key: c for c in selected.citations}
    citation = citations["dovesi_symmetry_lcao_1986"]
    assert citation.doi == "10.1002/qua.560290608"
    assert citation.print is True
    construction = citations["zicovich_wilson_crystalline_orbitals_1998"]
    assert construction.doi == "10.1002/(SICI)1097-461X(1998)67:5<299::AID-QUA3>3.0.CO;2-Q"
    assert construction.print is True
    if numerics == "periodic_orbital_sewing":
        mixing = citations["casassa_symmetry_wannier_2006"]
        assert mixing.doi == "10.1007/s00214-006-0119-z"
        assert mixing.print is True


def test_shared_symmetry_sources_are_routed(tmp_path: Path) -> None:
    selected = assemble(_plan(tmp_path, basis="sto-3g", method="rhf"), numerics=["shared_symmetry"])
    citations = {c.key: c for c in selected.citations}
    for key, doi in {
        "dovesi_symmetry_lcao_1986": "10.1002/qua.560290608",
        "casassa_symmetry_wannier_2006": "10.1007/s00214-006-0119-z",
    }.items():
        assert citations[key].doi == doi
        assert citations[key].print is True


def test_ccsd_t_route_pins_canonical_cc_papers(tmp_path: Path) -> None:
    """method='ccsd(t)' fires the canonical CC + (T) + DF papers, DOIs pinned.

    The route is keyed on the method name, so closed-shell RCCSD(T)
    (cpp/src/ccsd.cpp) and open-shell UCCSD(T) (cpp/src/uccsd.cpp) share it.
    DOIs pinned mechanically per CLAUDE.md section 8 (the pob-* audit lesson).
    """
    result = assemble(_plan(tmp_path, basis="cc-pvdz", method="ccsd(t)"))
    by_key = {c.key: c for c in result.citations}
    expected = {
        "purvis_bartlett_ccsd_1982": "10.1063/1.443164",
        "stanton_gauss_watts_bartlett_ccsd_1991": "10.1063/1.460620",
        "raghavachari_ccsdt_1989": "10.1016/S0009-2614(89)87395-6",
        "deprince_sherrill_df_ccsd_2013": "10.1021/ct400250u",
    }
    for key, doi in expected.items():
        assert key in by_key, f"{key} did not fire for ccsd(t)"
        assert by_key[key].doi == doi
        assert by_key[key].print is True  # reaches .out / .bibtex


def test_bounded_df_ccsdt_algorithm_citation_is_runtime_gated(
    tmp_path: Path,
) -> None:
    plan = _plan(tmp_path, basis="cc-pvdz", method="ccsd(t)")
    ordinary = {c.key for c in assemble(plan).citations}
    assert "gyevi_nagy_direct_ccsdt_2020" not in ordinary

    result = assemble(plan, extra_entries=["gyevi_nagy_direct_ccsdt_2020"])
    by_key = {c.key: c for c in result.citations}
    citation = by_key["gyevi_nagy_direct_ccsdt_2020"]
    assert citation.doi == "10.1021/acs.jctc.9b00957"
    assert citation.print is True


def test_native_periodic_diabatic_starting_gauge_has_its_specific_source(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="sto-3g", method="rhf")
    key = "zhu_tew_bloch_iao_wannier_2024"
    for properties in ([], ["wannier"], ["ibo"]):
        assert key not in {c.key for c in assemble(plan, properties=properties).citations}
    selected = assemble(plan, properties=["periodic_diabatic_wannier"])
    citation = {c.key: c for c in selected.citations}[key]
    assert citation.doi == "10.1021/acs.jpca.4c04555"
    assert citation.authors == ("Zhu, Andrew", "Tew, David P.")
    assert citation.print is True


def test_native_finite_gaussian_rhf_has_specific_integral_sources() -> None:
    db = load_default_database()
    required = {
        "obara_saika_integrals_1986": "10.1063/1.450106",
        "ahlrichs_general_os_2006": "10.1039/b605188j",
    }
    ordinary = {c.key for c in db.assemble(method="rhf", basis="sto-3g", periodic=True).citations}
    assert not required.keys() & ordinary
    selected = db.assemble(method="periodic-finite-gaussian-rhf", basis="sto-3g", periodic=True)
    by_key = {c.key: c for c in selected.citations}
    assert {"sun_berkelbach_gdf_2017", "mcmurchie_davidson_1978",
            "ewald_lattice_sum_1921", "pulay_diis_1982"} <= by_key.keys()
    for key, doi in required.items():
        assert by_key[key].doi == doi
        assert by_key[key].print is True


def test_native_finite_gaussian_pair_mp2_cites_source_and_periodic_pair_equations() -> None:
    db = load_default_database()
    rhf = {c.key for c in db.assemble(method="periodic-finite-gaussian-rhf", periodic=True).citations}
    selected = db.assemble(method="periodic-finite-gaussian-pno-mp2", periodic=True)
    by_key = {c.key: c for c in selected.citations}
    assert rhf <= by_key.keys()
    assert {"zhu_tew_bloch_iao_wannier_2024", "nejad_periodic_dlpno_pbc_2025", "riplinger_dlpno_2013"} <= by_key.keys()
    assert by_key["nejad_periodic_dlpno_pbc_2025"].doi == "10.1063/5.0290816"
    assert not {"liakos_dlpno_thresholds_2015", "guo_dlpno_t1_2018"} & by_key.keys()


def test_native_finite_gaussian_selected_space_ccsdt_cites_actual_chain() -> None:
    db = load_default_database()
    rhf = {c.key for c in db.assemble(method="periodic-finite-gaussian-rhf", periodic=True).citations}
    selected = db.assemble(method="periodic-finite-gaussian-ccsd(t)", periodic=True)
    by_key = {c.key: c for c in selected.citations}
    assert rhf <= by_key.keys()
    assert {"zhu_tew_bloch_iao_wannier_2024", "nejad_periodic_dlpno_pbc_2025",
            "riplinger_dlpno_2013", "purvis_bartlett_ccsd_1982",
            "stanton_gauss_watts_bartlett_ccsd_1991", "raghavachari_ccsdt_1989",
            "riplinger_triples_2013", "guo_dlpno_t1_2018"} <= by_key.keys()
    assert by_key["guo_dlpno_t1_2018"].doi == "10.1063/1.5011798"
    assert "liakos_dlpno_thresholds_2015" not in by_key


def test_native_finite_gaussian_pair_ccsd_cites_projected_ccsd_without_triples() -> None:
    db = load_default_database()
    mp2 = {c.key for c in db.assemble(method="periodic-finite-gaussian-pno-mp2", periodic=True).citations}
    selected = db.assemble(method="periodic-finite-gaussian-pno-ccsd", periodic=True)
    by_key = {c.key: c for c in selected.citations}
    assert mp2 <= by_key.keys()
    assert by_key["purvis_bartlett_ccsd_1982"].doi == "10.1063/1.443164"
    assert by_key["stanton_gauss_watts_bartlett_ccsd_1991"].doi == "10.1063/1.460620"
    assert not {"raghavachari_ccsdt_1989", "riplinger_triples_2013",
                "guo_dlpno_t1_2018", "liakos_dlpno_thresholds_2015"} & by_key.keys()


def test_native_finite_gaussian_pair_triples_cites_local_moments_and_occupied_couplings() -> None:
    db = load_default_database()
    ccsd = {c.key for c in db.assemble(method="periodic-finite-gaussian-pno-ccsd", periodic=True).citations}
    by_key = {c.key: c for c in db.assemble(method="periodic-finite-gaussian-pno-ccsd(t)", periodic=True).citations}
    assert ccsd <= by_key.keys()
    assert by_key["riplinger_triples_2013"].doi == "10.1063/1.4821834"
    assert by_key["guo_dlpno_t1_2018"].doi == "10.1063/1.5011798"
    assert "raghavachari_ccsdt_1989" in by_key
    assert "liakos_dlpno_thresholds_2015" not in by_key


def test_accsdt_route_pins_lambda_triples_papers(tmp_path: Path) -> None:
    """triples='A-CCSD(T)' routes the Lambda/asymmetric triples papers."""
    result = assemble(_plan(tmp_path, basis="cc-pvdz", method="a-ccsd(t)"))
    by_key = {c.key: c for c in result.citations}
    expected = {
        "purvis_bartlett_ccsd_1982": "10.1063/1.443164",
        "stanton_gauss_watts_bartlett_ccsd_1991": "10.1063/1.460620",
        "kucharski_bartlett_accsdt_1998": "10.1063/1.475961",
        "crawford_stanton_accsdt_1998": (
            "10.1002/(SICI)1097-461X(1998)70:4/5<601::AID-QUA6>3.0.CO;2-Z"
        ),
        "deprince_sherrill_df_ccsd_2013": "10.1021/ct400250u",
    }
    for key, doi in expected.items():
        assert key in by_key, f"{key} did not fire for a-ccsd(t)"
        assert by_key[key].doi == doi
        assert by_key[key].print is True
    assert "raghavachari_ccsdt_1989" not in by_key


def test_plain_ccsd_route_omits_triples_paper(tmp_path: Path) -> None:
    """Plain CCSD must not cite the Raghavachari (T) paper."""
    keys = {
        c.key
        for c in assemble(_plan(tmp_path, basis="cc-pvdz", method="ccsd")).citations
    }
    assert "purvis_bartlett_ccsd_1982" in keys
    assert "raghavachari_ccsdt_1989" not in keys


@pytest.mark.parametrize(
    "method",
    (
        "dlpno-mp2",
        "dlpno-ump2",
        "dlpno-ccsd",
        "dlpno-ccsd(t)",
        "dlpno-uccsd",
        "dlpno-uccsd(t)",
    ),
)
def test_every_dlpno_route_cites_the_threshold_convention(
    tmp_path: Path,
    method: str,
) -> None:
    """All six DLPNO routes expose the Liakos preset source (#448)."""
    result = assemble(_plan(tmp_path, basis="def2-svp", method=method))
    by_key = {citation.key: citation for citation in result.citations}

    citation = by_key["liakos_dlpno_thresholds_2015"]
    assert citation.doi == "10.1021/ct501129s"
    assert citation.print is True
    assert citation in result.printable


@pytest.mark.parametrize("method", ["dlpno-mp2", "dlpno-ump2"])
def test_dlpno_mp2_cites_its_same_name_preset_collision(
    tmp_path: Path, method: str
) -> None:
    """The artifact exposes Pinski's distinct MP2 threshold names (#448)."""

    result = assemble(_plan(tmp_path, basis="def2-svp", method=method))
    by_key = {citation.key: citation for citation in result.citations}

    citation = by_key["pinski_dlpno_mp2_thresholds_2019"]
    assert citation.doi == "10.1063/1.5086544"
    assert citation.print is True
    assert citation in result.printable


def test_saitow_open_shell_cc_background_does_not_fire_for_ump2(
    tmp_path: Path,
) -> None:
    """Independent-spin UMP2 must not claim Saitow's open-shell CC method."""

    ump2 = assemble(_plan(tmp_path, basis="def2-svp", method="dlpno-ump2"))
    uccsd = assemble(_plan(tmp_path, basis="def2-svp", method="dlpno-uccsd"))

    assert "saitow_openshell_dlpno_2017" not in {c.key for c in ump2.citations}
    assert "saitow_openshell_dlpno_2017" in {c.key for c in uccsd.citations}


def test_published_frozen_core_manual_entry_is_runtime_gated() -> None:
    """The official table source is printable but never a static method cite."""
    key = "orca_manual_frozen_core_6_1_1"
    db = load_default_database()

    ordinary = db.assemble(method="mp2", basis="def2-svp")
    assert key not in {citation.key for citation in ordinary.citations}

    routed = db.assemble(
        method="mp2",
        basis="def2-svp",
        extra_entries=(key,),
    )
    citation = {item.key: item for item in routed.citations}[key]
    assert citation.kind == "misc"
    assert citation.authors == ("Max-Planck-Institut für Kohlenforschung",)
    assert citation.version == "6.1.1"
    assert citation.publisher == "FACCTs GmbH"
    assert citation.url == (
        "https://www.faccts.de/docs/orca/6.1/manual/contents/"
        "essentialelements/frozencore.html"
    )
    assert citation.print is True
    assert citation in routed.printable


def test_frozen_core_source_gate_covers_every_correlated_option_route() -> None:
    """Defaults cite Table 2.69; explicit counts and all-electron do not."""
    from vibeqc import (
        CC3Options,
        CCSDOptions,
        CCSDTOptions,
        MP2Options,
        UMP2Options,
    )
    from vibeqc.dlpno.ccsd import DLPNOCCSDPilotOptions
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions
    from vibeqc.dlpno.mp2 import DLPNOMP2Options
    from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions
    from vibeqc.dlpno.uccsd_local_solver import LocalUCCSDOptions
    from vibeqc.dlpno.ump2 import DLPNOUMP2Options
    from vibeqc.runner import _frozen_core_citation_entries

    key = "orca_manual_frozen_core_6_1_1"
    routes = (
        ("mp2", MP2Options, "n_frozen_core"),
        ("scs-mp2", MP2Options, "n_frozen_core"),
        ("sos-mp2", UMP2Options, "n_frozen_core"),
        ("ccsd", CCSDOptions, "n_frozen_core"),
        ("ccsd(t)", CCSDOptions, "n_frozen_core"),
        ("cc3", CC3Options, "n_frozen_core"),
        ("ccsdt", CCSDTOptions, "n_frozen_core"),
        ("dlpno-mp2", DLPNOMP2Options, "n_frozen"),
        ("dlpno-mp2", DLPNOUMP2Options, "n_frozen"),
        ("dlpno-ccsd", DLPNOCCSDPilotOptions, "n_frozen"),
        ("dlpno-ccsd(t)", LocalCCSDOptions, "n_frozen"),
        ("dlpno-ccsd", DLPNOUCCSDPilotOptions, "n_frozen"),
        ("dlpno-ccsd(t)", LocalUCCSDOptions, "n_frozen"),
    )
    for method, factory, frozen_field in routes:
        assert _frozen_core_citation_entries(
            method, None, factory()
        ) == (key,), (method, factory.__name__)

        explicit = factory()
        setattr(explicit, frozen_field, 0)
        assert _frozen_core_citation_entries(
            method, None, explicit
        ) == (), (method, factory.__name__)

    for selector in (True, "published", "chemical-core"):
        assert _frozen_core_citation_entries("mp2", selector, None) == (key,)
    for selector in (False, 0, 2, "all-electron"):
        assert _frozen_core_citation_entries("mp2", selector, None) == ()
    for method in ("cc3", "ccsdt"):
        # run_job uses this explicit zero when active_space owns the
        # Hamiltonian partition, even if no route-options object was passed.
        assert _frozen_core_citation_entries(method, 0, None) == ()
    assert _frozen_core_citation_entries("rhf", None, None) == ()


def test_dlpno_t1_source_is_runtime_gated() -> None:
    """Only an executed iterative local-basis UCCSD (T1) adds Guo."""
    from types import SimpleNamespace

    from vibeqc.dlpno.ccsd import DLPNOCCSDPilotOptions
    from vibeqc.dlpno.ccsd_local_solver import LocalCCSDOptions
    from vibeqc.dlpno.uccsd import DLPNOUCCSDPilotOptions
    from vibeqc.dlpno.uccsd_local_solver import LocalUCCSDOptions
    from vibeqc.runner import _dlpno_triples_citation_entries

    keys = (
        "guo_dlpno_t1_2018",
        "guo_openshell_dlpno_triples_2020",
    )
    options = LocalUCCSDOptions(
        compute_triples=True,
        triples_mode="t1-iterative",
    )
    assert _dlpno_triples_citation_entries(
        "dlpno-ccsd(t)",
        options,
        SimpleNamespace(triples_executed=True),
    ) == keys
    assert _dlpno_triples_citation_entries(
        "dlpno-ccsd(t)",
        options,
        SimpleNamespace(triples_executed=False),
    ) == ()
    assert _dlpno_triples_citation_entries("dlpno-ccsd(t)", options) == ()
    for options in (
        LocalCCSDOptions(compute_triples=True, triples_mode="local"),
        LocalCCSDOptions(compute_triples=True, triples_mode="t1"),
        LocalCCSDOptions(compute_triples=True, triples_mode="exact"),
        LocalUCCSDOptions(compute_triples=True),
        LocalUCCSDOptions(compute_triples=True, triples_mode="t1"),
    ):
        assert _dlpno_triples_citation_entries(
            "dlpno-ccsd(t)",
            options,
            SimpleNamespace(triples_executed=True),
        ) == ()
    for options in (
        DLPNOCCSDPilotOptions(compute_triples=True),
        DLPNOUCCSDPilotOptions(compute_triples=True),
    ):
        assert _dlpno_triples_citation_entries(
            "dlpno-ccsd(t)", options
        ) == ()
    assert _dlpno_triples_citation_entries(
        "dlpno-ccsd",
        LocalUCCSDOptions(compute_triples=True, triples_mode="t1-iterative"),
        SimpleNamespace(triples_executed=True),
    ) == ()

    database = load_default_database()
    for route in ("dlpno-ccsd(t)", "dlpno-uccsd(t)"):
        static_keys = {
            citation.key
            for citation in database.assemble(method=route).citations
        }
        assert set(keys).isdisjoint(static_keys)
    dynamic_keys = {
        citation.key
        for citation in database.assemble(
            method="dlpno-uccsd(t)",
            extra_entries=keys,
        ).citations
    }
    assert set(keys) <= dynamic_keys


@pytest.mark.parametrize(
    "wrapper_name",
    ("run_scs_mp2", "run_sos_mp2", "run_scs_ump2", "run_sos_ump2"),
)
@pytest.mark.parametrize(
    ("selector", "expected"),
    (
        (None, ("orca_manual_frozen_core_6_1_1",)),
        (False, ()),
        (2, ()),
    ),
)
def test_scaled_mp2_convenience_citations_follow_frozen_core_selector(
    tmp_path: Path,
    monkeypatch,
    wrapper_name: str,
    selector: object,
    expected: tuple[str, ...],
) -> None:
    """Standalone SCS/SOS RMP2 and UMP2 emit the same dynamic source."""
    from types import SimpleNamespace

    import vibeqc as vq
    import vibeqc.output.citations as citation_api

    calls: list[dict[str, object]] = []
    sentinel = object()

    monkeypatch.setattr(vq, "run_mp2", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(vq, "run_ump2", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        citation_api,
        "emit_citations",
        lambda *args, **kwargs: calls.append(dict(kwargs)),
    )

    molecule = SimpleNamespace(atoms=(SimpleNamespace(Z=8),))
    basis = SimpleNamespace(name="def2-svp")
    result = getattr(vq, wrapper_name)(
        molecule,
        basis,
        object(),
        density_fit=False,
        frozen_core=selector,
        output=tmp_path / wrapper_name,
    )

    assert result is sentinel
    assert len(calls) == 1
    assert calls[0]["extra_entries"] == expected


def test_canonical_non_df_ccsd_omits_df_assembly_paper(tmp_path: Path) -> None:
    """cc_density_fit=False drops DePrince-Sherrill, keeps the CC papers.

    A canonical (CCSDOptions(density_fit=False)) run assembles exact
    four-index integrals, so citing the DF-CCSD assembly paper would claim
    an algorithm the run did not use (CLAUDE.md section 8: attribute,
    never claim). The defining CC + (T) papers still fire.
    """
    keys = {
        c.key
        for c in assemble(
            _plan(tmp_path, basis="cc-pvdz", method="ccsd(t)"),
            cc_density_fit=False,
        ).citations
    }
    assert "deprince_sherrill_df_ccsd_2013" not in keys
    assert "purvis_bartlett_ccsd_1982" in keys
    assert "stanton_gauss_watts_bartlett_ccsd_1991" in keys
    assert "raghavachari_ccsdt_1989" in keys


def test_libint_always_fires_after_software(tmp_path: Path) -> None:
    result = assemble(_plan(tmp_path, basis="sto-3g"))
    keys = [c.key for c in result.citations]
    assert "libint_valeev" in keys
    assert keys.index("libint_valeev") == 1


def test_pbe_run_pulls_libxc_and_pbe(tmp_path: Path) -> None:
    result = assemble(_plan(tmp_path, basis="sto-3g", functional="PBE"))
    keys = {c.key for c in result.citations}
    assert "libxc_2018" in keys
    assert "pbe_1996" in keys
    assert result.warnings == ()


@pytest.mark.parametrize("spelling", ["skala-1.1", "skala-1.1-rev1", "skala"])
def test_skala_cites_functional_and_host_interface_but_not_libxc(
    tmp_path: Path, spelling: str
) -> None:
    """External SKALA uses vibe-qc's grid and AO projector, but not libxc."""
    result = assemble(_plan(tmp_path, basis="sto-3g", functional=spelling))
    by_key = {citation.key: citation for citation in result.citations}
    keys = set(by_key)
    assert "luise_skala_2025" in keys
    assert by_key["luise_skala_2025"].doi == "10.48550/arXiv.2506.14665"
    assert by_key["luise_skala_2025"].print is True
    assert "poschel_skala_cp2k_2026" in keys
    assert (
        by_key["poschel_skala_cp2k_2026"].doi
        == "10.48550/arXiv.2608.19033"
    )
    assert by_key["poschel_skala_cp2k_2026"].print is True
    assert "sun_pyscf_2020" in keys
    assert by_key["sun_pyscf_2020"].doi == "10.1063/5.0006074"
    assert "libxc_2018" not in keys
    assert {"becke_grid_1988", "treutler_ahlrichs_1995"} <= keys
    assert result.warnings == ()


def test_pob_tzvp_rev2_pulls_both_basis_papers(tmp_path: Path) -> None:
    result = assemble(_plan(tmp_path, basis="pob-tzvp-rev2"))
    keys = {c.key for c in result.citations}
    assert "peintinger_pob_tzvp_2013" in keys
    assert "vilela_oliveira_pob_rev2_2019" in keys


def test_pm6_pm7_routes_carry_the_mopac_parameter_source() -> None:
    """Every PM6/PM7 route surfaces MOPAC, the source of the values (#440).

    ``docs/license.md`` used to declare the inline PM6 H/C/N/O/F values as
    *transcribed from Stewart 2007*. They are not: all 118 of them (88
    element + 30 diatomic) reproduce MOPAC's Apache-2.0
    ``parameters_for_PM6_C.F90`` exactly at its six published decimals, and
    none of them occurs anywhere in Stewart 2007. The same holds for all 95
    inline PM7 values (65 element + 30 diatomic) against
    ``parameters_for_PM7_C.F90`` since the 2026-09-06 pair-table correction
    (the pre-fix pair table carried 28 values of unrecorded origin). Apache-2.0 § 4
    attribution therefore has to reach the user, which means a live route --
    not merely a database entry (CLAUDE.md § 8: a citation without a route is
    silently dead weight).

    Pinned mechanically because that is exactly what the pob-* audit thread
    showed is needed: a provenance claim that only a human reads is a claim
    that silently rots.
    """
    db = load_default_database()
    entry = db.entries()["mopac_moussa_stewart_2026"]
    assert entry.doi == "10.21105/joss.08025"

    for method in ("pm6", "upm6", "pm7", "upm7", "pm6_seccm"):
        keys = {c.key for c in db.assemble(method=method).citations}
        assert "mopac_moussa_stewart_2026" in keys, method
        # The method paper stays: it says what the numbers mean. MOPAC says
        # where this copy of them came from. Both, not either.
        assert {"stewart_pm6_2007", "stewart_pm7_2013"} & keys, method

    # Negative control -- the same assembly path with the feature off. These
    # semiempirical methods ship parameters that are NOT MOPAC-derived (OMx is
    # transcribed from Dral 2016 Tables 1-3; MSINDO ships by permission of the
    # copyright holder; GFN2 is fetched from its own upstream), so the MOPAC
    # attribution must not leak onto them.
    for method in ("om1", "om2", "om3", "msindo", "gfn2_xtb", "dftb0"):
        keys = {c.key for c in db.assemble(method=method).citations}
        assert "mopac_moussa_stewart_2026" not in keys, method


def test_omx_routes_cite_the_per_table_sources_and_the_license_doc_agrees() -> None:
    """Each bundled OMx set cites its own table source, never MOPAC (#272).

    Maintainer ruling 2026-08-28: the OMx parameter metadata must not carry
    the PM6 MOPAC license string; the values come from paper tables, so cite
    the papers. The bundled sets are transcribed from Dral et al. 2016
    (Table 1 = OM1, originally Kolb & Thiel 1993; Table 2 = OM2, its first
    full publication, with Weber & Thiel 2000 as the method paper; Table 3 =
    OM3, originally Scholten 2003). This pins the three legs that have to
    agree -- the live citation routes, the runtime ``ParameterSetMetadata``,
    and ``docs/license.md`` -- so a future edit to any one of them cannot
    silently re-open the split.

    Independent verification of the first fix (2026-09-04) found the docs
    claiming "originally Weber & Thiel 2000, Table 2": that table is an
    orthogonalization-energy table for H3-, not a parameter table. Prose
    comparing repository strings to repository strings cannot catch that, so
    the database now carries a structured ``role`` on the OMx entries
    (maintainer wording 2026-09-06): exactly one ``parameter-source`` per
    route, and it is Dral 2016; the formalism papers are ``method``.
    """
    db = load_default_database()
    entries = db.entries()
    dral = entries["dral_omx_2016"]
    assert dral.doi == "10.1021/acs.jctc.5b01046"
    assert entries["kolb_thiel_om1_1993"].doi == "10.1002/jcc.540140704"
    assert entries["weber_thiel_om2_2000"].doi == "10.1007/s002149900083"
    assert entries["scholten_om3_2003"].kind == "phdthesis"

    per_table_source = {
        "om1": "kolb_thiel_om1_1993",
        "om2": "weber_thiel_om2_2000",
        "om3": "scholten_om3_2003",
    }
    # Structured provenance roles, not prose: the entry the shipped values
    # were transcribed from is the one and only parameter source on every
    # OMx route, and the original method papers are tagged as such.
    assert dral.role == "parameter-source"
    for key in per_table_source.values():
        assert entries[key].role == "method", key

    for method, original in per_table_source.items():
        assembled = db.assemble(method=method)
        keys = {c.key for c in assembled.citations}
        assert "dral_omx_2016" in keys, method
        assert original in keys, method
        assert "mopac_moussa_stewart_2026" not in keys, method
        sources = {c.key for c in assembled.citations if c.role == "parameter-source"}
        assert sources == {"dral_omx_2016"}, (method, sources)
        methods = {c.key for c in assembled.citations if c.role == "method"}
        assert original in methods, (method, methods)
        # Both reach the user-facing references block, not only .system.
        printable = {c.key for c in assembled.printable}
        assert {"dral_omx_2016", original} <= printable, method

    # Runtime metadata agrees with the routed source and inherits nothing
    # from the MOPAC-derived PM6 sets.
    from vibeqc.semiempirical.methods.omx_params import (
        load_om1_params,
        load_om2_params,
        load_om3_params,
    )

    for loader in (load_om1_params, load_om2_params, load_om3_params):
        meta = loader().metadata()
        assert meta.origin == "published"
        assert meta.doi_or_url == dral.doi
        assert meta.license == "published parameter table"
        assert "mopac" not in meta.license.lower()
        assert "apache" not in meta.license.lower()

    # The licensing inventory records the per-table source of every set.
    license_doc = (
        Path(__file__).resolve().parents[1] / "docs" / "license.md"
    ).read_text(encoding="utf-8")
    words = " ".join(license_doc.split())
    assert "OM1 values in `omx_params.py`" in words
    assert "OM2 values in `omx_params.py`" in words
    assert "OM3 values in `omx_params.py`" in words
    assert "Dral et al. 2016, Table 1" in words
    assert "Dral et al. 2016, Table 2" in words
    assert "Dral et al. 2016, Table 3" in words
    assert "10.1002/jcc.540140704" in words
    assert "10.1007/s002149900083" in words
    assert "inherits nothing from the PM6 sets" in words
    # The false attribution the 2026-09-04 verification found must not return:
    # Weber & Thiel 2000 is the OM2 method paper, its Table 2 is not the
    # parameter table the values came from.
    assert "Weber & Thiel 2000, Table 2" not in words
    assert "Weber & Thiel 2000 (method" in words
    # The retired generic row must not come back.
    assert "OM1/OM2/OM3 values transcribed from Dral et al. 2016 Tables 1-3 |" not in words


def test_mopac_attribution_is_user_visible_not_manifest_only() -> None:
    """The MOPAC entry reaches the printable references, not just .system.

    Apache-2.0 § 4(c)/(d) attribution that lands only in the internal
    provenance manifest has not been surfaced to anyone. ``print`` therefore
    has to be true (its default) for this entry -- pinned so a later
    ``print = false`` cannot quietly bury it (#440).
    """
    db = load_default_database()
    result = db.assemble(method="pm6")
    assert "mopac_moussa_stewart_2026" in {c.key for c in result.printable}


def test_b3lyp_pulls_becke_lyp_and_stephens(tmp_path: Path) -> None:
    result = assemble(_plan(tmp_path, basis="6-31g*", functional="B3LYP"))
    keys = {c.key for c in result.citations}
    assert "becke_1993" in keys
    assert "lee_yang_parr_1988" in keys
    assert "stephens_b3lyp_1994" in keys
    assert "libxc_2018" in keys
    # The flavor-disambiguation paper (which VWN parametrisation fills
    # the B3 correlation slot) rides along since the v0.12.0 flavor
    # audit (bare b3lyp = VWN5 / ORCA definition, unchanged).
    assert "hertwig_koch_1997" in keys


@pytest.mark.parametrize("spelling", ["b3lyp5", "b3lypg", "b3lyp/g"])
def test_b3lyp_flavor_aliases_cite_the_same_papers(
    tmp_path: Path, spelling: str
) -> None:
    """b3lyp5 (VWN5 / ORCA-TURBOMOLE-CRYSTAL flavor) and the b3lypg /
    b3lyp/g spellings of the default cite identically to bare b3lyp."""
    base = assemble(_plan(tmp_path, basis="6-31g*", functional="B3LYP"))
    alias = assemble(_plan(tmp_path, basis="6-31g*", functional=spelling))
    assert {c.key for c in alias.citations} == {c.key for c in base.citations}


def test_diis_default_fires(tmp_path: Path) -> None:
    result = assemble(_plan(tmp_path, basis="sto-3g"))
    keys = {c.key for c in result.citations}
    assert "pulay_diis_1980" in keys
    assert "pulay_diis_1982" in keys


def test_unknown_basis_emits_warning_but_does_not_raise(
    tmp_path: Path,
) -> None:
    result = assemble(_plan(tmp_path, basis="totally-made-up-basis"))
    assert any("totally-made-up-basis" in w for w in result.warnings)
    # The software + libint + DIIS routes still fire.
    keys = {c.key for c in result.citations}
    assert "vibeqc_software" in keys


def test_unknown_functional_emits_warning_but_does_not_raise(
    tmp_path: Path,
) -> None:
    result = assemble(_plan(tmp_path, basis="sto-3g", functional="MADE-UP-XC"))
    assert any("MADE-UP-XC" in w for w in result.warnings)


def test_dispersion_d3bj_pulls_both_grimme_papers(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="6-31g*", functional="PBE")
    db = load_default_database()
    result = db.assemble_from_plan(plan, dispersion="d3bj")
    keys = {c.key for c in result.citations}
    assert "grimme_d3_2010" in keys
    assert "grimme_d3bj_2011" in keys


def test_dispersion_d4_pulls_caldeweyher(tmp_path: Path) -> None:
    """d4 fires the full three-paper set upstream dftd4 asks users to
    cite — 2017 precursor + 2019 method + 2020 periodic extension
    (maintainer decision 2026-06-11; see the routes.methods comment)."""
    plan = _plan(tmp_path, basis="6-31g*", functional="PBE")
    db = load_default_database()
    result = db.assemble_from_plan(plan, dispersion="d4")
    keys = {c.key for c in result.citations}
    assert "caldeweyher_d4_2017" in keys
    assert "caldeweyher_d4_2019" in keys
    assert "caldeweyher_d4_2020" in keys


def test_dispersion_d4_param_fit_paper_fires(tmp_path: Path) -> None:
    """A parametrization fit outside the 2019 D4 method paper pulls its
    fit paper via routes.dispersion_params, alongside the method paper."""
    plan = _plan(tmp_path, basis="6-31g*", functional="r2scan")
    db = load_default_database()
    result = db.assemble_from_plan(plan, dispersion="d4", dispersion_params="r2scan")
    keys = {c.key for c in result.citations}
    assert "caldeweyher_d4_2019" in keys
    assert "ehlert_r2scan_d4_2021" in keys


def test_dispersion_d4_method_paper_params_add_nothing(tmp_path: Path) -> None:
    """A parametrization from the 2019 method paper itself (pbe) has no
    dispersion_params row — only the method paper fires, silently."""
    plan = _plan(tmp_path, basis="6-31g*", functional="PBE")
    db = load_default_database()
    result = db.assemble_from_plan(plan, dispersion="d4", dispersion_params="pbe")
    keys = {c.key for c in result.citations}
    assert "caldeweyher_d4_2019" in keys
    fit_keys = {
        entry
        for row in db.routes().get("dispersion_params", {}).values()
        for entry in row
    }
    assert not (keys & fit_keys), (
        "method-paper parametrization must not pull a separate fit paper"
    )
    assert not result.warnings or all(
        "dispersion_params" not in w for w in result.warnings
    )


def test_dispersion_params_ignored_without_dispersion(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="6-31g*", functional="r2scan")
    db = load_default_database()
    result = db.assemble_from_plan(plan, dispersion_params="r2scan")
    keys = {c.key for c in result.citations}
    assert "ehlert_r2scan_d4_2021" not in keys


class TestD4ParameterDoiCoverage:
    """Mechanical pin of the inline ``doi=`` provenance in
    ``vibeqc.dispersion_d4_parameters`` against the citation database
    (CLAUDE.md § 8): every DOI a D4 parameter row claims must exist as
    a database entry, and every parametrization fit OUTSIDE the 2019
    D4 method paper must have a ``routes.dispersion_params`` row that
    fires that DOI's entry. Guards against the pob-* failure mode —
    provenance strings that survive review with no mechanical test
    pinning them to the user-facing citation surface."""

    # The 2019 D4 method paper — parametrizations fit there are covered
    # by the methods.d4 route and need no dispersion_params row.
    METHOD_PAPER_DOI = "10.1063/1.5090222"

    @staticmethod
    def _param_rows() -> dict[str, str]:
        """name → doi for every D4 parameter row that records one."""
        from vibeqc.dispersion_d4_parameters import (
            get_d4_params,
            list_d4_functionals,
        )

        return {
            name: get_d4_params(name).doi
            for name in list_d4_functionals()
            if get_d4_params(name).doi
        }

    def test_every_d4_param_doi_has_a_database_entry(self) -> None:
        db = load_default_database()
        known_dois = {e.doi for e in db.entries().values() if e.doi}
        missing = {
            name: doi
            for name, doi in self._param_rows().items()
            if doi not in known_dois
        }
        assert not missing, (
            f"D4 parameter rows cite DOIs with no [entries.*] block in "
            f"database.toml: {missing} — add the entry (and a "
            f"routes.dispersion_params row) in the same merge, per "
            f"CLAUDE.md § 8."
        )

    def test_every_external_fit_doi_is_routed(self) -> None:
        from vibeqc.dispersion_d4_parameters import normalize_d4_key

        db = load_default_database()
        table = db.routes().get("dispersion_params", {})
        entries = db.entries()
        for name, doi in self._param_rows().items():
            if doi == self.METHOD_PAPER_DOI:
                continue
            route_key = f"d4:{normalize_d4_key(name)}"
            row = table.get(route_key)
            assert row, (
                f"D4 parametrization {name!r} was fit in {doi} (not the "
                f"2019 method paper) but routes.dispersion_params has no "
                f"{route_key!r} row — the fit paper would never reach a "
                f"user's references block."
            )
            routed_dois = {entries[k].doi for k in row}
            assert doi in routed_dois, (
                f"routes.dispersion_params[{route_key!r}] fires "
                f"{sorted(row)} (DOIs {sorted(d for d in routed_dois if d)}) "
                f"but the parameter row claims doi={doi}."
            )

    def test_no_stale_dispersion_params_routes(self) -> None:
        """Reverse direction: every route row maps back to a live D4
        parameter-table key, so renames/removals can't leave dead routes."""
        from vibeqc.dispersion_d4_parameters import list_d4_functionals

        db = load_default_database()
        live = set(list_d4_functionals())
        for route_key in db.routes().get("dispersion_params", {}):
            disp, _, param_key = route_key.partition(":")
            assert disp == "d4" and param_key, (
                f"routes.dispersion_params key {route_key!r} is not of the "
                f"documented '<disp>:<param>' form"
            )
            assert param_key in live, (
                f"routes.dispersion_params[{route_key!r}] references a "
                f"parametrization missing from dispersion_d4_parameters "
                f"— stale route?"
            )


def test_periodic_run_pulls_spglib(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="pob-tzvp", functional="PBE")
    db = load_default_database()
    result = db.assemble_from_plan(plan, periodic=True)
    keys = {c.key for c in result.citations}
    assert "togo_shinohara_tanaka_2024" in keys


def test_smearing_run_pulls_mermin(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="pob-tzvp", functional="PBE")
    db = load_default_database()
    result = db.assemble_from_plan(plan, periodic=True, uses_smearing=True)
    keys = {c.key for c in result.citations}
    assert "mermin_finite_temperature_dft_1965" in keys


def test_level_shift_pulls_saunders_hillier(tmp_path: Path) -> None:
    """A job run with a Saunders-Hillier level shift cites the defining
    1973 paper; a plain run does not."""
    plan = _plan(tmp_path, basis="6-31g*")
    db = load_default_database()

    result = db.assemble_from_plan(plan, uses_level_shift=True)
    keys = {c.key for c in result.citations}
    assert "saunders_hillier_levelshift_1973" in keys

    baseline = db.assemble_from_plan(plan)
    assert "saunders_hillier_levelshift_1973" not in {
        c.key for c in baseline.citations
    }


def test_uks_stability_adds_kohn_sham_stability_source() -> None:
    """The KS-only formalism fires for UKS stability, never plain UHF."""
    db = load_default_database()
    uks = {
        citation.key
        for citation in db.assemble(
            method="uks",
            basis="sto-3g",
            functional="pbe",
            uses_scf_stability=True,
        ).citations
    }
    assert "bauernschmitt_ahlrichs_ks_stability_1996" in uks
    assert {
        "seeger_pople_stability_1977",
        "lehtola_scf_overview_2020",
        "davidson_iterative_1975",
    } <= uks

    uhf = {
        citation.key
        for citation in db.assemble(
            method="uhf",
            basis="sto-3g",
            uses_scf_stability=True,
        ).citations
    }
    assert "bauernschmitt_ahlrichs_ks_stability_1996" not in uhf


def test_uses_ecp_pulls_libecpint(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="def2-tzvp", functional="PBE")
    db = load_default_database()
    result = db.assemble_from_plan(plan, uses_ecp=True)
    keys = {c.key for c in result.citations}
    assert "shaw_gilbert_libecpint" in keys


def test_uses_ase_pulls_ase_paper(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="6-31g*")
    db = load_default_database()
    result = db.assemble_from_plan(plan, uses_ase=True)
    keys = {c.key for c in result.citations}
    bibtex_keys = {c.bibtex_key for c in result.citations}
    assert "ase_2017" in keys
    assert "larsen_ase_2017" in bibtex_keys


def test_direct_scf_pulls_almlof_and_haser(tmp_path: Path) -> None:
    """When direct_scf=True, both the Almlöf 1982 and Häser-Ahlrichs
    1989 references fire."""
    plan = _plan(tmp_path, basis="def2-svp")
    db = load_default_database()
    result = db.assemble_from_plan(plan, direct_scf=True)
    keys = {c.key for c in result.citations}
    assert "almlof_direct_scf_1982" in keys
    assert "haser_ahlrichs_schwarz_1989" in keys


def test_direct_scf_false_does_not_pull_direct_refs(tmp_path: Path) -> None:
    """Without direct_scf=True the direct-SCF references are not cited."""
    plan = _plan(tmp_path, basis="def2-svp")
    db = load_default_database()
    result = db.assemble_from_plan(plan)  # direct_scf defaults to False
    keys = {c.key for c in result.citations}
    assert "almlof_direct_scf_1982" not in keys
    assert "haser_ahlrichs_schwarz_1989" not in keys


# ---------------------------------------------------------------------------
# Deduplication: each entry appears once
# ---------------------------------------------------------------------------


def test_assembled_list_has_no_duplicates(tmp_path: Path) -> None:
    # b3lyp pulls VWN 1980 via routes.functionals; pulling the LDA route
    # explicitly (if we ever do — we don't here) shouldn't duplicate.
    result = assemble(_plan(tmp_path, basis="6-31g*", functional="B3LYP"))
    keys = [c.key for c in result.citations]
    assert len(keys) == len(set(keys))


# ---------------------------------------------------------------------------
# Template substitution
# ---------------------------------------------------------------------------


def test_software_citation_resolves_version_template(
    tmp_path: Path,
) -> None:
    from vibeqc.banner import VIBEQC_VERSION

    result = assemble(_plan(tmp_path, basis="sto-3g"))
    software = result.citations[0]
    assert software.version == VIBEQC_VERSION
    # Year is the current year (best-effort).
    assert software.year and str(software.year).isdigit()


# ---------------------------------------------------------------------------
# Coverage gate: every functional / basis used in tests is routed.
# This is the CI gate that enforces the dev-chat discipline — adding a
# functional without updating database.toml fails here.
# ---------------------------------------------------------------------------

# Hand-picked from the v0.8.0 test surface. Update this list when a
# new feature lands; that fail is the explicit reminder to add the
# matching route.
_REQUIRED_FUNCTIONALS = (
    "LDA",
    "PBE",
    "PBE0",
    "B3LYP",
    "PW91",
    "B2PLYP",
    # meta-GGA + range-separated hybrids (v0.9.0 XC-library expansion).
    "TPSS",
    "r2scan",
    "wb97x",
    "wb97x-v",
    "wb97m-v",
    "vv10",
    # functionals wired in xc.cpp that were missing routes (§8 backfill).
    "blyp",
    "tpssh",
    "m06-l",
    "m06-2x",
    "r2scan0",
    "r2scanh",
    "wb97x-d",
    "cam-b3lyp",
    "camb3lyp",
    "lc-wpbe",
    "lcwpbe",
    "mn15",
    # revDSD-PBEP86-D4 double-hybrid SCF part (also a [routes.methods] key).
    "revdsd-pbep86",
    # B3LYP flavor spellings (v0.12.0 flavor audit): all three must
    # cite the same papers as bare b3lyp.
    "b3lyp5",
    "b3lypg",
    "b3lyp/g",
    # Microsoft SKALA-1.1 external-XC backend and its exact-revision aliases.
    "skala-1.1",
    "skala-1.1-rev1",
    "skala",
)
# Every method `run_job` accepts that carries a method-specific
# citation route. Mean-field methods (rhf / uhf / rks / uks) are
# deliberately absent — their citations come from the integral
# library + the functional, not a method-specific paper. Post-SCF
# (CCSD / FCI) and composite-3c methods each have a defining paper.
_REQUIRED_METHODS = (
    "direct_scf",
    "rohf",
    "roks",
    "ccsd",
    "ccsd(t)",
    "a-ccsd(t)",
    "fci",
    "hf-3c",
    "pbeh-3c",
    "b97-3c",
    "b3lyp-3c",
    "r2scan-3c",
    "wb97x-3c",
    "hse-3c",
    # Active-space wavefunction methods (method= in run_job).
    "selected_ci",
    "dmrg",
    "v2rdm",
    "transcorrelated_ci",
    "casci",
    "mrci",
    "casscf",
    "nevpt2",
    "caspt2",
    "caspt2_nac",
    # Fixed-space variational CISD (vibeqc.solvers.cisd; surfaced by the
    # semiempirical msindo_cisd path via assemble extra_entries).
    "cisd",
    # The direct experimental DFTB0-SECCM adapter combines the in-house
    # DFTB screening Hamiltonian with the finite cyclic-cluster construction.
    "dftb0_seccm",
    # MP2 lineage (standalone runners + run_double_hybrid SCF prep).
    "mp2",
    "ri-mp2",
    "scs-mp2",
    "sos-mp2",
    # Semicanonical ROHF-MP2 (vibeqc.cc.run_rohf_mp2; Knowles 1991).
    "rohf-mp2",
    # OVGF / GF2 Green's-function quasiparticle IPs (method="ovgf";
    # vibeqc.propagator). The ab-initio route cites Cederbaum 1975 + the
    # von Niessen-Schirmer-Cederbaum 1984 computational review.
    "ovgf",
    # Double-hybrid dispatchers.
    "b2plyp",
    "dsd-pbep86",
    "revdsd-pbep86",
    "pwpb95",
    # ωB97X-D (run_wb97x_d).
    "wb97x-d",
    # Non-default SCF accelerators (Pulay base + accelerator paper).
    "adiis",
    "kdiis",
    "ediis_diis",
    # GPW / GAPW (Lippert-Hutter / Quickstep). Fires when
    # ``jk_method='gpw'`` is used in ``run_periodic_job`` or when the
    # standalone ``run_periodic_rhf_gpw`` / ``run_periodic_rks_gpw_multi_k``
    # entries are called.
    "gpw",
    "gapw",
    # BIPOLE periodic Coulomb route. The runner fires this through the
    # explicit uses_bipole flag when ``jk_method='bipole'`` is selected.
    "bipole",
    # GFN2-xTB. Experimental + gated (python/vibeqc/semiempirical/methods/
    # gfn2.py), but its citation must still fire whenever a job actually uses
    # it — CLAUDE.md § 8.
    "gfn2_xtb",
    # NDDO semiempirical methods exposed by run_job. Even while PM6 / OMx
    # remain validation-gated for production science, their method papers must
    # fire whenever the executable routes are used.
    "pm6",
    "upm6",
    "pm7",
    "upm7",
    "om1",
    "om2",
    "om3",
    # MSINDO (Bredow/Geudtner/Jug INDO; closed-shell s/p, H–F). Its method
    # papers must fire whenever a job uses method="msindo" — CLAUDE.md § 8.
    "msindo",
    # MSINDO semiempirical Cyclic Cluster Model (preferred method="seccm",
    # normalized internally to legacy method="ccm",
    # vibeqc.semiempirical.methods.msindo_ccm). Peintinger & Bredow 2014 must
    # fire whenever a periodic CCM calculation is run — CLAUDE.md § 8.
    "ccm",
    "seccm",
    # Γ-CCM, the union-and-weight / Wigner-Seitz four-center ab-initio CCM
    # route. These are the ``method=`` selectors the -a drivers accept
    # (vibeqc.periodic.ccm.scf._CCM_ERI_METHODS). The public
    # method="aiccm", variant="four-center" arm passes
    # "aiccm2026dev-a" to citation assembly; the two library spellings remain
    # pinned because direct users owe the same CCM references.
    "aiccm2026dev-a",
    "union12",
    "aiccmdev",
    # χ-CCM, the finite-character (Γ-centred character-mesh) ab-initio CCM route. The
    # periodic runner fires the bare key for jk_method="aiccm2026dev-b" while GDF
    # and Ewald infrastructure citations are added by their feature flags.
    "aiccm2026dev-b",
    # Neutral Γ-CCM (ruling R1): the two producers of the one neutral
    # fitted-torus Hamiltonian. "real-gamma" is passed by the
    # periodic_runner.py assemble() call for method="aiccm",
    # variant="real-gamma" (and its warned legacy spelling
    # jk_method="real-gamma"), pinned end to end in
    # tests/test_feature_citation_end_to_end.py; "neutral-bloch" is passed
    # for variant="neutral-bloch" since M1b (the runner alias of that name
    # was retired to fail-closed in M1, D-2b, so a plain unit-cell GDF job
    # cannot reach the row). Both rows carry the CCM lineage pair; the GDF
    # stack rides on uses_gdf and real-Γ forces on uses_gradient. Both are
    # pinned end to end, with a plain-GDF negative control.
    "real-gamma",
    "neutral-bloch",
    # The compound post-HF keys below are each stamped onto a result
    # dataclass's ``backend`` field by periodic/chi/posthf.py, but
    # nothing passes them to assemble() yet: those drivers are library-only
    # (not reachable from run_periodic_job, and the module imports nothing
    # from vibeqc.output), so a -b post-HF job emits no citation surface at
    # all. Pinned ahead of that wiring for the same reason as the -a rows —
    # a routes.methods miss is silent. See the comment block on this family
    # in database.toml, and test_b_posthf_backend_labels_all_have_routes /
    # test_b_posthf_routes_all_have_a_producer below, which tie this list to
    # the labels that module can actually produce.
    "aiccm2026dev-b-ri-mp2",
    "aiccm2026dev-b-dlpno-mp2",
    "aiccm2026dev-b-dlpno-ccsd",
    "aiccm2026dev-b-dlpno-ccsd(t)",
    "aiccm2026dev-b-ccsd",
    "aiccm2026dev-b-ccsd(t)",
    "aiccm2026dev-b-ri-ump2",
    "aiccm2026dev-b-dlpno-ump2",
    "aiccm2026dev-b-dlpno-uccsd",
    "aiccm2026dev-b-dlpno-uccsd(t)",
    # Canonical (complete-domain-limit) unrestricted CC. Produced by
    # run_aiccm2026dev_b_uccsd / _uccsd_t; had no route until 2026-08-22.
    "aiccm2026dev-b-uccsd",
    "aiccm2026dev-b-uccsd(t)",
    # MACE MLIP (method="mace", vibeqc.mlip.mace). External pre-trained
    # model; its citation route must fire whenever a job uses it.
    "mace",
)
_REQUIRED_BASIS_SETS = (
    "STO-3G",
    "STO-6G",
    "3-21G",
    "6-31G*",
    "6-31G**",
    "6-311G**",
    "6-311+G(3df,2p)",
    "def2-svp",
    "def2-tzvp",
    "def2-tzvpp",
    "def2-qzvp",
    "def2-qzvpp",
    "def2-svpd",
    "def2-tzvpd",
    "def2-svp-jk",
    "def2-svp-jkfit",
    "def2-tzvp-jk",
    "def2-universal-jkfit",
    "def2-universal-jfit",
    "def2-svp-rifit",
    "def2-tzvp-rifit",
    "cc-pvdz",
    "cc-pvtz",
    "cc-pvqz",
    "cc-pv5z",
    "cc-pvdz-ri",
    "pob-tzvp",
    "pob-dzvp-rev2",
    "pob-tzvp-rev2",
)


@pytest.mark.parametrize("functional", _REQUIRED_FUNCTIONALS)
def test_required_functional_has_a_route(
    tmp_path: Path,
    functional: str,
) -> None:
    result = assemble(_plan(tmp_path, basis="sto-3g", functional=functional))
    # No warning that *this* functional was unrouted.
    bad = [w for w in result.warnings if functional in w]
    assert not bad, f"functional {functional!r} has no citation route"


@pytest.mark.parametrize("basis", _REQUIRED_BASIS_SETS)
def test_required_basis_has_a_route(tmp_path: Path, basis: str) -> None:
    result = assemble(_plan(tmp_path, basis=basis))
    bad = [w for w in result.warnings if basis.lower() in w.lower()]
    assert not bad, f"basis {basis!r} has no citation route"


def test_default_filtered_basis_name_uses_source_citation_route() -> None:
    db = load_default_database()
    source = db.assemble(method="rhf", basis="sto-3g")
    filtered = db.assemble(method="rhf", basis="_vibeqc_filtered_sto-3g_deadbeef")
    assert not [w for w in filtered.warnings if "basis set" in w]
    assert {c.key for c in filtered.citations} == {c.key for c in source.citations}


@pytest.mark.parametrize("method_key", _REQUIRED_METHODS)
def test_required_method_route_is_present(method_key: str) -> None:
    """Every key in _REQUIRED_METHODS must exist in the database's
    methods routes table and resolve without dangling references."""
    db = load_default_database()
    routes = db._routes.get("methods", {})
    assert method_key in routes, (
        f"method route {method_key!r} missing from database.toml"
    )
    # Validate entries resolve.
    for entry_key in routes[method_key]:
        assert entry_key in db._entries, (
            f"route methods.{method_key!r} references unknown entry {entry_key!r}"
        )


@pytest.mark.parametrize("method_key", _REQUIRED_METHODS)
def test_required_method_route_actually_fires(
    tmp_path: Path,
    method_key: str,
) -> None:
    """Regression for the method= routing bug: ``assemble()`` must
    actually walk ``routes.methods[method]`` and emit those entries
    — not merely have the route present in the database. Before this
    was fixed, the CCSD / direct-SCF routes existed but no job's
    ``.bibtex`` ever contained them."""
    db = load_default_database()
    expected = set(db._routes["methods"][method_key])
    result = db.assemble(method=method_key, basis="sto-3g")
    got = {c.key for c in result.citations}
    missing = expected - got
    assert not missing, (
        f"method {method_key!r} declares routes {sorted(expected)} "
        f"but assemble() did not emit {sorted(missing)}"
    )


def test_caspt2_route_includes_exact_ipea_references() -> None:
    db = load_default_database()
    result = db.assemble(method="caspt2", basis="sto-3g")
    emitted = {citation.key for citation in result.citations}
    assert {
        "ghigo_roos_malmqvist_ipea_2004",
        "nishimoto_ipea_derivatives_2023",
    } <= emitted


# ---------------------------------------------------------------------------
# Geometry-optimization optimizers + coordinate systems
# (vibeqc.geomopt, run_job(geom_opt=..., geom_coords=...)). Each
# keyword-selected optimizer / coordinate system that traces to a defining
# paper must carry a route that fires. sd / cg / (L-)BFGS / cartesian are
# textbook (or covered by the ASE BFGS + Pulay-forces routes) and carry no
# row on purpose.
# ---------------------------------------------------------------------------

_REQUIRED_OPTIMIZER_ROUTES = ("trust", "rfo", "ef", "prfo", "gdiis", "fire")
_REQUIRED_COORDINATE_ROUTES = ("dlc", "internal", "delocalized")


@pytest.mark.parametrize("opt_key", _REQUIRED_OPTIMIZER_ROUTES)
def test_required_optimizer_route_is_present(opt_key: str) -> None:
    """Every geom_opt optimizer with a defining paper has a
    ``routes.optimizers`` row that resolves without dangling references."""
    db = load_default_database()
    routes = db._routes.get("optimizers", {})
    assert opt_key in routes, (
        f"optimizer route {opt_key!r} missing from database.toml"
    )
    for entry_key in routes[opt_key]:
        assert entry_key in db._entries, (
            f"route optimizers.{opt_key!r} references unknown entry {entry_key!r}"
        )


@pytest.mark.parametrize("opt_key", _REQUIRED_OPTIMIZER_ROUTES)
def test_required_optimizer_route_actually_fires(opt_key: str) -> None:
    """``assemble(geom_optimizer=...)`` must actually walk
    ``routes.optimizers`` and emit those entries — not merely have the
    route present (CLAUDE.md § 8 step 5)."""
    db = load_default_database()
    expected = set(db._routes["optimizers"][opt_key])
    result = db.assemble(method="rhf", basis="sto-3g", geom_optimizer=opt_key)
    got = {c.key for c in result.citations}
    missing = expected - got
    assert not missing, (
        f"optimizer {opt_key!r} declares routes {sorted(expected)} "
        f"but assemble() did not emit {sorted(missing)}"
    )


@pytest.mark.parametrize("coord_key", _REQUIRED_COORDINATE_ROUTES)
def test_required_coordinate_route_actually_fires(coord_key: str) -> None:
    """``assemble(geom_coords=...)`` must emit the DLC + redundant-internals
    coordinate-system citations for every DLC alias."""
    db = load_default_database()
    expected = set(db._routes["coordinates"][coord_key])
    result = db.assemble(method="rhf", basis="sto-3g", geom_coords=coord_key)
    got = {c.key for c in result.citations}
    missing = expected - got
    assert not missing, (
        f"coordinate {coord_key!r} declares routes {sorted(expected)} "
        f"but assemble() did not emit {sorted(missing)}"
    )


def test_geomopt_rfo_dlc_run_cites_optimizer_and_coords() -> None:
    """An RFO/DLC geometry optimization surfaces Banerjee 1985 (RFO) plus
    the two delocalized-internal-coordinate papers, all user-visible."""
    db = load_default_database()
    result = db.assemble(
        method="rks",
        basis="sto-3g",
        functional="pbe",
        geom_optimizer="rfo",
        geom_coords="dlc",
        uses_gradient=True,
    )
    printable = {c.key for c in result.printable}
    assert {
        "banerjee_rfo_1985",
        "baker_kessi_delley_dlc_1996",
        "peng_ayala_schlegel_frisch_redundant_1996",
    } <= printable
    assert not list(result.warnings), result.warnings


def test_ef_prfo_routes_add_baker_on_top_of_banerjee() -> None:
    """Eigenvector-following / P-RFO cite Baker 1986 in addition to the
    Banerjee 1985 RFO foundation."""
    db = load_default_database()
    for opt in ("ef", "prfo"):
        keys = {
            c.key
            for c in db.assemble(
                method="rhf", basis="sto-3g", geom_optimizer=opt
            ).citations
        }
        assert {"banerjee_rfo_1985", "baker_ts_1986"} <= keys


def test_geomopt_optimizer_citations_silent_when_unset() -> None:
    """A plain SCF (no geom_opt / geom_coords) pulls none of the
    geometry-optimizer or coordinate-system papers."""
    db = load_default_database()
    keys = {c.key for c in db.assemble(method="rhf", basis="sto-3g").citations}
    for k in (
        "banerjee_rfo_1985",
        "baker_ts_1986",
        "csaszar_pulay_gdiis_1984",
        "bitzek_fire_2006",
        "more_sorensen_trust_1983",
        "baker_kessi_delley_dlc_1996",
        "peng_ayala_schlegel_frisch_redundant_1996",
    ):
        assert k not in keys


def test_cartesian_and_textbook_optimizers_carry_no_citation() -> None:
    """Cartesian coordinates and the textbook BFGS optimizer have no
    defining-paper route; selecting them must not warn or emit a geomopt
    citation."""
    db = load_default_database()
    result = db.assemble(
        method="rhf", basis="sto-3g", geom_optimizer="bfgs", geom_coords="cartesian"
    )
    keys = {c.key for c in result.citations}
    assert "baker_kessi_delley_dlc_1996" not in keys
    assert "banerjee_rfo_1985" not in keys
    assert not list(result.warnings), result.warnings


def test_mlip_suppresses_libint_and_diis() -> None:
    """An MLIP engine (method="mace") evaluates no Gaussian integrals and
    runs no SCF: ``assemble(uses_integrals=False, uses_scf=False)`` must
    drop the always-on libint + Pulay-DIIS routes while still firing the
    MACE method paper and the vibe-qc software cite — the references must
    reflect what actually ran (CLAUDE.md § 8)."""
    db = load_default_database()
    result = db.assemble(method="mace", uses_integrals=False, uses_scf=False)
    keys = {c.key for c in result.citations}
    # vibe-qc + the MACE *method* paper (static route) fire.
    assert "vibeqc_software" in keys
    assert "batatia_mace_2022" in keys
    # The wrong, always-on routes are suppressed for an MLIP.
    assert "libint_valeev" not in keys
    assert "pulay_diis_1980" not in keys
    assert "pulay_diis_1982" not in keys
    # The foundation-model paper is per-model (extra_entries), so it is NOT
    # in the static route output.
    assert "batatia_mace_mp_2024" not in keys


def test_dftb0_seccm_cites_hamiltonian_and_boundary_construction() -> None:
    db = load_default_database()
    keys = {
        citation.key
        for citation in db.assemble(
            method="dftb0_seccm", uses_integrals=True, uses_scf=False
        ).citations
    }

    assert {
        "porezag_dftb0_1995",
        "wolfsberg_helmholz_1952",
        "hehre_sto_ng_1969",
        "claff_dftb_ccm_2012",
        "bredow_geudtner_jug_ccm_2001",
        "peintinger_ccm_2014",
    } <= keys


@pytest.mark.parametrize(
    ("method_key", "expected"),
    [
        (
            "scc_dftb_seccm",
            {
                "porezag_dftb0_1995",
                "elstner_scc_dftb_1998",
                "irons_tuck_aitken_1969",
                "noga_ccm_1999",
                "bredow_geudtner_jug_ccm_2001",
                "peintinger_ccm_2014",
                # The route reports a Mermin free energy under smearing and,
                # since #302, records how far the applied occupation is from
                # the Aufbau one; both come from this paper's Eqs. (8) and
                # (10'), so it must reach the user's reference list the way
                # it already does for the molecular scc_dftb route.
                "weinert_fractional_occupations_1992",
            },
        ),
        (
            "pm6_seccm",
            {
                "stewart_pm6_2007",
                "bredow_geudtner_jug_ccm_2001",
                "peintinger_ccm_2014",
            },
        ),
        (
            "om2_seccm",
            {
                "weber_thiel_om2_2000",
                "dral_omx_2016",
                "bredow_geudtner_jug_ccm_2001",
                "peintinger_ccm_2014",
            },
        ),
        (
            "gfn2_seccm",
            {
                "bannwarth_gfn2_2019",
                "pauling_chemical_bond_1960",
                "bredow_geudtner_jug_ccm_2001",
                "peintinger_ccm_2014",
            },
        ),
    ],
)
def test_seccm_adapters_cite_method_and_boundary_construction(
    method_key: str, expected: set[str]
) -> None:
    db = load_default_database()
    keys = {
        citation.key
        for citation in db.assemble(
            method=method_key, uses_integrals=False, uses_scf=True
        ).citations
    }
    assert expected <= keys


def test_pm6_cites_dewar_thiel_nddo_multipoles() -> None:
    """The PM6/UPM6/PM6-SECCM routes fire the Dewar-Thiel two-centre
    NDDO multipole paper (Theor. Chim. Acta 46, 89 (1977); issue #420).

    The heavy-heavy and heavy-hydrogen two-centre blocks contract the
    AO-resolved MNDOD multipole tensor that paper defines, so every
    executable PM6 route must carry the reference alongside Stewart 2007.
    """
    db = load_default_database()
    entries = db.entries()
    assert "dewar_thiel_1977" in entries
    citation = entries["dewar_thiel_1977"]
    assert citation.doi == "10.1007/BF00548085"
    assert citation.print is True
    for method_key in ("pm6", "upm6", "pm6_seccm"):
        keys = {
            c.key
            for c in db.assemble(
                method=method_key, uses_integrals=False, uses_scf=True
            ).citations
        }
        assert "dewar_thiel_1977" in keys


def test_msindo_nddo_cites_multipole_sources() -> None:
    """The MSINDO NDDO variant cites both multipole sources."""
    database = load_default_database()
    voigt = database.entries()["voigt_1973"]
    assert voigt.doi == "10.1007/BF00527556"
    assert voigt.print is True
    plan = SemiempiricalRoutePlan.from_request("nddo")
    keys = {
        citation.key
        for citation in database.assemble(**plan.citation_assemble_kwargs).citations
    }

    assert {
        "ahlswede_jug_msindo_1_1999",
        "ahlswede_jug_msindo_2_1999",
        "dewar_thiel_1977",
        "voigt_1973",
    } <= keys

    indo = SemiempiricalRoutePlan.from_request("msindo")
    indo_keys = {
        citation.key
        for citation in database.assemble(**indo.citation_assemble_kwargs).citations
    }
    assert {"dewar_thiel_1977", "voigt_1973"}.isdisjoint(indo_keys)


def test_gfn2_cites_pauling_electronegativities() -> None:
    """The GFN2 routes fire the Pauling electronegativity reference
    (Pauling, The Nature of the Chemical Bond, 3rd ed., 1960; issue #433).

    Both executable GFN2 routes -- the molecular ``gfn2_xtb`` path and the
    periodic ``gfn2_seccm`` path -- build their H0 off-site EN factor
    (1 + 0.02*dEN^2) from the Pauling table, so each must carry the
    reference alongside Bannwarth 2019.
    """
    db = load_default_database()
    entries = db.entries()
    assert "pauling_chemical_bond_1960" in entries
    citation = entries["pauling_chemical_bond_1960"]
    assert citation.year == 1960
    for method_key in ("gfn2_xtb", "gfn2_seccm"):
        keys = {
            c.key
            for c in db.assemble(
                method=method_key, uses_integrals=False, uses_scf=True
            ).citations
        }
        assert "pauling_chemical_bond_1960" in keys



def test_periodic_gfn2_cites_its_lattice_summed_electrostatics() -> None:
    """The Bloch-periodic GFN2 kernel's own references reach the user.

    Issues #296/#338 replaced the home-cell g=0 shell gamma with an
    Ewald-split lattice sum of a short-range gamma.  Two things a periodic
    run now uses are not implied by the method key: the gamma functional
    form (Elstner 1998 by default, the maintainer's D1 decision) and the
    dimensional Ewald channel the cell selects.  Both must be cited.
    """
    db = load_default_database()

    def keys(dimension: int, form: str) -> set[str]:
        plan = SemiempiricalRoutePlan.from_request(
            "gfn2",
            boundary="periodic_gamma",
        ).with_periodic_electrostatics_runtime(
            periodic_dimension=dimension,
            shell_gamma_form=form,
        )
        return {
            citation.key
            for citation in db.assemble(
                **plan.citation_assemble_kwargs
            ).citations
        }

    bulk = keys(3, "elstner")
    assert {
        "bannwarth_gfn2_2019",
        "ewald_lattice_sum_1921",
        "elstner_scc_dftb_1998",
    } <= bulk
    # The bulk channel is plain 3-D Ewald: no slab or wire references.
    assert {
        "parry_2d_ewald_1975",
        "de_leeuw_perram_2d_ewald_1979",
        "rozzi_wire_coulomb_2006",
    }.isdisjoint(bulk)

    slab = keys(2, "elstner")
    assert {"parry_2d_ewald_1975", "de_leeuw_perram_2d_ewald_1979"} <= slab
    assert "rozzi_wire_coulomb_2006" not in slab

    wire = keys(1, "elstner")
    assert "rozzi_wire_coulomb_2006" in wire
    assert "parry_2d_ewald_1975" not in wire

    # The Klopman-Ohno form keeps the Ewald channel and drops Elstner: it is
    # the GFN2 paper's own kernel, already carried by bannwarth_gfn2_2019.
    klopman_ohno = keys(3, "klopman_ohno")
    assert "ewald_lattice_sum_1921" in klopman_ohno
    assert "elstner_scc_dftb_1998" not in klopman_ohno


def test_periodic_electrostatics_context_is_absent_until_the_driver_ran() -> None:
    """A plan built before the kernel ran claims no electrostatics source."""
    db = load_default_database()
    plan = SemiempiricalRoutePlan.from_request("gfn2", boundary="periodic_gamma")
    assert plan.shell_gamma_form is None
    assert "extra_entries" not in plan.citation_assemble_kwargs
    keys = {
        c.key
        for c in db.assemble(**plan.citation_assemble_kwargs).citations
    }
    assert "elstner_scc_dftb_1998" not in keys
    # And the runtime attachment is refused on a boundary that has no
    # Bloch-periodic lattice sum to describe.
    with pytest.raises(ValueError, match="Bloch-periodic boundary"):
        SemiempiricalRoutePlan.from_request(
            "gfn2", boundary="molecular"
        ).with_periodic_electrostatics_runtime(
            periodic_dimension=3, shell_gamma_form="elstner"
        )


@pytest.mark.parametrize(
    ("method", "family"),
    [
        ("scc_dftb", "madelung"),
        ("gfn2", "madelung"),
        ("gfn2", "ewald_gamma"),
    ],
)
def test_seccm_slab_citations_follow_the_selected_2d_kernel(
    method: str,
    family: str,
) -> None:
    db = load_default_database()

    def keys(dimension: int, electrostatics_family: str) -> set[str]:
        plan = SemiempiricalRoutePlan.from_request(
            method,
            boundary="seccm",
        ).with_seccm_runtime(
            periodic_dimension=dimension,
            electrostatics_family=electrostatics_family,
        )
        return {
            citation.key
            for citation in db.assemble(
                **plan.citation_assemble_kwargs
            ).citations
        }

    slab = keys(2, family)
    assert {
        "parry_2d_ewald_1975",
        "de_leeuw_perram_2d_ewald_1979",
    } <= slab

    # Dimension 1 fires the wire-Ewald route (methods.wire_ewald_1d,
    # 2026-08-27): the wire kernel's provenance is documented on the
    # Parry entry, so parry legitimately appears in 1-D - but never the
    # 2-D-only de Leeuw-Perram companion.
    wire = keys(1, family)
    assert "parry_2d_ewald_1975" in wire
    assert "de_leeuw_perram_2d_ewald_1979" not in wire

    for irrelevant in (keys(3, family), keys(2, "none"), keys(1, "none")):
        assert "parry_2d_ewald_1975" not in irrelevant
        assert "de_leeuw_perram_2d_ewald_1979" not in irrelevant


def test_msindo_seccm_1d_does_not_cite_the_parry_wire_ewald() -> None:
    """MSINDO's 1-D kernel is the frozen truncated lattice sum, not Parry.

    ``routes.methods.wire_ewald_1d`` credits Parry 1975 for the converged
    background-corrected wire Ewald (``detail::wire_madkonst_1d``,
    cpp/src/semiempirical/seccm/ewald_1d.h) that the SECCM adapters run at
    d = 1. MSINDO instead keeps ``indo::_madelung_potential_1d`` -- the
    truncated +/-1, +/-2 shell bare point-charge sum of ``ccm1dmadelsum.f``,
    which the wire header names as the very kernel it replaced. It performs
    no Ewald split, so it must cite neither 2-D slab paper. Its 2-D path does
    run the genuine Parry/Heyes machinery (``indo::_madkonst_2d``) and cites
    both. GitLab #442; CLAUDE.md section 8.
    """
    db = load_default_database()

    def keys(dimension: int, electrostatics_family: str) -> set[str]:
        plan = SemiempiricalRoutePlan.from_request(
            "ccm",
            boundary="seccm",
        ).with_seccm_runtime(
            periodic_dimension=dimension,
            electrostatics_family=electrostatics_family,
        )
        return {
            citation.key
            for citation in db.assemble(
                **plan.citation_assemble_kwargs
            ).citations
        }

    slab_keys = {"parry_2d_ewald_1975", "de_leeuw_perram_2d_ewald_1979"}

    # Only the real 2-D Parry/Heyes kernel earns the slab references.
    assert slab_keys <= keys(2, "madelung")

    for clean in (
        keys(1, "madelung"),
        keys(3, "madelung"),
        keys(1, "none"),
        keys(2, "none"),
        keys(3, "none"),
    ):
        assert clean.isdisjoint(slab_keys)


def test_msindo_seccm_truncated_1d_is_not_credited_as_exact_ewald() -> None:
    """The frozen truncated 1-D sum owes the CCM construction, not Ewald.

    MSINDO at d = 1 resolves ``madelung_truncated_1d``, the finite +/-1, +/-2
    shell bare point-charge sum of ``ccm1dmadelsum.f``. It performs no Ewald
    split and Janetzko, Bredow and Jug (2002, Eq. 12) define their exact
    Madelung matrices for m = 2, 3 only, so crediting either to it is a wrong
    attribution (GitLab vibe-qc#64). The genuinely executing 2-D and 3-D
    embeddings keep the full row, and the "none" family cites nothing of it.
    """
    db = load_default_database()

    def keys(dimension: int, electrostatics_family: str) -> set[str]:
        plan = SemiempiricalRoutePlan.from_request(
            "ccm",
            boundary="seccm",
        ).with_seccm_runtime(
            periodic_dimension=dimension,
            electrostatics_family=electrostatics_family,
        )
        kwargs = plan.citation_assemble_kwargs
        if dimension == 1 and electrostatics_family == "madelung":
            assert kwargs["seccm_electrostatics_kernel"] == "madelung_truncated_1d"
        return {citation.key for citation in db.assemble(**kwargs).citations}

    construction = "bredow_geudtner_jug_ccm_2001"
    exact = {"janetzko_ccm_long_range_2002", "ewald_lattice_sum_1921"}

    truncated = keys(1, "madelung")
    assert construction in truncated
    assert truncated.isdisjoint(exact)
    assert truncated.isdisjoint({"parry_2d_ewald_1975", "de_leeuw_perram_2d_ewald_1979"})

    for dimension in (2, 3):
        executing = keys(dimension, "madelung")
        assert construction in executing
        assert exact <= executing

    # The "none" family runs no embedding at all. Bredow 2001 may still
    # appear there: it is the CCM construction reference on the method's
    # own lineage row, which is not what this test polices.
    for dimension in (1, 2, 3):
        assert keys(dimension, "none").isdisjoint(exact)

    # ``methods.msindo`` carries no CCM lineage, so on a bare assembly the
    # construction reference can only arrive through the madelung route:
    # this pins which row fired, not just which keys the lineage supplies.
    def bare(kernel: str) -> set[str]:
        assembled = db.assemble(
            method="msindo", uses_scf=True, periodic=False,
            seccm_dimension=1 if kernel.endswith("1d") else 3,
            seccm_electrostatics_kernel=kernel,
        )
        return {citation.key for citation in assembled.citations}

    bare_truncated = bare("madelung_truncated_1d")
    assert construction in bare_truncated
    assert bare_truncated.isdisjoint(exact)
    assert exact | {construction} <= bare("madelung_ewald_3d")

    routes = db._routes["methods"]
    assert tuple(routes["ccm_madelung_truncated_1d"]) == (construction,)
    assert set(routes["ccm_madelung_embedding"]) == exact | {construction}


@pytest.mark.parametrize("method", ["scc_dftb", "gfn2"])
def test_seccm_mermin_citation_follows_runtime_temperature(method: str) -> None:
    db = load_default_database()

    def keys(temperature: float) -> set[str]:
        plan = SemiempiricalRoutePlan.from_request(
            method,
            boundary="seccm",
        ).with_seccm_runtime(
            periodic_dimension=2,
            electronic_temperature=temperature,
        )
        return {
            citation.key
            for citation in db.assemble(
                **plan.citation_assemble_kwargs
            ).citations
        }

    assert "mermin_finite_temperature_dft_1965" not in keys(0.0)
    assert "mermin_finite_temperature_dft_1965" in keys(0.002)


def test_heyes_slab_reference_waits_for_acquired_primary_record() -> None:
    """Issue #173: do not invent bibliography before the paper is acquired."""
    db = load_default_database()
    assert all(
        citation.doi != "10.1039/F29777301485"
        for citation in db.entries().values()
    )


def test_scc_dftb_cites_charge_relaxation() -> None:
    db = load_default_database()
    citations = db.assemble(
        method="scc_dftb", uses_integrals=False, uses_scf=True
    ).citations
    by_key = {citation.key: citation for citation in citations}

    assert "irons_tuck_aitken_1969" in by_key
    assert by_key["irons_tuck_aitken_1969"].doi == "10.1002/nme.1620010306"
    assert by_key["irons_tuck_aitken_1969"].print


@pytest.mark.parametrize("method_key", ["msindo", "ccm", "seccm"])
def test_indo_suppresses_libint_keeps_diis(method_key: str) -> None:
    """The INDO-family engines (MSINDO molecular + periodic CCM) evaluate
    analytic Slater (STO) integrals, not libint Gaussians, but DO run a
    Pulay-DIIS-accelerated SCF. So ``assemble(uses_integrals=False,
    uses_scf=True)`` — how runner.py calls it for these methods — must drop
    the always-on libint citation while keeping Pulay DIIS, and still fire
    the method's own route entries + the vibe-qc software cite (CLAUDE.md
    §8). This pins the ``_is_indo`` flag in run_job's citation assembly."""
    db = load_default_database()
    result = db.assemble(
        method=method_key,
        uses_integrals=False,
        uses_scf=True,
    )
    keys = {c.key for c in result.citations}
    # libint is suppressed (STO integrals, not Gaussian).
    assert "libint_valeev" not in keys
    # Pulay DIIS stays — these methods accelerate the SCF with it.
    assert "pulay_diis_1980" in keys
    # The vibe-qc software cite + the method's own route entries fire.
    assert "vibeqc_software" in keys
    for entry_key in db._routes["methods"][method_key]:
        assert entry_key in keys, (
            f"method {method_key!r} route entry {entry_key!r} did not fire"
        )


def test_ecp_runs_cite_the_pseudopotential_review_with_libecpint() -> None:
    """GitLab #642: an ECP run cites Dolg & Cao 2012 (the valence-only model
    Hamiltonian whose core charges Q = Z - n_core the properties now use)
    alongside libecpint; an all-electron run cites neither."""
    db = load_default_database()
    ecp = {c.key for c in db.assemble(method="rhf", basis="lanl2dz", uses_ecp=True).citations}
    assert "shaw_gilbert_libecpint" in ecp
    assert "dolg_cao_pseudopotentials_2011" in ecp
    printable = {c.key for c in db.assemble(method="rhf", basis="lanl2dz", uses_ecp=True).printable}
    assert "dolg_cao_pseudopotentials_2011" in printable
    ae = {c.key for c in db.assemble(method="rhf", basis="sto-3g", uses_ecp=False).citations}
    assert "dolg_cao_pseudopotentials_2011" not in ae
    assert "shaw_gilbert_libecpint" not in ae


def test_selected_ci_restricted_basis_cites_doci_via_extra_entries() -> None:
    """GitLab #639: ``SelectedCIOptions(spin_restricted=True)`` walks the
    seniority-zero (DOCI) basis, so the molecular runner adds Bytautas et al.
    2011 through ``extra_entries``; the default unrestricted basis cites the
    CIPSI/SHCI papers alone.  The entry is print=true (a Methods-section
    citation), so it must reach the printable references, not only the
    .system manifest."""
    db = load_default_database()
    restricted = db.assemble(
        method="selected_ci", basis="sto-3g", extra_entries=["bytautas_doci_2011"]
    )
    keys = {c.key for c in restricted.citations}
    assert "bytautas_doci_2011" in keys
    assert "huron_malrieu_cipsi_1973" in keys
    assert "bytautas_doci_2011" in {c.key for c in restricted.printable}
    doi = {c.key: c.doi for c in restricted.citations}["bytautas_doci_2011"]
    assert doi == "10.1063/1.3613706"

    default = {c.key for c in db.assemble(method="selected_ci", basis="sto-3g").citations}
    assert "bytautas_doci_2011" not in default
    assert "huron_malrieu_cipsi_1973" in default


def test_mlip_per_model_foundation_citation_via_extra_entries() -> None:
    """The foundation-model paper is model-dependent and added via
    ``extra_entries``: MPA-0 -> Batatia 2024, OFF23 -> Kovács 2023. The
    MACE method paper always fires; the wrong foundation paper never does.
    Empty / unknown model -> method paper only, no crash (M2 § 8)."""
    db = load_default_database()
    mp = {
        c.key
        for c in db.assemble(
            method="mace",
            uses_integrals=False,
            uses_scf=False,
            extra_entries=["batatia_mace_mp_2024"],
        ).citations
    }
    assert "batatia_mace_2022" in mp
    assert "batatia_mace_mp_2024" in mp
    assert "kovacs_mace_off_2023" not in mp

    off = {
        c.key
        for c in db.assemble(
            method="mace",
            uses_integrals=False,
            uses_scf=False,
            extra_entries=["kovacs_mace_off_2023"],
        ).citations
    }
    assert "batatia_mace_2022" in off
    assert "kovacs_mace_off_2023" in off
    assert "batatia_mace_mp_2024" not in off

    unknown = {
        c.key
        for c in db.assemble(
            method="mace", uses_integrals=False, uses_scf=False, extra_entries=[""]
        ).citations
    }
    assert "batatia_mace_2022" in unknown
    assert "batatia_mace_mp_2024" not in unknown


def test_mlip_foundation_citations_use_versions_of_record() -> None:
    """MACE method and model citations point to their versions of record.

    The semiempirical article cites the published JCP and JACS papers.  Keep
    the user-facing citation sidecar on those same records instead of sending
    readers to the earlier arXiv DOI aliases (issue #523).
    """
    entries = load_default_database().entries()

    mace_method = entries["batatia_mace_2022"]
    assert mace_method.doi == "10.52202/068431-0830"
    assert mace_method.url == "https://doi.org/10.52202/068431-0830"
    assert (
        mace_method.journal,
        mace_method.volume,
        mace_method.pages,
        mace_method.year,
    ) == (
        "Advances in Neural Information Processing Systems",
        35,
        "11423--11436",
        2022,
    )

    mace_mp = entries["batatia_mace_mp_2024"]
    assert mace_mp.doi == "10.1063/5.0297006"
    assert mace_mp.url == "https://doi.org/10.1063/5.0297006"
    assert (mace_mp.journal, mace_mp.volume, mace_mp.pages, mace_mp.year) == (
        "The Journal of Chemical Physics",
        163,
        "184110",
        2025,
    )

    mace_off = entries["kovacs_mace_off_2023"]
    assert mace_off.doi == "10.1021/jacs.4c07099"
    assert mace_off.url == "https://doi.org/10.1021/jacs.4c07099"
    assert (mace_off.journal, mace_off.volume, mace_off.pages, mace_off.year) == (
        "Journal of the American Chemical Society",
        147,
        "17598--17611",
        2025,
    )

    emitted = format_bibtex((mace_method, mace_mp, mace_off))
    assert "doi         = {10.52202/068431-0830}" in emitted
    assert "doi         = {10.1063/5.0297006}" in emitted
    assert "doi         = {10.1021/jacs.4c07099}" in emitted
    assert "10.48550/arXiv.2401.00096" not in emitted
    assert "10.48550/arXiv.2312.15211" not in emitted
    assert "10.48550/arXiv.2206.07697" not in emitted

    # Residual sweep (#523, 2026-09-02): pointing at the journal DOI is not
    # enough if the entry still carries the preprint's title and author
    # list. Crossref for 10.1021/jacs.4c07099 titles the version of record
    # "MACE-OFF: ..." (the preprint said "MACE-OFF23: ...") and lists
    # Yixuan Pu between Horton and Kapil.
    assert mace_off.title.startswith("MACE-OFF: Short-Range Transferable")
    assert "Pu, Yixuan" in mace_off.authors
    assert mace_off.authors.index("Horton, Joshua T.") + 1 == mace_off.authors.index("Pu, Yixuan")
    assert mace_off.issue == 21


def test_entanglement_bonds_citation_uses_version_of_record() -> None:
    """The MEAO bond-index paper is cited as Nat. Commun. 17, 4732 (2026).

    The entry recorded the arXiv:2501.15699 preprint while its own notes
    already named the published version. The same split #523 fixed for the
    MACE foundation models: a reader following the emitted ``.bibtex``
    landed on the preprint. Crossref-verified 2026-09-02.
    """
    entries = load_default_database().entries()
    entry = entries["entanglement_bonds_2025"]
    assert entry.kind == "article"
    assert entry.doi == "10.1038/s41467-026-73527-w"
    assert entry.url == "https://doi.org/10.1038/s41467-026-73527-w"
    assert (entry.journal, entry.volume, entry.pages, entry.year) == (
        "Nature Communications",
        17,
        "4732",
        2026,
    )
    assert list(entry.authors) == [
        "Ding, Lexin",
        "Matito, Eduard",
        "Schilling, Christian",
    ]
    emitted = format_bibtex((entry,))
    assert "doi         = {10.1038/s41467-026-73527-w}" in emitted
    assert "10.48550/arXiv.2501.15699" not in emitted
    assert "arxiv.org" not in emitted


def test_no_entry_routes_a_preprint_doi_when_its_notes_name_a_journal_version() -> None:
    """Sweep guard for the #523 split: an entry may keep an arXiv DOI only
    when no version of record is known. If its own ``notes`` field announces
    a published version, the DOI must be the journal one."""
    entries = load_default_database().entries()
    offenders = []
    for key, entry in entries.items():
        doi = (entry.doi or "").lower()
        notes = (entry.notes or "").lower()
        # "no journal version of record" (sun_ciah_2016) is the honest
        # preprint-only case and must not trip the guard.
        announces_journal = "published version" in notes or (
            "journal version of record" in notes
            and "no journal version of record" not in notes
        )
        if doi.startswith("10.48550/arxiv") and announces_journal:
            offenders.append(key)
    assert offenders == [], offenders


def test_default_run_still_cites_libint_and_diis(tmp_path: Path) -> None:
    """Guard the gating defaults: a normal SCF job (uses_integrals/uses_scf
    default True) must still pull libint + DIIS — the MLIP suppression must
    not leak into the ab-initio path."""
    result = assemble(_plan(tmp_path, basis="sto-3g"))
    keys = {c.key for c in result.citations}
    assert "libint_valeev" in keys
    assert "pulay_diis_1980" in keys


def test_dft_plus_u_route_fires_when_enabled() -> None:
    """``assemble(dft_plus_u=True)`` must surface both Dudarev 1998
    (rotationally-invariant formalism) and Cococcioni-Gironcoli 2005
    (linear-response U) — the two canonical DFT+U citations per
    CLAUDE.md § 8 and docs/user_guide/dft_plus_u.md."""
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g", dft_plus_u=True)
    keys = {c.key for c in result.citations}
    assert "dudarev_dft_plus_u_1998" in keys
    assert "cococcioni_gironcoli_2005" in keys


def test_ewald_ao_ft_route_fires_when_enabled() -> None:
    """The native EWALD_3D analytical Hartree J path uses the AO-pair
    FT machinery even when the user did not select GDF."""
    db = load_default_database()
    result = db.assemble(
        method="rhf",
        basis="sto-3g",
        periodic=True,
        uses_ewald_ao_ft=True,
    )
    keys = {c.key for c in result.citations}
    assert "sun_berkelbach_gdf_2017" in keys
    assert "mcmurchie_davidson_1978" in keys
    assert "helgaker_jorgensen_olsen_2000" in keys


@pytest.mark.parametrize(
    "route_flag", ["uses_gdf", "uses_rsgdf", "uses_gdf_2d"]
)
def test_gdf_routes_cite_global_occupation_rule(route_flag: str) -> None:
    """Native fitted routes use one global-BZ particle constraint."""
    db = load_default_database()
    result = db.assemble(
        method="rks",
        basis="sto-3g",
        functional="pbe",
        periodic=True,
        **{route_flag: True},
    )
    keys = {citation.key for citation in result.citations}
    assert "weinert_fractional_occupations_1992" in keys


def test_rsgdf_routes_range_separation_source() -> None:
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g", periodic=True, uses_rsgdf=True)
    keys = {citation.key for citation in result.printable}
    assert "ye_berkelbach_rsgdf_2021" in keys


def test_roks_route_cites_restricted_and_fractional_shell_rules() -> None:
    """ROKS carries both its Roothaan coupling and fixed-shell DFT source."""
    db = load_default_database()
    result = db.assemble(
        method="roks",
        basis="sto-3g",
        functional="pbe",
        periodic=True,
    )
    citations = {citation.key: citation for citation in result.citations}
    assert "roothaan_rohf_1960" in citations
    assert "guest_saunders_rohf_1974" in citations
    assert "weinert_fractional_occupations_1992" in citations
    assert citations["guest_saunders_rohf_1974"].doi == "10.1080/00268977400102171"


def test_slab_gdf_route_fires_truncation_and_2d_gauge_references() -> None:
    db = load_default_database()
    result = db.assemble(
        method="rks",
        basis="sto-3g",
        functional="pbe",
        periodic=True,
        uses_gdf_2d=True,
        uses_slab_ewald_2d=True,
    )
    keys = {citation.key for citation in result.citations}
    assert "sun_berkelbach_gdf_2017" in keys
    assert "spencer_alavi_truncated_coulomb_2008" in keys
    assert "sundararaman_arias_exx_2013" in keys
    assert "parry_2d_ewald_1975" in keys
    assert "de_leeuw_perram_2d_ewald_1979" in keys


def test_gilat_raubenheimer_numerics_route_fires_both_references() -> None:
    db = load_default_database()
    result = db.assemble(
        method="rks",
        basis="sto-3g",
        periodic=True,
        numerics=("gilat_raubenheimer",),
    )
    keys = {citation.key for citation in result.citations}
    assert "gilat_raubenheimer_1966" in keys
    assert "gilat_spectral_1972" in keys


def test_bipole_sr_range_screening_route() -> None:
    """Charge-pair Schwarz screening cites Sun Eq. 51 when it runs."""
    db = load_default_database()
    on = db.assemble(
        method="rhf",
        basis="sto-3g",
        periodic=True,
        uses_bipole=True,
        uses_bipole_sr_range=True,
    )
    assert "sun_range_separated_exchange_2023" in {c.key for c in on.citations}
    off = db.assemble(
        method="rhf",
        basis="sto-3g",
        periodic=True,
        uses_bipole=True,
    )
    assert "sun_range_separated_exchange_2023" not in {c.key for c in off.citations}


def test_bipole_route_fires_when_enabled() -> None:
    """BIPOLE is a periodic Coulomb route, not a mean-field method name,
    so ``run_periodic_job(..., jk_method="bipole")`` fires it through
    a dedicated assembly flag."""
    db = load_default_database()
    result = db.assemble(
        method="rhf",
        basis="sto-3g",
        periodic=True,
        uses_bipole=True,
    )
    keys = {c.key for c in result.citations}
    assert "pisani_crystal_1988" in keys
    assert "saunders_bipole_1992" in keys
    # The Ewald J/K split of the production BIPOLE route (its erfc/erf
    # arms and the #674 alpha/cutoff contract) is Ewald 1921.
    assert "ewald_lattice_sum_1921" in keys
    assert "dovesi_crystal14_2014" in keys
    # The zone-partition + exchange-truncation primary sources landed
    # with the exact_zone_bohr restriction (BIPOLE-EXACT-ZONE
    # increment 1, 2026-08-06); CLAUDE.md § 8 pins them mechanically.
    assert "dovesi_coulomb_multipole_1983" in keys
    assert "causa_exchange_truncation_1988" in keys


def test_dft_plus_u_route_silent_when_disabled() -> None:
    """The Dudarev / Cococcioni entries must NOT fire on a plain RHF
    job — only when the +U surface is explicitly enabled."""
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g")
    keys = {c.key for c in result.citations}
    assert "dudarev_dft_plus_u_1998" not in keys
    assert "cococcioni_gironcoli_2005" not in keys


def test_neb_driver_route_fires_when_enabled() -> None:
    """``assemble(uses_neb=True)`` must surface Henkelman+Jónsson 2000
    (improved-tangent NEB) plus Smidstrup 2014 (IDPP — the default
    initial-path constructor in vibeqc.run_neb). CLAUDE.md § 8 — the
    implementing chat that lands a citable algorithm wires the
    route + verifies it fires."""
    db = load_default_database()
    result = db.assemble(method="uhf", basis="sto-3g", uses_neb=True)
    keys = {c.key for c in result.citations}
    assert "henkelman_jonsson_neb_2000" in keys
    assert "smidstrup_idpp_2014" in keys
    assert not list(result.warnings), result.warnings


def test_neb_driver_route_silent_when_disabled() -> None:
    """A plain RHF job without NEB must NOT pull in the NEB papers."""
    db = load_default_database()
    result = db.assemble(method="uhf", basis="sto-3g")
    keys = {c.key for c in result.citations}
    assert "henkelman_jonsson_neb_2000" not in keys
    assert "smidstrup_idpp_2014" not in keys


def test_ci_neb_route_adds_climbing_paper_on_top_of_neb() -> None:
    """``assemble(uses_neb=True, uses_ci_neb=True)`` must surface
    the climbing-image NEB paper *in addition* to the base
    improved-tangent + IDPP bundle. CI-NEB fires only on top of
    plain NEB — never on its own."""
    db = load_default_database()
    result = db.assemble(method="uhf", basis="sto-3g", uses_neb=True, uses_ci_neb=True)
    keys = {c.key for c in result.citations}
    assert "henkelman_jonsson_neb_2000" in keys
    assert "smidstrup_idpp_2014" in keys
    assert "henkelman_uberuaga_jonsson_ci_neb_2000" in keys
    assert not list(result.warnings), result.warnings


def test_ci_neb_route_silent_when_disabled() -> None:
    """A plain NEB run (no climbing) must NOT include the CI-NEB
    paper — only the base improved-tangent + IDPP bundle."""
    db = load_default_database()
    result = db.assemble(method="uhf", basis="sto-3g", uses_neb=True)
    keys = {c.key for c in result.citations}
    assert "henkelman_uberuaga_jonsson_ci_neb_2000" not in keys


def test_soscf_route_fires_when_enabled() -> None:
    """``assemble(uses_soscf=True)`` surfaces Bacskay 1981 (quadratic
    SCF) + Neese 2000 (approximate SOSCF) — the second-order SCF path in
    cpp/src/soscf.hpp (RHFOptions.soscf_threshold). CLAUDE.md § 8."""
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g", uses_soscf=True)
    keys = {c.key for c in result.citations}
    assert "bacskay_qcscf_1981" in keys
    assert "neese_soscf_2000" in keys


def test_trah_route_fires_when_enabled() -> None:
    """``assemble(uses_trah=True)`` surfaces Helmich-Paris 2021 — the
    trust-region augmented-Hessian path in cpp/src/trah.hpp
    (RHFOptions.trah_threshold)."""
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g", uses_trah=True)
    keys = {c.key for c in result.citations}
    assert "helmich_paris_trah_2021" in keys


def test_soscf_trah_silent_when_disabled() -> None:
    """A plain SCF must not pull the second-order-SCF papers."""
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g")
    keys = {c.key for c in result.citations}
    assert "bacskay_qcscf_1981" not in keys
    assert "helmich_paris_trah_2021" not in keys


def test_tddft_route_fires_runge_gross_and_casida() -> None:
    """``assemble(uses_tddft=True)`` surfaces Runge-Gross 1984 + Casida
    1995 — the linear-response TDDFT formalism in python/vibeqc/tddft.py
    (run_tddft_casida). CLAUDE.md § 8."""
    db = load_default_database()
    result = db.assemble(
        method="rks", basis="def2-svp", functional="pbe", uses_tddft=True
    )
    keys = {c.key for c in result.citations}
    assert "runge_gross_1984" in keys
    assert "casida_tddft_1995" in keys
    # Without the TDA variant the Hirata-Head-Gordon paper does NOT fire.
    assert "hirata_headgordon_tda_1999" not in keys
    assert not list(result.warnings), result.warnings


def test_tddft_tda_variant_adds_hirata() -> None:
    """The Tamm-Dancoff variant (run_tddft_tda) fires Hirata-Head-Gordon
    1999 *in addition* to the Runge-Gross + Casida base bundle."""
    db = load_default_database()
    result = db.assemble(
        method="rks",
        basis="def2-svp",
        functional="pbe",
        uses_tddft=True,
        tddft_variant="tda",
    )
    keys = {c.key for c in result.citations}
    assert "runge_gross_1984" in keys
    assert "casida_tddft_1995" in keys
    assert "hirata_headgordon_tda_1999" in keys


def test_tddft_silent_when_disabled() -> None:
    """A plain ground-state run must not pull the TDDFT papers."""
    db = load_default_database()
    result = db.assemble(method="rks", basis="def2-svp", functional="pbe")
    keys = {c.key for c in result.citations}
    assert "runge_gross_1984" not in keys
    assert "casida_tddft_1995" not in keys


def test_cis_route_fires_foresman() -> None:
    """``assemble(uses_cis=True)`` surfaces Foresman-Head-Gordon-Pople-Frisch
    1992 — the wavefunction CIS excited states (vibeqc.excited / msindo_cis) and
    CIS excited-state gradients (vibeqc.excited_gradient). CLAUDE.md § 8."""
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g", uses_cis=True)
    keys = {c.key for c in result.citations}
    assert "foresman_cis_1992" in keys
    assert not list(result.warnings), result.warnings


def test_cis_silent_when_disabled() -> None:
    """A plain ground-state run must not pull the CIS paper."""
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g")
    keys = {c.key for c in result.citations}
    assert "foresman_cis_1992" not in keys


def test_msindo_cis_gradient_emits_citations(tmp_path: Path) -> None:
    """msindo_cis_gradient_fd(output=...) writes the .bibtex / .references
    siblings citing MSINDO + CIS, with libint suppressed (INDO uses no Gaussian
    integrals). CLAUDE.md § 8.5 — verify the citation reaches its destination."""
    from vibeqc.semiempirical.methods.msindo import msindo_cis_gradient_fd

    stem = tmp_path / "h2o_cis_grad"
    msindo_cis_gradient_fd(
        [8, 1, 1],
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]],
        1,
        spin="singlet",
        step=2e-3,
        output=stem,
    )
    bib = (tmp_path / "h2o_cis_grad.bibtex").read_text(encoding="utf-8")
    assert "foresman_cis_1992" in bib  # the CIS method paper
    assert "ahlswede_jug_msindo_1_1999" in bib  # MSINDO itself
    assert "valeev_libint" not in bib  # INDO: no Gaussian integrals


def test_conical_intersection_route_fires_levine() -> None:
    """``assemble(uses_conical_intersection=True)`` surfaces Levine-Coe-Martínez
    2008 — the penalty-function MECI optimizer (vibeqc.conical). CLAUDE.md § 8."""
    db = load_default_database()
    result = db.assemble(
        method="rhf", basis="sto-3g", uses_cis=True, uses_conical_intersection=True
    )
    keys = {c.key for c in result.citations}
    assert "levine_meci_2008" in keys
    assert "foresman_cis_1992" in keys  # the CIS states it couples
    assert not list(result.warnings), result.warnings


def test_conical_intersection_silent_when_disabled() -> None:
    """A plain CIS run must not pull the conical-intersection paper."""
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g", uses_cis=True)
    keys = {c.key for c in result.citations}
    assert "levine_meci_2008" not in keys


def test_msindo_meci_emits_citations(tmp_path: Path) -> None:
    """msindo_meci(output=...) writes siblings citing MSINDO + CIS + the
    conical-intersection penalty method. CLAUDE.md § 8.5."""
    from vibeqc.semiempirical.methods.msindo import msindo_meci

    stem = tmp_path / "h2o_meci"
    msindo_meci(
        [8, 1, 1],
        [[0.0, 0.0, 0.117], [0.0, 0.757, -0.467], [0.0, -0.757, -0.467]],
        lower_state=0,
        upper_state=1,
        spin="singlet",
        step=2e-3,
        gap_tol=1e-3,
        max_macro=2,
        max_micro=8,
        output=stem,
    )
    bib = (tmp_path / "h2o_meci.bibtex").read_text(encoding="utf-8")
    assert "levine_meci_2008" in bib  # the MECI optimizer
    assert "foresman_cis_1992" in bib  # the CIS states
    assert "ahlswede_jug_msindo_1_1999" in bib  # MSINDO itself


def test_mean_field_methods_have_no_method_route_warning(
    tmp_path: Path,
) -> None:
    """RHF / UHF / RKS / UKS must NOT produce a routing warning for
    the absence of a method-specific route — they are covered by the
    integral library + functional, by design."""
    db = load_default_database()
    for m in ("rhf", "uhf", "rks", "uks"):
        result = db.assemble(method=m, basis="sto-3g")
        bad = [w for w in result.warnings if "method" in w.lower()]
        assert not bad, (
            f"mean-field method {m!r} should not warn about a "
            f"missing method route; got {bad}"
        )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------


def test_write_bibtex_emits_one_entry_per_citation(tmp_path: Path) -> None:
    result = assemble(_plan(tmp_path, basis="6-31g*", functional="PBE"))
    target = write_bibtex(tmp_path / "job", result)
    assert target == (tmp_path / "job").with_suffix(".bibtex")
    body = target.read_text(encoding="utf-8")
    # One @<type>{... block per *printable* citation. write_bibtex emits
    # result.printable, not result.citations: entries flagged print = false
    # (e.g. eigen, link-time infrastructure) are confined to the .system
    # manifest and must not appear here.
    n_entries = body.count("@")
    assert n_entries >= len(result.printable)
    assert all(c.bibtex_key not in body for c in result.citations if not c.print)
    # bibtex_key of the software citation must appear.
    assert "peintinger_vibeqc" in body
    # libxc + PBE keys must appear in a PBE run.
    assert "lehtola_libxc_2018" in body
    assert "perdew_burke_ernzerhof_1996" in body


def test_write_references_emits_numbered_list(tmp_path: Path) -> None:
    result = assemble(_plan(tmp_path, basis="6-31g*", functional="PBE"))
    target = write_references(tmp_path / "job", result)
    assert target == (tmp_path / "job").with_suffix(".references")
    body = target.read_text(encoding="utf-8")
    assert "[1]" in body
    assert "[2]" in body
    assert "vibe-qc" in body.lower() or "vibeqc" in body.lower()
    # libxc + PBE citations appear in the plain text too.
    assert "Lehtola" in body
    assert "Perdew" in body


def test_format_references_block_starts_with_section_header() -> None:
    citations = (
        Citation(
            key="x",
            kind="article",
            bibtex_key="x_2020",
            authors=("Doe, Jane",),
            title="A study",
            year=2020,
            journal="J. Test",
            volume=1,
            pages="1--2",
        ),
    )
    block = format_references_block(citations)
    assert block.startswith("## References")
    assert "Doe, Jane" in block


def test_format_references_empty_list_is_graceful() -> None:
    block = format_references_block(())
    assert "## References" in block
    assert "no citations assembled" in block.lower()


def test_format_references_includes_warnings_in_plain_file() -> None:
    from vibeqc.output.citations.registry import AssembledCitations

    ac = AssembledCitations(citations=(), warnings=("synthetic warning",))
    body = format_references(ac)
    assert "synthetic warning" in body


# ---------------------------------------------------------------------------
# `print` flag — user-visibility gate on individual citations
# ---------------------------------------------------------------------------
#
# Per CLAUDE.md § 8 ("When you implement something citable") the optional
# `print` field on `[entries.<key>]` controls whether a citation lands in
# the user-facing references block or stays in the .system manifest only.
# These tests pin the contract end-to-end.


def test_citation_print_defaults_to_true():
    """Bare-bones Citation must default to print=True (backward compat)."""
    c = Citation(
        key="x",
        kind="article",
        bibtex_key="x_2020",
        authors=("Doe, J.",),
        title="T",
    )
    assert c.print is True


def test_to_jsonable_includes_print_flag():
    """Manifest serialisation must surface the flag for downstream consumers."""
    c = Citation(
        key="x",
        kind="article",
        bibtex_key="x_2020",
        authors=("Doe, J.",),
        title="T",
        print=False,
    )
    jb = c.to_jsonable()
    assert jb["print"] is False


def test_database_omits_print_field_means_true(tmp_path: Path):
    """An entry without `print` parses as printable — keeps the entire
    existing database backwards-compatible."""
    f = tmp_path / "db.toml"
    f.write_text(
        'schema_version = "1"\n'
        "[entries.foo]\n"
        'kind = "article"\n'
        'bibtex_key = "foo"\n'
        'authors = ["A, B"]\n'
        'title = "T"\n',
        encoding="utf-8",
    )
    db = load_database(f)
    assert db.entries()["foo"].print is True


def test_database_reads_print_false(tmp_path: Path):
    f = tmp_path / "db.toml"
    f.write_text(
        'schema_version = "1"\n'
        "[entries.foo]\n"
        'kind = "article"\n'
        'bibtex_key = "foo"\n'
        'authors = ["A, B"]\n'
        'title = "T"\n'
        "print = false\n",
        encoding="utf-8",
    )
    db = load_database(f)
    assert db.entries()["foo"].print is False


def test_database_rejects_non_boolean_print(tmp_path: Path):
    f = tmp_path / "db.toml"
    f.write_text(
        'schema_version = "1"\n'
        "[entries.foo]\n"
        'kind = "article"\n'
        'bibtex_key = "foo"\n'
        'authors = ["A, B"]\n'
        'title = "T"\n'
        'print = "no"\n',
        encoding="utf-8",
    )
    with pytest.raises(DatabaseError, match="'print' must be a boolean"):
        load_database(f)


def _two_citation_set(*, second_print: bool):
    """Build an AssembledCitations with two entries; second's print flag
    is the parameter under test."""
    from vibeqc.output.citations.registry import AssembledCitations

    visible = Citation(
        key="v",
        kind="article",
        bibtex_key="v_2020",
        authors=("Visible, Author",),
        title="Paper-relevant",
        year=2020,
        journal="J. Visible",
        volume=1,
        pages="1",
    )
    other = Citation(
        key="i",
        kind="software",
        bibtex_key="i_2024",
        authors=("Infra, Author",),
        title="Infrastructure",
        version="1.0",
        print=second_print,
    )
    return AssembledCitations(citations=(visible, other))


def test_printable_property_filters_to_print_true():
    ac = _two_citation_set(second_print=False)
    assert len(ac.citations) == 2
    assert len(ac.printable) == 1
    assert ac.printable[0].key == "v"


def test_printable_property_passes_through_when_all_true():
    ac = _two_citation_set(second_print=True)
    assert ac.printable == ac.citations


def test_eigen_is_internal_provenance_not_a_printed_reference():
    """Eigen is linked into the C++ core for every job, so it fires
    unconditionally from routes.libraries -- but it is link-time infrastructure,
    not a scientific dependency. It belongs in the .system manifest and must
    stay out of the references block a paper's Methods section copies from
    (CLAUDE.md § 8 step 3). This is the first real database entry exercising
    print = false; the tests above use synthetic entries."""
    db = load_default_database()
    result = db.assemble(method="rhf", basis="sto-3g")
    assert "eigen" in {c.key for c in result.citations}
    assert "eigen" not in {c.key for c in result.printable}
    assert not result.warnings
    rows = {r["key"]: r for r in citation_manifest_rows(result)}
    assert rows["eigen"]["print"] is False


def test_eigen_fires_for_a_semiempirical_job_too():
    """The core is linked regardless of method, so the unconditional route must
    not be accidentally gated on a Gaussian-integral job."""
    db = load_default_database()
    result = db.assemble(method="msindo", uses_integrals=False)
    assert "eigen" in {c.key for c in result.citations}
    assert "valeev_libint" not in {c.key for c in result.citations}


def test_write_references_omits_print_false(tmp_path: Path):
    ac = _two_citation_set(second_print=False)
    path = write_references(tmp_path / "job", ac)
    body = path.read_text(encoding="utf-8")
    assert "Visible, Author" in body
    assert "Infra, Author" not in body, body


def test_write_bibtex_omits_print_false(tmp_path: Path):
    ac = _two_citation_set(second_print=False)
    path = write_bibtex(tmp_path / "job", ac)
    body = path.read_text(encoding="utf-8")
    assert "v_2020" in body
    assert "i_2024" not in body, body


def test_format_references_block_omits_print_false():
    ac = _two_citation_set(second_print=False)
    block = format_references_block(ac)
    assert "Visible, Author" in block
    assert "Infra, Author" not in block, block


def test_raw_iterable_path_renders_all(tmp_path: Path):
    """When the writer gets a raw iterable (not an AssembledCitations),
    the caller has already chosen the subset — every entry renders, regardless
    of the print flag on individual items. This is the .system manifest path."""
    ac = _two_citation_set(second_print=False)
    # Pass the .citations tuple directly (raw iterable, bypasses .printable).
    path = write_references(tmp_path / "manifest", ac.citations)
    body = path.read_text(encoding="utf-8")
    assert "Visible, Author" in body
    assert "Infra, Author" in body


# ---------------------------------------------------------------------------
# CPCM / solvation routing — fires when uses_cpcm=True.
# ---------------------------------------------------------------------------


def test_cpcm_pulls_klamt_cossi_scalmani(tmp_path: Path) -> None:
    """run_job(..., solvent=...) sets uses_cpcm=True; the assembler
    must then surface the three CPCM-formulation papers."""
    plan = _plan(tmp_path, basis="def2-svp", functional="PBE")
    db = load_default_database()
    result = db.assemble_from_plan(plan, uses_cpcm=True)
    keys = {c.key for c in result.citations}
    assert "klamt_schuurmann_cosmo_1993" in keys
    assert "cossi_cpcm_2003" in keys
    assert "scalmani_frisch_csc_2010" in keys
    assert result.warnings == ()


def test_cpcm_off_does_not_pull_solvation_refs(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="def2-svp", functional="PBE")
    db = load_default_database()
    result = db.assemble_from_plan(plan)  # uses_cpcm defaults to False
    keys = {c.key for c in result.citations}
    assert "klamt_schuurmann_cosmo_1993" not in keys


def test_cpcm_unknown_variant_warns(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="sto-3g")
    db = load_default_database()
    result = db.assemble_from_plan(plan, uses_cpcm=True, solvent_variant="made-up-pcm")
    assert any("made-up-pcm" in w for w in result.warnings)


# ---------------------------------------------------------------------------
# SCF accelerator routing — non-DIIS accelerators must cite their paper.
# ---------------------------------------------------------------------------


def test_adiis_route_pulls_hu_yang(tmp_path: Path) -> None:
    plan = _plan(tmp_path, basis="sto-3g")
    db = load_default_database()
    result = db.assemble_from_plan(plan, scf_accelerator="adiis")
    bibtex_keys = {c.bibtex_key for c in result.citations}
    assert "hu_yang_adiis_2010" in bibtex_keys


def test_kdiis_route_pulls_kollmar(tmp_path: Path) -> None:
    """KDIIS must cite Kollmar 1997, the originating publication for the
    orbital-rotation-gradient DIIS variant.

    Garza & Scuseria 2012 ("Comparison of self-consistent field
    convergence acceleration techniques") is a comparative study of
    ADIIS / LIST / EDIIS+DIIS whose text never mentions Kollmar or
    KDIIS. It was routed here in error until 2026-07-10; keep it off
    this route.
    """
    plan = _plan(tmp_path, basis="sto-3g")
    db = load_default_database()
    result = db.assemble_from_plan(plan, scf_accelerator="kdiis")
    bibtex_keys = {c.bibtex_key for c in result.citations}
    assert "kollmar_kdiis_1997" in bibtex_keys
    assert "garza_scuseria_2012" not in bibtex_keys


def test_ediis_diis_route_pulls_garza_scuseria(tmp_path: Path) -> None:
    """The EDIIS+DIIS hybrid is what Garza & Scuseria 2012 actually
    recommends, so that is the route their entry belongs on."""
    plan = _plan(tmp_path, basis="sto-3g")
    db = load_default_database()
    result = db.assemble_from_plan(plan, scf_accelerator="ediis_diis")
    bibtex_keys = {c.bibtex_key for c in result.citations}
    assert "garza_scuseria_2012" in bibtex_keys
    assert "kudin_ediis_2002" in bibtex_keys


# ---------------------------------------------------------------------------
# Active-space + post-SCF method routes.
# ---------------------------------------------------------------------------


def test_selected_ci_pulls_cipsi() -> None:
    db = load_default_database()
    result = db.assemble(method="selected_ci", basis="sto-3g")
    keys = {c.key for c in result.citations}
    assert "huron_malrieu_cipsi_1973" in keys
    assert "holmes_tubman_umrigar_shci_2016" in keys


def test_dmrg_pulls_white_and_chan() -> None:
    db = load_default_database()
    result = db.assemble(method="dmrg", basis="sto-3g")
    keys = {c.key for c in result.citations}
    assert "white_dmrg_1992" in keys
    assert "chan_headgordon_dmrg_2002" in keys


def test_v2rdm_pulls_mazziotti() -> None:
    db = load_default_database()
    result = db.assemble(method="v2rdm", basis="sto-3g")
    keys = {c.key for c in result.citations}
    assert "mazziotti_v2rdm_2011" in keys


def test_mp2_route_fires_moller_plesset() -> None:
    db = load_default_database()
    result = db.assemble(method="mp2", basis="def2-svp")
    keys = {c.key for c in result.citations}
    assert "moller_plesset_1934" in keys


@pytest.mark.parametrize(
    ("entry", "doi"),
    [
        ("head_gordon_direct_mp2_1988", "10.1016/0009-2614(88)85250-3"),
        ("frisch_semidirect_mp2_1990", "10.1016/0009-2614(90)80030-H"),
        ("bintrim_direct_df_mp2_2022", "10.1021/acs.jctc.2c00640"),
    ],
)
def test_bounded_mp2_algorithm_citations_are_runtime_gated(
    entry: str,
    doi: str,
) -> None:
    db = load_default_database()
    ordinary = db.assemble(method="mp2", basis="def2-svp")
    assert entry not in {c.key for c in ordinary.citations}

    result = db.assemble(
        method="mp2",
        basis="def2-svp",
        extra_entries=[entry],
    )
    by_key = {c.key: c for c in result.citations}
    assert by_key[entry].doi == doi
    assert by_key[entry].print is True


def test_scs_mp2_route_adds_grimme() -> None:
    db = load_default_database()
    result = db.assemble(method="scs-mp2", basis="def2-svp")
    keys = {c.key for c in result.citations}
    assert "moller_plesset_1934" in keys
    assert "grimme_scs_mp2_2003" in keys


def test_pwpb95_route_carries_dh_bundle() -> None:
    db = load_default_database()
    result = db.assemble(method="pwpb95", basis="def2-tzvp")
    keys = {c.key for c in result.citations}
    assert "goerigk_grimme_pwpb95_2011" in keys
    assert "moller_plesset_1934" in keys


def test_wb97x_d_route_pulls_chg_dispersion() -> None:
    db = load_default_database()
    result = db.assemble(method="wb97x-d", basis="def2-svp")
    keys = {c.key for c in result.citations}
    assert "chai_headgordon_wb97xd_2008" in keys
    assert "grimme_dftd2_2006" in keys


# ---------------------------------------------------------------------------
# emit_citations helper — used by standalone runners (run_b2plyp etc.)
# ---------------------------------------------------------------------------


def test_emit_citations_writes_bibtex_and_references(tmp_path: Path) -> None:
    from vibeqc.output.citations import emit_citations

    stem = tmp_path / "wb97xd_job"
    citations, bib_path, ref_path, block = emit_citations(
        stem,
        method="wb97x-d",
        basis="def2-svp",
        functional="wb97x-d",
    )
    assert bib_path == stem.with_suffix(".bibtex")
    assert ref_path == stem.with_suffix(".references")
    assert bib_path.is_file()
    assert ref_path.is_file()
    assert "## References" in block
    body = bib_path.read_text(encoding="utf-8")
    assert "chai_headgordon_wb97xd_2008" in body
    assert "grimme_dftd2_2006" in body
    assert "peintinger_vibeqc" in body


# ---------------------------------------------------------------------------
# Forward-looking routes (roadmap features pre-staged in the 2026-06 citation
# expansion). These fire through the standard method= / functional= / basis=
# triggers and the new scf_guess / properties / acceleration / numerics /
# gradient / hessian / smearing_method hooks. They must stay silent by default.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method_key,expected",
    [
        ("cc2", {"christiansen_cc2_1995"}),
        ("cc3", {"koch_cc3_1997"}),
        (
            "dlpno-ccsd(t)",
            {"riplinger_dlpno_2013", "riplinger_sparsemaps_2016"},
        ),
        ("eom-ip-ccsd", {"stanton_bartlett_eomcc_1993", "stanton_gauss_eomip_1994"}),
        ("ccsdt", {"noga_bartlett_ccsdt_1987"}),
        ("mrci+q", {"werner_knowles_mrci_1988", "langhoff_davidson_1974"}),
    ],
)
def test_forward_method_ladder_routes(method_key, expected) -> None:
    db = load_default_database()
    keys = {c.key for c in db.assemble(method=method_key, basis="cc-pvtz").citations}
    assert expected <= keys, f"{method_key}: missing {expected - keys}"


@pytest.mark.parametrize(
    "accel,expected",
    [
        ("anderson", "anderson_mixing_1965"),
        ("broyden", "broyden_1965"),
        ("ot", "vandevondele_ot_2003"),
        ("oda", "cances_lebris_oda_2000"),
        ("r_cdiis", "chupin_adaptive_diis_2021"),
        ("ad_cdiis", "chupin_adaptive_diis_2021"),
        ("adiis_diis", "hu_yang_adiis_2010"),
    ],
)
def test_forward_scf_accelerator_routes(accel, expected) -> None:
    db = load_default_database()
    keys = {
        c.key
        for c in db.assemble(
            method="rhf", basis="sto-3g", scf_accelerator=accel
        ).citations
    }
    assert expected in keys


def test_dft_integration_grid_fires_for_any_functional() -> None:
    """Becke 1988 multicenter integration + Treutler-Ahlrichs 1995 fire for
    any KS-DFT run (the XC grid), but not for pure Hartree-Fock."""
    db = load_default_database()
    dft = {
        c.key
        for c in db.assemble(method="rks", basis="def2-svp", functional="pbe").citations
    }
    assert {"becke_grid_1988", "treutler_ahlrichs_1995"} <= dft
    hf = {c.key for c in db.assemble(method="rhf", basis="sto-3g").citations}
    assert "becke_grid_1988" not in hf


def test_scf_guess_routes() -> None:
    db = load_default_database()
    sad = {
        c.key
        for c in db.assemble(method="rhf", basis="sto-3g", scf_guess="sad").citations
    }
    assert "vanlenthe_sad_2006" in sad
    sap = {
        c.key
        for c in db.assemble(method="rhf", basis="sto-3g", scf_guess="sap").citations
    }
    assert {"lehtola_sap_2019", "lehtola_visscher_engel_sap_2020"} <= sap
    # silent by default
    plain = {c.key for c in db.assemble(method="rhf", basis="sto-3g").citations}
    assert "vanlenthe_sad_2006" not in plain


def test_properties_routes() -> None:
    db = load_default_database()
    r = db.assemble(
        method="rks",
        basis="def2-svp",
        functional="pbe",
        properties=["nmr_shielding", "nci", "hirshfeld"],
    )
    keys = {c.key for c in r.citations}
    assert {"ditchfield_giao_1974", "wolinski_hinton_pulay_giao_1990"} <= keys
    assert "johnson_nci_2010" in keys
    assert "hirshfeld_1977" in keys
    # silent by default
    plain = {
        c.key
        for c in db.assemble(method="rks", basis="def2-svp", functional="pbe").citations
    }
    assert "johnson_nci_2010" not in plain

    # COOP / COHP routes
    coop_keys = {
        c.key
        for c in db.assemble(
            method="rks",
            basis="def2-svp",
            functional="pbe",
            properties=["coop"],
        ).citations
    }
    assert "hughbanks_hoffmann_coop_1983" in coop_keys
    assert "dronskowski_cohp_1993" in coop_keys
    assert "deringer_cohp_2011" in coop_keys
    assert "maintz_lobster3_2016" in coop_keys

    cohp_keys = {
        c.key
        for c in db.assemble(
            method="rks",
            basis="def2-svp",
            functional="pbe",
            properties=["cohp"],
        ).citations
    }
    assert "hughbanks_hoffmann_coop_1983" not in cohp_keys  # COOP original only on coop
    assert "dronskowski_cohp_1993" in cohp_keys
    assert "deringer_cohp_2011" in cohp_keys
    assert "maintz_lobster3_2016" in cohp_keys
    # silent by default
    plain = {
        c.key
        for c in db.assemble(method="rks", basis="def2-svp", functional="pbe").citations
    }
    assert "dronskowski_cohp_1993" not in plain


def test_bond_analysis_routes_pin_dois() -> None:
    """v0.22.0 BOND property routes fire their defining papers, DOIs pinned.

    DOIs pinned mechanically per CLAUDE.md section 8 (the pob-* audit
    lesson) -- every DOI below was verified against Crossref / the arXiv
    abstract page on 2026-07-11 when the original hand-entered metadata
    for four of these entries turned out to be wrong or hallucinated.
    """
    db = load_default_database()
    expected_by_route = {
        "mayer_bond_order": {
            "mayer_bond_order_1983": "10.1016/0009-2614(83)80005-0",
            "mayer_bond_order_2007": "10.1002/jcc.20494",
        },
        "wiberg": {
            "wiberg_bond_index_1968": "10.1016/0040-4020(68)88057-3",
        },
        "delocalization_index": {
            "matito_esi_2007": "10.1039/B605086G",
            "outeiral_di_2018": "10.1039/C8SC01338A",
        },
        "morokuma_eda": {
            "morokuma_eda_1971": "10.1063/1.1676210",
        },
        "entanglement": {
            "legeza_entanglement_2003": "10.1103/PhysRevB.68.195116",
        },
        "multiorbital_correlation": {
            "szalay_multiorbital_2017": "10.1038/s41598-017-02447-z",
        },
        "entanglement_bonds": {
            # Version of record since #523 (Nat. Commun. 17, 4732 (2026));
            # the entry recorded the arXiv:2501.15699 preprint until then.
            "entanglement_bonds_2025": "10.1038/s41467-026-73527-w",
        },
        "nbo": {
            "reed_weinhold_npa_1985": "10.1063/1.449486",
            "foster_weinhold_nho_1980": "10.1021/ja00544a007",
            "reed_curtiss_weinhold_nbo_1988": "10.1021/cr00088a005",
        },
    }
    for route, expected in expected_by_route.items():
        by_key = {
            c.key: c
            for c in db.assemble(
                method="rhf",
                basis="sto-3g",
                properties=[route],
            ).citations
        }
        for key, doi in expected.items():
            assert key in by_key, f"{key} did not fire for properties=[{route!r}]"
            assert by_key[key].doi == doi
            assert by_key[key].print is True  # reaches .out / .bibtex
    # silent by default -- no bond-analysis paper on a plain SCF
    plain_scf = {
        c.key for c in db.assemble(method="rhf", basis="sto-3g").citations
    }
    for expected in expected_by_route.values():
        for key in expected:
            assert key not in plain_scf


def test_acceleration_and_numerics_routes() -> None:
    db = load_default_database()
    chi = db.assemble(method="rhf", basis="sto-3g", numerics=["chi_occupied_symmetry"])
    assert "casassa_symmetry_wannier_2006" in {c.key for c in chi.citations}
    group = db.assemble(method="rhf", basis="sto-3g", numerics=["chi_torus_symmetry_group"])
    assert {"casassa_symmetry_wannier_2006", "dovesi_symmetry_lcao_1986"} <= {
        c.key for c in group.citations
    }
    assert "casassa_symmetry_wannier_2006" not in {
        c.key for c in db.assemble(method="rhf", basis="sto-3g").citations
    }
    acc = {
        c.key
        for c in db.assemble(
            method="rks",
            basis="def2-svp",
            functional="pbe",
            acceleration=["rij", "rijcosx"],
        ).citations
    }
    assert {"whitten_1973", "eichkorn_rij_1995", "neese_rijcosx_2009"} <= acc
    num = {
        c.key
        for c in db.assemble(
            method="rhf", basis="cc-pvqz", numerics=["pivoted_cholesky"]
        ).citations
    }
    assert "lehtola_pivoted_cholesky_2019" in num
    gr = {
        c.key
        for c in db.assemble(
            method="rhf", basis="sto-3g", numerics=["generalized_regular_kgrid"]
        ).citations
    }
    assert {"wisesa_grids_2016", "wang_grids_2021"} <= gr
    kdb = {
        c.key
        for c in db.assemble(
            method="rhf", basis="sto-3g", numerics=["kpoint_database"]
        ).citations
    }
    assert {"monkhorst_pack_1976", "mehl_kpoint_database_1990"} <= kdb
    lob = {
        c.key
        for c in db.assemble(
            method="rhf", basis="sto-3g", numerics=["lobpcg"]
        ).citations
    }
    assert "knyazev_lobpcg_2001" in lob
    # The k-point rows that KPoints.citation_numerics emits. The producer side
    # is pinned by tests/test_kpoints.py; this pins that the routes answer.
    kppra = {
        c.key
        for c in db.assemble(
            method="rhf", basis="sto-3g", periodic=True, numerics=["kppra"]
        ).citations
    }
    assert "curtarolo_aflow_2012" in kppra
    hpkot = {
        c.key
        for c in db.assemble(
            method="rhf", basis="sto-3g", periodic=True, numerics=["hpkot_band_path"]
        ).citations
    }
    assert "hinuma_seekpath_2017" in hpkot


def test_gradient_and_hessian_driver_routes() -> None:
    db = load_default_database()
    g = {
        c.key
        for c in db.assemble(method="rhf", basis="sto-3g", uses_gradient=True).citations
    }
    assert {"pulay_forces_1969", "feynman_forces_1939"} <= g
    h = {
        c.key
        for c in db.assemble(method="rhf", basis="sto-3g", uses_hessian=True).citations
    }
    assert "wilson_decius_cross_1955" in h
    # silent by default
    plain = {c.key for c in db.assemble(method="rhf", basis="sto-3g").citations}
    assert "pulay_forces_1969" not in plain


def test_smearing_method_variant_route() -> None:
    """uses_smearing fires Mermin; a non-Fermi-Dirac broadening additionally
    cites its defining paper via smearing_method."""
    db = load_default_database()
    mp = {
        c.key
        for c in db.assemble(
            method="rks",
            basis="def2-svp",
            functional="pbe",
            periodic=True,
            uses_smearing=True,
            smearing_method="methfessel_paxton",
        ).citations
    }
    assert "mermin_finite_temperature_dft_1965" in mp
    assert "methfessel_paxton_1989" in mp


def test_forward_basis_family_routes() -> None:
    db = load_default_database()
    for basis, expect in [
        ("pcseg-2", "jensen_pcseg_2014"),
        ("pcs-2", "jensen_pcs_2008"),
        ("lanl2dz", "hay_wadt_lanl_1985"),
        ("ano-r2", "zobel_ano_r_2020"),
        ("cc-pv(t+d)z", "dunning_tightd_2001"),
        ("x2c-tzvpall", "pollak_weigend_x2c_2017"),
    ]:
        keys = {c.key for c in db.assemble(method="rhf", basis=basis).citations}
        assert expect in keys, f"basis {basis}: missing {expect}"
        bad = [
            w
            for w in db.assemble(method="rhf", basis=basis).warnings
            if basis.lower() in w.lower()
        ]
        assert not bad, f"basis {basis} warned: {bad}"


# ---------------------------------------------------------------------------
# Producer guard for the χ-CCM (aiccm2026dev-b) post-HF method routes.
#
# ``test_required_method_route_actually_fires`` proves a key resolves through
# ``routes.methods`` when something passes it to ``assemble()``. It cannot see
# whether anything ever does, nor whether a label some driver really stamps is
# missing from the table — and a ``routes.methods`` miss is silent (step 5b of
# ``CitationDatabase.assemble``), so a gap in that direction drops the method
# papers with no warning at all.
#
# That is not hypothetical: ``run_aiccm2026dev_b_uccsd`` / ``_uccsd_t`` stamp
# ``aiccm2026dev-b-uccsd`` / ``...-uccsd(t)`` and had no route until
# 2026-08-22, so those two would have dropped both CCM papers plus
# Purvis-Bartlett and Raghavachari the moment a citation surface was wired up.
#
# This guard closes that direction by deriving the producible label set from
# the driver module's own source. Scope is deliberately the post-HF module:
# ``periodic/chi/scf.py`` also stamps ``backend`` labels
# (``-four_center`` / ``-ri`` / ``-rijcosx``), but those describe the SCF
# two-electron backend, and the runner cites those jobs through the bare
# ``aiccm2026dev-b`` key rather than the label.
# ---------------------------------------------------------------------------

_B_POSTHF_SOURCE = (
    Path(__file__).resolve().parents[1]
    / "python"
    / "vibeqc"
    / "periodic"
    / "chi"
    / "posthf.py"
)
_B_LABEL_PREFIX = "aiccm2026dev-b-"


def _const_strings(node: ast.AST) -> set[str] | None:
    """Every string value ``node`` can evaluate to, or None if not static.

    Handles the two shapes the driver module uses to pick a label part:
    a plain literal, and ``"a" if cond else "b"``.
    """

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return {node.value}
    if isinstance(node, ast.IfExp):
        body = _const_strings(node.body)
        orelse = _const_strings(node.orelse)
        if body is None or orelse is None:
            return None
        return body | orelse
    return None


def _resolve_label_expr(
    node: ast.AST,
    names: dict[str, set[str]],
) -> set[str] | None:
    """Expand a ``backend=`` expression to the labels it can produce."""

    direct = _const_strings(node)
    if direct is not None:
        return direct
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = [""]
        for piece in node.values:
            if isinstance(piece, ast.Constant) and isinstance(piece.value, str):
                options = {piece.value}
            elif isinstance(piece, ast.FormattedValue) and isinstance(
                piece.value, ast.Name
            ):
                options = names.get(piece.value.id)
                if options is None:
                    return None
            else:
                return None
            parts = [prefix + tail for prefix in parts for tail in options]
        return set(parts)
    return None


def _producible_b_posthf_labels() -> set[str]:
    """Labels ``vibeqc.periodic.chi.posthf`` can stamp on ``backend``.

    Fails loudly rather than silently under-reporting: an unresolvable
    ``backend=`` expression raises, so a future refactor into a shape this
    reader cannot follow surfaces as a test failure asking for a human look,
    not as a quietly shrunken guard.
    """

    return _producible_labels((_B_POSTHF_SOURCE,), _B_LABEL_PREFIX)


def _producible_labels(
    sources: "tuple[Path, ...]",
    prefix: str,
) -> set[str]:
    """Labels the given modules can stamp on ``backend``, filtered by prefix.

    Generalised from the -b reader so the Γ-CCM (-a) correlation drivers get
    the same protection. Unlike χ, whose post-HF drivers live in one module,
    the -a labels are spread across ``periodic/ccm/{mp2,ump2,...}.py``, so this
    takes several sources.
    """

    labels: set[str] = set()
    for source in sources:
        labels |= _labels_in_source(source, prefix)
    return labels


def _labels_in_source(source: "Path", prefix: str) -> set[str]:
    tree = ast.parse(source.read_text(encoding="utf-8"))
    labels: set[str] = set()

    for scope in ast.walk(tree):
        if not isinstance(
            scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)
        ):
            continue

        # String-valued locals in this scope, e.g.
        #   method = "ccsd(t)" if with_triples else "ccsd"
        #   prefix = "dlpno-" if local else ""
        names: dict[str, set[str]] = {}
        for node in ast.walk(scope):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                values = _const_strings(node.value)
                if isinstance(target, ast.Name) and values is not None:
                    names[target.id] = values

        for node in ast.walk(scope):
            # Dataclass field default:  backend: str = "aiccm2026dev-b-ri-mp2"
            if (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == "backend"
                and node.value is not None
            ):
                expr: ast.AST | None = node.value
            # Constructor / dataclasses.replace keyword:  backend=<expr>
            elif isinstance(node, ast.keyword) and node.arg == "backend":
                expr = node.value
            else:
                continue

            resolved = _resolve_label_expr(expr, names)
            if resolved is None:
                raise AssertionError(
                    "cannot statically resolve a backend= expression at "
                    f"{source.name}:{getattr(expr, 'lineno', '?')}. "
                    "Extend _resolve_label_expr so this guard keeps covering "
                    "the AICCM post-HF citation routes."
                )
            labels |= {v for v in resolved if v.startswith(prefix)}

    return labels


# ---------------------------------------------------------------------------
# The same guard, both directions, for the Γ-CCM (-a) correlation drivers.
#
# Until M4b (#778) the -a line had no compound rows at all, and database.toml
# said so explicitly: they would have been "guesswork" because nothing stamped
# such a label. Now the drivers stamp and the rows exist, so the two sides can
# drift exactly as the -b pair was written to prevent -- with the extra wrinkle
# that ONE result class serves TWO lineages (run_ccm_mp2 stamps the bare
# four-centre key, run_ccm_ri_mp2 replaces it with the neutral-RI one), so a
# regression here does not merely drop citations, it can attribute a
# neutral-RI number to the union-and-weight papers. That is the R1/D89
# conflation, and it would be invisible in any energy test.
# ---------------------------------------------------------------------------

_A_POSTHF_SOURCES = tuple(
    Path(__file__).resolve().parents[1] / "python" / "vibeqc" / "periodic" / "ccm" / name
    for name in ("mp2.py", "ump2.py", "ccsd.py", "uccsd.py",
                 "dlpno.py", "dlpno_ump2.py", "dlpno_ccsd.py",
                 "dlpno_uccsd.py")
)
_A_LABEL_PREFIX = "aiccm2026dev-a-"


def _producible_a_posthf_labels() -> set[str]:
    return _producible_labels(_A_POSTHF_SOURCES, _A_LABEL_PREFIX)


def test_a_posthf_backend_labels_all_have_routes() -> None:
    """Every label the Γ-CCM correlation drivers stamp must carry a route."""

    produced = _producible_a_posthf_labels()
    assert produced, "found no -a backend labels — the AST reader has gone stale"

    routes = load_default_database()._routes["methods"]
    missing = sorted(label for label in produced if label not in routes)
    assert not missing, (
        "periodic/ccm stamps backend labels with no routes.methods row: "
        f"{missing}. assemble() would drop their method citations silently — "
        "add a row to database.toml."
    )


def test_a_posthf_routes_all_have_a_producer() -> None:
    """And no compound -a row without a driver that stamps it.

    database.toml used to carry a note that the -a compound keys were absent
    "on purpose: -a has no producer emitting such backend labels, so their
    spelling would be guesswork". This is what keeps that honest now that they
    exist: a row nothing can emit is guesswork again.
    """

    produced = _producible_a_posthf_labels()
    routes = load_default_database()._routes["methods"]
    compound = {key for key in routes if key.startswith(_A_LABEL_PREFIX)}
    orphans = sorted(compound - produced)
    assert not orphans, (
        f"database.toml has compound -a rows nothing stamps: {orphans}. "
        "Either a driver stopped stamping, or the row was written ahead of "
        "the code that would emit it."
    )


def test_a_bare_and_ri_lineages_never_share_a_route() -> None:
    """The bare four-centre and neutral-RI keys must stay distinct rows.

    Ruling R1 / D89: these are different constructions, not two spellings of
    one. Collapsing them would cite the union-and-weight papers for a neutral
    fitted-torus number.
    """

    routes = load_default_database()._routes["methods"]
    bare, ri = routes["aiccm2026dev-a-mp2"], routes["aiccm2026dev-a-ri-mp2"]
    assert bare != ri, "the bare and neutral-RI MP2 rows have become identical"
    # The RI row owes the fitting machinery the bare one does not.
    assert "feyereisen_rimp2_1993" in ri
    assert "feyereisen_rimp2_1993" not in bare
    # Both owe the CCM lineage.
    for row in (bare, ri):
        assert "bredow_geudtner_jug_ccm_2001" in row
        assert "peintinger_ccm_2014" in row


def test_b_posthf_backend_labels_all_have_routes() -> None:
    """Every label the -b post-HF drivers stamp must carry a method route.

    Direction the ``_REQUIRED_METHODS`` tests cannot check: a produced label
    with no row resolves silently to the generic periodic set, dropping the
    method papers CLAUDE.md § 8 makes mandatory.
    """

    produced = _producible_b_posthf_labels()
    assert produced, "found no backend labels — the AST reader has gone stale"

    routes = load_default_database()._routes["methods"]
    missing = sorted(label for label in produced if label not in routes)
    assert not missing, (
        "periodic/chi/posthf.py stamps backend labels with no "
        f"routes.methods row: {missing}. assemble() would drop their method "
        "citations silently — add a row to database.toml."
    )


def test_b_posthf_routes_all_have_a_producer() -> None:
    """And the reverse: no compound -b row without a driver that stamps it.

    Keeps the table honest about the difference between a row pre-positioned
    for code that exists (these) and a row nothing will ever emit.
    """

    produced = _producible_b_posthf_labels()
    routes = load_default_database()._routes["methods"]
    compound = {key for key in routes if key.startswith(_B_LABEL_PREFIX)}
    orphans = sorted(compound - produced)
    assert not orphans, (
        f"routes.methods carries compound χ-CCM keys with no producer: "
        f"{orphans}. Either a driver stopped stamping them or the spelling "
        "drifted; remove the row or fix the spelling."
    )


# ---------------------------------------------------------------------------
# AICCM front door M0 (handovers/HANDOVER_AICCM_STANDARD_METHOD.md): ruling R1
# at the database level.
# ---------------------------------------------------------------------------


def test_neutral_producers_share_one_lineage_row() -> None:
    """R1: real-gamma and neutral-bloch produce the same neutral Γ-CCM
    Hamiltonian (Theorem 1), so their routes.methods rows must stay identical
    and equal to the CCM lineage pair. Producer-specific machinery (the GDF
    stack, forces) rides on uses_gdf / uses_gradient, never on these rows."""
    from vibeqc.output.citations.registry import load_default_database

    routes = load_default_database()._routes["methods"]
    lineage = ("bredow_geudtner_jug_ccm_2001", "peintinger_ccm_2014")
    assert routes["real-gamma"] == lineage
    assert routes["neutral-bloch"] == lineage
    assert routes["real-gamma"] == routes["aiccm2026dev-b"], (
        "the neutral producers and the chi line share the CCM lineage row"
    )


@pytest.mark.parametrize("method", ["rohf", "roks"])
@pytest.mark.parametrize("guess", ["auto", "sap", "sad", "patom", "hueckel", "minao"])
def test_ro_guess_execution_citation_routes(method, guess):
    from vibeqc import Atom, Molecule
    from vibeqc.rohf import ROHFOptions
    from vibeqc.runner import _detect_scf_guess
    mol = Molecule([Atom(8, [0., 0., 0.])], multiplicity=3)
    route = _detect_scf_guess(method, mol, ROHFOptions(initial_guess=guess))
    assembled = load_default_database().assemble(
        method=method, basis="sto-3g", scf_guess=route,
    )
    keys = {c.key for c in assembled.citations}
    expected = {
        "auto": "vanlenthe_sad_2006", "sad": "vanlenthe_sad_2006",
        "patom": "vanlenthe_sad_2006", "sap": "lehtola_sap_2019",
        "hueckel": "hoffmann_extended_huckel_1963", "minao": "knizia_ibo_2013",
    }
    assert expected[guess] in keys


@pytest.mark.parametrize("periodic,restricted", [(False, False), (True, False), (True, True)])
def test_auto_capabilities_share_execution_and_citation_resolution(periodic, restricted):
    from vibeqc import Atom, Molecule, InitialGuess
    from vibeqc.guess import select_initial_guess
    molecule = Molecule([Atom(1, [0., 0., 0.]), Atom(1, [0., 0., 1.4])])
    kinds = (InitialGuess.HCORE,) if restricted else tuple(InitialGuess.__members__.values())
    selection = select_initial_guess(molecule, "AUTO", is_periodic=periodic, supported=kinds)
    expected = InitialGuess.HCORE if restricted else (InitialGuess.SAD if periodic else InitialGuess.PATOM)
    assert selection.effective == expected
    result = load_default_database().assemble(method="rhf", basis="sto-3g", scf_guess=selection.effective.name.lower())
    keys = {citation.key for citation in result.citations}
    assert ("vanlenthe_sad_2006" in keys) == (expected != InitialGuess.HCORE)
    assert "lehtola_sap_2019" not in keys


@pytest.mark.parametrize("guess", ["sad", "patom"])
def test_ecp_atomic_guess_cites_construction_and_actual_basis_operator(guess):
    keys = {c.key for c in load_default_database().assemble(
        method="rhf", basis="def2-svp", scf_guess=guess, uses_ecp=True,
    ).citations}
    assert {"vanlenthe_sad_2006", "andrae_ecp_1990", "shaw_gilbert_libecpint"} <= keys

def test_iepa_pno_norm_cites_its_own_paper(tmp_path: Path) -> None:
    """`pno_norm="iepa"` is the LPNO-CEPA convention, and cites it (#701).

    `tcut_pno` cuts pair-density occupation numbers, so which density was
    used is part of the method actually run. The IEPA norm comes from Neese,
    Wennmohs and Hansen, J. Chem. Phys. 130, 114108 (2009), Eqs. 18-19, which
    is a different paper from `neese_lpno_2009` (the LPNO-CCSD paper,
    J. Chem. Phys. 131, 064103).
    """
    db = load_default_database()
    plan = _plan(tmp_path, basis="def2-svp", method="dlpno-ccsd")
    result = db.assemble_from_plan(plan, pno_norm="iepa")
    by_key = {citation.key: citation for citation in result.citations}

    citation = by_key["neese_lpno_cepa_2009"]
    assert citation.doi == "10.1063/1.3086717"
    assert citation.print is True
    assert citation in result.printable
    assert result.warnings == ()
    # Not the LPNO-CCSD paper, which carries the other 2009 DOI.
    assert by_key.keys().isdisjoint({"neese_lpno_2009"})


@pytest.mark.parametrize("norm", [None, "legacy", "mp2"])
def test_non_iepa_pno_norms_do_not_pull_the_lpno_cepa_paper(
    tmp_path: Path, norm: str | None
) -> None:
    """The default density is vibe-qc's own, so it attributes nothing.

    `"legacy"` deliberately has no `[routes.pno_norm]` row: routing it would
    attribute vibe-qc's own choice of density to a paper that does not make
    it. `"mp2"` is Riplinger and Neese's Eq. 23, already cited by the DLPNO
    method route. Neither may drag in the LPNO-CEPA paper, and neither may
    warn about a missing route.
    """
    db = load_default_database()
    plan = _plan(tmp_path, basis="def2-svp", method="dlpno-ccsd")
    result = db.assemble_from_plan(plan, pno_norm=norm)
    keys = {citation.key for citation in result.citations}

    assert "neese_lpno_cepa_2009" not in keys
    assert "riplinger_dlpno_2013" in keys  # the DLPNO route, always present
    assert result.warnings == ()
