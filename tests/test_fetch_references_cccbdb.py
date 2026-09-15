"""``vibeqc.fetch.references`` — CCCBDB parser + client tests.

Pinned-replay tests against ``tests/data/cccbdb/h2o_exp2.html``,
captured live from ``cccbdb.nist.gov/exp2x.asp?casno=7732185`` on
2026-05-09. Tests must NEVER hit the live network — the client's
``html_path=`` escape hatch loads HTML from disk.

Pins:
  1. Parser extracts every property we promise (thermochem,
     vibrational, IE, dipole, polarizability, geometry).
  2. ``ExperimentalReference`` constructed from parsed data carries
     full provenance (CCCBDB + NIST DOI + CC0-equivalent license).
  3. Atomization energy is derived from Hf°(0K) and matches the
     CODATA D₀ for H₂O within rounding.
  4. ``MoleculeNotFound`` raised on a 500/404 page (CCCBDB returns
     200 with an error skeleton when the CAS doesn't match).
  5. Cache round-trip: a fetched reference re-parses identically
     when read back from cache (parser is deterministic).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

# The CCCBDB parser imports beautifulsoup4 + the lxml tree builder (the
# ``[fetch]`` optional extra, pulled in below via
# ``vibeqc.fetch.references.parsers`` / ``client_cccbdb``). Skip this whole
# module when they are absent so ``pytest --collect-only`` stays green on a
# base install / the CI runner, which does not install ``[fetch]``.
pytest.importorskip("bs4")
pytest.importorskip("lxml")

from examples.regression.core.spec import (
    ExperimentalReference,
    Provenance,
)
from vibeqc.fetch.cache import FetchCache, FetchCacheMiss, SOURCE_DB_CCCBDB
from vibeqc.fetch.references.atomic_enthalpies import (
    atomic_h_f_0k,
    derive_atomization_kj_per_mol,
)
from vibeqc.fetch.references.canonical_molecules import (
    CANONICAL_MOLECULES,
    composition_from_formula,
    find,
)
from vibeqc.fetch.references.client_cccbdb import (
    MoleculeNotFound,
    fetch_cccbdb,
)
from vibeqc.fetch.references.parsers import (
    CCCBDBNoData,
    parse_exp2x_html,
)


H2O_HTML_PATH = Path(__file__).resolve().parent / "data" / "cccbdb" / "h2o_exp2.html"


# ---------------------------------------------------------------------------
# Parser — pinned values from the H2O snapshot
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2o_parsed():
    html = H2O_HTML_PATH.read_text(encoding="iso-8859-1")
    return parse_exp2x_html(html)


def test_parser_identity(h2o_parsed):
    assert h2o_parsed["formula"] == "H2O"
    assert h2o_parsed["name"] == "Water"


def test_parser_thermochem(h2o_parsed):
    # CCCBDB H2O page (Release 22): values match CODATA exactly.
    assert h2o_parsed["enthalpy_of_formation_298_kj_per_mol"] == pytest.approx(-241.81)
    assert h2o_parsed["enthalpy_of_formation_298_uncertainty_kj_per_mol"] == pytest.approx(0.03)
    assert h2o_parsed["enthalpy_of_formation_0_kj_per_mol"] == pytest.approx(-238.90)
    assert h2o_parsed["entropy_298_j_per_mol_per_k"] == pytest.approx(188.84)
    assert h2o_parsed["heat_capacity_298_j_per_mol_per_k"] == pytest.approx(33.60)


def test_parser_vibrational(h2o_parsed):
    # H2O has 3 fundamental modes — bend (~1595) + symmetric (~3657) +
    # antisymmetric (~3756) cm⁻¹. Sorted ascending by parser.
    assert h2o_parsed["vibrational_fundamentals_cm_inv"] == (1595.0, 3657.0, 3756.0)
    # Harmonics tabulated separately (don't conflate per § 12).
    assert h2o_parsed["vibrational_harmonics_cm_inv"] == (1649.0, 3832.0, 3943.0)
    # IR intensities follow page (fundamental) order, not sorted.
    assert h2o_parsed["ir_intensities_km_per_mol"] == (2.9, 62.5, 41.7)


def test_parser_ionization_energy(h2o_parsed):
    assert h2o_parsed["ionization_energy_ev"] == pytest.approx(12.621)
    assert h2o_parsed["ionization_energy_uncertainty_ev"] == pytest.approx(0.002)


def test_parser_proton_affinity(h2o_parsed):
    assert h2o_parsed["proton_affinity_kj_per_mol"] == pytest.approx(691.0)


def test_parser_dipole(h2o_parsed):
    # H2O dipole = 1.857 Debye (Shostak/Ebenstein 1991).
    assert h2o_parsed["dipole_moment_debye"] == pytest.approx(1.857)


def test_parser_polarizability(h2o_parsed):
    assert h2o_parsed["polarizability_au"] == pytest.approx(1.501)


def test_parser_geometry(h2o_parsed):
    # H2O has equivalent OH bonds (CCCBDB tabulates one with the
    # symmetry equivalence implicit). Bond angle = 104.4776°.
    assert h2o_parsed["bond_lengths_ang"] == (("O", "H", 0.958),)
    assert h2o_parsed["bond_angles_deg"] == (("H", "O", "H", 104.4776),)


# ---------------------------------------------------------------------------
# Error handling — 500 / 404 pages
# ---------------------------------------------------------------------------

ERROR_500_HTML = (
    '<html><head><title>500 - Internal server error.</title></head>'
    '<body><h1>Server Error</h1></body></html>'
)


def test_parser_raises_on_500_page():
    with pytest.raises(CCCBDBNoData, match="error page"):
        parse_exp2x_html(ERROR_500_HTML)


def test_parser_raises_on_missing_h1():
    """Page exists but lacks the per-molecule master header — likely a
    landing-page or form page that ignored ``?casno=``."""
    landing = '<html><head><title>CCCBDB</title></head><body><h1>Welcome</h1></body></html>'
    with pytest.raises(CCCBDBNoData, match="lacks the 'Experimental data for X"):
        parse_exp2x_html(landing)


# ---------------------------------------------------------------------------
# Atomic enthalpy lookup + atomization derivation
# ---------------------------------------------------------------------------

def test_atomic_enthalpy_h_o():
    """CODATA 2018 atomic Hf°(0K) for H and O."""
    assert atomic_h_f_0k("H") == pytest.approx(216.034)
    assert atomic_h_f_0k("O") == pytest.approx(246.79)


def test_atomic_enthalpy_unknown_raises():
    with pytest.raises(KeyError, match="no atomic Hf"):
        atomic_h_f_0k("Es")  # einsteinium — clearly out of scope


def test_atomization_h2o_d0():
    """AE(H2O, 0K) = 2·Hf°(H) + Hf°(O) − Hf°(H2O, 0K) ≈ 917.76 kJ/mol.

    Equals 219.35 kcal/mol — matches CODATA D₀ for water.
    The QC-textbook 232.4 kcal/mol is D_e (electronic, ZPE-removed)
    = D_0 + ZPE; we expose D_0 here, caller adds ZPE if they want
    D_e.
    """
    ae_kj = derive_atomization_kj_per_mol(
        composition={"H": 2, "O": 1},
        h_f_0k_molecule_kj_per_mol=-238.90,
    )
    assert ae_kj == pytest.approx(917.758, abs=0.01)
    assert ae_kj / 4.184 == pytest.approx(219.35, abs=0.01)


def test_composition_from_formula():
    assert composition_from_formula("H2O") == {"H": 2, "O": 1}
    assert composition_from_formula("CH4") == {"C": 1, "H": 4}
    # Two-letter symbol case — Cl2 is a single chlorine pair.
    assert composition_from_formula("HCl") == {"H": 1, "Cl": 1}


# ---------------------------------------------------------------------------
# fetch_cccbdb — end-to-end via html_path escape hatch
# ---------------------------------------------------------------------------

def test_fetch_cccbdb_offline(tmp_path):
    """No live HTTP — read HTML from disk, return ExperimentalReference
    with full provenance + derived atomization."""
    cache = FetchCache(root=tmp_path, ttl_seconds=3600)
    ref = fetch_cccbdb(
        cas="7732-18-5",
        cache=cache,
        use_cache=False,
        html_path=H2O_HTML_PATH,
    )
    assert isinstance(ref, ExperimentalReference)
    assert ref.cas == "7732-18-5"
    assert ref.formula == "H2O"
    assert ref.kind == "experimental"
    assert ref.atomization_energy_kcal_per_mol == pytest.approx(219.35, abs=0.01)
    # Provenance carries the NIST DOI + license verbatim.
    assert ref.provenance.source_db == SOURCE_DB_CCCBDB
    assert ref.provenance.original_reference == "doi:10.18434/T47C7Z"
    assert ref.provenance.license == "NIST SRD"
    assert ref.provenance.fetched_at.endswith("Z")
    # D_0 vs D_e distinction notes are in the provenance notes block.
    assert "D_0" in ref.provenance.notes
    assert "ZPE" in ref.provenance.notes


def test_fetch_cccbdb_writes_cache(tmp_path):
    cache = FetchCache(root=tmp_path, ttl_seconds=3600)
    ref = fetch_cccbdb(
        cas="7732-18-5",
        cache=cache,
        use_cache=False,
        html_path=H2O_HTML_PATH,
    )
    cache_path = cache.path_for(SOURCE_DB_CCCBDB, "7732-18-5")
    assert cache_path.is_file()
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "raw" in payload
    assert "html" in payload["raw"]
    # Now read back via cache_only mode — no html_path, no HTTP.
    ref2 = fetch_cccbdb(
        cas="7732-18-5",
        cache=cache,
        use_cache=True,
        cache_only=True,
    )
    # Identical extraction (parser is deterministic).
    assert ref2.formula == ref.formula
    assert ref2.atomization_energy_kcal_per_mol == ref.atomization_energy_kcal_per_mol
    assert ref2.vibrational_fundamentals_cm_inv == ref.vibrational_fundamentals_cm_inv


def test_fetch_cccbdb_cache_only_misses_cleanly(tmp_path):
    cache = FetchCache(root=tmp_path, ttl_seconds=3600)
    with pytest.raises(FetchCacheMiss):
        fetch_cccbdb(
            cas="9999-99-9",
            cache=cache,
            cache_only=True,
        )


def test_fetch_cccbdb_500_page_raises_molecule_not_found(tmp_path):
    """A 500/404 page (CCCBDB serves these on bad CAS) → MoleculeNotFound."""
    error_path = tmp_path / "error.html"
    error_path.write_text(ERROR_500_HTML, encoding="iso-8859-1")
    cache = FetchCache(root=tmp_path, ttl_seconds=3600)
    with pytest.raises(MoleculeNotFound, match="9999-99-9"):
        fetch_cccbdb(
            cas="9999-99-9",
            cache=cache,
            use_cache=False,
            html_path=error_path,
        )


# ---------------------------------------------------------------------------
# Canonical molecule table — sanity
# ---------------------------------------------------------------------------

def test_canonical_h2o():
    h2o = find("h2o")
    assert h2o.cas == "7732-18-5"
    assert h2o.formula == "H2O"
    assert "exp2x.asp" in h2o.cccbdb_url


def test_canonical_set_size():
    """8 v1 molecules per § 10."""
    assert len(CANONICAL_MOLECULES) == 8
    slugs = {m.slug for m in CANONICAL_MOLECULES}
    assert {"h2o", "ch4", "nh3", "hf", "o2", "o3", "co2", "h2co"} <= slugs


def test_canonical_unique_cas():
    """No accidental duplicate CAS numbers."""
    cas_list = [m.cas for m in CANONICAL_MOLECULES]
    assert len(cas_list) == len(set(cas_list))


# ---------------------------------------------------------------------------
# Cartesian geometry parsing (phase-2 polish step 5)
# ---------------------------------------------------------------------------

def test_parser_cartesian_geometry(h2o_parsed):
    """CCCBDB H2O exp2x has an explicit Cartesian table — labels
    "O1"/"H2"/"H3" with Z atomic numbers + Å coordinates."""
    cart = h2o_parsed["cartesian_geometry_ang"]
    assert len(cart) == 3
    o, h2, h3 = cart
    assert o == ("O1", 8, 0.0, 0.0, 0.1173)
    assert h2 == ("H2", 1, 0.0, 0.7572, -0.4692)
    assert h3 == ("H3", 1, 0.0, -0.7572, -0.4692)


def test_geometry_bridge_h2o():
    """Build a ``MoleculeSpec`` directly from the H2O CCCBDB record."""
    from vibeqc.fetch.references import experimental_geometry_to_molecule_spec

    ref = fetch_cccbdb(
        cas="7732-18-5", use_cache=False, html_path=H2O_HTML_PATH,
    )
    spec = experimental_geometry_to_molecule_spec(ref, quick=True)

    # Slug + identity.
    assert spec.id == "cccbdb_7732185"
    assert spec.family == "molecule"
    assert len(spec.atoms) == 3
    syms = [a.symbol for a in spec.atoms]
    assert syms == ["O", "H", "H"]
    assert [a.z for a in spec.atoms] == [8, 1, 1]
    # Cartesian preserved exactly.
    assert spec.atoms[0].xyz_ang == (0.0, 0.0, 0.1173)
    # Closed-shell from electron parity.
    assert spec.multiplicity == 1
    assert spec.charge == 0
    # Quick mode → sto-3g.
    assert spec.recommended_basis == "sto-3g"
    # Provenance carries CCCBDB DOI + bridge note.
    assert spec.provenance is not None
    assert spec.provenance.original_reference == "doi:10.18434/T47C7Z"
    assert "experimental_geometry_to_molecule_spec" in spec.provenance.notes


def test_geometry_bridge_overrides():
    """Caller can override charge / multiplicity for open-shell species."""
    from vibeqc.fetch.references import experimental_geometry_to_molecule_spec

    ref = fetch_cccbdb(
        cas="7732-18-5", use_cache=False, html_path=H2O_HTML_PATH,
    )
    # Hypothetical: pretend H2O is a triplet for a benchmarking exercise.
    spec = experimental_geometry_to_molecule_spec(
        ref, overrides={"multiplicity": 3, "family": "molecule_open_shell"},
    )
    assert spec.multiplicity == 3
    assert spec.family == "molecule_open_shell"


def test_geometry_bridge_slug_override():
    """Custom slug for emission filename control."""
    from vibeqc.fetch.references import experimental_geometry_to_molecule_spec

    ref = fetch_cccbdb(
        cas="7732-18-5", use_cache=False, html_path=H2O_HTML_PATH,
    )
    spec = experimental_geometry_to_molecule_spec(ref, slug="h2o_nist")
    assert spec.id == "h2o_nist"


def test_geometry_bridge_refuses_empty_cartesians():
    """Empty Cartesian table → clear error message."""
    from vibeqc.fetch.references import experimental_geometry_to_molecule_spec

    ref = ExperimentalReference(
        cas="0000-00-0", formula="X", name="Test",
        # cartesian_geometry_ang left as default ()
        provenance=Provenance(
            source_db="manual", source_id="test", source_url="",
            original_reference="", license="", fetched_at="2026-05-09T00:00:00Z",
            fetcher_version="0.0.0",
        ),
    )
    with pytest.raises(ValueError, match="cartesian_geometry_ang is empty"):
        experimental_geometry_to_molecule_spec(ref)
