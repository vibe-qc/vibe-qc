"""``examples.regression.core.expected_bridge`` — projects an
:class:`ExperimentalReference` into ``ExpectedRef.published_*``.

Pins:
  1. Construction-required ``provenance`` (the no-default-factory
     contract — passing nothing raises with a citable error message).
  2. Each ``ReferenceQuantity`` projects to the right unit and rounds
     trip cleanly through CODATA 2018.
  3. Missing properties → ``return expected unchanged`` (caller treats
     ``published_energy_ha is None`` as "not available").
  4. Citation string carries the NIST CCCBDB DOI for CCCBDB sources.
"""
from __future__ import annotations

import pytest

from examples.regression.core.expected_bridge import (
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    HARTREE_TO_KJ_PER_MOL,
    attach_reference_to_expected,
    ev_to_hartree,
    kcal_per_mol_to_hartree,
    kj_per_mol_to_hartree,
)
from examples.regression.core.spec import (
    ExperimentalReference,
    ExpectedRef,
    Provenance,
)


# ---------------------------------------------------------------------------
# Schema construction contract
# ---------------------------------------------------------------------------

def _h2o_cccbdb_provenance() -> Provenance:
    return Provenance(
        source_db="CCCBDB",
        source_id="7732-18-5",
        source_url="https://cccbdb.nist.gov/exp1.asp?casno=7732185",
        original_reference="doi:10.18434/T47C7Z",
        license="NIST SRD",
        fetched_at="2026-05-09T12:00:00Z",
        fetcher_version="0.1.0",
    )


def test_experimental_reference_requires_provenance():
    with pytest.raises(ValueError, match="provenance is required"):
        ExperimentalReference(
            cas="7732-18-5", formula="H2O", name="Water",
        )


def test_experimental_reference_smoke():
    """Minimal happy-path construction with the canonical H2O fields
    we'll see from CCCBDB. Pins the field names so a rename downstream
    has to update the test too."""
    ref = ExperimentalReference(
        cas="7732-18-5",
        formula="H2O",
        name="Water",
        kind="experimental",
        atomization_energy_kcal_per_mol=232.4,        # CCCBDB H2O
        atomization_energy_uncertainty_kcal_per_mol=0.1,
        ionization_energy_ev=12.621,
        dipole_moment_debye=1.8546,
        vibrational_fundamentals_cm_inv=(1594.6, 3657.1, 3755.9),
        provenance=_h2o_cccbdb_provenance(),
    )
    assert ref.cas == "7732-18-5"
    assert ref.kind == "experimental"
    assert ref.bond_lengths_ang == ()      # default empty tuple
    assert ref.provenance.source_db == "CCCBDB"


# ---------------------------------------------------------------------------
# Unit conversions — CODATA 2018, must match the rest of vibe-qc
# ---------------------------------------------------------------------------

def test_unit_constants_match_codata_2018():
    # Same values used in python/vibeqc/scf_log.py and runner.py.
    assert HARTREE_TO_EV == 27.211386245988
    assert HARTREE_TO_KCAL_PER_MOL == 627.5094740631
    assert HARTREE_TO_KJ_PER_MOL == 2625.499639


def test_round_trip_kcal_per_mol():
    # 1 Hartree = 627.5094740631 kcal/mol exactly by definition.
    assert kcal_per_mol_to_hartree(HARTREE_TO_KCAL_PER_MOL) == pytest.approx(1.0, abs=1e-12)


def test_round_trip_kj_per_mol():
    assert kj_per_mol_to_hartree(HARTREE_TO_KJ_PER_MOL) == pytest.approx(1.0, abs=1e-12)


def test_round_trip_ev():
    assert ev_to_hartree(HARTREE_TO_EV) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Bridge — project ref → ExpectedRef.published_*
# ---------------------------------------------------------------------------

def _empty_expected(quantity_basis="sto-3g") -> ExpectedRef:
    return ExpectedRef(
        system_id="h2o",
        basis=quantity_basis,
        method_id="rks-lda",
        kmesh=(0, 0, 0),
    )


def test_bridge_atomization_energy_kcal_to_hartree():
    ref = ExperimentalReference(
        cas="7732-18-5", formula="H2O", name="Water",
        atomization_energy_kcal_per_mol=232.4,
        provenance=_h2o_cccbdb_provenance(),
    )
    out = attach_reference_to_expected(
        ref, _empty_expected(), quantity="atomization_energy",
    )
    assert out.published_energy_ha == pytest.approx(232.4 / HARTREE_TO_KCAL_PER_MOL, rel=1e-12)
    # Citation carries the NIST CCCBDB DOI verbatim.
    assert "doi:10.18434/T47C7Z" in out.published_source
    assert "atomization_energy" in out.published_source
    assert "experimental" in out.published_source


def test_bridge_ionization_energy_ev_to_hartree():
    ref = ExperimentalReference(
        cas="7732-18-5", formula="H2O", name="Water",
        ionization_energy_ev=12.621,
        provenance=_h2o_cccbdb_provenance(),
    )
    out = attach_reference_to_expected(
        ref, _empty_expected(), quantity="ionization_energy",
    )
    assert out.published_energy_ha == pytest.approx(12.621 / HARTREE_TO_EV, rel=1e-12)


def test_bridge_enthalpy_298_kj_to_hartree():
    ref = ExperimentalReference(
        cas="7732-18-5", formula="H2O", name="Water",
        enthalpy_of_formation_298_kj_per_mol=-241.826,   # CCCBDB H2O(g)
        provenance=_h2o_cccbdb_provenance(),
    )
    out = attach_reference_to_expected(
        ref, _empty_expected(), quantity="enthalpy_of_formation_298",
    )
    assert out.published_energy_ha == pytest.approx(-241.826 / HARTREE_TO_KJ_PER_MOL, rel=1e-12)


def test_bridge_missing_quantity_returns_unchanged():
    """No atomization energy → ``published_*`` stays at defaults.
    Caller treats ``None`` as "not available"; never raise on a
    missing field, never silently fabricate a number."""
    ref = ExperimentalReference(
        cas="7732-18-5", formula="H2O", name="Water",
        # atomization_energy_kcal_per_mol left None
        provenance=_h2o_cccbdb_provenance(),
    )
    expected = _empty_expected()
    out = attach_reference_to_expected(
        ref, expected, quantity="atomization_energy",
    )
    # Same instance returned (defensive: assert by-value too).
    assert out is expected
    assert out.published_energy_ha is None
    assert out.published_source == ""


def test_bridge_unknown_quantity_raises():
    ref = ExperimentalReference(
        cas="7732-18-5", formula="H2O", name="Water",
        atomization_energy_kcal_per_mol=232.4,
        provenance=_h2o_cccbdb_provenance(),
    )
    with pytest.raises(ValueError, match="unknown reference quantity"):
        attach_reference_to_expected(
            ref, _empty_expected(), quantity="not_a_quantity",  # type: ignore[arg-type]
        )


def test_bridge_computed_kind_propagates_to_citation():
    """``ref.kind`` lands in the citation string so the report writer
    can disambiguate experimental vs computed columns at a glance."""
    ref = ExperimentalReference(
        cas="7732-18-5", formula="H2O", name="Water",
        kind="computed",
        atomization_energy_kcal_per_mol=233.0,
        provenance=_h2o_cccbdb_provenance(),
    )
    out = attach_reference_to_expected(
        ref, _empty_expected(), quantity="atomization_energy",
    )
    assert "(computed)" in out.published_source
