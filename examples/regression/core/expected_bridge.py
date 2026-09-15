"""Bridge phase-2 :class:`ExperimentalReference` records into the
existing phase-1 :class:`ExpectedRef.published_*` columns.

The regression suite already renders ``ExpectedRef.published_energy_ha``
(see :mod:`examples.regression.core.report`); this module is the
adapter layer that takes a fetched experimental record and projects
one of its scalar properties (atomization energy, enthalpy of
formation, ionization / electron affinity, etc.) into Hartree, fills
``published_source`` with the NIST DOI, and returns a fresh
``ExpectedRef``.

A single ``ExperimentalReference`` can populate multiple
``ExpectedRef`` columns (one per quantity) — call this function once
per (system_id, basis, method_id, quantity) the report cares about.

Unit conversions use the CODATA 2018 values that the rest of vibe-qc
uses (see ``python/vibeqc/molecule.py``, ``python/vibeqc/scf_log.py``).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from .spec import ExperimentalReference, ExpectedRef


# CODATA 2018 — kept identical to the values used elsewhere in vibe-qc.
HARTREE_TO_EV: float = 27.211386245988
HARTREE_TO_KCAL_PER_MOL: float = 627.5094740631
HARTREE_TO_KJ_PER_MOL: float = 2625.499639


def kcal_per_mol_to_hartree(x: float) -> float:
    """Convert kcal/mol → Hartree (per particle)."""
    return x / HARTREE_TO_KCAL_PER_MOL


def kj_per_mol_to_hartree(x: float) -> float:
    """Convert kJ/mol → Hartree (per particle)."""
    return x / HARTREE_TO_KJ_PER_MOL


def ev_to_hartree(x: float) -> float:
    """Convert eV → Hartree."""
    return x / HARTREE_TO_EV


# Quantity selector — what scalar property of the reference to project
# into ``ExpectedRef.published_energy_ha``. The choice is the caller's
# responsibility; the report writer uses it to decide what the column
# semantically means (atomization energy is NOT a total SCF energy and
# must not be compared to one).
ReferenceQuantity = Literal[
    "atomization_energy",
    "enthalpy_of_formation_298",
    "enthalpy_of_formation_0",
    "ionization_energy",
    "electron_affinity",
    "proton_affinity",
]


_NIST_CCCBDB_DOI = "doi:10.18434/T47C7Z"
_NIST_CCCBDB_CITATION = (
    "NIST Standard Reference Database 101 (CCCBDB) Release 22, "
    "Editor R. D. Johnson III, May 2022, doi:10.18434/T47C7Z. "
    "Cite per https://cccbdb.nist.gov/citation.asp."
)


def attach_reference_to_expected(
    ref: ExperimentalReference,
    expected: ExpectedRef,
    *,
    quantity: ReferenceQuantity = "atomization_energy",
) -> ExpectedRef:
    """Project one scalar from ``ref`` into ``expected.published_*`` and
    return a new ``ExpectedRef`` (the input is frozen so we ``replace``).

    Returns the input unchanged when the requested quantity isn't
    populated on ``ref`` (the report writer treats ``published_energy_ha
    is None`` as "not available"). Set ``ref.kind`` to ``"experimental"``
    or ``"computed"`` to disambiguate the column header at the report
    level — this bridge doesn't leak that into the projected number.
    """
    val_ha = _project_to_hartree(ref, quantity)
    if val_ha is None:
        return expected
    cite = (
        _NIST_CCCBDB_CITATION
        if ref.provenance.source_db == "CCCBDB"
        else (
            f"{ref.provenance.source_db}:{ref.provenance.source_id} "
            f"({ref.provenance.original_reference})"
        )
    )
    return replace(
        expected,
        published_energy_ha=val_ha,
        published_source=f"{quantity} ({ref.kind}) — {cite}",
    )


def _project_to_hartree(
    ref: ExperimentalReference,
    quantity: ReferenceQuantity,
) -> "float | None":
    """Pick the right field, do the unit conversion, return Hartree.

    Returns ``None`` when the reference doesn't carry that quantity.
    The report writer treats ``None`` as "not available" — keep the
    behaviour silent (don't raise) so missing CCCBDB fields don't
    abort a sweep.
    """
    if quantity == "atomization_energy":
        if ref.atomization_energy_kcal_per_mol is None:
            return None
        return kcal_per_mol_to_hartree(ref.atomization_energy_kcal_per_mol)
    if quantity == "enthalpy_of_formation_298":
        if ref.enthalpy_of_formation_298_kj_per_mol is None:
            return None
        return kj_per_mol_to_hartree(ref.enthalpy_of_formation_298_kj_per_mol)
    if quantity == "enthalpy_of_formation_0":
        if ref.enthalpy_of_formation_0_kj_per_mol is None:
            return None
        return kj_per_mol_to_hartree(ref.enthalpy_of_formation_0_kj_per_mol)
    if quantity == "ionization_energy":
        if ref.ionization_energy_ev is None:
            return None
        return ev_to_hartree(ref.ionization_energy_ev)
    if quantity == "electron_affinity":
        if ref.electron_affinity_ev is None:
            return None
        return ev_to_hartree(ref.electron_affinity_ev)
    if quantity == "proton_affinity":
        if ref.proton_affinity_kj_per_mol is None:
            return None
        return kj_per_mol_to_hartree(ref.proton_affinity_kj_per_mol)
    raise ValueError(
        f"unknown reference quantity {quantity!r}; "
        f"see ReferenceQuantity in expected_bridge.py"
    )
