"""Tests for Materials Project property-field client (VFETCH-X3).

Tests the MP summary-record augmentation without hitting the live API:
synthetic MP response dicts exercise the property folding, open-shell
refinement, and Provenance.notes merging.
"""

from __future__ import annotations

from dataclasses import replace as _replace
from pathlib import Path

import pytest
from vibeqc.fetch.client_mp import _augment_spec_with_mp_properties

from examples.regression.core.spec import AtomFrac, PeriodicSpec, Provenance

# ---------- Helpers --------------------------------------------------------


def _make_spec(*, id: str = "mp-1265", a: float = 4.194) -> PeriodicSpec:
    """Conventional 8-atom MgO rocksalt cell."""
    return PeriodicSpec(
        id=id,
        family="rocksalt",
        lattice_ang=((a, 0.0, 0.0), (0.0, a, 0.0), (0.0, 0.0, a)),
        space_group="Fm-3m",
        atoms=(
            AtomFrac(symbol="Mg", z=12, frac=(0.0, 0.0, 0.0)),
            AtomFrac(symbol="Mg", z=12, frac=(0.0, 0.5, 0.5)),
            AtomFrac(symbol="Mg", z=12, frac=(0.5, 0.0, 0.5)),
            AtomFrac(symbol="Mg", z=12, frac=(0.5, 0.5, 0.0)),
            AtomFrac(symbol="O", z=8, frac=(0.5, 0.5, 0.5)),
            AtomFrac(symbol="O", z=8, frac=(0.5, 0.0, 0.0)),
            AtomFrac(symbol="O", z=8, frac=(0.0, 0.5, 0.0)),
            AtomFrac(symbol="O", z=8, frac=(0.0, 0.0, 0.5)),
        ),
        provenance=Provenance(
            source_db="OPTIMADE/mp",
            source_id="mp-1265",
            source_url="https://optimade.materialsproject.org/structures/mp-1265",
            original_reference="",
            license="CC-BY-4.0",
            fetched_at="2026-06-12T00:00:00Z",
            fetcher_version="0.1.0",
            notes="",
        ),
    )


# ---------- Open-shell refinement -----------------------------------------


def test_mp_magnetic_updates_open_shell():
    """MP says magnetic + total_mag > 0.1 → is_open_shell=True."""
    spec = _make_spec()
    mp = {"is_magnetic": True, "total_magnetization": 2.5}
    out = _augment_spec_with_mp_properties(spec, mp)
    assert out.is_open_shell is True


def test_mp_magnetic_but_zero_mag_stays_closed():
    """MP says magnetic but total_mag ≤ 0.1 → is_open_shell=False."""
    spec = _make_spec()
    mp = {"is_magnetic": True, "total_magnetization": 0.05}
    out = _augment_spec_with_mp_properties(spec, mp)
    assert out.is_open_shell is False


def test_mp_non_magnetic_stays_closed():
    """MP says non-magnetic → is_open_shell=False."""
    spec = _make_spec()
    mp = {"is_magnetic": False, "total_magnetization": 0.0}
    out = _augment_spec_with_mp_properties(spec, mp)
    assert out.is_open_shell is False


def test_mp_no_magnetic_fields_does_not_crash():
    """Missing magnetic fields → default closed-shell, no crash."""
    spec = _make_spec()
    out = _augment_spec_with_mp_properties(spec, {})
    assert out.is_open_shell is False


# ---------- Property notes -------------------------------------------------


def test_mp_band_gap_in_notes():
    """band_gap appended to Provenance.notes."""
    spec = _make_spec()
    mp = {"band_gap": 5.123}
    out = _augment_spec_with_mp_properties(spec, mp)
    assert "MP band_gap=5.123 eV" in out.provenance.notes


def test_mp_formation_energy_in_notes():
    """formation_energy_per_atom appended to Provenance.notes."""
    spec = _make_spec()
    mp = {"formation_energy_per_atom": -3.01}
    out = _augment_spec_with_mp_properties(spec, mp)
    assert "MP formation_energy_per_atom=-3.010 eV" in out.provenance.notes


def test_mp_energy_above_hull_in_notes():
    """energy_above_hull appended to Provenance.notes."""
    spec = _make_spec()
    mp = {"energy_above_hull": 0.0}
    out = _augment_spec_with_mp_properties(spec, mp)
    assert "MP energy_above_hull=0.000 eV/atom" in out.provenance.notes


def test_mp_all_properties_in_notes():
    """Multiple MP properties correctly concatenated in notes."""
    spec = _make_spec()
    mp = {
        "band_gap": 5.1,
        "formation_energy_per_atom": -3.0,
        "energy_above_hull": 0.001,
    }
    out = _augment_spec_with_mp_properties(spec, mp)
    notes = out.provenance.notes
    assert "MP band_gap=5.100 eV" in notes
    assert "MP formation_energy_per_atom=-3.000 eV" in notes
    assert "MP energy_above_hull=0.001 eV/atom" in notes


def test_mp_properties_preserve_existing_notes():
    """MP properties appended after existing provenance.notes."""
    spec = _make_spec()
    spec = _replace(
        spec,
        provenance=_replace(
            spec.provenance,
            notes="Existing note.",
        ),
    )
    mp = {"band_gap": 2.0}
    out = _augment_spec_with_mp_properties(spec, mp)
    assert out.provenance.notes.startswith("Existing note.")
    assert "MP band_gap=2.000 eV" in out.provenance.notes


def test_mp_no_properties_keeps_notes_unchanged():
    """Empty MP dict does not mutate provenance.notes."""
    spec = _make_spec()
    spec = _replace(
        spec,
        provenance=_replace(spec.provenance, notes="pristine"),
    )
    out = _augment_spec_with_mp_properties(spec, {})
    assert out.provenance.notes == "pristine"
