"""Smoke tests for vibe_basis.io.structures.

The full 39-compound database is loaded; we assert basic invariants
that protect against accidental data corruption when someone edits
``structures.py`` (e.g. forgetting to update the asymm-unit when
adding an entry, mixing up Z values, swapping a fluorite for an
antifluorite).

Per-compound chemistry-physics checks (e.g. "MgO at 4.217 Å"
matches PT2013) live in the Stage 0 recipe driver, not here — to
keep this suite a fast smoke and CRYSTAL-free.
"""

from __future__ import annotations

import pytest

from vibe_basis.io.structures import (
    STRUCTURES,
    Structure,
    StructureAtom,
    all_structures,
    in_table,
)


def test_thirtynine_structures_load():
    """The database has the expected count and is non-empty."""
    items = all_structures()
    assert len(items) == 39
    assert all(isinstance(s, Structure) for s in items)


def test_structures_dict_matches_all_structures():
    """``STRUCTURES`` and ``all_structures()`` are the same data,
    just exposed in two shapes."""
    by_name = {s.name: s for s in all_structures()}
    assert set(STRUCTURES.keys()) == set(by_name.keys())
    for name, s in STRUCTURES.items():
        assert s.name == name


def test_all_structures_have_unique_names():
    names = [s.name for s in all_structures()]
    assert len(names) == len(set(names)), \
        f"duplicate names: {[n for n in names if names.count(n) > 1]}"


def test_in_table_filters_correctly():
    """``in_table('T4')`` returns only entries that list T4."""
    t4 = in_table("PT2013-T4")
    assert len(t4) >= 1, "PT2013 T4 should have ≥1 compound"
    assert all("PT2013-T4" in s.pob_tables for s in t4)


def test_every_structure_has_lattice_parameters_in_angstrom():
    for s in all_structures():
        assert s.a > 0.0, f"{s.name}: a must be positive"
        assert s.b > 0.0, f"{s.name}: b must be positive"
        assert s.c > 0.0, f"{s.name}: c must be positive"
        # Sanity: experimental lattice constants for these compounds
        # are all between 3 Å (diamond) and ~8 Å (anti-fluorite K₂S).
        assert 2.0 < s.a < 10.0, f"{s.name}: implausible a={s.a}"


def test_every_structure_has_unit_cell_and_asymm_unit():
    """Both views must be populated. AFM compounds skip emission
    in the .d12 step but their structures still carry full data."""
    for s in all_structures():
        assert len(s.unit_cell) > 0, f"{s.name}: empty unit_cell"
        assert all(isinstance(a, StructureAtom) for a in s.unit_cell)
        assert len(s.crystal_asymm_unit) > 0, \
            f"{s.name}: empty crystal_asymm_unit"
        assert 1 <= s.crystal_spacegroup <= 230, \
            f"{s.name}: invalid spacegroup {s.crystal_spacegroup}"


def test_fractional_coords_in_unit_interval():
    """All fractional coordinates must be in [0, 1) by convention.
    Off-by-one bugs in symmetry expansion show up here first."""
    for s in all_structures():
        for atom in s.unit_cell:
            for c in atom.fxyz:
                assert -1e-9 <= c < 1.0 + 1e-9, \
                    f"{s.name}: frac coord out of [0,1): {atom.fxyz}"


def test_atomic_numbers_are_physical():
    """Z must be a positive integer ≤ 103 (we go up to actinides)."""
    for s in all_structures():
        for atom in s.unit_cell + s.crystal_asymm_unit:
            assert isinstance(atom.Z, int)
            assert 1 <= atom.Z <= 103, f"{s.name}: implausible Z={atom.Z}"


def test_lattice_matrix_is_3x3():
    """``lattice_matrix_angstrom`` returns a 3×3 matrix; for cubic
    cells it's a·I."""
    mgo = STRUCTURES["MgO"]
    M = mgo.lattice_matrix_angstrom()
    assert len(M) == 3 and all(len(r) == 3 for r in M)
    # MgO is cubic, lattice 4.217 Å → diagonal = 4.217, off-diag = 0.
    assert M[0][0] == pytest.approx(mgo.a)
    assert M[1][1] == pytest.approx(mgo.a)
    assert M[2][2] == pytest.approx(mgo.a)
    assert M[0][1] == pytest.approx(0.0)
    assert M[1][0] == pytest.approx(0.0)


def test_afm_compounds_carry_pattern_tag():
    """The 6 AFM TM oxides must have a non-empty ``afm_pattern``
    so the .d12 emitter can skip them."""
    afm_names = {"MnO", "FeO", "CoO", "NiO", "MnS", "MnSe"}
    for name in afm_names:
        s = STRUCTURES.get(name)
        assert s is not None, f"missing AFM compound: {name}"
        assert s.afm_pattern, f"{name}: AFM compound but afm_pattern is empty"


def test_cubic_compounds_have_valid_spacegroups():
    """Cubic test set uses space groups 216 (zincblende), 221
    (anti-ReO3 / CsCl), 224 (cuprite), 225 (rocksalt / fluorite /
    antifluorite), 227 (diamond). Anything else in this set is a
    data bug."""
    expected = {216, 221, 224, 225, 227}
    for s in all_structures():
        if s.crystal_system.lower() == "cubic":
            assert s.crystal_spacegroup in expected, (
                f"{s.name}: cubic but unexpected spacegroup "
                f"{s.crystal_spacegroup}"
            )
