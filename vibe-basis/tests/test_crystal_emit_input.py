"""Tests for vibe_basis.backends.crystal.emit_input.

Verifies the deck-builder against the CRYSTAL23-manual format and
against a known-good fixture (MgO at pob-TZVP RHF — the same shape
PT2013 SI Table 2 documents). AFM compounds, missing space-group
data, and unknown methods must all skip emission rather than emit
a deck CRYSTAL would reject.
"""

from __future__ import annotations

import textwrap

import pytest

from vibe_basis.backends.crystal import emit_input
from vibe_basis.io.structures import STRUCTURES, all_structures


def test_mgo_emits_canonical_pob_tzvp_rhf_d12():
    """MgO RHF/POB-TZVP — the canonical shape PT2013 / VO2019 / our
    Phase 14h fixtures all use."""
    mgo = STRUCTURES["MgO"]
    deck = emit_input(mgo, "pob-tzvp", "rhf")
    assert deck is not None

    expected = textwrap.dedent("""\
        MgO — RHF on POB-TZVP (CRYSTAL23 parity)
        CRYSTAL
        0 0 0
        225
        4.217000
        2
        12 0.000000 0.000000 0.000000
        8 0.500000 0.500000 0.500000
        BASISSET
        POB-TZVP
        TOLINTEG
        9 9 9 18 54
        SHRINK
        8 8
        TOLDEE
        8
        END
    """)
    assert deck == expected


def test_diamond_emits_single_atom_asymm_unit():
    """C in diamond — space group 227, one atom in the asymm unit."""
    diamond = STRUCTURES["C-diamond"]
    deck = emit_input(diamond, "pob-tzvp", "rhf")
    assert deck is not None
    assert "227" in deck
    # Diamond's asymm unit is one C at (1/8, 1/8, 1/8).
    assert "6 0.125000 0.125000 0.125000" in deck


def test_dft_method_inserts_dft_block():
    """A DFT-functional method like ``pbe`` must produce a DFT block;
    HF must not."""
    mgo = STRUCTURES["MgO"]
    rhf_deck = emit_input(mgo, "pob-tzvp", "rhf")
    pbe_deck = emit_input(mgo, "pob-tzvp", "pbe")
    assert "DFT" not in rhf_deck
    assert "\nDFT\nPBE\nEND\n" in pbe_deck


def test_pw1pw_recognized_as_dft_functional():
    """PW1PW is the hybrid behind the pob-TZVP fits; must be a
    recognized CRYSTAL functional (not silently fall through to
    None)."""
    mgo = STRUCTURES["MgO"]
    deck = emit_input(mgo, "pob-tzvp", "pw1pw")
    assert deck is not None
    assert "PW1PW" in deck


def test_unknown_method_returns_none():
    """Bogus method → None, so the optimizer doesn't ship a deck
    CRYSTAL would reject silently."""
    mgo = STRUCTURES["MgO"]
    assert emit_input(mgo, "pob-tzvp", "quack") is None


def test_afm_compounds_return_none():
    """The 6 AFM TM oxides need ATOMSPIN (R3-blocked) — emitter
    must skip them, not emit a deck that converges to the wrong
    state."""
    afm_names = ["MnO", "FeO", "CoO", "NiO", "MnS", "MnSe"]
    for name in afm_names:
        s = STRUCTURES[name]
        assert emit_input(s, "pob-tzvp", "rhf") is None, \
            f"{name}: AFM compound should skip emission"


def test_structure_without_crystal_data_returns_none(monkeypatch):
    """A structure with ``crystal_spacegroup=0`` (unset) must skip
    emission rather than crash."""
    mgo = STRUCTURES["MgO"]
    # Frozen dataclass — build a sibling with spacegroup zeroed.
    from dataclasses import replace
    blank = replace(mgo, crystal_spacegroup=0)
    assert emit_input(blank, "pob-tzvp", "rhf") is None


def test_shrink_and_toldee_are_configurable():
    """Stage 0 uses (8, 8) per the pob convention; an optimizer
    might want tighter / looser at experiment time."""
    mgo = STRUCTURES["MgO"]
    deck = emit_input(mgo, "pob-tzvp", "rhf", shrink=12, toldee=10)
    assert "\nSHRINK\n12 12\n" in deck
    assert "\nTOLDEE\n10\n" in deck


def test_basis_name_is_uppercased():
    """CRYSTAL keywords are case-sensitive uppercase."""
    mgo = STRUCTURES["MgO"]
    deck = emit_input(mgo, "pob-tzvp-rev2", "rhf")
    assert "POB-TZVP-REV2" in deck
    # And the lower-cased form should NOT appear (it would be
    # rejected by CRYSTAL's basis-keyword parser).
    assert "pob-tzvp-rev2" not in deck


def test_every_non_afm_cubic_structure_emits():
    """Inventory invariant: every non-AFM cubic structure must
    produce a deck. If this regresses, .d12 generation broke."""
    emitted = 0
    skipped_afm = 0
    skipped_other = 0
    for s in all_structures():
        if s.crystal_system.lower() != "cubic":
            continue
        deck = emit_input(s, "pob-tzvp", "rhf")
        if deck is None:
            if s.afm_pattern:
                skipped_afm += 1
            else:
                skipped_other += 1
                pytest.fail(
                    f"{s.name}: non-AFM cubic but emit_input returned None"
                )
        else:
            emitted += 1
    # Phase 1+2+3 total 39 cubic compounds, 6 of which are AFM.
    assert emitted == 33
    assert skipped_afm == 6
    assert skipped_other == 0


def test_emitted_deck_ends_with_end_keyword():
    """The closing ``END`` is required for CRYSTAL to parse the
    deck. Easy to drop with a copy/paste — guard against regression."""
    mgo = STRUCTURES["MgO"]
    deck = emit_input(mgo, "pob-tzvp", "rhf")
    assert deck.rstrip().endswith("END")
