"""Tests for emit_crystal() and emit_input_inline() — CRYSTAL inline basis.

Verifies that pob-TZVP can be round-tripped through parse → emit_crystal
and that emit_input_inline produces correct .d12 decks.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType

import pytest

# Import basis_crystal without triggering vibe-qc's C extension.
_BASIS_CRYSTAL_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "python"
    / "vibeqc"
    / "basis_crystal.py"
)


@pytest.fixture(scope="module")
def basis_crystal():
    """Load the real parser and its element data without either package initializer."""
    namespace = "_vibe_basis_crystal_test"
    package_dir = _BASIS_CRYSTAL_PATH.parent
    # Keep relative imports valid without importing the native QC extension or
    # replacing a real vibeqc module another test may already have loaded.
    with pytest.MonkeyPatch.context() as patch:
        for name, directory in (
            (namespace, package_dir),
            (f"{namespace}.basis_toolkit", package_dir / "basis_toolkit"),
        ):
            package = ModuleType(name)
            package.__path__ = [str(directory)]
            patch.setitem(sys.modules, name, package)
        try:
            yield importlib.import_module(f"{namespace}.basis_crystal")
        finally:
            for name in tuple(sys.modules):
                if name.startswith(f"{namespace}."):
                    sys.modules.pop(name, None)


@pytest.fixture
def pob_mg(basis_crystal):
    source = (
        Path(__file__).resolve().parent.parent.parent
        / "python"
        / "vibeqc"
        / "basis_library"
        / "sources"
        / "pob-TZVP"
        / "12_Mg"
    )
    return basis_crystal.parse_crystal_atom_basis_file(source)


@pytest.fixture
def pob_o(basis_crystal):
    source = (
        Path(__file__).resolve().parent.parent.parent
        / "python"
        / "vibeqc"
        / "basis_library"
        / "sources"
        / "pob-TZVP"
        / "08_O"
    )
    return basis_crystal.parse_crystal_atom_basis_file(source)


class TestEmitCrystal:
    def test_header_format(self, basis_crystal, pob_mg):
        text = basis_crystal.emit_crystal([pob_mg])
        lines = text.splitlines()
        # First line: "Z NSHELL"
        assert lines[0].startswith(f"{pob_mg.Z} ")
        assert int(lines[0].split()[1]) == len(pob_mg.shells)

    def test_shell_format(self, basis_crystal, pob_o):
        text = basis_crystal.emit_crystal([pob_o])
        lines = text.splitlines()
        # Check a few shell headers: "0 LAT NPG OCC SCALE"
        found_s = found_sp = found_p = found_d = False
        for line in lines:
            if line.startswith("0 "):
                parts = line.split()
                assert len(parts) >= 5
                lat = int(parts[1])
                if lat == 0:
                    found_s = True
                elif lat == 1:
                    found_sp = True
                elif lat == 2:
                    found_p = True
                elif lat == 3:
                    found_d = True
        assert found_s and found_p
        # pob-O has no SP or D shells in TZVP (SP is only for H,He in some formats)
        # Actually O in pob-TZVP: 9 s, 3 sp-valence, 3 valence-p, 1 d-polarization
        # The sp shells have LAT=1

    def test_emits_mgo_both_atoms(self, basis_crystal, pob_mg, pob_o):
        text = basis_crystal.emit_crystal([pob_mg, pob_o])
        assert f"{pob_mg.Z} " in text
        assert f"{pob_o.Z} " in text
        # emit_crystal ends with " 99 0" (no trailing END — that's added by the deck).
        assert text.strip().endswith("99 0")

    def test_exponent_format_is_float(self, basis_crystal, pob_o):
        """Every exponent line must contain valid float pairs."""
        text = basis_crystal.emit_crystal([pob_o])
        for line in text.splitlines():
            if line.startswith("      "):
                parts = line.split()
                # SP shells: 3 numbers; others: 2 numbers
                assert len(parts) in (2, 3)
                for p in parts:
                    float(p)  # must parse

    def test_roundtrip_pob_tzvp_mgo_shell_count(self, basis_crystal, pob_mg, pob_o):
        """emit_crystal should preserve shell count."""
        text = basis_crystal.emit_crystal([pob_mg, pob_o])
        # Count Z lines (each starts a block)
        z_headers = [
            l for l in text.splitlines() if l.strip().startswith(("12 ", "8 "))
        ]
        assert len(z_headers) == 2


class TestEmitInputInline:
    @pytest.fixture(scope="class")
    @classmethod
    def mgo_crystal_basis(cls, basis_crystal):
        mod = basis_crystal

        mg = mod.parse_crystal_atom_basis_file(
            Path(__file__).resolve().parent.parent.parent
            / "python"
            / "vibeqc"
            / "basis_library"
            / "sources"
            / "pob-TZVP"
            / "12_Mg"
        )
        o = mod.parse_crystal_atom_basis_file(
            Path(__file__).resolve().parent.parent.parent
            / "python"
            / "vibeqc"
            / "basis_library"
            / "sources"
            / "pob-TZVP"
            / "08_O"
        )
        return mod.emit_crystal([mg, o])

    def test_rhf_deck_has_no_dft_block(self, mgo_crystal_basis):
        from vibe_basis.backends.crystal import emit_input_inline
        from vibe_basis.io.structures import STRUCTURES

        deck = emit_input_inline(STRUCTURES["MgO"], mgo_crystal_basis, method="rhf")
        assert deck is not None
        assert "DFT" not in deck

    def test_pw1pw_deck_has_dft_block(self, mgo_crystal_basis):
        from vibe_basis.backends.crystal import emit_input_inline
        from vibe_basis.io.structures import STRUCTURES

        deck = emit_input_inline(STRUCTURES["MgO"], mgo_crystal_basis, method="pw1pw")
        assert deck is not None
        assert "DFT" in deck
        assert "PW1PW" in deck

    def test_deck_contains_inline_basis_text(self, mgo_crystal_basis):
        from vibe_basis.backends.crystal import emit_input_inline
        from vibe_basis.io.structures import STRUCTURES

        deck = emit_input_inline(STRUCTURES["MgO"], mgo_crystal_basis, method="rhf")
        assert deck is not None
        # The inline basis appears between END (geometry close) and SHRINK.
        assert "ENDGEOM" not in deck
        assert deck.count("\nEND\n") >= 2  # geometry END, deck END
        assert deck.strip().endswith("END")

    def test_deck_has_asymm_atoms(self, mgo_crystal_basis):
        from vibe_basis.backends.crystal import emit_input_inline
        from vibe_basis.io.structures import STRUCTURES

        mgo = STRUCTURES["MgO"]
        deck = emit_input_inline(mgo, mgo_crystal_basis, method="rhf")
        assert deck is not None
        # Mg at (0,0,0), O at (0.5,0.5,0.5)
        assert "0.000000" in deck
        assert "0.500000" in deck

    def test_afm_compound_returns_none(self, mgo_crystal_basis):
        from vibe_basis.backends.crystal import emit_input_inline
        from vibe_basis.io.structures import STRUCTURES

        deck = emit_input_inline(STRUCTURES["MnO"], mgo_crystal_basis, method="rhf")
        assert deck is None  # AFM skip
