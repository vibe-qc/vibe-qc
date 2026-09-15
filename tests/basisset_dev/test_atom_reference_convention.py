"""The free-atom reference convention, on both sides of the tier seam.

A cohesive energy is ``SUM E_atom - E_bulk``, so the free-atom reference
is half the answer, and it is only comparable to an external reference
computed under the **same** convention. The plane-wave work measured the
cost of getting this wrong: spin-restricting the bromine reference alone
moved KBr's atomization energy by **37 kJ/mol**
(``handovers/HANDOVER_GPAW_PW_REFERENCE.md``, GPAW-PWREF-002), against a
gate that has to hold to 1 kJ/mol.

Two things are pinned here.

**The emitted CRYSTAL deck.** An open-shell atom must be aspherical and
spin-polarised: ``SYMMREMO`` to drop the symmetry operators, ``UHF`` (or
``SPIN`` in the DFT block) for unrestricted spin, and ``SPINLOCK`` to fix
the occupancy. The CRYSTAL23 manual requires it for the single-atom
route: *"UHF and SPINLOCK must be used to define a reasonable orbital
occupancy."* Before 2026-07-26 the emitter shipped none of them, so every
open-shell free atom was computed spin-restricted and spherical, and the
error was invisible because those SCFs converge perfectly well.

**The two multiplicity tables.** vibe-basis cannot import vibe-qc
(``vibe-basis/tests/test_no_vibeqc_dependency.py``), so the ground-state
table is duplicated. This test is the only place that can see both, and
it is what stops them drifting.
"""

from __future__ import annotations

import pytest

from vibeqc.atomization import _GROUND_STATE_MULTIPLICITY as VIBEQC_TABLE

crystal_atom = pytest.importorskip(
    "vibe_basis.backends.crystal_atom",
    reason="vibe-basis is an optional [basisopt] extra",
)
GROUND_STATE_MULTIPLICITY = crystal_atom.GROUND_STATE_MULTIPLICITY
emit_input_atom = crystal_atom.emit_input_atom

# A throwaway inline basis; the deck's spin/symmetry keywords are what
# is under test, not the basis itself.
BASIS = "8 1\n0 0 3 6.0 1.0\n 100.0 1.0\n 10.0 1.0\n 1.0 1.0\n"


# ---------------------------------------------------------------------------
# The duplicated table must not drift
# ---------------------------------------------------------------------------


def test_multiplicity_tables_match_across_the_tier_seam():
    assert GROUND_STATE_MULTIPLICITY == VIBEQC_TABLE, (
        "vibe_basis.backends.crystal_atom.GROUND_STATE_MULTIPLICITY has "
        "drifted from vibeqc.atomization._GROUND_STATE_MULTIPLICITY. They are "
        "duplicated because vibe-basis may not import vibe-qc; update both."
    )


def test_the_table_is_physically_right_where_it_matters():
    """Spot-check the elements the cohesive test sets actually use."""
    # Closed shells.
    for Z in (2, 4, 10, 12, 18, 20, 36):
        assert GROUND_STATE_MULTIPLICITY[Z] == 1
    # Doublets: one unpaired electron.
    for Z in (1, 3, 9, 11, 17, 19, 35):
        assert GROUND_STATE_MULTIPLICITY[Z] == 2
    # Carbon and oxygen are triplets; nitrogen is a quartet. Getting
    # oxygen wrong would corrupt every oxide in the test set.
    assert GROUND_STATE_MULTIPLICITY[6] == 3
    assert GROUND_STATE_MULTIPLICITY[8] == 3
    assert GROUND_STATE_MULTIPLICITY[7] == 4


# ---------------------------------------------------------------------------
# The emitted deck
# ---------------------------------------------------------------------------


def test_open_shell_atom_is_aspherical_and_spin_polarised_dft():
    """Oxygen, a triplet, at PBE."""
    deck = emit_input_atom(8, BASIS, method="pbe")
    assert "SYMMREMO" in deck        # aspherical
    assert "\nSPIN\n" in deck        # unrestricted DF
    assert "SPINLOCK\n2 50" in deck  # nalpha - nbeta = 2

    # SYMMREMO belongs to the geometry block, i.e. before its END and
    # before the basis; misplaced, CRYSTAL would reject the deck.
    assert deck.index("SYMMREMO") < deck.index("END")


def test_open_shell_atom_uses_uhf_at_hartree_fock():
    """The HF route has no DFT block to put SPIN in."""
    deck = emit_input_atom(8, BASIS, method="rhf")
    assert "UHF" in deck
    assert "DFT" not in deck
    assert "SPINLOCK\n2 50" in deck
    assert "SYMMREMO" in deck


def test_closed_shell_atom_gets_none_of_it():
    """A restricted symmetric solution is already correct for Mg."""
    deck = emit_input_atom(12, BASIS, method="pbe")
    assert "SYMMREMO" not in deck
    assert "SPINLOCK" not in deck
    assert "\nSPIN\n" not in deck
    assert "UHF" not in deck


def test_spinlock_value_is_the_unpaired_count():
    """NSPIN is nalpha - nbeta, i.e. multiplicity - 1."""
    for Z, expected in ((1, 1), (6, 2), (7, 3), (8, 2), (9, 1)):
        deck = emit_input_atom(Z, BASIS, method="rhf")
        assert f"SPINLOCK\n{expected} 50" in deck, f"Z={Z}"


def test_caller_can_override_the_spin_state():
    """Needed for a non-ground state, or an element off the table."""
    deck = emit_input_atom(8, BASIS, method="pbe", n_unpaired=0)
    assert "SPINLOCK" not in deck and "SYMMREMO" not in deck

    deck = emit_input_atom(26, BASIS, method="pbe", n_unpaired=4)  # Fe, off-table
    assert "SPINLOCK\n4 50" in deck


def test_spinlock_cycles_are_tunable():
    deck = emit_input_atom(8, BASIS, method="pbe", spinlock_cycles=30)
    assert "SPINLOCK\n2 30" in deck


def test_untabulated_element_without_an_explicit_spin_is_refused():
    """Guessing a transition metal's ground state would be worse than
    failing: it is exactly the silent-wrong-number class this fixes."""
    with pytest.raises(ValueError, match="no tabulated free-atom ground state"):
        emit_input_atom(26, BASIS, method="pbe")


def test_negative_unpaired_count_is_refused():
    with pytest.raises(ValueError, match="n_unpaired must be >= 0"):
        emit_input_atom(8, BASIS, method="pbe", n_unpaired=-1)


def test_unknown_method_still_refused():
    with pytest.raises(ValueError, match="unknown method"):
        emit_input_atom(8, BASIS, method="not-a-functional")


# ---------------------------------------------------------------------------
# The engine reports the convention it used
# ---------------------------------------------------------------------------


def test_engine_tags_the_convention_it_actually_used():
    """`atom_reference` must describe the *occupancy*, not just the box.

    The previous tag was ``"p1_box"``, which says where the atom sits and
    nothing about how it was occupied. An auditor comparing against the
    plane-wave oracle needs the part that can be wrong.
    """
    engines = pytest.importorskip("vibe_basis.engines.crystal23")

    class _Recorder(engines.Crystal23Engine):
        captured: dict = {}

        def _run_deck(self, deck, name, *, method, **detail):
            _Recorder.captured = dict(detail)
            from vibe_basis.engine import EngineEnergy

            return EngineEnergy(
                energy=-1.0, ok=True, engine=self.name, detail=dict(detail)
            )

    engine = _Recorder(transport=object())

    engine.atom_energy(BASIS, 8, method="pbe")     # triplet
    assert _Recorder.captured["atom_reference"] == "p1_box_aspherical_spin_polarised"
    assert _Recorder.captured["n_unpaired"] == 2

    engine.atom_energy(BASIS, 12, method="pbe")    # closed shell
    assert _Recorder.captured["atom_reference"] == "p1_box_closed_shell"
    assert _Recorder.captured["n_unpaired"] == 0
