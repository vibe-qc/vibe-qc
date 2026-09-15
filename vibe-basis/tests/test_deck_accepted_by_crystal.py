"""Deck-structure invariants CRYSTAL enforces but text inspection does not.

Every test here pins something a real CRYSTAL23 run rejected. None of it
was catchable by reading the emitted text, and none of the existing
tests caught it, because they compare decks to expected strings that
were themselves never run.

Found while converging the counterpoise ghost shell against a local
CRYSTAL23 v1.0.1, i.e. the first time these decks were fed to the
program they are written for.
"""

from __future__ import annotations

import pytest

from vibe_basis.backends.crystal import (
    _CRYSTAL_DFT_FUNCTIONALS,
    _NEEDS_HUGEGRID,
    emit_input_inline,
)
from vibe_basis.backends.crystal_atom import (
    emit_input_atom,
    emit_input_atom_counterpoise,
)
from vibe_basis.io.structures import STRUCTURES

# An inline basis block as vibeqc.basis_crystal.emit_crystal writes one:
# per-element shells, terminated by the ` 99 0` sentinel.
# Shell records are `ITYB LAT NG CHE SCAL`: general basis, shell type
# (0 = S, 2 = P), primitive count, electrons in the shell, scale factor.
# The electron counts must be physical -- CRYSTAL rejects more than two
# in an s shell -- and must sum to Z.
INLINE = (
    "3 2\n"                                    # Li, 3 electrons
    "0 0 1 2.0 1.0\n 10.0 1.0\n"               #   1s2
    "0 0 1 1.0 1.0\n 0.5 1.0\n"                #   2s1
    "9 3\n"                                    # F, 9 electrons
    "0 0 1 2.0 1.0\n 200.0 1.0\n"              #   1s2
    "0 0 1 2.0 1.0\n 10.0 1.0\n"               #   2s2
    "0 2 1 5.0 1.0\n 1.0 1.0\n"                #   2p5
    " 99 0\n"
)
LIF = STRUCTURES["LiF"]


def _all_decks():
    """One of each emitter, covering the open- and closed-shell paths."""
    return {
        "inline": emit_input_inline(LIF, INLINE, method="pbe"),
        "atom_open": emit_input_atom(9, INLINE, method="pbe"),
        # Chemically a fiction; this is a deck-shape check, and
        # forcing a closed shell is how the no-SPINLOCK path is covered.
        "atom_closed": emit_input_atom(9, INLINE, method="pbe", n_unpaired=0),
        "cp_open": emit_input_atom_counterpoise(LIF, 2, INLINE, method="pbe"),
        "cp_closed": emit_input_atom_counterpoise(LIF, 1, INLINE, method="rhf"),
    }


# ---------------------------------------------------------------------------
# No blank lines
# ---------------------------------------------------------------------------


def test_no_deck_contains_a_blank_line():
    """CRYSTAL reads the SCF section as one keyword per line, so a blank
    line is an *empty keyword*::

        ERROR **** READM2 **** KEYWORD            NOT ALLOWED

    Every emitter here interpolates optional blocks that are empty when
    the option is off (no DFT block at Hartree-Fock, no SPINLOCK for a
    closed shell), and each of those left a blank line behind.
    """
    for name, deck in _all_decks().items():
        assert deck is not None, name
        blanks = [i for i, line in enumerate(deck.splitlines(), 1) if not line.strip()]
        assert not blanks, f"{name}: blank line(s) at {blanks}"


# ---------------------------------------------------------------------------
# ENDBS
# ---------------------------------------------------------------------------


def test_inline_basis_is_closed_with_endbs():
    """``99 0`` ends the basis *data*; the basis *section* still needs a
    terminator before the SCF keywords start.

    Without it CRYSTAL reads ``DFT`` as a basis-section keyword::

        ERROR **** INPBAS **** KEYWORD          NOT ALLOWED
    """
    for name, deck in _all_decks().items():
        lines = [ln.strip() for ln in deck.splitlines()]
        assert "ENDBS" in lines, f"{name}: no ENDBS"
        assert lines.index("99 0") < lines.index("ENDBS"), (
            f"{name}: ENDBS must follow the 99 0 basis sentinel"
        )


def test_scf_keywords_come_after_endbs():
    """A DFT block placed before ENDBS is inside the basis section."""
    deck = emit_input_atom(9, INLINE, method="pbe")
    lines = [ln.strip() for ln in deck.splitlines()]
    assert lines.index("ENDBS") < lines.index("DFT")
    assert lines.index("ENDBS") < lines.index("SPINLOCK")


# ---------------------------------------------------------------------------
# The SCAN family
# ---------------------------------------------------------------------------


def test_r2scan_is_an_accepted_functional():
    """The pob cohesive reference set is computed at r2SCAN, so the
    emitter has to accept it. CRYSTAL23 supports SCAN and r2SCAN plus
    r2SCAN hybrids at 10 / 25 / 50 % HF exchange."""
    for f in ("scan", "r2scan", "scan0", "r2scanh", "r2scan0", "r2scan50"):
        assert f in _CRYSTAL_DFT_FUNCTIONALS, f
    deck = emit_input_atom(9, INLINE, method="r2scan")
    assert "R2SCAN" in deck


def test_hugegrid_is_off_by_default_because_the_reference_set_omits_it():
    """The manual recommends HUGEGRID for SCAN; the protocol does not use it.

    Not one of the 1759 decks in the pob cohesive-energy reference set
    carries HUGEGRID, and those decks produced every published number
    this work is gated against. When the goal is reproducing a result,
    matching its protocol beats following the manual, so the default is
    off and the manual's advice is opt-in.
    """
    assert _NEEDS_HUGEGRID <= _CRYSTAL_DFT_FUNCTIONALS
    for f in sorted(_NEEDS_HUGEGRID):
        assert "HUGEGRID" not in emit_input_atom(9, INLINE, method=f), f
        assert "HUGEGRID" in emit_input_atom(9, INLINE, method=f, hugegrid=True), f


def test_hugegrid_is_never_offered_for_non_scan_functionals():
    """Even opted in, it applies only where the manual prescribes it."""
    for f in ("pbe", "pbe0", "pw1pw", "b3lyp"):
        assert "HUGEGRID" not in emit_input_atom(9, INLINE, method=f, hugegrid=True), f


def test_tolinteg_matches_the_reference_set_by_default():
    """Every reference deck uses 9 9 9 18 54, far tighter than CRYSTAL's
    own defaults. The difference is not negligible at the sub-kJ/mol
    tolerance a cohesive-energy gate works to."""
    for deck in _all_decks().values():
        assert "TOLINTEG" in deck
        lines = [ln.strip() for ln in deck.splitlines()]
        assert lines[lines.index("TOLINTEG") + 1] == "9 9 9 18 54"


def test_tolinteg_can_be_dropped_for_crystal_defaults():
    assert "TOLINTEG" not in emit_input_atom(9, INLINE, method="pbe", tolinteg=None)
    assert "TOLINTEG" not in emit_input_inline(LIF, INLINE, method="pbe", tolinteg=None)


# ---------------------------------------------------------------------------
# Live CRYSTAL, when one is installed
# ---------------------------------------------------------------------------


def _crystal_binary():
    import shutil
    from pathlib import Path

    for candidate in (Path.home() / "bin" / "crystal", "crystal"):
        found = shutil.which(str(candidate))
        if found:
            return found
    return None


@pytest.mark.skipif(_crystal_binary() is None, reason="no CRYSTAL binary")
def test_a_real_crystal_accepts_the_deck_structure():
    """Feed a deck to CRYSTAL and assert it gets past input parsing.

    Checks that the *input* was accepted, not that the SCF converged:
    this is a deck-structure test, and tying it to an energy would make
    it slow and chemistry-dependent. All three bugs it guards showed up
    as input-parse errors.

    The assertion is **positive** -- CRYSTAL must reach the point of
    reporting a basis it built. An earlier version only asserted the
    absence of two error strings and passed vacuously when the deck
    failed for a third reason (an element with no basis block), which
    is precisely the trap an absence-only assertion sets.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    # Z must be an element the INLINE basis actually covers, or CRYSTAL
    # fails on the missing basis rather than on deck structure.
    deck = emit_input_atom(9, INLINE, method="pbe")
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "INPUT").write_text(deck)
        with (d / "INPUT").open() as fh:
            proc = subprocess.run(
                [_crystal_binary()], stdin=fh, capture_output=True,
                text=True, timeout=600, cwd=d,
            )
    out = proc.stdout
    errors = [ln.strip() for ln in out.splitlines() if "ERROR ****" in ln]
    assert not errors, "CRYSTAL rejected the deck:\n" + "\n".join(errors)
    assert "NUMBER OF AO" in out, (
        "CRYSTAL never reported a built basis, so the deck did not survive "
        "input processing:\n" + out[-2000:]
    )


# ---------------------------------------------------------------------------
# The bulk and atom emitters must agree
# ---------------------------------------------------------------------------


def test_bulk_and_atom_decks_choose_the_same_integration_grid():
    """A cohesive energy subtracts a bulk from its own free atoms.

    If one side is integrated on HUGEGRID and the other on the default,
    the difference does not cancel: it surfaces as a few kJ/mol that
    reads as chemistry. The plane-wave reference work learned this
    concretely, holding atoms at a fixed cutoff while sweeping the solid
    and producing a spurious rising curve
    (``handovers/HANDOVER_GPAW_PW_REFERENCE.md``).

    Only the atom emitter carried the grid keyword when it was first
    added, so this pins the pair rather than either one alone.
    """
    for f in sorted(_CRYSTAL_DFT_FUNCTIONALS):
        for opt in (False, True):
            bulk = emit_input_inline(LIF, INLINE, method=f, hugegrid=opt)
            atom = emit_input_atom(9, INLINE, method=f, hugegrid=opt)
            cp = emit_input_atom_counterpoise(LIF, 2, INLINE, method=f,
                                              hugegrid=opt)
            grids = {"bulk": "HUGEGRID" in bulk, "atom": "HUGEGRID" in atom,
                     "counterpoise": "HUGEGRID" in cp}
            assert len(set(grids.values())) == 1, f"{f} hugegrid={opt}: {grids}"
            assert grids["bulk"] is (opt and f in _NEEDS_HUGEGRID), f


@pytest.mark.skipif(_crystal_binary() is None, reason="no CRYSTAL binary")
def test_a_real_crystal_accepts_the_bulk_deck():
    """The bulk half of a cohesive energy, through the same gate.

    ``emit_input_atom`` was the only emitter fed to CRYSTAL when the five
    deck defects were found; ``emit_input_inline`` shares the basis and
    SCF blocks with it and was fixed alongside, but sharing a fix is not
    the same as being tested.
    """
    import subprocess
    import tempfile
    from pathlib import Path

    deck = emit_input_inline(LIF, INLINE, method="pbe", shrink=4)
    assert deck is not None
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        (d / "INPUT").write_text(deck)
        with (d / "INPUT").open() as fh:
            proc = subprocess.run(
                [_crystal_binary()], stdin=fh, capture_output=True,
                text=True, timeout=600, cwd=d,
            )
    errors = [ln.strip() for ln in proc.stdout.splitlines() if "ERROR ****" in ln]
    assert not errors, "CRYSTAL rejected the bulk deck:\n" + "\n".join(errors)
    assert "NUMBER OF AO" in proc.stdout


# ---------------------------------------------------------------------------
# Non-symmorphic space groups: ORIGIN + TRASREMO
# ---------------------------------------------------------------------------


def test_counterpoise_emits_origin_and_trasremo_before_atombsse():
    """ATOMBSSE on a non-symmorphic group needs the translations gone.

    ATOMBSSE carves a 0D cluster out of the crystal, and an operator
    carrying a translation cannot map a finite cluster onto itself, so in
    a non-symmorphic group the survivors do not close::

        ERROR **** MULTIP **** SYMMOPS DO NOT FORM A GROUP

    That killed every space-group-227 system in the pob set (C-diamond,
    Si, Ge) at every rung of the ghost-shell ladder, while the symmorphic
    rocksalt (225) and zincblende (216) systems were unaffected. The
    reference set's own diamond decks carry the fix: ORIGIN, then
    TRASREMO, immediately before ATOMBSSE.

    Emitted unconditionally rather than keyed to a space-group table.
    Measured on CRYSTAL23 v1.0.1, PBE/pob-TZVP-REV2, nstar=5 rmax=4.0:

        Si   (SG 227, non-symmorphic)  with:    -289.21198632 Ha
                                       without: MULTIP error
        LiCl (SG 225, symmorphic)      with:    -459.94485757623 Ha
                                       without: -459.94485757623 Ha

    Bit-identical on the symmorphic system, so there is nothing to gate
    on: on a group with no translational operators ORIGIN has nothing to
    minimise and TRASREMO nothing to remove.
    """
    deck = emit_input_atom_counterpoise(LIF, 2, INLINE, method="pbe")
    lines = [ln.strip() for ln in deck.splitlines()]
    assert lines.index("ORIGIN") < lines.index("TRASREMO") < lines.index("ATOMBSSE")
    # ...and inside the geometry block, before its terminator.
    assert lines.index("ATOMBSSE") < lines.index("END")


def test_trasremo_can_be_disabled():
    """Opt-out, for anyone reproducing a deck that predates this."""
    deck = emit_input_atom_counterpoise(LIF, 2, INLINE, method="pbe", trasremo=False)
    assert "TRASREMO" not in deck and "ORIGIN" not in deck
    assert "ATOMBSSE" in deck
