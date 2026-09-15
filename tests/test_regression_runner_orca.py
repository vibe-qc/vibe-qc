from __future__ import annotations

from examples.regression.core.runner_orca import (
    _is_double_hybrid,
    _orca_blocks,
    _orca_simpleinput,
)
from examples.regression.core.spec import MethodSpec


def test_orca_pure_dft_df_uses_auxj_basis():
    method = MethodSpec(id="rks-pbe-df", scf="rks", xc="pbe", df=True)

    simple = _orca_simpleinput(method, "def2-svp")

    assert simple == "PBE RI def2-svp TightSCF SP def2/J"


def test_orca_hybrid_and_hf_df_keep_rijk_auxjk():
    hybrid = MethodSpec(id="rks-b3lyp-df", scf="rks", xc="b3lyp", df=True)
    rhf = MethodSpec(id="rhf-df", scf="rhf", df=True)

    assert _orca_simpleinput(hybrid, "def2-svp") == (
        "B3LYP RIJK def2-svp TightSCF SP def2/JK"
    )
    assert _orca_simpleinput(rhf, "def2-svp") == (
        "HF RIJK def2-svp TightSCF SP def2/JK"
    )


# ---------------------------------------------------------------------------
# BUG 36 / BUG 36 regress: double-hybrid ORCA decks need AutoAux
# ---------------------------------------------------------------------------
# ORCA 6.1 requires an auxiliary basis for the RI-MP2 correlation step
# in double-hybrid functionals.  Without AutoAux (or an explicit /C
# basis) it exits with:
#   ERROR: RI-MP2 needs an AuxC basis but none was defined!
# The SCF-side DF aux (def2/JK) covers the JK fitting; the /C
# (correlation-fitting) aux is separate.  AutoAux generates one
# automatically from the orbital basis (Stoychev, Auer, Neese,
# JCTC 13, 554 (2017)).


def test_double_hybrid_orca_decks_pin_aux_and_all_electron_protocol():
    """Every calibrated double hybrid gets AutoAux plus NoFrozenCore."""
    for xc in ("b2plyp", "dsd-pbep86", "pwpb95"):
        method = MethodSpec(id=f"rks-{xc}", scf="rks", xc=xc)
        deck = _orca_simpleinput(method, "def2-svp")
        tokens = deck.split()
        assert "AutoAux" in tokens
        assert tokens.count("NoFrozenCore") == 1


def test_open_double_hybrid_orca_deck_pins_all_electron_protocol():
    """The unrestricted double-hybrid mapping uses the same MP2 model."""
    method = MethodSpec(id="uks-pwpb95", scf="uks", xc="pwpb95")
    deck = _orca_simpleinput(method, "def2-svp")
    tokens = deck.split()
    assert "AutoAux" in tokens
    assert tokens.count("NoFrozenCore") == 1


def test_regular_dft_does_not_add_autoux():
    """Regular DFT must NOT add AutoAux — only double-hybrids need it."""
    method = MethodSpec(id="rks-pbe", scf="rks", xc="pbe")
    deck = _orca_simpleinput(method, "def2-svp")
    assert "AutoAux" not in deck.split()
    assert "NoFrozenCore" not in deck.split()


def test_double_hybrid_with_df_gets_both_rijk_and_autoux():
    """DF double hybrid keeps SCF fitting and the all-electron MP2 model."""
    method = MethodSpec(id="rks-b2plyp-df", scf="rks", xc="b2plyp", df=True)
    deck = _orca_simpleinput(method, "def2-svp")
    tokens = deck.split()
    assert "RIJK" in tokens
    assert "def2/JK" in tokens
    assert "AutoAux" in tokens
    assert tokens.count("NoFrozenCore") == 1


def test_double_hybrid_without_df_still_gets_autoux():
    """Non-DF SCF still needs AutoAux and all-electron MP2 in ORCA."""
    method = MethodSpec(id="rks-b2plyp", scf="rks", xc="b2plyp", df=False)
    deck = _orca_simpleinput(method, "def2-svp")
    tokens = deck.split()
    assert "RIJK" not in tokens
    assert "RI" not in tokens
    assert "AutoAux" in tokens
    assert tokens.count("NoFrozenCore") == 1


def test_is_double_hybrid_detects_all_three():
    """_is_double_hybrid returns True for all supported double hybrids."""
    for xc in ("b2plyp", "dsd-pbep86", "pwpb95"):
        method = MethodSpec(id=f"rks-{xc}", scf="rks", xc=xc)
        assert _is_double_hybrid(method) is True


def test_is_double_hybrid_returns_false_for_regular_dft():
    """_is_double_hybrid returns False for regular DFT functionals."""
    for xc in ("pbe", "b3lyp", "lda", "blyp"):
        method = MethodSpec(id=f"rks-{xc}", scf="rks", xc=xc)
        assert _is_double_hybrid(method) is False
    # HF is not a double hybrid
    assert _is_double_hybrid(MethodSpec(id="rhf", scf="rhf")) is False


# ---------------------------------------------------------------------------
# BUG 109: CCSD(T) ORCA decks need %mdci MaxIter
# ---------------------------------------------------------------------------
# ORCA's 50-iteration CC default is too tight — cyclopropene
# CCSD(T)/def2-TZVP hits the wall at residual >0.003 and exits 126.
# The simple-input must emit "CCSD(T)" and the orcablocks channel
# must carry %mdci MaxIter 100.


def test_ccsdt_orca_deck_has_ccsdt_keyword():
    """CCSD(T) ORCA simple-input must contain the CCSD(T) keyword."""
    method = MethodSpec(id="ccsdt", scf="rhf", post="ccsd(t)")
    deck = _orca_simpleinput(method, "def2-svp")
    assert "CCSD(T)" in deck.split()


def test_ccsdt_orca_deck_has_nofrozencore():
    """CCSD(T) deck must use NoFrozenCore for all-electron correlation."""
    method = MethodSpec(id="ccsdt", scf="rhf", post="ccsd(t)")
    deck = _orca_simpleinput(method, "def2-svp")
    assert "NoFrozenCore" in deck.split()


def test_ccsdt_orca_deck_has_tightscf():
    """CCSD(T) deck must request TightSCF."""
    method = MethodSpec(id="ccsdt", scf="rhf", post="ccsd(t)")
    deck = _orca_simpleinput(method, "def2-svp")
    assert "TightSCF" in deck.split()


def test_ccsdt_orca_deck_has_sp():
    """CCSD(T) deck must request single-point."""
    method = MethodSpec(id="ccsdt", scf="rhf", post="ccsd(t)")
    deck = _orca_simpleinput(method, "def2-svp")
    assert "SP" in deck.split()


def test_mp2_deck_unchanged():
    """MP2 deck must still work — not broken by the CCSD(T) branch."""
    method = MethodSpec(id="mp2", scf="rhf", post="mp2")
    deck = _orca_simpleinput(method, "def2-svp")
    tokens = deck.split()
    assert "MP2" in tokens
    assert "NoFrozenCore" in tokens
    assert "TightSCF" in tokens


def test_correlated_orca_blocks_pin_fc_none_for_every_route():
    """Block-form FC_NONE is authoritative where NoFrozenCore is ignored."""

    routes = (
        MethodSpec(id="mp2", scf="rhf", post="mp2"),
        MethodSpec(id="ump2", scf="uhf", post="mp2"),
        MethodSpec(id="ccsd", scf="rhf", post="ccsd"),
        MethodSpec(id="uccsdt", scf="uhf", post="ccsd(t)"),
        MethodSpec(id="b2plyp", scf="rks", xc="b2plyp"),
        MethodSpec(id="u-pwpb95", scf="uks", xc="pwpb95"),
    )
    for method in routes:
        blocks = _orca_blocks(method, 200)
        assert blocks.count("%method") == 1
        assert blocks.count("FrozenCore FC_NONE") == 1


def test_uncorrelated_orca_blocks_do_not_claim_frozen_core_policy():
    method = MethodSpec(id="pbe", scf="rks", xc="pbe")
    blocks = _orca_blocks(method, 200)
    assert "%method" not in blocks
    assert "FrozenCore" not in blocks
