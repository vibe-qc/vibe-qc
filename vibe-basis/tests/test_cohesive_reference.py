"""ATOMBSSE labels in CRYSTAL's primitive cell, and the reference-tree reader.

The reader is exercised on a small synthetic tree with the shapes of the pob
cohesive reference set: an energy log citing each run, a bulk deck relaxed
with OPTGEOM, counterpoise decks whose geometry lists the atoms in another
order, a truncated atom output, and a bulk energy that the log gives only in
its arithmetic. No reference data is in the repository.
"""

from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest

from vibe_basis.backends.crystal_atom import (
    emit_input_atom_counterpoise,
    primitive_cell_labels,
)
from vibe_basis.engines.crystal23 import Crystal23Engine
from vibe_basis.io.cohesive_reference import (
    conventional_cubic_a,
    deck_settings,
    final_primitive_cell,
    load_reference_system,
)
from vibe_basis.io.structures import STRUCTURES
from vibe_basis.pipeline import HARTREE_TO_KJ_PER_MOL
from vibe_basis.transports.base import TransportError

K2O = STRUCTURES["K2O"]                      # asymmetric unit (O, K)
K2O_K_FIRST = replace(K2O, crystal_asymm_unit=K2O.crystal_asymm_unit[::-1])
INLINE = "8 1\n0 0 1 2.0 1.0\n 1.0 1.0\n19 1\n0 0 1 2.0 1.0\n 1.0 1.0\n99 0\n"


# ---------------------------------------------------------------------------
# IAT is a primitive-cell label
# ---------------------------------------------------------------------------


def test_primitive_cell_labels_run_through_one_orbit_at_a_time():
    assert primitive_cell_labels(K2O) == [8, 19, 19]
    assert primitive_cell_labels(K2O_K_FIRST) == [19, 19, 8]
    # A centred cell that is only the asymmetric unit cannot say.
    assert primitive_cell_labels(replace(K2O, unit_cell=K2O.crystal_asymm_unit)) is None


def test_the_counterpoise_deck_takes_the_label_and_the_element_it_names():
    deck = emit_input_atom_counterpoise(K2O_K_FIRST, 3, INLINE, method="pbe")
    assert "ATOMBSSE\n3 30 10.0\nSYMMREMO" in deck   # O: open shell
    with pytest.raises(ValueError, match="is Z=8, not Z=19"):
        emit_input_atom_counterpoise(K2O_K_FIRST, 3, INLINE, method="pbe", Z=19)
    unexpanded = replace(K2O_K_FIRST, unit_cell=K2O.crystal_asymm_unit)
    with pytest.raises(ValueError, match="pass Z"):
        emit_input_atom_counterpoise(unexpanded, 3, INLINE, method="pbe")
    deck = emit_input_atom_counterpoise(unexpanded, 3, INLINE, method="pbe", Z=8)
    assert "ATOMBSSE\n3 30 10.0\nSYMMREMO" in deck


class _Capture:
    def __init__(self):
        self.decks = []

    def run(self, work_dir, command, dest, **kwargs):
        self.decks += [p.read_text() for p in Path(work_dir).glob("*.d12")]
        raise TransportError("captured")


def test_the_engine_finds_o_at_label_3_when_k_comes_first(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert Crystal23Engine._atom_index(K2O, 8) == 1
    assert Crystal23Engine._atom_index(K2O_K_FIRST, 8) == 3
    assert Crystal23Engine._atom_index(K2O_K_FIRST, 19) == 1
    t = _Capture()
    eng = Crystal23Engine(transport=t, bsse_nstar=10, bsse_rmax=7.5)
    eng.atom_energy(INLINE, 8, "pbe", host=K2O_K_FIRST)
    unexpanded = replace(K2O_K_FIRST, unit_cell=K2O.crystal_asymm_unit)
    eng.atom_energy(INLINE, 8, "pbe", host=unexpanded, atom_label=3)
    assert all("ATOMBSSE\n3 10 7.5\n" in d for d in t.decks) and len(t.decks) == 2


# ---------------------------------------------------------------------------
# A synthetic reference tree
# ---------------------------------------------------------------------------

E_BULK, E_K, E_O = -1203.25, -599.6, -75.06
A_START, A_PRIM = 6.40, 4.5
BANNER = " *                              CRYSTAL23                          *\n"
END = " EEEEEEEEEE TERMINATION  DATE 01 01 2026 TIME 00:00:00.0\n"


def _geometry(asym: list[tuple[int, str]], cell: str) -> str:
    rows = "".join(f"   {z}  {xyz}\n" for z, xyz in asym)
    return f"K2O\nCRYSTAL\n   0 0 0\n 225\n   {cell}\n   {len(asym)}\n{rows}"


def _atom_list(rows: list[tuple[int, str]]) -> str:
    body = "".join(
        f"      {i} T  {z:2d} {sym:<2s}    0.0E+00  0.0E+00  0.0E+00\n"
        for i, (z, sym) in enumerate(rows, start=1)
    )
    return (
        f" ATOMS IN THE ASYMMETRIC UNIT    2 - ATOMS IN THE UNIT CELL:    {len(rows)}\n"
        "     ATOM                 X/A                 Y/B                 Z/C\n"
        " *******************************************************************************\n"
        + body
    )


def _scf(energy: float, cycles: int = 7) -> str:
    return (
        f" CYCLE {cycles} TOTAL ENERGY(DFT)(AU)(  {cycles}) {energy:.13E} DE-1.0E-09\n"
        f" == SCF ENDED - CONVERGENCE ON ENERGY      E(AU) {energy:.13E} CYCLES   {cycles}\n"
    )


@pytest.fixture
def tree(tmp_path):
    d = tmp_path / "K2O"
    d.mkdir()
    K_first = [(19, "0.25 0.25 0.25"), (8, "0.0 0.0 0.0")]
    O_first = K_first[::-1]
    # Bulk: relaxed with OPTGEOM, keyword basis, SHRINK 8 24.
    (d / "K2O_pob_r2SCAN.d12").write_text(
        _geometry(K_first, f"{A_START}") + "OPTGEOM\nENDOPT\nBASISSET\npob-tzvp-rev2\n"
        "DFT\nr2SCAN\nENDDFT\nTOLINTEG\n 9 9 9 18 54\nSHRINK\n 8 24\nEND\n"
    )
    final = (
        " FINAL OPTIMIZED GEOMETRY - DIMENSIONALITY OF THE SYSTEM      3\n"
        " LATTICE PARAMETERS (ANGSTROMS AND DEGREES) - BOHR = 0.5291772083 ANGSTROM\n"
        "         A              B              C           ALPHA      BETA       GAMMA\n"
        f"     {A_PRIM:.8f}     {A_PRIM:.8f}     {A_PRIM:.8f}    60.000000  60.000000  60.000000\n"
    )
    (d / "K2O_pob_r2SCAN.111.out").write_text(
        BANNER + _atom_list([(19, "K"), (19, "K"), (8, "O")]) + _scf(E_BULK) + final + END
    )
    # K counterpoise atom: deck lists K first (IAT 1); converged output.
    (d / "K2O_pob_r2SCAN_atombsse_K.d12").write_text(
        _geometry(K_first, f"{A_START}") + "ATOMBSSE\n  1 20 7.5\nBASISSET\npob-tzvp-rev2\nEND\n"
    )
    (d / "K2O_pob_r2SCAN_atombsse_K.222.out").write_text(
        BANNER + _atom_list([(19, "K"), (0, "XX")]) + _scf(E_K) + END
    )
    # O counterpoise atom: deck lists O first (IAT 1 again); the cited run is
    # gone, a truncated rerun of the same deck is there.
    (d / "K2O_pob_r2SCAN_atombsse_O.d12").write_text(
        _geometry(O_first, f"{A_START}") + "ATOMBSSE\n  1 10 7.5\nSYMMREMO\nBASISSET\npob-tzvp-rev2\nEND\n"
    )
    (d / "K2O_pob_r2SCAN_atombsse_O.444.out").write_text(
        BANNER + _atom_list([(8, "O"), (0, "XX")]) + f" Y.DF 12 ETOT(AU) {E_O:.12E} DETOT -1.0E-07\n"
    )
    total = (2 * E_K + E_O - E_BULK) * HARTREE_TO_KJ_PER_MOL
    log = (
        "K2O_pob_r2SCAN_atombsse_K.222.out: TOTAL ENERGY(DFT)(AU)(  7) "
        f"{E_K:.13E} DE-1.0E-09\n"
        f"K2O_pob_r2SCAN_atombsse_O.333.out: TOTAL ENERGY(DFT)(AU)(  9) {E_O:.13E} DE-2.0E-07\n"
        f"E_atom K2O = ({E_K:.13E}*2{E_O:+.13E}-{E_BULK:.13E})*2625.5 = {total:.1f} kJ/mol r2SCAN pob\n"
    )
    return tmp_path, log, round(total, 1)


def test_deck_settings_and_final_cell():
    s = deck_settings(
        _geometry([(19, "0.25 0.25 0.25"), (8, "0 0 0")], "6.4")
        + "OPTGEOM\nENDOPT\nBASISSET\npob-tzvp-rev2\nDFT\nr2SCAN\nENDDFT\n"
        "TOLINTEG\n 9 9 9 18 54\nSHRINK\n 8 24\nEND\n"
    )
    assert (s.spacegroup, s.lattice, s.optgeom, s.basis_keyword) == (225, (6.4,), True, "pob-tzvp-rev2")
    assert (s.functional, s.tolinteg, s.shrink, s.atombsse) == ("r2SCAN", (9, 9, 9, 18, 54), (8, 24), None)
    assert conventional_cubic_a((A_PRIM,) * 3 + (60.0,) * 3, 225) == pytest.approx(A_PRIM * math.sqrt(2))
    assert conventional_cubic_a((4.1,) * 3 + (90.0,) * 3, 221) == 4.1
    assert final_primitive_cell("no geometry") is None


def test_a_published_value_reassembles_from_the_runs_the_log_cites(tree):
    root, log, published = tree
    r = load_reference_system(root, "K2O", published, log)
    assert r.counts == {19: 2, 8: 1} and r.formula_units == 1
    assert r.cohesive_kj() == pytest.approx(published, abs=0.05)
    # The bulk is the run whose energy is in the arithmetic, at its relaxed cell.
    assert r.bulk.output_name == "K2O_pob_r2SCAN.111.out"
    assert r.bulk.structure.a == pytest.approx(A_PRIM * math.sqrt(2))
    assert r.bulk.settings.shrink == (8, 24)
    k, o = sorted(r.atoms, key=lambda c: c.Z, reverse=True)
    # Both decks say IAT 1; the element comes from each run, not the bulk's order.
    assert (k.Z, k.settings.atombsse) == (19, (1, 20, 7.5))
    assert (o.Z, o.settings.atombsse) == (8, (1, 10, 7.5))
    assert (k.output_status, o.output_status) == ("converged", "missing")
    assert o.structure.a == A_START
    # A replacement energy goes in where the reference energy went.
    shifted = r.cohesive_kj({k.output_name: E_K - 0.001})
    assert shifted == pytest.approx(r.cohesive_kj() - 2 * 0.001 * HARTREE_TO_KJ_PER_MOL)


def test_a_value_the_log_does_not_give_is_refused(tree):
    root, log, published = tree
    with pytest.raises(LookupError):
        load_reference_system(root, "K2O", published + 5.0, log)


# ---------------------------------------------------------------------------
# Relaxing a bulk
# ---------------------------------------------------------------------------


def test_optgeom_closes_the_geometry_input_as_the_reference_decks_do():
    from vibe_basis.backends.crystal import emit_input_inline

    deck = emit_input_inline(STRUCTURES["MgO"], "12 0\n99 0\n", "r2scan", optgeom=True, toldeg=5)
    assert "8 0.500000 0.500000 0.500000\nOPTGEOM\nTOLDEG\n5\nENDOPT\nEND\n12 0" in deck
    assert "OPTGEOM" not in emit_input_inline(STRUCTURES["MgO"], "12 0\n99 0\n", "r2scan")
    with pytest.raises(ValueError, match="OPTGEOM keyword"):
        emit_input_inline(STRUCTURES["MgO"], "12 0\n99 0\n", "r2scan", toldeg=5)


class _Finished:
    """A transport whose job 'ran': it writes the output text it was given."""

    def __init__(self, out_text):
        self.out_text = out_text

    def run(self, work_dir, command, dest, **kwargs):
        from types import SimpleNamespace

        out = Path(dest) / "job"
        out.mkdir(parents=True, exist_ok=True)
        (out / "MgO.out").write_text(self.out_text)
        return SimpleNamespace(output_dir=out)


def test_a_relaxation_that_did_not_converge_is_not_an_energy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scf = BANNER + _scf(-275.5) + END
    ok = Crystal23Engine(transport=_Finished(scf + " * OPT END - CONVERGED * E(AU): -275.5\n"),
                         optgeom=True).crystal_energy("12 0\n99 0\n", STRUCTURES["MgO"], "r2scan")
    assert ok.ok and ok.energy == -275.5
    bad = Crystal23Engine(transport=_Finished(scf), optgeom=True).crystal_energy(
        "12 0\n99 0\n", STRUCTURES["MgO"], "r2scan")
    assert not bad.ok and bad.failure_mode == "optgeom_not_converged"
    sp = Crystal23Engine(transport=_Finished(scf)).crystal_energy("12 0\n99 0\n", STRUCTURES["MgO"], "r2scan")
    assert sp.ok
