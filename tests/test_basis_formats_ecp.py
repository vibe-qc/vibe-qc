"""ORCA and NWChem basis text carries the record whole: shells and ECPs.

``vibeqc.basis_toolkit`` writes a QVF-Basis record as an ORCA ``%basis``
block (``NewGTO`` plus ``NewECP``) or as NWChem ``basis`` plus ``ECP``
blocks, so a reference calculation in either program can use exactly the
basis set and the pseudopotential vibe-qc used. Until 2026-10 both writers
dropped every ECP, rounded to 12 significant digits, and the ORCA one wrote
0-based primitive indices and had no label for l >= 7.

What is pinned:

* every bundled record, the 46 with ECPs among them, reads back from either
  text to the identical doubles (an SP shell comes back as its S and P
  shells over the same exponents, the same basis functions);
* the labels are each program's own: ORCA ``s p d f g h i k`` and integers
  from l = 8 (ORCA 6.1 manual, section 2.7.6), NWChem ``S P D F G H I K L``
  with no ``J``; ORCA primitive rows count from 1;
* an independent consumer agrees: PySCF reading the NWChem text reproduces
  the Ag V_ECP that ``examples/regression/pob_ag_ecp_reference.py`` built
  straight from the Bredow CRYSTAL records (``tests/data/
  pob_ag_ecp_reference.json``), trace and generalized eigenvalues.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vibeqc.basis_toolkit import from_nwchem, from_orca, library, to_nwchem, to_orca
from vibeqc.basis_toolkit.exporter_nwchem import loss_report_nwchem, nwchem_label
from vibeqc.basis_toolkit.exporter_orca import loss_report_orca, orca_label

STEMS = library.stems()
REFERENCE = Path(__file__).resolve().parent / "data" / "pob_ag_ecp_reference.json"


def _shells(data):
    """Shells per element with SP split into S and P, as exact floats."""
    out = {}
    for sym, element in data.elements.items():
        rows = []
        for shell in element.shells:
            am, prims = list(shell.angular_momentum), shell.primitives
            if am == [0, 1]:
                n = len(prims) // 2
                rows.append((0, tuple((p.exponent, p.coefficient) for p in prims[:n])))
                rows.append((1, tuple((p.exponent, p.coefficient) for p in prims[n:])))
            else:
                rows.append((am[0], tuple((p.exponent, p.coefficient) for p in prims)))
        out[sym] = rows
    return out


def _ecps(data):
    out = {}
    for sym, entry in (data.ecps or {}).items():
        channels = {
            int(p.angular_momentum): tuple((t.power, t.exponent, t.coefficient) for t in p.terms)
            for p in entry.potentials
        }
        out[sym] = (entry.n_core_electrons, entry.angular_momentum_max, sorted(channels.items()))
    return out


def test_the_bundled_ecp_records_are_all_covered():
    assert sum(1 for _, r in library.iter_records() if r.ecps) >= 46


@pytest.mark.parametrize("stem", STEMS)
def test_orca_text_reads_back_to_the_record(stem):
    record = library.load_record(stem)
    again = from_orca(to_orca(record), name=stem)
    assert _shells(again) == _shells(record)
    assert _ecps(again) == _ecps(record)
    for sym, element in record.elements.items():
        assert (again.elements[sym].ecp_id is None) == (element.ecp_id is None), sym


@pytest.mark.parametrize("stem", STEMS)
def test_nwchem_text_reads_back_to_the_record(stem):
    record = library.load_record(stem)
    again = from_nwchem(to_nwchem(record), name=stem)
    assert _shells(again) == _shells(record)
    # Every bundled ECP has its local channel one above its highest
    # projector, so the l NWChem leaves implicit comes back unchanged.
    assert _ecps(again) == _ecps(record)
    assert again.name == stem


def test_each_program_gets_its_own_shell_labels():
    assert [orca_label(l) for l in range(9)] == ["S", "P", "D", "F", "G", "H", "I", "K", "8"]
    assert [nwchem_label(l) for l in range(9)] == ["S", "P", "D", "F", "G", "H", "I", "K", "L"]
    # cc-pV5Z-RIFIT carries l = 7 and l = 8 fitting shells.
    record = library.load_record("cc-pv5z-rifit")
    orca_heads = {line.split()[0] for line in to_orca(record).splitlines()
                  if len(line.split()) == 2 and line.split()[1].isdigit()}
    assert {"K", "8"} <= orca_heads and "J" not in orca_heads
    nwchem_heads = {line.split()[1] for line in to_nwchem(record).splitlines()
                    if len(line.split()) == 2 and line.split()[1].isalpha()}
    assert {"K", "L"} <= nwchem_heads and "J" not in nwchem_heads


def test_orca_rows_count_from_one_and_keep_every_digit():
    text = to_orca(library.load_record("pob-tzvp-rev2"))
    lines = text.splitlines()
    assert lines[:3] == ["%basis", "  NewGTO H", "    S  3"]
    assert [line.split()[0] for line in lines[3:6]] == ["1", "2", "3"]
    assert "0.97098845035" in text  # #291's Rh value, unrounded
    block = text[text.index("NewECP Ag"):]
    assert block.splitlines()[1].split() == ["N_core", "28"]
    assert block.splitlines()[2].split() == ["lmax", "g"]


def test_ecps_are_no_longer_reported_lost():
    record = library.load_record("def2-svp")
    assert "ecps" not in loss_report_orca(record).fields_dropped
    assert "ecps" not in loss_report_nwchem(record).fields_dropped
    assert "NewECP Ag" in to_orca(record) and "\nECP\n" in to_nwchem(record)


def test_orca_import_accepts_atomic_numbers_and_refuses_what_it_cannot_resolve():
    text = "%basis\n NewGTO 8\n  s 1\n   1 1.5 1.0\n end\nend\n"
    assert set(from_orca(text).elements) == {"O"}
    with pytest.raises(ValueError, match="SCALE"):
        from_orca("NewGTO O\n S 1 SCALE 1.2\n 1 1.5 1.0\nend\n")
    with pytest.raises(ValueError, match="built-in"):
        from_orca('NewGTO O\n "def2-TZVP"\nend\n')


@pytest.mark.parametrize("stem", ["pob-tzvp", "pob-tzvp-rev2", "pob-dzvp-rev2"])
def test_pyscf_reading_the_nwchem_text_reproduces_the_independent_ag_ecp(stem):
    """PySCF as the consumer: our NWChem text must give the operator PySCF
    built from the CRYSTAL source records with its own parser (#515). The
    pob-DZVP-rev2 record has no eigenvalue reference yet; its trace is the
    PySCF anchor recorded on #292."""
    np = pytest.importorskip("numpy")
    scipy_linalg = pytest.importorskip("scipy.linalg")
    gto = pytest.importorskip("pyscf.gto")

    reference = json.loads(REFERENCE.read_text())
    geometry = reference["geometry_bohr"]
    text = to_nwchem(library.load_record(stem))
    basis_text, ecp_text = text.split("\nECP\n", 1)
    mol = gto.M(
        atom=[("Ag", geometry["Ag"]), ("Cl", geometry["Cl"])], unit="Bohr", spin=0, verbose=0,
        basis={s: gto.basis.parse(basis_text, symb=s) for s in ("Ag", "Cl")},
        ecp={"Ag": gto.basis.parse_ecp("ECP\n" + ecp_text, symb="Ag")},
    )
    v = mol.intor("ECPscalar")
    if stem in reference["bases"]:
        expected = reference["bases"][stem]
        assert mol.nao == expected["n_ao"]
        assert float(np.trace(v)) == pytest.approx(expected["trace"], abs=1e-10)
        eig = np.sort(scipy_linalg.eigh(v, mol.intor("int1e_ovlp"), eigvals_only=True))
        assert float(np.abs(eig - np.asarray(expected["generalized_eigenvalues"])).max()) < 1e-10
    else:
        assert float(np.trace(v)) == pytest.approx(111.9109001638, abs=1e-9)
