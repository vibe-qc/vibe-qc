"""CRYSTAL basis-set input from a QVF-Basis record, shells and ECPs.

``vibeqc.basis_toolkit.to_crystal`` writes a record as the per-element
blocks CRYSTAL reads after the geometry (CRYSTAL23 manual, "Basis set
input" pp. 23-25, "Effective core pseudo-potentials" pp. 83-84), so a
CRYSTAL reference calculation can use exactly the basis and ECP vibe-qc
used.

What is pinned:

* every record CRYSTAL can hold reads back through the CRYSTAL readers
  (``parse_crystal_inline_basis``, ``from_crystal_atom``) to the identical
  shells and the same ECP operator, and its CHE sum per element is the
  neutral valence count CRYSTAL requires of a periodic cell (note 12);
* the pob records, which came from CRYSTAL text, give the Bredow files'
  numbers and electron counts back;
* what CRYSTAL cannot hold (shells above g, more primitives than its NG
  limits, an ECP projector above g) is refused, not written;
* the ground-state table that assigns CHE is the one the SAD guess uses
  (``cpp/src/guess.cpp``).
"""

from __future__ import annotations

import copy
import re
from pathlib import Path

import pytest

from vibeqc.basis_crystal import (
    CrystalAtomBasis,
    CrystalShell,
    parse_crystal_atom_basis_file,
    parse_crystal_inline_basis,
)
from vibeqc.basis_toolkit import (
    HarmonicType,
    from_crystal_atom,
    library,
    loss_report_crystal,
    to_crystal,
)
from vibeqc.basis_toolkit import exporter_crystal
from vibeqc.basis_toolkit.exporter_crystal import ground_state_electrons_per_l, to_crystal_atoms

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "python" / "vibeqc" / "basis_library" / "sources"


def _writable():
    out = []
    for stem in library.stems():
        try:
            to_crystal(library.load_record(stem))
        except ValueError:
            continue
        out.append(stem)
    return out


WRITABLE = _writable()


def _shells(element):
    return [
        (list(s.angular_momentum), [(p.exponent, p.coefficient) for p in s.primitives])
        for s in element.shells
    ]


def _operator(entry):
    """The ECP as an operator: core size and the nonzero terms per channel,
    the local one under "local"."""
    if entry is None:
        return None
    lmax = int(entry.angular_momentum_max)
    channels = {}
    for potential in entry.potentials:
        key = "local" if int(potential.angular_momentum) == lmax else int(potential.angular_momentum)
        terms = sorted(
            (t.power, t.exponent, t.coefficient) for t in potential.terms if t.coefficient != 0.0
        )
        if terms:
            channels[key] = terms
    return int(entry.n_core_electrons), channels


def test_the_main_ecp_families_are_writable():
    assert len(WRITABLE) >= 70
    assert {"def2-svp", "def2-tzvp", "lanl2dz", "pob-tzvp-rev2", "pob-dzvp-rev2", "vdzp"} <= set(WRITABLE)


@pytest.mark.parametrize("stem", WRITABLE)
def test_crystal_text_reads_back_to_the_record(stem):
    record = library.load_record(stem)
    atoms = parse_crystal_inline_basis(to_crystal(record), source=stem)
    assert [a.element_symbol for a in atoms] == list(record.elems_sorted())
    for atom in atoms:
        sym = atom.element_symbol
        again = from_crystal_atom(atom)
        assert _shells(again.elements[sym]) == _shells(record.elements[sym]), sym
        entry = record.ecps.get(record.elements[sym].ecp_id) if record.ecps else None
        assert _operator((again.ecps or {}).get(sym)) == _operator(entry), sym
        n_core = int(entry.n_core_electrons) if entry is not None else 0
        assert sum(s.occupancy for s in atom.shells) == atom.Z - n_core, sym


@pytest.mark.parametrize("family", ["pob-TZVP", "pob-TZVP-rev2", "pob-DZVP-rev2"])
def test_pob_records_give_the_bredow_files_back(family):
    """The bundled pob records were converted from these CRYSTAL files; the
    CRYSTAL text written from the record carries their numbers, and the
    same electron count per element (the CHE split may differ: Bredow's Ag
    is 4d9 5s2, the ground state 4d10 5s1)."""
    record = library.load_record(family.lower())
    for path in sorted((SOURCES / family).glob("*_*")):
        source = parse_crystal_atom_basis_file(path)
        (written,) = to_crystal_atoms(record, [source.element_symbol])
        assert written.has_ecp == source.has_ecp
        assert [(s.shell_type, s.exponents, s.coefficients, s.coefficients_p) for s in written.shells] == [
            (s.shell_type, s.exponents, s.coefficients, s.coefficients_p) for s in source.shells
        ], path.name
        assert sum(s.occupancy for s in written.shells) == sum(s.occupancy for s in source.shells)
        if source.has_ecp:
            assert written.ecp.znuc == source.ecp.znuc
            assert sorted((str(t.ell), t.alpha, t.coefficient, t.n_pow) for t in written.ecp.terms) == sorted(
                (str(t.ell), t.alpha, t.coefficient, t.n_pow) for t in source.ecp.terms
            ), path.name


def test_che_is_the_ground_state_filled_in_record_order():
    (o,) = to_crystal_atoms(library.load_record("def2-svp"), ["O"])
    assert [(s.shell_type, s.occupancy) for s in o.shells] == [
        ("S", 2.0), ("S", 2.0), ("S", 0.0), ("P", 4.0), ("P", 0.0), ("D", 0.0)
    ]
    (ag,) = to_crystal_atoms(library.load_record("def2-svp"), ["Ag"])
    per_l = {}
    for s in ag.shells:
        per_l[s.shell_type] = per_l.get(s.shell_type, 0.0) + s.occupancy
    assert per_l == {"S": 3.0, "P": 6.0, "D": 10.0, "F": 0.0}  # [Kr]4d10 5s1 outside 28
    assert ag.ecp.znuc == 19.0
    (h,) = to_crystal_atoms(library.load_record("sto-3g"), ["H"])
    assert [s.occupancy for s in h.shells] == [1.0]
    own = to_crystal_atoms(library.load_record("sto-3g"), ["H"], occupancies={"H": [0.0]})
    assert [s.occupancy for s in own[0].shells] == [0.0]


def test_lanl2dz_keeps_its_r_minus_two_terms():
    """NKL is the literal power of r: the Hay-Wadt local terms with
    Gaussian n = 0 are NKL = -2, and Cl is a 200+Z valence-only atom."""
    text = to_crystal(library.load_record("lanl2dz"), ["Cl"])
    rows = [line.split() for line in text.splitlines()]
    assert rows[0][0] == "217" and rows[1] == ["INPUT"] and rows[2][0] == "7.0"
    assert {r[-1] for r in rows[3:] if len(r) == 3 and r[-1].lstrip("-").isdigit()} >= {"-2", "-1", "0"}


def test_what_crystal_cannot_hold_is_refused():
    with pytest.raises(ValueError, match="no shell type"):
        to_crystal(library.load_record("cc-pv5z"), ["C"])
    with pytest.raises(ValueError, match="at most 10"):
        to_crystal(library.load_record("cc-pvdz"), ["Na"])
    with pytest.raises(ValueError, match="no closed-core occupation"):
        ground_state_electrons_per_l(57, 47)
    with pytest.raises(ValueError, match="1-98"):
        to_crystal(library.load_record("def2-mtzvp"), ["Es"])
    record = copy.deepcopy(library.load_record("6-31g*"))
    assert "harmonic_type" not in loss_report_crystal(record).fields_dropped
    record.elements["C"].shells[-1].harmonic_type = HarmonicType.CARTESIAN
    report = loss_report_crystal(record)
    assert "harmonic_type" in report.fields_dropped and "ecps" not in report.fields_dropped


def test_the_ground_state_tables_are_the_sad_guess_tables():
    """exporter_crystal mirrors ground_config_per_l / valence_config_per_l."""
    source = (ROOT / "cpp" / "src" / "guess.cpp").read_text()
    fill = re.search(r"fill_l = \{([^}]*)\}", source).group(1)
    assert tuple(int(x) for x in fill.split(",")) == exporter_crystal._FILL_L

    def table(opening: str) -> dict[int, tuple[int, ...]]:
        start = source.index(opening)
        block = source[start:source.index("};", start)]
        return {
            int(z): tuple(int(x) for x in v.split(","))
            for z, v in re.findall(r"\{(\d+),\s*\{([\d,\s]+)\}\}", block)
        }

    assert table("anomalies = {") == exporter_crystal._ANOMALIES
    assert table("cores{") == exporter_crystal._CLOSED_CORES


def test_the_conventional_atomic_number_is_read_as_crystal_reads_it():
    """Z is NAT mod 100; 200 <= NAT < 1000 carries an ECP (manual, basis
    set input, notes 3 and 5). NAT 108 used to read as Z = 108 and 306 as
    Z = 106."""
    from vibeqc.basis_crystal import parse_crystal_atom_basis

    shell = "0 0 1 2.0 1.0\n1.0 1.0\n"
    second = parse_crystal_atom_basis("108 1\n" + shell, "O")
    assert (second.Z, second.has_ecp) == (8, False)
    ecp = parse_crystal_atom_basis("306 1\nINPUT\n4.0 0 1 0 0 0 0\n1.0 1.0 0\n" + shell, "C")
    assert (ecp.Z, ecp.has_ecp) == (6, True)
    assert parse_crystal_atom_basis("1006 1\n" + shell, "C").Z == 6
    with pytest.raises(ValueError, match="conventional atomic number 99"):
        parse_crystal_atom_basis("99 1\n" + shell, "Es")


def test_hand_built_atoms_keep_their_che():
    from vibeqc.basis_crystal import emit_crystal

    atom = CrystalAtomBasis(Z=1, has_ecp=False, shells=[
        CrystalShell("S", 1.0, 1.0, exponents=[1.24], coefficients=[1.0]),
    ])
    assert emit_crystal([atom]).splitlines()[1] == "0 0   1  1.0 0.0"
