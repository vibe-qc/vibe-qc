"""Read a CRYSTAL cohesive-energy reference tree: decks, outputs, energy log.

M2's engine rung (#224, rung 2b) asks an engine to reproduce the published
pob cohesive energies. Those numbers come from a tree of CRYSTAL runs, one
directory per compound, plus an energy log (``TE.dat``) that records which
outputs each published number was assembled from. The tree is not part of
this repository; the caller passes its path.

What this module reads, and why each piece is needed:

* **Which outputs make a published number.** :func:`cited_outputs` finds the
  log line that produces the value and the output names cited above it. File
  names alone are not enough: the tree holds several runs per compound, and
  the log is the only record of which were combined.
* **How each run was set up.** :func:`deck_settings` reads a deck: the
  basis (``BASISSET`` keyword or inline block), functional, ``TOLINTEG``,
  ``SHRINK``, ``OPTGEOM``, ``ATOMBSSE IAT NSTAR RMAX``, the spin and the
  SCF-convergence input (``FMIXING``, ``ANDERSON``, ``ATOMSPIN``,
  ``SPINLOCK`` ...). The reference set varies all of these per system, and
  ``NSTAR``/``RMAX`` even per atom. The deck is the input the output echoes
  (:func:`echoed_input`): a ``.d12`` beside it may have been edited after
  the run, and in this set some were (NSTAR, RMAX, cell, basis, SCF input).
  The ``.d12`` file is read only for a run whose output does not echo it.
* **Where the bulk ended up.** A bulk deck with ``OPTGEOM`` reports its
  energy at the relaxed cell (:func:`final_primitive_cell`), and that is the
  cell a single point must use. The ghost cluster of a counterpoise atom is
  cut from the cell its own deck states, which in this set is sometimes the
  relaxed cell and sometimes the starting one.

A cohesive energy is assembled as the log does it, per primitive cell:
sum over atoms of n_el * E(el, counterpoise) - E(bulk), with n_el counted
from the bulk output's primitive-cell atom list.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional, Sequence

from ..backends.crystal import parse_output
from ..pipeline import HARTREE_TO_KJ_PER_MOL
from .library_bridge import parse_d12_geometry_text
from .structures import Structure
from ..backends.crystal_atom import _ELEMENT_SYMBOLS

_Z_BY_SYMBOL = {s: z for z, s in _ELEMENT_SYMBOLS.items()}

__all__ = [
    "DeckSettings",
    "ReferenceComponent",
    "ReferenceSystem",
    "cited_outputs",
    "conventional_cubic_a",
    "deck_settings",
    "echoed_input",
    "final_primitive_cell",
    "load_reference_system",
]

# Space groups whose conventional cubic cell is face- or body-centred; CRYSTAL
# prints the primitive cell, so the conventional a is a_p * sqrt(2) or
# a_p * 2/sqrt(3) (ITA, Bravais lattice of each group).
_F_CUBIC = frozenset({196, 202, 203, 209, 210, 216, 219, 225, 226, 227, 228})
_I_CUBIC = frozenset({197, 199, 204, 206, 211, 214, 217, 220, 229, 230})

_NUMBER = r"-?\d*\.\d+(?:[EeDd][+-]?\d+)?"


@dataclass(frozen=True)
class DeckSettings:
    """The parts of a CRYSTAL deck that decide its energy."""

    spacegroup: int
    lattice: tuple[float, ...]
    optgeom: bool
    atombsse: Optional[tuple[int, int, float]]
    basis_keyword: Optional[str]
    inline_basis: Optional[str]
    functional: Optional[str]
    tolinteg: Optional[tuple[int, ...]]
    shrink: Optional[tuple[int, int]]
    #: SCF-section input other than TOLINTEG, SHRINK and buffer sizes, verbatim.
    scf: tuple[str, ...] = ()
    #: Spin-polarised: ``SPIN`` in the DFT block, or ``UHF``.
    spin: bool = False
    #: Unpaired electrons: the ``SPINLOCK`` value, 0 when not spin-polarised.
    n_unpaired: Optional[int] = None
    #: ``SYMMREMO`` in the geometry input.
    symmremo: bool = False


#: SCF-section keywords :class:`DeckSettings` keeps elsewhere or that only
#: size buffers, each followed by one value line.
_NOT_SCF = frozenset({"TOLINTEG", "SHRINK", "EXCHSIZE", "BIPOSIZE"})


def echoed_input(out_text: str) -> Optional[str]:
    """The deck an output echoes before CRYSTAL's banner, or ``None``.

    The reference runs print their input ahead of the banner (and, for MPI
    runs, ahead of ``N PROCESSORS WORKING``). The deck starts on the title
    line above the ``CRYSTAL`` keyword.
    """
    lines = out_text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*\*{30,}", line):
            return None
        if line.strip() == "CRYSTAL" and i > 0:
            start = i - 1
            break
    if start is None:
        return None
    for j in range(start, len(lines)):
        if "PROCESSORS WORKING" in lines[j] or re.match(r"^\s*\*{30,}", lines[j]):
            return "\n".join(lines[start:j]).rstrip() + "\n"
    return None


def _scf_section(lines: list[str]) -> tuple[tuple[str, ...], bool]:
    """``(scf lines, spin)``: the input after the DFT block or basis, before the last END."""
    upper = [line.strip().upper() for line in lines]
    last_end = max((i for i, u in enumerate(upper) if u == "END"), default=len(lines))
    dft = next((i for i, u in enumerate(upper) if u == "DFT"), None)
    spin = False
    if dft is not None:
        close = next(
            (i for i in range(dft + 1, last_end) if upper[i] in ("END", "ENDDFT")), last_end
        )
        spin = "SPIN" in upper[dft + 1:close]
        first = close + 1
    else:
        basis_end = max(
            (i for i, u in enumerate(upper[:last_end]) if u in ("ENDBS", "99 0") or re.match(r"^99\s+0$", u)),
            default=None,
        )
        keyword = next((i for i, u in enumerate(upper) if u == "BASISSET"), None)
        first = (basis_end + 1) if basis_end is not None else (keyword + 2 if keyword is not None else last_end)
        if first < len(upper) and upper[first] == "ENDBS":
            first += 1
    out: list[str] = []
    skip = 0
    for line, u in zip(lines[first:last_end], upper[first:last_end]):
        if skip:
            skip -= 1
            continue
        if u in _NOT_SCF:
            skip = 1
            continue
        if u:
            out.append(line.strip())
    spin = spin or "UHF" in out
    return tuple(out), spin


def deck_settings(text: str) -> DeckSettings:
    """Read the energy-deciding settings of a periodic ``.d12`` deck."""
    lines = text.splitlines()
    try:
        spacegroup = int(lines[3].split()[0])
        lattice = tuple(float(x) for x in lines[4].split())
    except (IndexError, ValueError):
        raise ValueError("not a CRYSTAL periodic deck: no space group / lattice on lines 4-5") from None

    def after(keyword: str) -> Optional[str]:
        m = re.search(rf"^\s*{keyword}\s*\n(.*)$", text, re.M)
        return m.group(1) if m else None

    bsse = after("ATOMBSSE")
    atombsse = None
    if bsse is not None:
        iat, nstar, rmax = bsse.split()[:3]
        atombsse = (int(iat), int(nstar), float(rmax))
    keyword = after("BASISSET")
    inline = None
    if keyword is None:
        end = next(
            (i for i, l in enumerate(lines) if i > 5 and l.strip().upper() in ("END", "ENDGEOM")),
            None,
        )
        stop = next(
            (k for k in range(end + 1, len(lines)) if re.match(r"^\s*99\s+0\s*$", lines[k])),
            None,
        ) if end is not None else None
        if stop is not None:
            inline = "\n".join(lines[end + 1:stop + 1]) + "\n"
    functional = after("DFT")
    tolinteg = after("TOLINTEG")
    shrink = after("SHRINK")
    scf, spin = _scf_section(lines)
    n_unpaired: Optional[int] = 0 if not spin else None
    for i, line in enumerate(scf[:-1]):
        if line.upper() == "SPINLOCK":
            n_unpaired = int(scf[i + 1].split()[0])
    geometry_end = next(
        (i for i, l in enumerate(lines) if i > 5 and l.strip().upper() in ("END", "ENDGEOM", "BASISSET")),
        len(lines),
    )
    return DeckSettings(
        spacegroup=spacegroup,
        lattice=lattice,
        optgeom=bool(re.search(r"^\s*OPTGEOM\s*$", text, re.M)),
        atombsse=atombsse,
        basis_keyword=keyword.split()[0] if keyword else None,
        inline_basis=inline,
        functional=functional.split()[0] if functional else None,
        tolinteg=tuple(int(x) for x in tolinteg.split()) if tolinteg else None,
        shrink=tuple(int(x) for x in shrink.split()[:2]) if shrink else None,
        scf=scf,
        spin=spin,
        n_unpaired=n_unpaired,
        symmremo=any(l.strip().upper() == "SYMMREMO" for l in lines[:geometry_end]),
    )


def final_primitive_cell(out_text: str) -> Optional[tuple[float, ...]]:
    """``(a, b, c, alpha, beta, gamma)`` of the final optimized geometry, or None."""
    i = out_text.rfind("FINAL OPTIMIZED GEOMETRY")
    if i < 0:
        return None
    block = out_text[i:i + 4000].splitlines()
    for k, line in enumerate(block):
        if "ALPHA" in line and "GAMMA" in line and k + 1 < len(block):
            values = block[k + 1].split()
            if len(values) >= 6:
                return tuple(float(x) for x in values[:6])
    return None


def conventional_cubic_a(primitive: Sequence[float], spacegroup: int) -> float:
    """Conventional cubic lattice constant from CRYSTAL's primitive cell."""
    if not 195 <= spacegroup <= 230:
        raise ValueError(f"space group {spacegroup} is not cubic")
    a = float(primitive[0])
    if spacegroup in _F_CUBIC:
        return a * math.sqrt(2.0)
    if spacegroup in _I_CUBIC:
        return a * 2.0 / math.sqrt(3.0)
    return a


def _cited_energy(rest: str) -> Optional[float]:
    """The energy a log line cites: the last number before ``DE``/``DETOT``."""
    head = re.split(r"\bDE", rest)[0]
    values = re.findall(_NUMBER, head)
    return float(values[-1].replace("D", "E")) if values else None


def cited_outputs(
    log_text: str,
    system: str,
    value: float,
    *,
    labels: Sequence[str] = ("pob", "r2SCAN"),
    tolerance: float = 0.051,
) -> tuple[str, list[tuple[str, Optional[float]]], list[float]]:
    """The log line that yields ``value`` for ``system``, and what it was built from.

    The line must read ``E_atom <system> ... = <value> kJ/mol <label...>``
    with every one of ``labels`` in its label and no ``seg`` variant. Returns
    the line, the ``(output name, cited energy)`` pairs between it and the
    previous ``E_atom`` line in log order, and the energies (Ha) in the line's
    own arithmetic, which name a run the block does not list. Raises
    ``LookupError`` when no line matches.
    """
    lines = log_text.splitlines()
    for i, line in enumerate(lines):
        head, sep, label = line.partition("kJ/mol")
        if not sep or "E_atom" not in head:
            continue
        name = head.replace("E_atom", "").split("=")[0].split()
        if not name or name[0] != system:
            continue
        if not all(lab in label for lab in labels) or "seg" in label:
            continue
        m = re.search(rf"=\s*({_NUMBER})\s*$", head.strip())
        if not m or abs(float(m.group(1)) - value) > tolerance:
            continue
        cited: list[tuple[str, Optional[float]]] = []
        j = i - 1
        while j >= 0 and "E_atom" not in lines[j] and not lines[j].startswith("---"):
            f = re.match(r"^\s*(\S+?\.(?:out|sum))\b:?(.*)$", lines[j])
            if f:
                cited.append((f.group(1), _cited_energy(f.group(2))))
            j -= 1
        expression = head.split("=")[1] if head.count("=") >= 2 else ""
        terms = [float(x.replace("D", "E")) for x in re.findall(r"\d*\.\d+[EeDd][+-]?\d+", expression)]
        return line.strip(), cited[::-1], terms
    raise LookupError(f"no log line gives {system} = {value} kJ/mol with labels {labels}")


@dataclass(frozen=True)
class ReferenceComponent:
    """One run a published number was assembled from.

    ``energy_ha`` is the energy the log cites, which is what the published
    number used. ``output_status`` says what the run's own output shows:
    ``converged``, a parser failure mode (``non_converged``, ``truncated``,
    ``no_energy_line``), ``crashed`` when the output records an error after
    its last cycle, or ``missing`` when the cited output is not in the tree.
    ``output_energy_ha`` is that output's last energy, when it has one.
    """

    role: str                        # "bulk" or "atom"
    output_name: str
    output: Optional[Path]
    deck: Optional[Path]
    energy_ha: float
    output_status: str
    output_energy_ha: Optional[float]
    settings: Optional[DeckSettings]
    structure: Optional[Structure]   # the cell this run used (relaxed, for a bulk OPTGEOM)
    Z: Optional[int] = None          # the counterpoise atom's element, for role "atom"
    #: Where ``settings`` were read: ``"output"`` (its echoed input) or ``"d12 file"``.
    deck_source: Optional[str] = None
    #: Settings in which the ``.d12`` file beside the output differs from the echo.
    deck_file_differs: tuple[str, ...] = ()

    @property
    def family(self) -> str:
        """Basis family from the run's name: ``pob``, ``seg``, ``segmod`` or ``?``."""
        for fam in ("segmod", "seg", "pob"):
            if f"_{fam}_" in self.output_name:
                return fam
        return "?"


@dataclass(frozen=True)
class ReferenceSystem:
    """A published cohesive energy and the runs behind it."""

    name: str
    published_kj: float
    log_line: str
    components: tuple[ReferenceComponent, ...]
    counts: dict[int, int]

    @property
    def bulk(self) -> ReferenceComponent:
        return next(c for c in self.components if c.role == "bulk")

    @property
    def atoms(self) -> tuple[ReferenceComponent, ...]:
        return tuple(c for c in self.components if c.role == "atom")

    @property
    def formula_units(self) -> int:
        """Formula units per primitive cell: the gcd of the element counts."""
        return math.gcd(*self.counts.values())

    def cohesive_kj(self, energies: Optional[dict[str, float]] = None) -> float:
        """Sum of n_el * E(atom) minus E(bulk), kJ/mol per formula unit.

        ``energies`` maps an output's name to a replacement energy (Ha), so a
        reproduction is assembled exactly as the reference was.
        """
        energies = energies or {}

        def e(c: ReferenceComponent) -> float:
            return energies.get(c.output_name, c.energy_ha)

        atoms = {c.Z: e(c) for c in self.atoms}
        missing = set(self.counts) - set(atoms)
        if missing:
            raise ValueError(f"{self.name}: no counterpoise atom for Z={sorted(missing)}")
        total = sum(n * atoms[z] for z, n in self.counts.items()) - e(self.bulk)
        return total * HARTREE_TO_KJ_PER_MOL / self.formula_units

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({c.family for c in self.components}))


def _deck_for(folder: Path, output_name: str) -> Optional[Path]:
    stem = re.sub(r"\.\d+\.(out|sum)$", "", Path(output_name).name)
    deck = folder / f"{stem}.d12"
    return deck if deck.exists() else None


def _output_state(text: str) -> tuple[str, Optional[float]]:
    parsed = parse_output(text)
    energy = parsed.energy
    if energy is None:
        last = re.findall(r"ETOT\(AU\)\s+(" + _NUMBER + ")", text)
        energy = float(last[-1]) if last else None
    if parsed.ok:
        return "converged", energy
    tail = text[text.rfind("ETOT(AU)"):] if "ETOT(AU)" in text else text[-2000:]
    if re.search(r"illegal value|ERROR|Error termination", tail):
        return "crashed", energy
    if parsed.converged:  # converged, but the file never finished
        return "truncated", energy
    if energy is not None and parsed.failure_mode == "no_energy_line":
        return "truncated", energy
    return parsed.failure_mode or "unknown", energy


def _differs(a: DeckSettings, b: DeckSettings) -> tuple[str, ...]:
    def basis(s: DeckSettings) -> Optional[str]:
        return s.basis_keyword or re.sub(r"\s+", " ", s.inline_basis or "").strip()

    names = [f for f in DeckSettings.__dataclass_fields__ if f not in ("basis_keyword", "inline_basis")]
    out = [f for f in names if getattr(a, f) != getattr(b, f)]
    return tuple(out + (["basis"] if basis(a) != basis(b) else []))


def _component(folder: Path, name: str, energy: float) -> ReferenceComponent:
    output = folder / Path(name).name
    status, out_energy = "missing", None
    out_text = None
    if output.exists():
        out_text = output.read_text(encoding="latin-1", errors="replace")
        status, out_energy = _output_state(out_text)
    else:
        output = None
    deck = _deck_for(folder, name)
    file_settings = deck_settings(deck.read_text(encoding="latin-1")) if deck is not None else None
    settings = structure = source = None
    differs: tuple[str, ...] = ()
    echo = echoed_input(out_text) if out_text is not None else None
    if echo is not None:
        settings, structure, source = deck_settings(echo), parse_d12_geometry_text(echo), "output"
        if file_settings is not None:
            differs = _differs(settings, file_settings)
    elif deck is not None:
        settings = file_settings
        structure = parse_d12_geometry_text(deck.read_text(encoding="latin-1"))
        source = "d12 file"
    by_name = re.search(r"_atombsse_([A-Z][a-z]?)\b", Path(name).name)
    role = "atom" if by_name or (settings is not None and settings.atombsse) else "bulk"
    Z = None
    if role == "atom":
        # The cluster's real atom is the first row of an ATOMBSSE output's
        # atom list; IAT itself is a label in that deck's own primitive cell.
        if output is not None:
            labels = _primitive_labels(output.read_text(encoding="latin-1", errors="replace"))
            Z = next((z for z in labels if z > 0), None)
        if Z is None and by_name:
            Z = _Z_BY_SYMBOL.get(by_name.group(1))
        if Z is None:
            raise ValueError(f"{name}: cannot tell which element the counterpoise atom is")
        if by_name and _Z_BY_SYMBOL.get(by_name.group(1)) not in (None, Z):
            raise ValueError(f"{name}: the output's atom is Z={Z}, the name says {by_name.group(1)}")
    return ReferenceComponent(
        role=role, output_name=Path(name).name, output=output, deck=deck,
        energy_ha=float(energy), output_status=status, output_energy_ha=out_energy,
        settings=settings, structure=structure, Z=Z,
        deck_source=source, deck_file_differs=differs,
    )


def _primitive_labels(out_text: str) -> list[int]:
    """Z of each atom of the primitive cell, in CRYSTAL's label order."""
    i = out_text.find("ATOMS IN THE UNIT CELL:")
    n = int(re.search(r"ATOMS IN THE UNIT CELL:\s*(\d+)", out_text[i:]).group(1))
    labels: list[int] = []
    for line in out_text[i:].splitlines()[1:]:
        m = re.match(r"^\s*\d+\s+[TF]\s+(\d+)\s+[A-Z]{1,2}\s", line)
        if m:
            labels.append(int(m.group(1)) % 100)
            if len(labels) == n:
                break
    return labels


def _find_output(folder: Path, stem: str, energy: float) -> Optional[str]:
    """The output of deck ``stem`` whose final energy is ``energy`` (Ha)."""
    for out in sorted(folder.glob(f"{stem}.*")):
        if out.suffix not in (".out", ".sum"):
            continue
        _, e = _output_state(out.read_text(encoding="latin-1", errors="replace"))
        if e is not None and abs(e - energy) < 1e-8 * max(1.0, abs(energy)):
            return out.name
    return None


def load_reference_system(
    tree: Path | str,
    system: str,
    published_kj: float,
    log_text: str,
    *,
    directory: Optional[str] = None,
    labels: Sequence[str] = ("pob", "r2SCAN"),
) -> ReferenceSystem:
    """The runs behind ``system``'s published value, read from ``tree``.

    ``directory`` names the compound's directory when it differs from the
    name in the log (``C_diamond`` for ``C-d``). A counterpoise atom's
    element is read from its output (the cluster's real atom) or its name:
    ``ATOMBSSE IAT`` labels an atom of that deck's own primitive cell, and
    the decks of one system do not always list their atoms in one order.
    """
    folder = Path(tree) / (directory or system)
    line, cited, terms = cited_outputs(log_text, system, published_kj, labels=labels)
    components = [_component(folder, name, e) for name, e in cited if e is not None]
    bulks = [c for c in components if c.role == "bulk"]
    if not bulks:
        # The bulk energy is in the line's own arithmetic but not listed above it.
        listed = [c.energy_ha for c in components]
        rest = [-x for x in terms if all(abs(-x - e) > 1e-9 * abs(e) for e in listed)]
        found = _find_output(folder, f"{system}_pob_r2SCAN", rest[0]) if len(rest) == 1 else None
        if found is None:
            raise ValueError(f"{system}: no bulk run among the cited outputs or the arithmetic")
        components.insert(0, _component(folder, found, rest[0]))
        bulks = components[:1]
    if len(bulks) != 1:
        raise ValueError(f"{system}: expected one bulk run, found {len(bulks)}")
    bulk = bulks[0]
    if bulk.output is None:
        # The cited run's output is gone; a rerun of the same deck that ended
        # on the same energy is the same calculation.
        stem = re.sub(r"\.\d+\.(out|sum)$", "", bulk.output_name)
        rerun = _find_output(folder, stem, bulk.energy_ha)
        if rerun is None:
            raise ValueError(
                f"{system}: the cited bulk output {bulk.output_name} is not in the "
                f"tree and no rerun of {stem} ends on its energy {bulk.energy_ha}"
            )
        bulk = replace(_component(folder, rerun, bulk.energy_ha), output_name=bulk.output_name)
    if bulk.settings is not None and bulk.settings.optgeom and bulk.structure is not None:
        cell = final_primitive_cell(bulk.output.read_text(encoding="latin-1", errors="replace")) if bulk.output else None
        if cell is None:
            raise ValueError(f"{system}: OPTGEOM bulk run {bulk.output_name} without a final geometry")
        a = conventional_cubic_a(cell, bulk.settings.spacegroup)
        bulk = replace(bulk, structure=replace(bulk.structure, a=a, b=a, c=a))
    order = _primitive_labels(bulk.output.read_text(encoding="latin-1", errors="replace"))
    counts: dict[int, int] = {}
    for z in order:
        counts[z] = counts.get(z, 0) + 1
    resolved = [bulk if c.role == "bulk" else c for c in components]
    return ReferenceSystem(
        name=system, published_kj=published_kj, log_line=line,
        components=tuple(resolved), counts=counts,
    )
