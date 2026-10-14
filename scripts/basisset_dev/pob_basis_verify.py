#!/usr/bin/env python3
"""Verify the bundled pob-* records against the upstream Bredow-group sources.

Three pob basis sets are bundled in vibe-qc:

  * pob-TZVP        Peintinger / Vilela Oliveira / Bredow JCC 34, 451 (2013);
                    Rb-I: Laun / Vilela Oliveira / Bredow JCC 39, 1285 (2018)
  * pob-TZVP-rev2   Vilela Oliveira / Laun / Peintinger / Bredow JCC 40, 2364 (2019)
                    and the fifth-period, sixth-period and lanthanide papers (#288)
  * pob-DZVP-rev2   Vilela Oliveira / Laun / Peintinger / Bredow JCC 40, 2364 (2019)

The authoritative numbers live in ``python/vibeqc/basis_library/sources/``,
in the per-element CRYSTAL-format files distributed by the Bredow group.
Those files are byte-identical to the basis listings in the supporting
information of the cited papers (cross-checked against the JCC 23153 SI
Section 2 by hand for H, Li, Be, B, C, N, O, F, Na, Mg, Al, Si, P, S,
Cl, K, Ca, Sc, Ti, V).

The bundled library stores each set as a QVF-Basis record
(``basis_library/qvf/<name>.qvf.json``) and renders the ``.g94`` libint
reads from it (#406). This script parses the sources with its own
Decimal-based reader, so it shares no code with the importer it checks,
and compares every exponent and contraction coefficient of the record and
of the rendering with the source to a configurable tolerance.

With ``--regenerate`` it rebuilds the record from the sources through
``vibeqc.basis_toolkit.from_crystal_file`` (orbital shells and, for the
Z+200 records, the ECP through the #207 bridge), keeps the existing
record's description, references and provenance, and re-renders the
library. The regenerated record must then agree with the sources, which
the same run checks.

Usage::

    .venv/bin/python scripts/basisset_dev/pob_basis_verify.py
    .venv/bin/python scripts/basisset_dev/pob_basis_verify.py --basis pob-TZVP-rev2 --atol 1e-12
    .venv/bin/python scripts/basisset_dev/pob_basis_verify.py --regenerate
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "vibeqc" / "basis_library"
SOURCES = PACKAGE_ROOT / "sources"
BASIS_DIR = PACKAGE_ROOT / "basis"
QVF_DIR = PACKAGE_ROOT / "qvf"

# CRYSTAL shell-type -> Gaussian g94 angular-momentum letter
SHELL_LETTER = {0: "S", 1: "SP", 2: "P", 3: "D", 4: "F"}
LETTER_TO_AM = {"S": [0], "SP": [0, 1], "P": [1], "D": [2], "F": [3]}

ELEMENT_BY_Z = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe", 55: "Cs", 56: "Ba", 57: "La",
    58: "Ce", 59: "Pr", 60: "Nd", 61: "Pm", 62: "Sm", 63: "Eu", 64: "Gd",
    65: "Tb", 66: "Dy", 67: "Ho", 68: "Er", 69: "Tm", 70: "Yb", 71: "Lu",
    72: "Hf", 73: "Ta", 74: "W", 75: "Re", 76: "Os", 77: "Ir", 78: "Pt",
    79: "Au", 80: "Hg", 81: "Tl", 82: "Pb", 83: "Bi", 84: "Po", 85: "At",
    86: "Rn",
}


@dataclass
class Primitive:
    exponent: Decimal
    coeffs: tuple[Decimal, ...]  # 1 entry for S/P/D/F, 2 for SP


@dataclass
class Shell:
    crystal_type: int           # 0=s, 1=sp, 2=p, 3=d, 4=f
    occupation: Decimal
    scale: Decimal
    primitives: list[Primitive] = field(default_factory=list)

    @property
    def letter(self) -> str:
        return SHELL_LETTER[self.crystal_type]


@dataclass
class ElementBasis:
    z: int
    shells: list[Shell] = field(default_factory=list)

    @property
    def symbol(self) -> str:
        return ELEMENT_BY_Z[self.z]


# ---------- CRYSTAL parser ---------------------------------------------------


def parse_crystal_file(path: Path) -> ElementBasis:
    """Parse one per-element CRYSTAL-format basis file (Bredow source).

    Line-oriented to tolerate the `0 3 1 0.0 1 0` typo (scale `1.0`
    mistyped as two tokens) that appears in the Bredow `16_S` files of
    both pob-TZVP and pob-TZVP-rev2.
    """
    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    li = iter(lines)
    first = re.split(r"\s+", next(li))
    z, nshells = int(first[0]), int(first[1])
    if z > 200:
        # CRYSTAL's Z+200 header: an INPUT pseudopotential precedes the
        # shells (ZNUC M M0 M1 M2 M3 M4, then one "alpha C n" row per term).
        # Only the orbital shells are compared here; the ECP's conversion is
        # pinned by tests/test_basis_ecp_sidecars.py against PySCF.
        z -= 200
        if next(li).upper() != "INPUT":
            raise ValueError(f"{path}: Z+200 header without an INPUT ECP block")
        counts = re.split(r"\s+", next(li))
        for _ in range(sum(int(tok) for tok in counts[1:7])):
            next(li)
    el = ElementBasis(z=z)
    for _ in range(nshells):
        header = re.split(r"\s+", next(li))
        ityp = int(header[0])
        if ityp != 0:
            raise ValueError(f"{path}: only ityp=0 (all-electron) supported, got {ityp}")
        ltype = int(header[1])
        nprim = int(header[2])
        occ_tok = header[3]
        scale_tok = header[4]
        # Tolerate "1 0" typo for "1.0" in scale field
        if "." not in scale_tok and len(header) > 5 and header[5].isdigit():
            scale_tok = f"{scale_tok}.{header[5]}"
        occ = Decimal(_token_to_str(occ_tok))
        scale = Decimal(_token_to_str(scale_tok))
        shell = Shell(crystal_type=ltype, occupation=occ, scale=scale)
        ncoeff = 2 if ltype == 1 else 1
        for _ in range(nprim):
            row = re.split(r"\s+", next(li))
            exp = Decimal(_token_to_str(row[0]))
            coeffs = tuple(Decimal(_token_to_str(row[k])) for k in range(1, 1 + ncoeff))
            shell.primitives.append(Primitive(exponent=exp, coeffs=coeffs))
        el.shells.append(shell)
    return el


def _token_to_str(tok: str) -> str:
    """Pass through, normalising scientific notation keys (D->E)."""
    return tok.replace("D", "E").replace("d", "e")


# ---------- g94 parser -------------------------------------------------------

_G94_HEADER_RE = re.compile(r"^([A-Z][a-z]?)\s+0\s*$")
_G94_SHELL_RE = re.compile(r"^\s*([A-Z][A-Z]?)\s+(\d+)\s+([\d.]+)\s*$")


def parse_g94_file(path: Path) -> dict[str, ElementBasis]:
    """Parse a Gaussian-94 basis file. Returns {symbol: ElementBasis}."""
    out: dict[str, ElementBasis] = {}
    blocks = re.split(r"^\s*\*\*\*\*\s*$", path.read_text(), flags=re.MULTILINE)
    for block in blocks:
        lines = [ln.rstrip() for ln in block.splitlines() if ln.strip() and not ln.startswith("!")]
        if not lines:
            continue
        m = _G94_HEADER_RE.match(lines[0])
        if not m:
            continue
        symbol = m.group(1)
        z = next(z for z, sym in ELEMENT_BY_Z.items() if sym == symbol)
        el = ElementBasis(z=z)
        i = 1
        while i < len(lines):
            ms = _G94_SHELL_RE.match(lines[i])
            if not ms:
                raise ValueError(f"{path}: cannot parse shell header {lines[i]!r}")
            letter, nprim_s, scale_s = ms.group(1), ms.group(2), ms.group(3)
            ltype = {"S": 0, "SP": 1, "P": 2, "D": 3, "F": 4}[letter]
            nprim = int(nprim_s)
            ncoeff = 2 if ltype == 1 else 1
            shell = Shell(
                crystal_type=ltype,
                occupation=Decimal(0),  # g94 doesn't carry occupation; default 0
                scale=Decimal(scale_s),
            )
            i += 1
            for _ in range(nprim):
                parts = re.split(r"\s+", lines[i].strip())
                exp = Decimal(_token_to_str(parts[0]))
                coeffs = tuple(Decimal(_token_to_str(p)) for p in parts[1 : 1 + ncoeff])
                shell.primitives.append(Primitive(exponent=exp, coeffs=coeffs))
                i += 1
            el.shells.append(shell)
        out[symbol] = el
    return out


# ---------- record reader ---------------------------------------------------


def parse_record(path: Path) -> dict[str, ElementBasis]:
    """Read a QVF-Basis record into the same shape as :func:`parse_g94_file`.

    The record stores SP shells blocked (N s-coefficients, then N
    p-coefficients over the same exponents); here they fold back into
    two-coefficient primitives. Floats are turned into Decimals through
    ``repr`` so an exactly representable source value compares equal.
    """
    import json

    record = json.loads(path.read_text(encoding="utf-8"))
    out: dict[str, ElementBasis] = {}
    for symbol, element in record["elements"].items():
        z = next(z for z, sym in ELEMENT_BY_Z.items() if sym == symbol)
        el = ElementBasis(z=z)
        for shell in element["shells"]:
            am = shell["angular_momentum"]
            letter = {str(v): k for k, v in LETTER_TO_AM.items()}[str(am)]
            ltype = {"S": 0, "SP": 1, "P": 2, "D": 3, "F": 4}[letter]
            prims = shell["primitives"]
            out_shell = Shell(crystal_type=ltype, occupation=Decimal(0), scale=Decimal(1))
            if ltype == 1:
                n = len(prims) // 2
                for i in range(n):
                    out_shell.primitives.append(Primitive(
                        exponent=Decimal(repr(float(prims[i]["exponent"]))),
                        coeffs=(Decimal(repr(float(prims[i]["coefficient"]))),
                                Decimal(repr(float(prims[i + n]["coefficient"])))),
                    ))
            else:
                for prim in prims:
                    out_shell.primitives.append(Primitive(
                        exponent=Decimal(repr(float(prim["exponent"]))),
                        coeffs=(Decimal(repr(float(prim["coefficient"]))),),
                    ))
            el.shells.append(out_shell)
        out[symbol] = el
    return out


# ---------- comparison -------------------------------------------------------


@dataclass
class Diff:
    element: str
    shell_idx: int
    prim_idx: int
    field: str  # "exponent" | "coeff_0" | "coeff_1"
    source: Decimal
    shipped: Decimal

    @property
    def abs_diff(self) -> Decimal:
        return abs(self.source - self.shipped)


def compare(source: ElementBasis, shipped: ElementBasis, *, atol: Decimal) -> list[Diff]:
    """Compare two ElementBasis objects, returning per-number diffs above atol.

    CRYSTAL scales exponents by the shell's scale factor; the shipped
    numbers are the scaled ones.
    """
    diffs: list[Diff] = []
    if source.z != shipped.z:
        raise ValueError(f"Z mismatch: {source.z} vs {shipped.z}")
    if len(source.shells) != len(shipped.shells):
        raise ValueError(
            f"{source.symbol}: shell count {len(source.shells)} vs {len(shipped.shells)}"
        )
    for si, (s_src, s_ship) in enumerate(zip(source.shells, shipped.shells, strict=True)):
        if s_src.crystal_type != s_ship.crystal_type:
            raise ValueError(
                f"{source.symbol} shell {si}: type {s_src.letter} vs {s_ship.letter}"
            )
        if len(s_src.primitives) != len(s_ship.primitives):
            raise ValueError(
                f"{source.symbol} shell {si}: nprim {len(s_src.primitives)} vs "
                f"{len(s_ship.primitives)}"
            )
        for pi, (p_src, p_ship) in enumerate(
            zip(s_src.primitives, s_ship.primitives, strict=True)
        ):
            src_exp = p_src.exponent * s_src.scale if s_src.scale != 1 else p_src.exponent
            if abs(src_exp - p_ship.exponent) > atol:
                diffs.append(
                    Diff(source.symbol, si, pi, "exponent", src_exp, p_ship.exponent)
                )
            for ci, (c_src, c_ship) in enumerate(zip(p_src.coeffs, p_ship.coeffs, strict=True)):
                if abs(c_src - c_ship) > atol:
                    diffs.append(
                        Diff(source.symbol, si, pi, f"coeff_{ci}", c_src, c_ship)
                    )
    return diffs


# ---------- driver -----------------------------------------------------------


@dataclass
class Verdict:
    basis: str
    surface: str
    elements_checked: int
    elements_missing: list[str]
    elements_extra: list[str]
    diffs: list[Diff]


def _source_elements(basis_name: str) -> list[ElementBasis]:
    src_dir = SOURCES / basis_name
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    return [parse_crystal_file(p) for p in sorted(
        src_dir.glob("[0-9][0-9]_*"), key=lambda p: int(p.name.split("_")[0]))]


def verify_surface(basis_name: str, surface: str, shipped: dict[str, ElementBasis], *, atol: Decimal) -> Verdict:
    source_symbols: list[str] = []
    diffs: list[Diff] = []
    for src_el in _source_elements(basis_name):
        sym = src_el.symbol
        source_symbols.append(sym)
        if sym not in shipped:
            continue
        diffs.extend(compare(src_el, shipped[sym], atol=atol))
    missing = [s for s in source_symbols if s not in shipped]
    extra = [s for s in shipped if s not in source_symbols]
    return Verdict(basis_name, surface, len(source_symbols) - len(missing), missing, extra, diffs)


def regenerate(basis_name: str) -> Path:
    """Rebuild the record from ``sources/`` and re-render the library.

    Shells and ECPs come from ``from_crystal_file`` per element (the #207
    bridge converts the ECP); description, references and provenance are
    kept from the record being replaced, with the retrieval date reset to
    today. The pob-DZVP-rev2 elements the archive lacks (Si, Cr-Br, from
    the Basis Set Exchange) are kept from the old record too.
    """
    import datetime as dt

    from vibeqc.basis_toolkit import from_crystal_file, library

    stem = basis_name.lower()
    old = library.load_record(stem)
    new_elements = {}
    new_ecps = {}
    for path in sorted((SOURCES / basis_name).glob("[0-9][0-9]_*"),
                       key=lambda p: int(p.name.split("_")[0])):
        atom = from_crystal_file(path, name=stem)
        new_elements.update(atom.elements)
        if atom.ecps:
            new_ecps.update(atom.ecps)
    for sym, element in old.elements.items():
        new_elements.setdefault(sym, element)
    for sym, entry in (old.ecps or {}).items():
        new_ecps.setdefault(sym, entry)
    old.elements = dict(sorted(new_elements.items(), key=lambda kv: next(
        z for z, s in ELEMENT_BY_Z.items() if s == kv[0])))
    old.ecps = new_ecps or None
    for sym, element in old.elements.items():
        element.ecp_id = sym if sym in (old.ecps or {}) else None
    if old.provenance is not None:
        old.provenance.retrieval_date = dt.date.today().isoformat()
    path = library.save_record(old, library.record_path(stem))
    library.render_library(library.renderings_dir())
    return path


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", action="append", default=None,
                        help="basis name to verify (default: all three pob-* sets)")
    parser.add_argument("--atol", type=str, default="1e-9",
                        help="absolute tolerance for the parity check (Decimal-parseable)")
    parser.add_argument("--regenerate", action="store_true",
                        help="rebuild the record from sources/ and re-render, then verify")
    args = parser.parse_args()
    atol = Decimal(args.atol)
    bases = args.basis or ["pob-TZVP", "pob-TZVP-rev2", "pob-DZVP-rev2"]
    rc = 0
    for basis in bases:
        stem = basis.lower()
        if args.regenerate:
            path = regenerate(basis)
            print(f"regenerated {path.relative_to(PACKAGE_ROOT.parents[2])} and re-rendered basis/")
        surfaces = {
            "record": parse_record(QVF_DIR / f"{stem}.qvf.json"),
            "rendering": parse_g94_file(BASIS_DIR / f"{stem}.g94"),
        }
        print(f"\n=== {basis} ===")
        for surface, shipped in surfaces.items():
            verdict = verify_surface(basis, surface, shipped, atol=atol)
            print(f"  [{surface}] source elements {verdict.elements_checked + len(verdict.elements_missing)},"
                  f" present {verdict.elements_checked}")
            if verdict.elements_missing:
                print(f"    MISSING: {verdict.elements_missing}")
                rc = 2
            if verdict.elements_extra:
                print(f"    extra (not in sources/): {verdict.elements_extra}")
            if verdict.diffs:
                print(f"    DIFFS (>atol={atol}): {len(verdict.diffs)}")
                for d in verdict.diffs[:20]:
                    print(f"      {d.element} shell{d.shell_idx} prim{d.prim_idx} {d.field}: "
                          f"src={d.source} shipped={d.shipped} |diff|={d.abs_diff}")
                if len(verdict.diffs) > 20:
                    print(f"      ... and {len(verdict.diffs) - 20} more")
                rc = max(rc, 1)
            else:
                print("    parity: CLEAN within tolerance")
    raise SystemExit(rc)


if __name__ == "__main__":
    cli()
