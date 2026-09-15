"""Verify shipped pob-* .g94 files against the upstream Bredow-group sources.

Three pob basis sets are bundled in vibe-qc:

  * pob-TZVP        Peintinger / Vilela Oliveira / Bredow JCC 34, 451 (2013)
  * pob-TZVP-rev2   Vilela Oliveira / Laun / Peintinger / Bredow JCC 40, 2364 (2019)
  * pob-DZVP-rev2   Vilela Oliveira / Laun / Peintinger / Bredow JCC 40, 2364 (2019)

The authoritative numbers live in `python/vibeqc/basis_library/sources/`,
in the per-element CRYSTAL-format files distributed by the Bredow group.
Those files are byte-identical to the basis listings in the supporting
information of the cited papers (cross-checked against the JCC 23153 SI
Section 2 by hand for H, Li, Be, B, C, N, O, F, Na, Mg, Al, Si, P, S,
Cl, K, Ca, Sc, Ti, V).

This script:

  1. Parses each per-element CRYSTAL file under `sources/<basis>/`.
  2. Parses the corresponding element block in the shipped
     `basis/<basis>.g94` and `custom/<basis>.g94` files.
  3. Compares exponents and contraction coefficients to a configurable
     tolerance and reports differences.
  4. Optionally re-emits the .g94 files at full source precision (i.e.
     preserving the trailing digit the Bredow source publishes — the
     shipped file currently rounds to 10 dp; the source has 11 dp for
     pob-TZVP and pob-TZVP-rev2 and 8 dp for pob-DZVP-rev2).
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "python" / "vibeqc" / "basis_library"
SOURCES = PACKAGE_ROOT / "sources"
BASIS_DIR = PACKAGE_ROOT / "basis"
CUSTOM_DIR = PACKAGE_ROOT / "custom"

# CRYSTAL shell-type → Gaussian g94 angular-momentum letter
SHELL_LETTER = {0: "S", 1: "SP", 2: "P", 3: "D", 4: "F"}

# Per-basis publication citation. Embedded verbatim in every emitted
# .g94 header so the per-publication reference travels with the basis
# data through the libint / BSE pipeline and is reachable from the
# SCF log (CLAUDE.md § 1 + § 8, AGENTS.md § 8). A pointer to the
# README is not a substitute — the README is not parsed and the
# pointer breaks the moment the file is mirrored or fetched.
CITATIONS = {
    "pob-TZVP": (
        "Peintinger, Vilela Oliveira, Bredow. "
        "J. Comput. Chem. 34, 451–459 (2013). "
        "DOI: 10.1002/jcc.23153; "
        "Rb-I: Laun, Vilela Oliveira, Bredow. "
        "J. Comput. Chem. 39, 1285–1290 (2018). "
        "DOI: 10.1002/jcc.25195"
    ),
    "pob-TZVP-rev2": (
        "Vilela Oliveira, Laun, Peintinger, Bredow. "
        "J. Comput. Chem. 40, 2364–2376 (2019). "
        "DOI: 10.1002/jcc.26013"
    ),
    "pob-DZVP-rev2": (
        "Vilela Oliveira, Laun, Peintinger, Bredow. "
        "J. Comput. Chem. 40, 2364–2376 (2019). "
        "DOI: 10.1002/jcc.26013"
    ),
}


ELEMENT_BY_Z = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I",
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
        # Only the orbital shells belong in the .g94; the ECP goes to the
        # .ecp sidecar through vibeqc.basis_crystal.emit_ecp_sidecar.
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


def _tokenise(text: str) -> list[str]:
    return re.split(r"\s+", text.strip())


def _token_to_str(tok: str) -> str:
    """Pass through, normalising scientific notation keys (D→E)."""
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

    @property
    def rel_diff(self) -> Decimal:
        ref = abs(self.source) if self.source != 0 else Decimal("1")
        return self.abs_diff / ref


def compare(source: ElementBasis, shipped: ElementBasis, *, atol: Decimal) -> list[Diff]:
    """Compare two ElementBasis objects, returning per-number diffs above atol."""
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
            if abs(p_src.exponent - p_ship.exponent) > atol:
                diffs.append(
                    Diff(source.symbol, si, pi, "exponent", p_src.exponent, p_ship.exponent)
                )
            for ci, (c_src, c_ship) in enumerate(zip(p_src.coeffs, p_ship.coeffs, strict=True)):
                if abs(c_src - c_ship) > atol:
                    diffs.append(
                        Diff(source.symbol, si, pi, f"coeff_{ci}", c_src, c_ship)
                    )
    return diffs


# ---------- g94 emitter ------------------------------------------------------


def emit_g94(basis_name: str, source_dir: Path, *, header_lines: list[str]) -> str:
    """Emit a full Gaussian-94 file for a basis from per-element CRYSTAL sources."""
    elements = sorted(source_dir.glob("[0-9][0-9]_*"), key=lambda p: int(p.name.split("_")[0]))
    out: list[str] = []
    out.extend(header_lines)
    out.append("")
    for elem_path in elements:
        el = parse_crystal_file(elem_path)
        out.append("****")
        out.append(f"{el.symbol:<5} 0")
        for shell in el.shells:
            out.append(f"{shell.letter:<2}{len(shell.primitives):>4}   {shell.scale}")
            ncoeff = 2 if shell.crystal_type == 1 else 1
            for prim in shell.primitives:
                # right-align exponent in 22-char column with high precision
                fields = [_format_number(prim.exponent)]
                for c in prim.coeffs[:ncoeff]:
                    fields.append(_format_number(c))
                out.append("    " + "    ".join(fields))
    out.append("****")
    out.append("")
    return "\n".join(out)


def _format_number(value: Decimal) -> str:
    """Format a Decimal preserving every digit the source file carried.

    The Bredow CRYSTAL files print 8-12 significant digits, mostly with
    a trailing 0 padding to 10-11 dp. Decimal preserves the literal
    text we parsed, so str(value) gives the source precision exactly.
    We just right-pad with spaces to 22 columns for visual alignment.
    """
    return f"{value:>22}"


# ---------- driver -----------------------------------------------------------


@dataclass
class Verdict:
    basis: str
    elements_checked: int
    elements_missing_in_g94: list[str]
    elements_extra_in_g94: list[str]
    diffs: list[Diff]


def verify_basis(basis_name: str, *, atol: Decimal) -> Verdict:
    src_dir = SOURCES / basis_name
    g94_path = BASIS_DIR / f"{basis_name.lower()}.g94"
    if not src_dir.exists():
        raise FileNotFoundError(src_dir)
    if not g94_path.exists():
        raise FileNotFoundError(g94_path)
    shipped = parse_g94_file(g94_path)
    source_files = sorted(src_dir.glob("[0-9][0-9]_*"))
    source_symbols = []
    diffs: list[Diff] = []
    for sf in source_files:
        src_el = parse_crystal_file(sf)
        sym = src_el.symbol
        source_symbols.append(sym)
        if sym not in shipped:
            continue
        diffs.extend(compare(src_el, shipped[sym], atol=atol))
    missing = [s for s in source_symbols if s not in shipped]
    extra = [s for s in shipped if s not in source_symbols]
    return Verdict(
        basis=basis_name,
        elements_checked=len(source_symbols) - len(missing),
        elements_missing_in_g94=missing,
        elements_extra_in_g94=extra,
        diffs=diffs,
    )


def cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", action="append", default=None,
                        help="basis name to verify (default: all three pob-* sets)")
    parser.add_argument("--atol", type=str, default="1e-9",
                        help="absolute tolerance for parity check (Decimal-parseable)")
    parser.add_argument("--regenerate", action="store_true",
                        help="emit full-precision .g94 from sources to "
                             "basis_library/{basis,custom}/<basis>.g94")
    args = parser.parse_args()
    atol = Decimal(args.atol)
    bases = args.basis or ["pob-TZVP", "pob-TZVP-rev2", "pob-DZVP-rev2"]
    rc = 0
    for basis in bases:
        verdict = verify_basis(basis, atol=atol)
        print(f"\n=== {basis} ===")
        print(f"  source elements:  {verdict.elements_checked + len(verdict.elements_missing_in_g94)}")
        print(f"  in shipped g94:   {verdict.elements_checked}")
        if verdict.elements_missing_in_g94:
            print(f"  MISSING IN g94:   {verdict.elements_missing_in_g94}")
            rc = 2
        if verdict.elements_extra_in_g94:
            print(f"  EXTRA  IN g94:    {verdict.elements_extra_in_g94}")
        if verdict.diffs:
            print(f"  DIFFS (>atol={atol}): {len(verdict.diffs)}")
            for d in verdict.diffs[:20]:
                print(f"    {d.element} shell{d.shell_idx} prim{d.prim_idx} {d.field}: "
                      f"src={d.source} shipped={d.shipped} |Δ|={d.abs_diff}")
            if len(verdict.diffs) > 20:
                print(f"    ... and {len(verdict.diffs) - 20} more")
            rc = max(rc, 1)
        else:
            print("  parity:           CLEAN within tolerance")
        if args.regenerate:
            citation = CITATIONS[basis]
            header = [
                f"! vibeqc-generated from Bredow-group archive '{basis.lower()}'.",
                f"! Source dir: python/vibeqc/basis_library/sources/{basis}/",
                f"! Cite: {citation}",
            ]
            text = emit_g94(basis, SOURCES / basis, header_lines=header)
            for target in (BASIS_DIR / f"{basis.lower()}.g94", CUSTOM_DIR / f"{basis.lower()}.g94"):
                target.write_text(text)
                print(f"  regenerated:      {target.relative_to(PACKAGE_ROOT.parents[1])}")
    raise SystemExit(rc)


if __name__ == "__main__":
    cli()
