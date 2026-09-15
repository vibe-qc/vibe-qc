"""Extend the bundled def2 orbital files beyond Kr with their def2-ECP.

libint's def2 files stop at Kr. The Basis Set Exchange carries the Rb-Rn
blocks (Turbomole 7.3 data) together with the def2-ECP blocks the valence
sets pair with. This script appends the BSE heavy blocks to libint's file
*byte for byte unchanged* for H-Kr, so no existing light-element number
moves, and writes the ECP blocks as the ``.ecp`` sidecar the molecular SCF
wrappers attach.

Inputs are the raw BSE ``gaussian94`` responses saved as
``<dir>/<name>.g94`` (one per basis, elements 37-86; the diffuse ``-D``
sets carry no lanthanides). Outputs land in ``basis_library/custom/`` and
``basis_library/basis/``::

    python scripts/basisset_dev/merge_def2_heavy_blocks.py <dir>

Provenance and the per-element def2-ECP references go into the file
headers; ``docs/license.md`` § 3a lists the same.
"""

from __future__ import annotations

import re
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIBRARY = REPO / "python" / "vibeqc" / "basis_library"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_ecp_g94 import split_blocks

NAMES = [
    "def2-sv(p)", "def2-svp", "def2-svpd",
    "def2-tzvp", "def2-tzvpd", "def2-tzvpp", "def2-tzvppd",
    "def2-qzvp", "def2-qzvpd", "def2-qzvpp", "def2-qzvppd",
]

_ECP_REFS = """\
!   Y-Cd (ecp-28), Hf-Hg (ecp-60): D. Andrae, U. Haeussermann, M. Dolg,
!     H. Stoll, H. Preuss, Theor. Chim. Acta 77, 123 (1990)
!   In-Sb (ecp-28), Tl-Bi (ecp-60): B. Metz, H. Stoll, M. Dolg,
!     J. Chem. Phys. 113, 2563 (2000)
!   Te-Xe (ecp-28), Po-Rn (ecp-60): K. A. Peterson, D. Figgen, E. Goll,
!     H. Stoll, M. Dolg, J. Chem. Phys. 119, 11113 (2003)
!   Rb (ecp-28), Cs (ecp-46): T. Leininger, A. Nicklass, W. Kuechle,
!     H. Stoll, M. Dolg, A. Bergner, Chem. Phys. Lett. 255, 274 (1996)
!   Sr (ecp-28), Ba (ecp-46): M. Kaupp, P. v. R. Schleyer, H. Stoll,
!     H. Preuss, J. Chem. Phys. 94, 1360 (1991)
!   La-Lu (ecp-28 / ecp-46): M. Dolg, H. Stoll, A. Savin, H. Preuss,
!     Theor. Chim. Acta 75, 173 (1989); M. Dolg, H. Stoll, H. Preuss,
!     J. Chem. Phys. 90, 1730 (1989)
"""


def _bse_version(text: str) -> str:
    m = re.search(r"^!\s+Version:\s+(\S+)", text, re.MULTILINE)
    return m.group(1) if m else "?"


def _header(name: str, kind: str, bse_text: str, heavy: list[str]) -> str:
    today = datetime.now(UTC).date().isoformat()
    span = f"{heavy[0]}-{heavy[-1]} ({len(heavy)} elements)"
    if kind == "orbital":
        what = (
            f"! {name}: libint 2.13.1's H-Kr file, unchanged, followed by the\n"
            f"! {span} valence blocks from the Basis Set Exchange\n"
            f"! (BSE {_bse_version(bse_text)}, Turbomole 7.3 data). The heavy blocks are\n"
            f"! valence-only and pair with the def2-ECP in {name}.ecp, which the\n"
            f"! molecular SCF wrappers attach per element.\n"
        )
    else:
        what = (
            f"! {name} ECPs: the def2-ECP blocks for {span}, from the\n"
            f"! Basis Set Exchange record of {name} (BSE {_bse_version(bse_text)}).\n"
            f"! Per-element originating publications:\n{_ECP_REFS}"
        )
    return (
        "! ----------------------------------------------------------\n"
        f"! Basis set: {name}{' ECPs' if kind == 'ecp' else ''}\n"
        f"! Originating publication: F. Weigend, R. Ahlrichs, Phys. Chem. Chem.\n"
        f"!   Phys. 7, 3297 (2005); DOI: 10.1039/b508541a"
        + (
            "; diffuse augmentation: D. Rappoport, F. Furche, J. Chem. Phys.\n"
            "!   133, 134105 (2010); DOI: 10.1063/1.3484283"
            if name.endswith("d") else ""
        )
        + "\n"
        + what
        + f"! Fetched from: https://www.basissetexchange.org (REST API), {today}\n"
        "! License: numerical parameter data from published scientific\n"
        "!   papers, distributed by BSE under CC-BY-4.0; see docs/license.md\n"
        "! ----------------------------------------------------------\n"
    )


def _element_symbols(lines: list[str]) -> list[str]:
    return [
        m.group(1).capitalize()
        for line in lines
        if (m := re.match(r"^\s*([A-Z][A-Za-z]?)\s+0\s*$", line))
    ]


def merge(name: str, bse_dir: Path) -> tuple[int, int]:
    bse_path = bse_dir / f"{name}.g94"
    libint_path = LIBRARY / "basis" / f"{name}.g94"
    bse_text = bse_path.read_text()
    if bse_text.lstrip().startswith("{"):
        raise RuntimeError(f"{bse_path}: BSE returned an error, not a basis")
    orbital_lines, ecp_lines = split_blocks(bse_text.splitlines())
    # Drop BSE's own comment header; the vibe-qc header carries provenance.
    while orbital_lines and (
        orbital_lines[0].startswith("!") or not orbital_lines[0].strip()
    ):
        orbital_lines.pop(0)
    heavy = _element_symbols(orbital_lines)
    if not heavy or not ecp_lines:
        raise RuntimeError(f"{name}: no heavy orbital blocks or no ECP blocks parsed")

    libint_text = libint_path.read_text()
    if not libint_text.endswith("\n"):
        libint_text += "\n"
    light = _element_symbols(libint_text.splitlines())
    if set(light) & set(heavy):
        raise RuntimeError(f"{name}: libint file already carries {set(light) & set(heavy)}")

    orbital_out = (
        _header(name, "orbital", bse_text, heavy)
        + libint_text
        + "\n".join(orbital_lines).rstrip("\n")
        + "\n"
    )
    ecp_out = (
        _header(name, "ecp", bse_text, heavy)
        + "\n"
        + "\n".join(ecp_lines).rstrip("\n")
        + "\n"
    )
    for target in (LIBRARY / "custom", LIBRARY / "basis"):
        (target / f"{name}.g94").write_text(orbital_out)
        (target / f"{name}.ecp").write_text(ecp_out)
    n_ecp = len(re.findall(r"^\s*[A-Z][A-Za-z]?-ECP", "\n".join(ecp_lines), re.MULTILINE))
    return len(heavy), n_ecp


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    bse_dir = Path(argv[1])
    for name in NAMES:
        n_orb, n_ecp = merge(name, bse_dir)
        print(f"{name:12s} +{n_orb:2d} heavy orbital blocks, {n_ecp:2d} ECP blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
