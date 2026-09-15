"""Split BSE-format Gaussian-94 files into orbital + ECP halves.

Background
----------

Many BSE-distributed ``.g94`` files for ECP-bearing bases (vDZP,
LANL2DZ, dhf-*, x2c-*, Sapporo-DKH3, Cologne-DKH2, SARC*) bundle
ORBITAL basis blocks AND ECP definition blocks into the same file.
The format looks like::

    H     0
    S    3   1.00
       ...
    ****
    ... (more orbital blocks, separated by `****`)
    ****
    Rn    0          ← last orbital block
       ...
    ****
    B-ECP    3    2   ← FIRST ECP block; no preceding `****`
    f potential
       3
       ...

libint2's ``BasisSet`` parser handles only orbital blocks; it errors
out on ``<Sym>-ECP <ncore> <lmax>`` lines with ``"invalid angular
momentum label"``. That blocks the whole file from loading.

This tool splits each affected ``.g94`` into:

* ``<name>.g94``  — orbital blocks only (libint-loadable).
* ``<name>.ecp``  — every ECP block, verbatim (vibe-qc's libecpint
                    layer reads this on demand).

Idempotent: a ``.g94`` with no ECP block is left untouched. A
``.g94`` already split (no ECP block + sister ``.ecp`` exists) is
left untouched. Re-running is safe.

Use::

    python scripts/basisset_dev/split_ecp_g94.py python/vibeqc/basis_library/basis

Wired into ``scripts/setup_basis_library.sh`` so every promotion of
``custom/`` → ``basis/`` runs the split automatically.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Iterable
from pathlib import Path

# Matches an ECP block header line. The BSE convention is::
#     <Symbol>-ECP <lmax> <ncore>
# where Symbol is one or two letters with an optional leading capital,
# lmax is the maximum angular momentum used in the ECP expansion and
# ncore is the number of core electrons replaced.  The order is lmax
# first: "RB-ECP 3 28" is the def2-ECP's 28-electron rubidium core, and
# "CU-ECP 4 10" the PP family's 10-electron copper core.  This splitter
# only detects the line, but ``vibeqc.ecp_metadata`` reads the fields,
# and an earlier revision of this comment had them the wrong way round.
# NB: BSE-format files come in two case conventions for element
# symbols — standard mixed-case (``Na``, ``Mg``, ``Si``) and the
# Pople-era all-caps form used by LANL2DZ and friends (``NA``, ``MG``,
# ``SI``). Both are accepted: the second character can be any case.
ECP_HEADER_RE = re.compile(r"^\s*([A-Z][A-Za-z]?)-ECP\s+(\d+)\s+(\d+)\s*$")

# Matches an orbital-block header line: ``<Symbol>  0``. The trailing
# ``0`` is the BSE convention for "no offset". Symbol is 1-2 chars.
ORBITAL_HEADER_RE = re.compile(r"^\s*([A-Z][A-Za-z]?)\s+0\s*$")

# Block delimiter (only present in orbital sections).
DELIMITER_RE = re.compile(r"^\s*\*{4}\s*$")


def has_ecp_block(lines: list[str]) -> bool:
    """True if any line in the file matches ``ECP_HEADER_RE``."""
    return any(ECP_HEADER_RE.match(line) for line in lines)


def split_blocks(lines: list[str]) -> tuple[list[str], list[str]]:
    """Walk a .g94 line-by-line, partitioning each block into either
    the orbital stream or the ECP stream.

    Block layout in BSE-format files comes in two patterns:

    * **vdzp pattern**: all orbital blocks first, separated by ``****``
      delimiters, then all ECP blocks contiguously at the end.
    * **lanl2dz / dhf-* pattern**: each element's orbital block is
      followed immediately by its own ECP block; orbital and ECP
      blocks alternate per element.

    The state machine handles both: any line matching
    ``ECP_HEADER_RE`` flips to the ECP stream until the next orbital
    block starts; any line matching ``ORBITAL_HEADER_RE`` flips back
    to the orbital stream. ``****`` delimiters terminate the current
    orbital block but stay in the orbital stream so libint sees them.

    Header comments (``!`` lines and blank lines before any block)
    go to the orbital stream by default — they're cosmetic and
    libint ignores them.
    """
    orbital: list[str] = []
    ecp: list[str] = []
    state = "pre_block"  # one of "pre_block" | "orbital" | "ecp"

    def _next_nonblank(start: int) -> str:
        """Return the next non-blank line after index ``start``, or ''."""
        for k in range(start + 1, len(lines)):
            s = lines[k].strip()
            if s:
                return s
        return ""

    for i, line in enumerate(lines):
        L = line.rstrip("\n").rstrip("\r")
        if ECP_HEADER_RE.match(L):
            state = "ecp"
            ecp.append(line)
            continue
        if ORBITAL_HEADER_RE.match(L):
            # Lookahead: if the next non-blank line is `<Sym>-ECP`,
            # this `<Sym>  0` line introduces an ECP block (BSE
            # convention) and belongs to the ECP stream — NOT the
            # orbital stream. Without this, the orbital file ends
            # up with N orphan `<Sym>  0` headers and libint then
            # hits the next line (an `<Sym>-ECP` header that
            # leaked in some other way, or the next orbital
            # header) and reports "invalid angular momentum label".
            nxt = _next_nonblank(i)
            if ECP_HEADER_RE.match(nxt):
                state = "ecp"
                ecp.append(line)
            else:
                state = "orbital"
                orbital.append(line)
            continue
        if DELIMITER_RE.match(L):
            # `****` belongs to whatever block just ended (orbital
            # in practice; ECP blocks have no delimiter).
            if state == "ecp":
                # Defensive: an unexpected **** inside what we
                # thought was the ECP stream most likely means a
                # missed orbital boundary. Send it to orbital.
                orbital.append(line)
                state = "pre_block"
            else:
                orbital.append(line)
                state = "pre_block"
            continue
        # Continuation line (primitive, comment, etc.): route by
        # current stream.
        if state == "ecp":
            ecp.append(line)
        else:
            orbital.append(line)

    return orbital, ecp


def split_one(g94_path: Path, *, dry_run: bool = False) -> bool:
    """Split a single .g94 file into orbital + ECP. Returns True if a
    split happened (or would happen in dry-run); False if the file
    has no ECP blocks.
    """
    text = g94_path.read_text()
    lines = text.splitlines(keepends=True)
    if not has_ecp_block(lines):
        return False

    orbital_lines, ecp_lines = split_blocks(lines)

    # Trim trailing blank lines on both sides; cosmetic, eases visual
    # diff. Keep one trailing newline.
    for buf in (orbital_lines, ecp_lines):
        while buf and buf[-1].strip() == "":
            buf.pop()
        if buf and not buf[-1].endswith("\n"):
            buf[-1] = buf[-1] + "\n"

    # Header on the ECP side, identifying provenance.
    ecp_header = (
        f"! ECP block(s) split out of {g94_path.name} by\n"
        f"! scripts/basisset_dev/split_ecp_g94.py.\n"
        f"!\n"
    )

    ecp_path = g94_path.with_suffix(".ecp")

    if dry_run:
        return True

    g94_path.write_text("".join(orbital_lines))
    ecp_path.write_text(ecp_header + "".join(ecp_lines))
    return True


def is_already_split(g94_path: Path) -> bool:
    """True if a sister ``.ecp`` already exists AND the .g94 has no
    ECP blocks left. Used to skip already-split files on re-runs.
    """
    if not g94_path.with_suffix(".ecp").exists():
        return False
    text = g94_path.read_text()
    return not has_ecp_block(text.splitlines())


def split_directory(directory: Path, *, dry_run: bool = False) -> dict[str, int]:
    """Split every ECP-bearing ``.g94`` in ``directory``.

    Returns counts: ``{"split": N_split, "already": N_already_split,
    "no_ecp": N_with_no_ecp}``.
    """
    counts = {"split": 0, "already": 0, "no_ecp": 0}
    for g94 in sorted(directory.glob("*.g94")):
        if is_already_split(g94):
            counts["already"] += 1
            continue
        if split_one(g94, dry_run=dry_run):
            counts["split"] += 1
            print(f"  split: {g94.name}  →  {g94.name} + {g94.stem}.ecp")
        else:
            counts["no_ecp"] += 1
    return counts


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "directory", type=Path,
        help="directory containing .g94 files to split (e.g. "
             "python/vibeqc/basis_library/basis/)",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="report what would be split, do not write")
    args = p.parse_args(argv)
    if not args.directory.is_dir():
        print(f"error: {args.directory} is not a directory", file=sys.stderr)
        return 2
    print(f"scanning {args.directory} for ECP-bearing .g94 files…")
    counts = split_directory(args.directory, dry_run=args.dry_run)
    total = sum(counts.values())
    print(
        f"done: {counts['split']} split, {counts['already']} already split, "
        f"{counts['no_ecp']} have no ECP blocks  ({total} .g94 files total)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
