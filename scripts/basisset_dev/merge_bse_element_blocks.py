"""Complete a bundled basis with the element blocks libint's file omits.

Some libint files are shorter than the published record because the missing
elements were added by a *later, separate* paper than the parent set. Three
such gaps, each one element, each its own publication:

    cc-pVQZ   Ca   Koput, Peterson, J. Phys. Chem. A 106, 9595 (2002)
    cc-pV6Z   Be   Prascher, Woon, Peterson, Dunning,
                   Theor. Chem. Acc. 128, 69 (2011)
    STO-3G    Xe   Pietro, Blurock, Hout, Hehre,
                   Inorg. Chem. 20, 3650 (1981)

That is why they are absent rather than an oversight upstream, and it is why
each needs its own citation route rather than inheriting the parent set's.

Method, following ``merge_def2_heavy_blocks.py``: libint's element data is
copied **verbatim** under a provenance header, then the Basis Set Exchange
blocks for the missing elements are appended. No existing element's numbers
move, which ``tests/test_basis_integrity.py`` pins.

Usage::

    pip install basis-set-exchange          # or: pip install 'vibe-qc[bse]'
    python scripts/basisset_dev/merge_bse_element_blocks.py
    ./scripts/setup_basis_library.sh --target-dir python/vibeqc/basis_library

Idempotent: rerunning rewrites the same custom/ files from libint + BSE.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIBRARY = REPO / "python" / "vibeqc" / "basis_library"
CUSTOM = LIBRARY / "custom"

# (vibe-qc stem, BSE name, {Z: symbol}, one-line reference for the header)
TARGETS = [
    (
        "cc-pvqz", "cc-pVQZ", {20: "Ca"},
        "J. Koput, K. A. Peterson, J. Phys. Chem. A 106, 9595 (2002); "
        "DOI: 10.1021/jp026283u",
    ),
    (
        "cc-pv6z", "cc-pV6Z", {4: "Be"},
        "B. P. Prascher, D. E. Woon, K. A. Peterson, T. H. Dunning, "
        "Theor. Chem. Acc. 128, 69 (2011); DOI: 10.1007/s00214-010-0764-0",
    ),
    (
        "sto-3g", "STO-3G", {54: "Xe"},
        "W. J. Pietro, E. S. Blurock, R. F. Hout, W. J. Hehre, "
        "Inorg. Chem. 20, 3650 (1981); DOI: 10.1021/ic50225a013",
    ),
    (
        "pob-dzvp-rev2", "pob-DZVP-rev2",
        {14: "Si", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni",
         29: "Cu", 30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se",
         35: "Br"},
        "D. Vilela Oliveira, J. Laun, M. F. Peintinger, T. Bredow, "
        "J. Comput. Chem. 40, 2364 (2019); DOI: 10.1002/jcc.26013",
    ),
]

# Bases whose source file is vibe-qc's own custom/ rather than libint's.
CUSTOM_SOURCED = {"pob-dzvp-rev2"}


# pob-DZVP-rev2 is completed from BSE because its copy was *verified* against
# our Bredow-archive-derived file first: all 19 shared elements agree as shell
# multisets, and the 13 missing ones come from the same single publication
# (Vilela Oliveira 2019) that already covers what we ship.
#
# pob-TZVP is deliberately NOT here. BSE's copy of it is an upstream
# distribution, and upstream pob carries the column-swap defect on the sulfur
# d-polarisation entry that this project fixed by regenerating from the
# Bredow-group archive (docs/roadmap.md "Pob S d-polarisation column-swap
# fix", commit f059b37; basis_library/sources/README.md). Measured here: our
# sulfur is 5s4p1d, BSE's is 5s4p with no d function at all -- the swapped
# entry had a zero exponent, so the polarisation function vanished. Importing
# the 16 missing Rb-I blocks from a source demonstrably affected by that
# defect, with no local counterpart to check them against, is exactly the
# unverifiable import this workstream exists to avoid. Those blocks should
# come from the Bredow archive (Laun 2018), which sources/ does not yet hold.


def _libint_basis_dir() -> Path:
    root = REPO / "third_party" / "libint" / "install" / "share" / "libint"
    for candidate in sorted(root.glob("*/basis")):
        return candidate
    raise SystemExit(
        f"libint basis directory not found under {root}; "
        "run ./scripts/build_libint.sh first"
    )


def _strip_generated_header(text: str) -> str:
    """Drop a leading run of ``!`` comment lines and blank lines.

    Keeps re-runs idempotent: without it, merging a file this script already
    wrote would prepend a second provenance header on top of the first.
    """
    lines = text.splitlines(keepends=True)
    i = 0
    while i < len(lines) and (not lines[i].strip() or lines[i].lstrip().startswith("!")):
        i += 1
    return "".join(lines[i:])


def main() -> int:
    try:
        from basis_set_exchange import api, version
    except ImportError:
        raise SystemExit(
            "this script needs the basis-set-exchange distribution:\n"
            "    pip install 'vibe-qc[bse]'"
        ) from None

    libint_dir = _libint_basis_dir()
    catalogue = str(version())
    today = _dt.date.today().isoformat()
    CUSTOM.mkdir(parents=True, exist_ok=True)

    for stem, bse_name, elements, reference in TARGETS:
        source = (CUSTOM / f"{stem}.g94") if stem in CUSTOM_SOURCED else (
            libint_dir / f"{stem}.g94"
        )
        if not source.is_file():
            print(f"  !! {stem}: {source} missing, skipped")
            continue

        libint_text = source.read_text(errors="replace")
        if stem in CUSTOM_SOURCED:
            # Re-merging our own output would stack provenance headers.
            libint_text = _strip_generated_header(libint_text)
        added = api.get_basis(
            name=bse_name, elements=sorted(elements), fmt="gaussian94",
            header=False,
        )
        symbols = ", ".join(elements[z] for z in sorted(elements))
        header = (
            "! ----------------------------------------------------------\n"
            f"! Basis set: {stem}\n"
            f"! {stem}: the previous bundled file's element data unchanged,\n"
            f"! followed\n"
            f"! by the {symbols} block(s) from the Basis Set Exchange. libint\n"
            "! omits these because they were added to the set by a later and\n"
            "! separate publication, not because the data is in doubt:\n"
            f"!   {symbols}: {reference}\n"
            f"! Fetched from the basis-set-exchange distribution, catalogue\n"
            f"!   version {catalogue}, on {today}. No network access involved.\n"
            "! License: numerical parameter data from published scientific\n"
            "!   papers; see docs/license.md section 3a.\n"
            "! ----------------------------------------------------------\n"
        )
        target = CUSTOM / f"{stem}.g94"
        target.write_text(header + libint_text + added)
        origin = "custom/" if stem in CUSTOM_SOURCED else "libint"
        print(f"  + {stem}: {origin} + {symbols} -> {target.relative_to(REPO)}")

    print("\nNow promote:")
    print("  ./scripts/setup_basis_library.sh --target-dir python/vibeqc/basis_library")
    return 0


if __name__ == "__main__":
    sys.exit(main())
