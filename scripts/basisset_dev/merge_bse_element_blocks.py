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

Method, following ``merge_def2_heavy_blocks.py``: the bundled record's
element data is kept **unchanged**, the Basis Set Exchange blocks for the
missing elements are appended, and the record's provenance gains the BSE
upstream. No existing element's numbers move.

Usage::

    pip install basis-set-exchange          # or: pip install 'vibe-qc[bse]'
    python scripts/basisset_dev/merge_bse_element_blocks.py
    ./scripts/setup_basis_library.sh --target-dir python/vibeqc/basis_library

Idempotent: an element the record already carries is not fetched again.
"""

from __future__ import annotations

import datetime as _dt
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
LIBRARY = REPO / "python" / "vibeqc" / "basis_library"
QVF = LIBRARY / "qvf"

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




def main() -> int:
    try:
        from basis_set_exchange import api, version
    except ImportError:
        raise SystemExit(
            "this script needs the basis-set-exchange distribution:\n"
            "    pip install 'vibe-qc[bse]'"
        ) from None

    from vibeqc.basis_toolkit import Upstream, from_g94, library

    catalogue = str(version())
    today = _dt.date.today().isoformat()

    for stem, bse_name, elements, reference in TARGETS:
        record = library.load_record(stem)
        wanted = {elements[z] for z in elements} - set(record.elements)
        if not wanted:
            print(f"  = {stem}: already carries {', '.join(elements.values())}")
            continue
        added = api.get_basis(
            name=bse_name, elements=sorted(z for z in elements if elements[z] in wanted),
            fmt="gaussian94", header=False,
        )
        blocks = from_g94(added, name=stem)
        for sym, element in blocks.elements.items():
            record.elements[sym] = element
        symbols = ", ".join(sorted(wanted))
        record.description = (record.description + "\n\n" if record.description else "") + (
            f"{symbols} added from the Basis Set Exchange (catalogue {catalogue}, {today}): "
            f"libint omits these because a later, separate publication added them: {reference}"
        )
        record.provenance.upstream.append(Upstream(
            name="basis-set-exchange", version=catalogue, url="https://www.basissetexchange.org",
            identifier=bse_name, detail=f"{symbols} block(s)"))
        library.save_record(record, library.record_path(stem))
        print(f"  + {stem}: + {symbols} -> qvf/{stem}{library.RECORD_SUFFIX} "
              f"(route the publication's key in database.toml)")

    library.render_library(library.renderings_dir())
    print("\nRe-rendered basis/. Check with: python scripts/render_basis_library.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
