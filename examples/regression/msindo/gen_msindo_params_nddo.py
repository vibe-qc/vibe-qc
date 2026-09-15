"""Regenerate ``msindo_params_nddo.json`` — the NDDO parameter overrides.

MSINDO's optional **NDDO** mode is a *separate parametrization*: when the
``NDDO`` keyword is present, ``parset.f`` calls ``NDDOPARAM`` (``nddoparam.f``),
which OVERWRITES the global parameter arrays (valence/one-centre Slater
exponents, frozen-core exponents, ionization potentials, resonance ``K``
betas, the ``AL`` anti-penetration matrix) with NDDO-specific values for
Z = 1..18 (H, Li–F, Na–Cl; the noble gases He/Ne/Ar are left at zero — NDDO is
not parametrized for them).  This is *build-time tooling* (CLAUDE.md §10): it
reads the maintainer's local MSINDO source, which is not in the repo; the
committed ``msindo_params_nddo.json`` is the shipped, citable data.

vibe-qc applies these as overrides on top of the INDO base bundle
(``msindo_params.json``) when ``nddo=True`` — exactly mirroring ``IF (NDDO)
CALL NDDOPARAM`` (which overwrites the same arrays in the Fortran).

The NDDO parameters are part of the published MSINDO parameter set (Ahlswede &
Jug, J. Comput. Chem. 20, 563 & 572 (1999); Bredow, Geudtner & Jug and later
papers; © Mulliken Center for Theoretical Chemistry, University of Bonn) —
bundled by permission as published data (docs/license.md).

Usage::

    MSINDO_SRC=/path/to/2025e .venv/bin/python \\
        examples/regression/msindo/gen_msindo_params_nddo.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

SRC = (Path(os.environ.get("MSINDO_SRC", "/path/to/msindo/2025e"))
       / "source" / "nddoparam.f")
_RAW = SRC.read_text()
# Strip Fortran comments line-by-line FIRST (so a "! Na Mg ..." label line in the
# middle of an array constructor is removed before continuations are joined),
# then fold fixed-form "&" continuation lines into one logical line.
_NOCOMMENT = re.sub(r"!.*", "", _RAW)
TEXT = re.sub(r"\n\s*&", " ", _NOCOMMENT)

# Scalar element-indexed arrays NDDOPARAM overrides (NAME(1:18) = (/ ... /)).
SCALAR_ARRAYS = [
    "MUS", "MUP", "MUD", "MUSE", "MUPE", "MUDE",
    "TAU1S", "TAU2S", "TAU2P",
    "IPOTS", "IPOTP", "IPOTD", "SCP3D",
    "KSS", "KPS", "KPP", "KDS", "KDP", "KDD",
]
# AL is the anti-penetration matrix AL(central, partner) — nddoparam declares it
# as AL(K, 1:18) with K = the CENTRAL atom (1..17 = H..Cl) and the 1:18 axis the
# PARTNER atomic number (constant within partner shell-groups), matching datas.f's
# AL(central_zk, partner_range) and deltah.f's AL(ANK, ANL).  We collapse the
# partner axis into the engine's 11 shell-groups, producing the SAME per-central
# 11-tuple structure as the INDO bundle (so it overrides _AL directly).
N_AL_ROWS = 17
N_ELEM = 18
# Partner shell-groups (must match msindo._AL_GROUPS / gen_msindo_params.AL_GROUPS).
AL_GROUPS = [(1, 2), (3, 5), (6, 10), (11, 12), (13, 18), (19, 20),
             (21, 30), (31, 36), (37, 38), (39, 48), (49, 54)]


def _floats_after(decl: str) -> list[float]:
    """Extract the float list from a ``NAME(...) = (/ ... /)`` constructor.

    Finds ``decl`` in the source, captures up to the closing ``/)``, strips
    ``!`` comment lines and ``&`` continuations, and returns the numbers.
    """
    m = re.search(re.escape(decl) + r"\s*=\s*\(/(.*?)/\)", TEXT, re.DOTALL)
    if m is None:
        raise ValueError(f"array constructor not found: {decl!r}")
    body = m.group(1)
    # Drop Fortran comments (everything after ! to end-of-line) and continuations.
    body = re.sub(r"!.*", "", body)
    body = body.replace("&", " ")
    nums = re.findall(r"[-+]?\d+\.\d+(?:[eE][-+]?\d+)?", body)
    vals = [float(x) for x in nums]
    return vals


def main() -> None:
    scalars: dict[str, list[float]] = {}
    for name in SCALAR_ARRAYS:
        vals = _floats_after(f"{name}(1:18)")
        if len(vals) != N_ELEM:
            raise ValueError(f"{name}: expected {N_ELEM} values, got {len(vals)}")
        scalars[name] = vals

    # al_rows[k] = AL(central = k+1, partner = 1..18).
    al_rows: list[list[float]] = []
    for k in range(1, N_AL_ROWS + 1):
        vals = _floats_after(f"AL({k},1:18)")
        if len(vals) != N_ELEM:
            raise ValueError(f"AL({k}): expected {N_ELEM} values, got {len(vals)}")
        al_rows.append(vals)

    def al_tuple(z: int) -> list[float]:
        """Per-central-Z AL collapsed to the 11 partner shell-groups (one
        representative partner per group; 0 for partner groups beyond the
        nddoparam Z=1..18 table)."""
        row = al_rows[z - 1]  # AL(central=z, partner=1..18)
        return [row[lo - 1] if lo - 1 < N_ELEM else 0.0 for lo, _ in AL_GROUPS]

    # Build per-element override records for Z where the element is NDDO-
    # parametrized (MUS != 0 — excludes He/Ne/Ar and the unset Z=19+).
    elements: dict[str, dict] = {}
    for i in range(N_ELEM):
        z = i + 1
        if scalars["MUS"][i] == 0.0:
            continue
        rec: dict = {name: scalars[name][i] for name in SCALAR_ARRAYS}
        rec["AL"] = al_tuple(z)  # per-central 11-tuple over partner shell-groups
        elements[str(z)] = rec

    bundle = {
        "_provenance": (
            "MSINDO NDDO parameter overrides parsed from nddoparam.f "
            "(published MSINDO parameter set: Ahlswede & Jug, J. Comput. Chem. "
            "20, 563 & 572 (1999) and later papers; "
            "© Mulliken Center for Theoretical Chemistry, University of Bonn). "
            "Regenerate: examples/regression/msindo/gen_msindo_params_nddo.py. "
            "Applied as overrides on msindo_params.json when nddo=True, mirroring "
            "MSINDO's IF(NDDO) CALL NDDOPARAM."
        ),
        "units": "Slater exponents bohr^-1; energies (IPOT*) Hartree.",
        "note": (
            "Overrides only — arrays NDDOPARAM does not set (FCP*, LS/MP/ND, "
            "TAU3*/4*) are inherited from the INDO base bundle. NDDO is "
            "parametrized for H, Li-F, Na-Cl; noble gases are excluded. AL is the "
            "per-central 11-tuple over partner shell-groups (same structure as the "
            "INDO bundle), so it overrides msindo._AL directly."
        ),
        "al_groups": AL_GROUPS,
        "elements": elements,
    }
    out = Path(__file__).resolve().parents[3] / (
        "python/vibeqc/semiempirical/methods/msindo_params_nddo.json")
    out.write_text(json.dumps(bundle, indent=2) + "\n")
    print(f"wrote {out} ({len(elements)} NDDO-parametrized elements: "
          f"{sorted(int(z) for z in elements)})")


if __name__ == "__main__":
    main()
