"""Converge the ATOMBSSE ghost shell (``nstar``, ``rmax``) per system.

    .venv/bin/python examples/basisset_dev/converge_bsse_shell.py LiF pbe

Emits counterpoise free-atom decks with vibe-basis's own emitter over a
ladder of ``(nstar, rmax)``, runs each through a local CRYSTAL binary,
and prints where the energy stops moving. Uses the reference-set lattice
constants, so the converged setting is the one production runs need.

Why this exists
---------------
The published pob cohesive energies use ``ATOMBSSE`` free atoms, and the
reference set picks the ghost-shell extent **per system** (LiF 30/10.0,
MgO 10/5.0). Inheriting one system's value for another is not a detail:
an under-converged ghost shell leaves part of the BSSE uncorrected, and
BSSE is the quantity the correction exists to remove.

Cost
----
This is a **multi-hour job**, and knowing why matters before you start.
``SYMMREMO`` strips the symmetry, so a counterpoise atom is a
symmetry-free cluster: 27 atoms and 343 AOs already at LiF's cheapest
ladder point. Measured on an M-series laptop with CRYSTAL23 v1.0.1,
roughly 4-5 minutes per point at PBE.

At r2SCAN with ``HUGEGRID`` it is far worse -- a single point did not
finish one SCF cycle in twenty minutes. So the recommended protocol is
to **converge at PBE, then validate the chosen setting once at
r2SCAN**: the BSSE's dependence on ghost-shell extent is a
basis-overlap property and only weakly functional-dependent. For a full
test set, run this on the queue rather than a laptop (CLAUDE.md § 15).
"""

from __future__ import annotations

import dataclasses
import re
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from vibeqc.basis_crystal import emit_crystal, parse_crystal_atom_basis_file
from vibe_basis.backends.crystal_atom import emit_input_atom_counterpoise
from vibe_basis.io.structures import STRUCTURES

CRYSTAL = Path.home() / "bin" / "crystal"
SRC = Path(__file__).resolve().parents[2] / "python/vibeqc/basis_library/sources/pob-TZVP-rev2"
WORK = Path(os.environ.get("VQ_WORKDIR", tempfile.gettempdir())) / "bsse_shell"
HA_KJ = 2625.4996394798254

E_RE = re.compile(r"TOTAL ENERGY\([A-Z]+\)\(AU\)\(\s*\d+\)\s+(-?\d+\.\d+E[+-]\d+)")
SCF_OK = re.compile(r"SCF ENDED - CONVERGENCE ON (ENERGY|DENSITY MATRIX)")

# Reference-set lattice constants (r2SCAN-optimised), from
# ~/gitlab/references-cohesive/<sys>/<sys>_pob_r2SCAN*.d12
REF_A = {"LiF": 4.04501971, "MgO": 4.22389871}

# (nstar, rmax) ladder. The reference set spans 10/5.0 to 60/10.0.
LADDER = [(5, 4.0), (10, 5.0), (15, 6.0), (20, 7.5), (30, 10.0)]


def basis_text(Zs):
    atoms = []
    for Z in Zs:
        matches = sorted(SRC.glob(f"{Z:02d}_*"))
        if not matches:
            raise SystemExit(f"no pob-TZVP-rev2 source for Z={Z}")
        atoms.append(parse_crystal_atom_basis_file(str(matches[0])))
    return emit_crystal(atoms)


def run(deck: str, tag: str):
    d = WORK / tag
    d.mkdir(parents=True, exist_ok=True)
    (d / "INPUT").write_text(deck)
    t0 = time.perf_counter()
    try:
        with (d / "INPUT").open() as fh, (d / "out.txt").open("w") as out:
            subprocess.run([str(CRYSTAL)], stdin=fh, stdout=out,
                           stderr=subprocess.STDOUT, timeout=900, cwd=d)
    except subprocess.TimeoutExpired:
        return None, "timeout", time.perf_counter() - t0
    text = (d / "out.txt").read_text(errors="ignore")
    m = E_RE.findall(text)
    if not m:
        reason = "no_energy"
        for line in text.splitlines():
            if "ERROR" in line:
                reason = line.strip()[:70]
                break
        return None, reason, time.perf_counter() - t0
    if not SCF_OK.search(text):
        return float(m[-1]), "not_converged", time.perf_counter() - t0
    return float(m[-1]), "ok", time.perf_counter() - t0


def main():
    system = sys.argv[1] if len(sys.argv) > 1 else "LiF"
    func = sys.argv[2] if len(sys.argv) > 2 else "pbe"
    struct = dataclasses.replace(STRUCTURES[system], a=REF_A[system],
                                 b=REF_A[system], c=REF_A[system])
    Zs = [a.Z for a in struct.crystal_asymm_unit]
    bt = basis_text(Zs)

    print(f"# {system}: a={REF_A[system]} A, pob-TZVP-REV2, r2SCAN, ATOMBSSE sweep")
    for idx, Z in enumerate(Zs, start=1):
        print(f"\n## atom {idx}: Z={Z}")
        prev = None
        for nstar, rmax in LADDER:
            deck = emit_input_atom_counterpoise(
                struct, idx, bt, method=func, nstar=nstar, rmax=rmax)
            e, status, dt = run(deck, f"{system}_Z{Z}_{nstar}_{rmax}")
            if e is None:
                print(f"  nstar={nstar:>3} rmax={rmax:>4}  FAILED ({status})  {dt:.0f}s")
                continue
            delta = "" if prev is None else f"  d={(e - prev) * HA_KJ:+8.3f} kJ/mol"
            print(f"  nstar={nstar:>3} rmax={rmax:>4}  E={e:.9f} Ha  [{status}]{delta}  {dt:.0f}s")
            prev = e


if __name__ == "__main__":
    main()
