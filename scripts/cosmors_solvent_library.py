#!/usr/bin/env python3
"""Build the COSMO-RS solvent geometry library.

Optimizes each solvent of the library in the gas phase at BPW91/def2-SVP (or,
with ``--conductor``, as an ideal conductor: ``epsilon = inf``, the COSMO FINE
cavity at the ``klamt1998`` radii, the state COSMO-RS describes, at the price
of a fine-cavity gradient per step) and writes
``python/vibeqc/solvation/cosmors/solvent_geometries.py``.
The starting structures are built here from standard bond lengths and ideal
angles, or taken from ASE's G2 collection where it has the molecule; only the
optimized coordinates enter the library, with the protocol recorded on each
entry.

Usage::

    .venv/bin/python scripts/cosmors_solvent_library.py --workdir /tmp/solvents
    .venv/bin/python scripts/cosmors_solvent_library.py --workdir /tmp/solvents --solvent water --solvent benzene
    .venv/bin/python scripts/cosmors_solvent_library.py --workdir /tmp/solvents --assemble

Each solvent's result is written to ``<workdir>/<name>.json`` as it finishes,
so a run can be resumed, and ``--assemble`` regenerates the module from every
result present. ``--jobs N`` runs N solvents at once, each in its own process.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "python" / "vibeqc" / "solvation" / "cosmors" / "solvent_geometries.py"
ANG_TO_BOHR = 1.8897261254578281

FUNCTIONAL = "bpw91"
BASIS = "def2-svp"


# --------------------------------------------------------------------------
# Starting structures (angstrom)
# --------------------------------------------------------------------------

def _tet_h(center, others, n, r=1.09):
    """``n`` hydrogens completing a tetrahedron around ``center``."""
    c = np.asarray(center, float)
    vs = [np.asarray(o, float) - c for o in others]
    vs = [v / np.linalg.norm(v) for v in vs]
    out = []
    if len(vs) == 2 and n == 2:
        b = -(vs[0] + vs[1]); b /= np.linalg.norm(b)
        p = np.cross(vs[0], vs[1]); p /= np.linalg.norm(p)
        ang = math.radians(109.47 / 2)
        for sgn in (+1, -1):
            out.append(c + r * (math.cos(ang) * b + sgn * math.sin(ang) * p))
    elif len(vs) == 1 and n == 3:
        a = -vs[0]
        t = np.array([1.0, 0, 0]) if abs(a[0]) < 0.9 else np.array([0, 1.0, 0])
        p1 = np.cross(a, t); p1 /= np.linalg.norm(p1)
        p2 = np.cross(a, p1)
        ang = math.radians(180 - 109.47)
        for k in range(3):
            phi = 2 * math.pi * k / 3
            out.append(c + r * (math.cos(ang) * a + math.sin(ang) * (math.cos(phi) * p1 + math.sin(phi) * p2)))
    elif len(vs) == 3 and n == 1:
        b = -(vs[0] + vs[1] + vs[2]); b /= np.linalg.norm(b)
        out.append(c + r * b)
    else:
        raise ValueError((len(vs), n))
    return [tuple(float(x) for x in v) for v in out]


def _chain(n, r=1.53, ang=112.0):
    half = math.radians(ang / 2)
    return [(i * r * math.sin(half), (i % 2) * r * math.cos(half), 0.0) for i in range(n)]


def _alkane(n):
    c = _chain(n)
    atoms = [(6, p) for p in c]
    for i, p in enumerate(c):
        others = [c[j] for j in (i - 1, i + 1) if 0 <= j < n]
        atoms += [(1, h) for h in _tet_h(p, others, 4 - len(others))]
    return atoms


def _alkanol(n):
    c = _chain(n)
    atoms = [(6, p) for p in c]
    for i, p in enumerate(c):
        others = [c[j] for j in (i - 1, i + 1) if 0 <= j < n]
        hs = _tet_h(p, others, 4 - len(others))
        if i == 0:
            o = np.asarray(p) + (np.asarray(hs[0]) - np.asarray(p)) / 1.09 * 1.43
            atoms.append((8, tuple(float(x) for x in o)))
            atoms.append((1, _tet_h(o, [p], 3, r=0.96)[0]))
            hs = hs[1:]
        atoms += [(1, h) for h in hs]
    return atoms


def _isopropanol():
    c0 = (0.0, 0.0, 0.0)
    c1 = (1.53 * math.sin(math.radians(55)), 1.53 * math.cos(math.radians(55)), 0.0)
    c2 = (-1.53 * math.sin(math.radians(55)), 1.53 * math.cos(math.radians(55)), 0.0)
    hs0 = _tet_h(c0, [c1, c2], 2)
    o = np.asarray(c0) + (np.asarray(hs0[0]) - np.asarray(c0)) / 1.09 * 1.43
    atoms = [(6, c0), (6, c1), (6, c2), (8, tuple(float(x) for x in o)), (1, hs0[1])]
    atoms.append((1, _tet_h(o, [c0], 3, r=0.96)[0]))
    atoms += [(1, h) for h in _tet_h(c1, [c0], 3)]
    atoms += [(1, h) for h in _tet_h(c2, [c0], 3)]
    return atoms


def _dmf():
    c = (0.0, 0.0, 0.0)
    o = (1.23 * math.cos(math.radians(60)), 1.23 * math.sin(math.radians(60)), 0.0)
    n = (-1.35, 0.0, 0.0)
    h = (0.55, -0.95, 0.0)
    cm1 = (-2.0, 1.25, 0.0)
    cm2 = (-2.0, -1.25, 0.0)
    atoms = [(6, c), (8, o), (7, n), (1, h), (6, cm1), (6, cm2)]
    atoms += [(1, x) for x in _tet_h(cm1, [n], 3)]
    atoms += [(1, x) for x in _tet_h(cm2, [n], 3)]
    return atoms


def _thf():
    r = 1.20
    ring = []
    for k in range(5):
        a = math.radians(90 + 72 * k)
        ring.append([r * math.cos(a), r * math.sin(a), 0.0])
    ring[3][2] = 0.45
    order = [tuple(x) for x in ring]
    atoms = [(8, order[0])] + [(6, x) for x in order[1:]]
    for i in range(1, 5):
        atoms += [(1, h) for h in _tet_h(order[i], [order[i - 1], order[(i + 1) % 5]], 2)]
    return atoms


def _cyclohexane():
    r, z = 1.45, 0.25
    cs = [(r * math.cos(math.radians(60 * k)), r * math.sin(math.radians(60 * k)),
           z * (1 if k % 2 == 0 else -1)) for k in range(6)]
    atoms = [(6, c) for c in cs]
    for i in range(6):
        atoms += [(1, h) for h in _tet_h(cs[i], [cs[i - 1], cs[(i + 1) % 6]], 2)]
    return atoms


def _toluene():
    rc, rh = 1.395, 2.48
    atoms = [(6, (rc * math.cos(math.radians(60 * k)), rc * math.sin(math.radians(60 * k)), 0.0))
             for k in range(6)]
    atoms += [(1, (rh * math.cos(math.radians(60 * k)), rh * math.sin(math.radians(60 * k)), 0.0))
              for k in range(1, 6)]
    cm = (rc + 1.51, 0.0, 0.0)
    atoms.append((6, cm))
    atoms += [(1, h) for h in _tet_h(cm, [(rc, 0.0, 0.0)], 3)]
    return atoms


def _dcm():
    c = (0.0, 0.0, 0.0)
    cl = [(0.0, 1.46, 1.03), (0.0, -1.46, 1.03)]
    return [(6, c)] + [(17, x) for x in cl] + [(1, h) for h in _tet_h(c, cl, 2)]


def _chloroform():
    c = (0.0, 0.0, 0.0)
    s, cz = math.sin(math.radians(109)), math.cos(math.radians(109))
    cls = [(1.77 * s * math.cos(2 * math.pi * k / 3), 1.77 * s * math.sin(2 * math.pi * k / 3), 1.77 * cz)
           for k in range(3)]
    return [(6, c)] + [(17, x) for x in cls] + [(1, h) for h in _tet_h(c, cls, 1)]


def _g2(name):
    from ase.build import molecule

    a = molecule(name)
    return [(int(z), tuple(float(x) for x in p)) for z, p in zip(a.numbers, a.positions)]


def starting_structures() -> dict[str, list]:
    return {
        "water": _g2("H2O"),
        "methanol": _g2("CH3OH"),
        "ethanol": _g2("CH3CH2OH"),
        "isopropanol": _isopropanol(),
        "n-butanol": _alkanol(4),
        "dmf": _dmf(),
        "acetonitrile": _g2("CH3CN"),
        "acetone": _g2("CH3COCH3"),
        "thf": _thf(),
        "dichloromethane": _dcm(),
        "chloroform": _chloroform(),
        "carbon-tetrachloride": _g2("CCl4"),
        "benzene": _g2("C6H6"),
        "toluene": _toluene(),
        "n-hexane": _alkane(6),
        "n-pentane": _alkane(5),
        "cyclohexane": _cyclohexane(),
    }


# --------------------------------------------------------------------------
# Optimization
# --------------------------------------------------------------------------

def optimize_one(name: str, workdir: Path, functional: str, basis: str, conductor: bool) -> Path:
    """Optimize ``name``; write ``<workdir>/<name>.json``."""
    import vibeqc as vq
    from vibeqc.solvation.cosmors import KLAMT_1998
    from vibeqc.solvation.cosmors.job import conductor_solvent_model

    atoms = starting_structures()[name]
    mol = vq.Molecule(
        [vq.Atom(z, tuple(c * ANG_TO_BOHR for c in xyz)) for z, xyz in atoms], 0, 1
    )
    stem = workdir / f"opt_{name}"
    res = vq.run_job(
        mol, basis=basis, method="rks", functional=functional, optimize=True,
        solvent=conductor_solvent_model(KLAMT_1998) if conductor else None,
        output=str(stem), progress=False, citations=False, write_molden_file=False,
    )
    medium = (
        "as an ideal conductor (epsilon=inf, COSMO FINE cavity at klamt1998 radii)"
        if conductor else "in the gas phase"
    )
    lines = stem.with_suffix(".xyz").read_text().splitlines()
    n = int(lines[0].strip())
    xyz = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines[2:2 + n]])
    zs = [z for z, _ in atoms]
    payload = {
        "name": name,
        "protocol": f"{functional}/{basis} optimized {medium}, vibe-qc {vq.__version__}",
        "energy_hartree": float(res.energy),
        "atoms": [[int(z), *(float(c) for c in (r * ANG_TO_BOHR))] for z, r in zip(zs, xyz)],
    }
    out = workdir / f"{name}.json"
    out.write_text(json.dumps(payload, indent=1))
    return out


def assemble(workdir: Path, module: Path) -> None:
    # Only this script's own result files: run_job writes ``opt_<name>.*``
    # sidecars (``.population.json`` among them) next to them.
    results = sorted(p for p in workdir.glob("*.json") if not p.name.startswith("opt_"))
    entries = [json.loads(p.read_text()) for p in results]
    entries = [e for e in entries if {"name", "atoms", "protocol", "energy_hartree"} <= set(e)]
    lines = [
        '"""Solvent geometries for the COSMO-RS solvent library (bohr).',
        "",
        "Generated by ``scripts/cosmors_solvent_library.py``; do not edit by hand.",
        "Each entry records the protocol its geometry was optimized at. The",
        "coordinates are vibe-qc's own results: a starting structure built from",
        "standard bond lengths (or ASE's G2 collection), optimized at the protocol",
        "each entry records.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        "",
        "@dataclass(frozen=True)",
        "class SolventGeometry:",
        '    """One library solvent: atoms as ``(Z, (x, y, z))`` in bohr."""',
        "",
        "    name: str",
        "    atoms: tuple[tuple[int, tuple[float, float, float]], ...]",
        "    protocol: str",
        "    energy_hartree: float",
        "",
        "",
        "SOLVENT_GEOMETRIES: dict[str, SolventGeometry] = {",
    ]
    for e in entries:
        lines.append(f"    {e['name']!r}: SolventGeometry(")
        lines.append(f"        name={e['name']!r},")
        lines.append(f"        protocol={e['protocol']!r},")
        lines.append(f"        energy_hartree={e['energy_hartree']!r},")
        lines.append("        atoms=(")
        for z, x, y, zc in e["atoms"]:
            lines.append(f"            ({z}, ({x:.8f}, {y:.8f}, {zc:.8f})),")
        lines.append("        ),")
        lines.append("    ),")
    lines.append("}")
    lines.append("")
    module.write_text("\n".join(lines))
    print(f"wrote {module} with {len(entries)} solvents")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, type=Path)
    ap.add_argument("--solvent", action="append", default=None)
    ap.add_argument("--functional", default=FUNCTIONAL)
    ap.add_argument("--basis", default=BASIS)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--conductor", action="store_true",
                    help="optimize as an ideal conductor instead of in the gas phase")
    ap.add_argument("--assemble", action="store_true", help="only regenerate the module")
    ap.add_argument("--module", type=Path, default=MODULE)
    ap.add_argument("--_one", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    args.workdir.mkdir(parents=True, exist_ok=True)

    if args._one:
        optimize_one(args._one, args.workdir, args.functional, args.basis, args.conductor)
        return 0
    if not args.assemble:
        names = args.solvent or sorted(starting_structures())
        todo = [n for n in names if not (args.workdir / f"{n}.json").is_file()]

        def run(name):
            cmd = [sys.executable, str(Path(__file__).resolve()), "--workdir", str(args.workdir),
                   "--functional", args.functional, "--basis", args.basis, "--_one", name]
            if args.conductor:
                cmd.append("--conductor")
            log = args.workdir / f"{name}.log"
            with log.open("w") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT)
            print(f"{name}: {'ok' if rc == 0 else f'FAILED (see {log})'}", flush=True)
            return rc

        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            codes = list(pool.map(run, todo))
        if any(codes):
            return 1
    assemble(args.workdir, args.module)
    return 0


if __name__ == "__main__":
    sys.exit(main())
