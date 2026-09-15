#!/usr/bin/env python
"""Interaction-range parameterization demo on the AICCM-2026 test set.

The cyclic cluster can be sized by a **real-space interaction radius** ``R_c``
instead of an explicit mesh ``nrep`` — the real-space dual of choosing a k-mesh
density. This is finite-group/lattice geometry, not a Γ-CCM/χ-CCM construction
equivalence. The minimal cluster whose supercell Wigner-Seitz cell
encloses a sphere of radius ``R_c`` around every atom is derived automatically.

Part 1 (geometry, all 28 systems, no SCF — fast)
    For a common ``R_c`` it derives the uniform cluster for every test-set
    crystal (using each system's intrinsic periodicity ``dim``) and reports the
    dual Monkhorst–Pack k-mesh, achieved inscribed radius ``r_in``, and
    equivalent k-spacing ``Δk = π/R_c``. One knob, all 7 crystal systems.

Part 2 (the duality, cubic 3-D systems)
    For cubic systems whose tabulated ``nrep_4c`` is already uniform, find the
    ``R_c`` that reproduces it and confirm cluster ≡ k-mesh of the same size.

Part 3 (energy convergence, cheap 1-D systems — opt-in with --scan)
    A real ``R_c`` → energy/atom convergence scan (scalable HF-CCM) — the CCM
    analogue of a k-point convergence study.

Usage
-----
    python demo_interaction_range.py [--radius R_c_bohr] [--scan] [--out FILE]

Reference: Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014),
doi:10.1002/jcc.23550. Derivation: docs/aiccm2026dev_a_followon.md § A.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testset  # noqa: E402

from vibeqc.periodic.ccm import (  # noqa: E402
    ccm_interaction_range_scan,
    interplanar_spacings,
    kspacing_for_interaction_range,
    nrep_for_interaction_range,
    wsc_inscribed_radius,
)


def _unit_vectors(system) -> np.ndarray:
    """Rows = lattice vectors a_i (PeriodicSystem stores them as columns)."""
    return np.asarray(system.lattice, dtype=float).T


def part1_geometry(radius_bohr: float) -> list:
    """Derive the uniform cluster for every test-set system at a common R_c."""
    rows = []
    for name, meta in testset.SYSTEMS.items():
        try:
            system = testset.build(name)
        except Exception as exc:  # ASE-built systems without the [ase] extra
            rows.append({"system": name, "skipped": str(exc)[:60]})
            continue
        dim = int(meta.get("dim", system.dim))
        L = _unit_vectors(system)
        d = interplanar_spacings(L)
        nrep = nrep_for_interaction_range(L, radius_bohr, dim=dim)
        Lc = np.asarray(nrep, dtype=float)[:, None] * L
        r_in = wsc_inscribed_radius(Lc)
        rows.append({
            "system": name,
            "dim": dim,
            "klass": meta.get("klass"),
            "d_bohr": [round(float(x), 3) for x in d],
            "nrep_radius": list(nrep),
            "kmesh_equivalent": list(nrep),       # cluster (N1,N2,N3) ≡ MP mesh
            "nrep_4c_tabulated": list(meta.get("nrep_4c", ())),
            "n_atoms": int(np.prod(nrep)) * int(meta.get("atoms", 0)),
            "r_in_bohr": round(float(r_in), 3),
            "kspacing_bohr_inv": round(kspacing_for_interaction_range(radius_bohr), 4),
        })
    return rows


def _radius_reproducing(L: np.ndarray, nrep_target, dim: int):
    """Smallest R_c whose derived cluster equals nrep_target, or None.

    Per direction, ``⌈2R_c/d_i⌉ = N_i`` holds for ``R_c ∈ ((N_i-1) d_i/2, N_i d_i/2]``.
    A single R_c reproduces an anisotropic target only if those intervals (over
    the periodic, ``N_i>1`` directions) intersect — return the interval's lower
    edge (just-reaching R_c), else None (the tabulated mesh is anisotropic and
    has no single isotropic-radius preimage).
    """
    d = interplanar_spacings(L)
    lo, hi = 0.0, np.inf
    for i in range(3):
        if i >= dim:
            continue
        Ni = int(nrep_target[i])
        lo = max(lo, 0.5 * (Ni - 1) * d[i])
        hi = min(hi, 0.5 * Ni * d[i])
    if lo >= hi:
        return None
    return float(lo)  # the just-reaching radius (exclusive lower edge)


def part2_duality() -> list:
    """For cubic systems with a uniform tabulated mesh, find the R_c preimage."""
    rows = []
    for name in ("c-diamond", "si-diamond", "mgo", "nacl-rocksalt", "sic-zb"):
        meta = testset.SYSTEMS.get(name)
        if meta is None:
            continue
        try:
            system = testset.build(name)
        except Exception:
            continue
        dim = int(meta.get("dim", system.dim))
        L = _unit_vectors(system)
        nrep_4c = meta["nrep_4c"]
        rc = _radius_reproducing(L, nrep_4c, dim)
        rows.append({
            "system": name,
            "nrep_4c": list(nrep_4c),
            "kmesh_tabulated": list(meta.get("kmesh", ())),
            "radius_reproducing_nrep_4c_bohr": (round(rc, 3) if rc is not None else None),
        })
    return rows


def part3_scan(systems, radii_bohr) -> dict:
    """Real R_c → energy/atom convergence scan (scalable HF-CCM)."""
    out = {}
    for name in systems:
        meta = testset.SYSTEMS[name]
        system = testset.build(name)
        basis = meta.get("basis", "sto-3g")
        # cap non-periodic directions at the system's intrinsic dim
        scan = ccm_interaction_range_scan(
            system, basis, radii_bohr, method="aiccm2026dev-a",
        )
        out[name] = {
            "basis": basis,
            "dim": int(meta.get("dim", system.dim)),
            "series": scan.as_records(),
            "converged_radius_1e4": scan.converged_radius(1e-4),
            "converged_radius_1e5": scan.converged_radius(1e-5),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--radius", type=float, default=7.0,
                    help="common interaction radius R_c (bohr) for Part 1")
    ap.add_argument("--scan", action="store_true",
                    help="run the energy-convergence scan (Part 3, cheap 1-D)")
    ap.add_argument("--scan-systems", nargs="+", default=["h-chain"],
                    help="systems to scan in Part 3 (default: h-chain)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the full result JSON here")
    args = ap.parse_args()

    result = {"radius_bohr": args.radius}

    print(f"\n=== Part 1: uniform cluster from R_c = {args.radius} bohr "
          f"(Δk = {kspacing_for_interaction_range(args.radius):.4f} bohr⁻¹) ===")
    print(f"{'system':16s} {'dim':>3} {'d_i (bohr)':>22} {'nrep(R_c)':>11} "
          f"{'≡ k-mesh':>10} {'nrep_4c':>10} {'r_in':>6}")
    geom = part1_geometry(args.radius)
    for r in geom:
        if "skipped" in r:
            print(f"{r['system']:16s}  -- skipped ({r['skipped']})")
            continue
        print(f"{r['system']:16s} {r['dim']:>3} "
              f"{str(r['d_bohr']):>22} {str(r['nrep_radius']):>11} "
              f"{str(r['kmesh_equivalent']):>10} {str(r['nrep_4c_tabulated']):>10} "
              f"{r['r_in_bohr']:>6.2f}")
    result["part1_geometry"] = geom

    print("\n=== Part 2: which R_c reproduces the tabulated cubic nrep_4c? ===")
    dual = part2_duality()
    for r in dual:
        rc = r["radius_reproducing_nrep_4c_bohr"]
        print(f"{r['system']:14s} nrep_4c={str(r['nrep_4c']):>10} "
              f"≡ k-mesh {str(r['nrep_4c']):>10}  "
              f"reproduced at R_c ≳ {rc if rc is not None else 'N/A (anisotropic)'} bohr")
    result["part2_duality"] = dual

    if args.scan:
        print("\n=== Part 3: R_c → energy/atom convergence (scalable HF-CCM) ===")
        # h-chain is the canonical fast 1-D demo (dim=1 → stays a chain at any R_c).
        # Add more systems on the command line if you want a heavier scan.
        radii = [6.0, 9.0, 12.0, 15.0, 18.0, 24.0, 30.0, 36.0]
        scan = part3_scan(args.scan_systems, radii)
        for name, blk in scan.items():
            print(f"\n  {name} ({blk['basis']}, dim={blk['dim']}):")
            print(f"    {'R_c':>5} {'nrep':>10} {'n_at':>5} {'E/atom':>13} {'ΔE':>11}")
            prev = None
            for rec in blk["series"]:
                e = rec["energy_per_atom"]
                de = (f"{e - prev:+.2e}" if prev is not None else "    --    ")
                prev = e
                print(f"    {rec['radius_bohr']:5.0f} {str(rec['nrep']):>10} "
                      f"{rec['n_atoms']:5d} {e:13.7f} {de:>11}")
            print(f"    converged R_c @1e-4={blk['converged_radius_1e4']}  "
                  f"@1e-5={blk['converged_radius_1e5']} bohr")
        result["part3_scan"] = scan

    if args.out:
        args.out.write_text(json.dumps(result, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
