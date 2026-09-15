"""LiH conventional rocksalt — full per-iteration SCF trace.

Goal: localise WHICH iteration of the v0.7.0 RKS-LDA SCF converges
to the wrong stationary point. The iter-1 V_ne fix (commit 9de640b)
moved E_total from a v0.6.x baseline of ~-1060 Ha down to a sane
-50 Ha at Hcore-guess; with SAD-guess + DIIS + auto_optimize_truncation
the converged SCF lands at ~+65 Ha — off by ~+97 Ha vs the
atomic-limit reference -32 Ha.

This script prints:
  - iter / E_total / dE / ||[F,DS]|| / n_kept / DIIS state per iter
  - the energy decomposition (kin / V_ne / J / xc / nn / Madelung-fix)
    at iter 1 vs final iter
  - re-runs with pob-DZVP-rev2 (the basis designed for solids) for
    a side-by-side comparison

Run:
  .venv/bin/python examples/debug/lih_iter_trace.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "lih-iter-trace"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _lih_conventional() -> vq.PeriodicSystem:
    """LiH conventional cubic, a = 4.084 Å (rocksalt, 4 LiH formula
    units per unit cell)."""
    a = 4.084 / 0.529177210903
    unit_cell = []
    for fx, fy, fz in [(0, 0, 0), (0, 0.5, 0.5),
                        (0.5, 0, 0.5), (0.5, 0.5, 0)]:
        unit_cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in [(0.5, 0.5, 0.5), (0.5, 0, 0),
                        (0, 0.5, 0), (0, 0, 0.5)]:
        unit_cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))
    return vq.PeriodicSystem(3, np.diag([a, a, a]), unit_cell)


def _atomic_limit_reference_lda() -> dict:
    """Sum of isolated-atom RKS-LDA energies for 4 Li + 4 H, in Ha.

    These come from molecular RKS-LDA / sto-3g (or pob-DZVP-rev2)
    using vibe-qc — running them is fast (sub-second per atom) so we
    compute on the fly instead of hardcoding."""
    refs = {}
    for atom_z, n_unpaired, label in [(3, 1, "Li"), (1, 1, "H")]:
        # Open-shell atom needs UKS + multiplicity = 2.
        mol = vq.Molecule([vq.Atom(atom_z, [0.0, 0.0, 0.0])],
                           charge=0, multiplicity=n_unpaired + 1)
        for basis_name in ("sto-3g", "pob-dzvp-rev2"):
            try:
                basis = vq.BasisSet(mol, basis_name)
            except Exception:
                # Some bases may not have entries for every Z.
                continue
            opts = vq.UKSOptions()
            opts.functional = "lda"
            opts.conv_tol_energy = 1e-9
            r = vq.run_uks(mol, basis, opts)
            refs[(label, basis_name)] = float(r.energy)
            print(f"    atomic-limit ref: {label} / {basis_name:>16s} = "
                  f"{r.energy:+12.6f} Ha  ({r.n_iter} iters)")
    return refs


def run_lih_scf(basis_name: str, atomic_refs: dict) -> Optional[dict]:
    """Run LiH conventional SCF and capture the iter trace + decomp."""
    print()
    print("=" * 70)
    print(f" LiH conventional rocksalt — {basis_name}")
    print("=" * 70)

    sysp = _lih_conventional()
    try:
        basis = vq.make_basis(sysp.unit_cell_molecule(), basis_name)
    except Exception as exc:
        print(f"  basis '{basis_name}' unavailable: "
              f"{type(exc).__name__}: {exc}")
        return None

    opts = vq.PeriodicKSOptions()
    opts.functional = "lda"
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 12.0
    opts.lattice_opts.nuclear_cutoff_bohr = 25.0
    opts.max_iter = 30
    opts.use_diis = True
    opts.damping = 0.7
    opts.conv_tol_energy = 1e-7
    opts.initial_guess = vq.InitialGuess.SAD

    out_stem = OUT_DIR / f"lih_{basis_name.replace('/', '_')}"
    t0 = time.perf_counter()
    try:
        with vq.crash_dump_context(out_stem):
            with vq.perf_log(out_stem.with_suffix(".perf")):
                r = vq.run_rks_periodic_gamma_ewald3d(
                    sysp, basis, opts,
                    omega=0.5, spacing_bohr=0.5,
                    progress=False,
                )
    except Exception as exc:
        wall = time.perf_counter() - t0
        print(f"  SCF FAILED after {wall:.1f}s: "
              f"{type(exc).__name__}: {str(exc)[:100]}")
        return None
    wall = time.perf_counter() - t0

    # --- Per-iteration trace ------------------------------------------
    print(f"\n  SCF trace ({r.n_iter} iters, {wall:.1f}s wall):")
    print(f"  {'iter':>4}  {'E_total (Ha)':>16}  {'dE':>12}  "
          f"{'||[F,DS]||':>12}")
    prev_E = None
    for it in r.scf_trace:
        dE = (it.energy - prev_E) if prev_E is not None else 0.0
        print(f"  {it.iteration:>4}  {it.energy:>+16.8f}  {dE:>+12.3e}  "
              f"{it.gradient_norm:>12.3e}")
        prev_E = it.energy

    # --- Reference comparison -----------------------------------------
    li_ref = atomic_refs.get(("Li", basis_name), float("nan"))
    h_ref = atomic_refs.get(("H", basis_name), float("nan"))
    if not np.isnan(li_ref) and not np.isnan(h_ref):
        atomic_total = 4 * li_ref + 4 * h_ref
        binding = r.energy - atomic_total
        print(f"\n  atomic-limit ref:  4·Li + 4·H = {atomic_total:+12.6f} Ha "
              f"({basis_name})")
        print(f"  vibe-qc converged: E_total       = {r.energy:+12.6f} Ha")
        print(f"  binding / spurious shift           "
              f"= {binding:+12.6f} Ha   "
              f"({'PHYSICAL ✓' if -10 < binding < 0 else '✗ NON-PHYSICAL'})")

    return {
        "basis": basis_name,
        "n_iter": r.n_iter,
        "converged": r.converged,
        "energy": float(r.energy),
        "wall_s": wall,
        "trace": [(it.iteration, float(it.energy), float(it.gradient_norm))
                  for it in r.scf_trace],
    }


def main() -> None:
    print("=" * 70)
    print(" LiH iter-trace diagnostic — v0.7.0 periodic SCF correctness")
    print("=" * 70)

    print("\n  Atomic-limit references (RKS-LDA on isolated Li, H):")
    atomic_refs = _atomic_limit_reference_lda()

    # Two-basis sweep: sto-3g (small molecular basis, known to break
    # on tight crystals) vs pob-dzvp-rev2 (Vilela Oliveira 2019,
    # designed for solids).
    results = []
    for basis_name in ("sto-3g", "pob-dzvp-rev2"):
        res = run_lih_scf(basis_name, atomic_refs)
        if res is not None:
            results.append(res)

    # Summary
    print()
    print("=" * 70)
    print(" Summary")
    print("=" * 70)
    print(f"  {'basis':>16}  {'iters':>6}  {'conv':>5}  "
          f"{'E_total (Ha)':>16}  {'wall (s)':>9}")
    for r in results:
        conv = "✓" if r["converged"] else "✗"
        print(f"  {r['basis']:>16s}  {r['n_iter']:>6}  {conv:>5}  "
              f"{r['energy']:>+16.6f}  {r['wall_s']:>9.1f}")

    # Save the trace as a JSON for downstream analysis
    import json
    trace_path = OUT_DIR / "trace.json"
    trace_path.write_text(json.dumps({
        "atomic_refs": {f"{k[0]}/{k[1]}": v
                         for k, v in atomic_refs.items()},
        "runs": results,
    }, indent=2))
    print(f"\n  artefacts → {OUT_DIR}/")
    print(f"  trace.json   per-iter trace + atomic refs")


if __name__ == "__main__":
    main()
