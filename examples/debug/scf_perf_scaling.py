"""Periodic-SCF perf-knob scan — characterise wall-time vs precision.

Runs the same NaCl/STO-3G/RKS-LDA SCF at varying values of the
three EWALD_3D performance knobs:

    spacing_bohr  ∈ {0.3, 0.4, 0.5, 0.6, 0.8}
                  (FFT grid resolution; smaller = more grid pts)
    cutoff_bohr   ∈ {8, 10, 12, 15}
                  (real-space AO-pair cutoff for the lattice ERIs)
    omega         ∈ {0.3, 0.5, 0.7}
                  (Ewald α split — higher = more weight to short-range)

For each setting (one knob varied at a time around a sensible
center), reports per-config:

    wall (s)  |  iter  |  E (Ha)  |  ΔE vs reference (Ha)

The reference is the tightest-knob run (spacing=0.3, cutoff=15,
omega=0.5). Anything that diverges from it by more than ~1e-3 Ha
on default-tolerance cases is the precision cost of the
coarsening — useful for picking trade-off points.

Wall: ~3-10 min total on a laptop. Each config wraps in
crash_dump_context + perf_log so failures leave artifacts.

Run:
    .venv/bin/python examples/debug/scf_perf_scaling.py

CSV output:
    output/perf-scaling/summary.csv
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "perf-scaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "summary.csv"

# --- System: NaCl conventional cubic --------------------------------
A_BOHR = 5.640 / 0.529177210903
NA_FRAC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
CL_FRAC = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]


def build_system() -> vq.PeriodicSystem:
    cell = (
        [vq.Atom(11, [f * A_BOHR for f in p]) for p in NA_FRAC] +
        [vq.Atom(17, [f * A_BOHR for f in p]) for p in CL_FRAC]
    )
    return vq.PeriodicSystem(3, A_BOHR * np.eye(3), cell)


def make_opts(cutoff_bohr: float = 12.0) -> vq.PeriodicKSOptions:
    o = vq.PeriodicKSOptions()
    o.functional = "LDA"
    o.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    o.lattice_opts.cutoff_bohr = cutoff_bohr
    o.lattice_opts.nuclear_cutoff_bohr = max(25.0, 2 * cutoff_bohr)
    o.conv_tol_energy = 1e-7
    o.max_iter = 30
    return o


@dataclass
class PerfRow:
    label: str
    spacing_bohr: float
    cutoff_bohr: float
    omega: float
    converged: Optional[bool]
    n_iter: Optional[int]
    energy_ha: Optional[float]
    wall_s: float
    note: str


def run_one(label: str, system: vq.PeriodicSystem, basis: vq.BasisSet,
            *, spacing: float, cutoff: float, omega: float) -> PerfRow:
    print(f"\n[{label}]  spacing={spacing}  cutoff={cutoff}  omega={omega}",
          flush=True)
    out_stem = OUT_DIR / label
    rec = PerfRow(
        label=label, spacing_bohr=spacing, cutoff_bohr=cutoff, omega=omega,
        converged=None, n_iter=None, energy_ha=None,
        wall_s=0.0, note="",
    )
    t0 = time.perf_counter()
    try:
        with vq.crash_dump_context(out_stem):
            with vq.perf_log(out_stem.with_suffix(".perf")):
                result = vq.run_rks_periodic_scf(
                    system, basis, vq.KPoints.gamma(system),
                    make_opts(cutoff_bohr=cutoff),
                    progress=False,
                    spacing_bohr=spacing,
                    omega=omega,
                )
        rec.wall_s = time.perf_counter() - t0
        rec.converged = bool(result.converged)
        rec.n_iter = int(result.n_iter)
        rec.energy_ha = float(result.energy)
        rec.note = "ok" if rec.converged else "max_iter"
    except Exception as exc:
        rec.wall_s = time.perf_counter() - t0
        rec.converged = False
        rec.note = f"{type(exc).__name__}: {str(exc)[:50]}"

    flag = "✓" if rec.converged else "✗"
    print(f"  {flag}  iter={rec.n_iter}  E={rec.energy_ha}  "
          f"wall={rec.wall_s:.1f}s  {rec.note}")
    return rec


def main() -> None:
    print("=" * 72)
    print(" Periodic-SCF perf-knob scan — NaCl/STO-3G/RKS-LDA/Γ")
    print("=" * 72)

    system = build_system()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    print(f"  cell:  {len(system.unit_cell)} atoms (4 Na + 4 Cl)")
    print(f"  basis: sto-3g, {basis.nbasis} bf per cell")

    # Center / reference: tightest knobs.
    ref_spacing, ref_cutoff, ref_omega = 0.3, 15.0, 0.5

    configs = [
        # Reference
        (f"00_ref_s{ref_spacing}_c{ref_cutoff:.0f}_o{ref_omega}",
         ref_spacing, ref_cutoff, ref_omega),

        # spacing sweep (cutoff + omega = ref)
        (f"01_spacing_0.4", 0.4, ref_cutoff, ref_omega),
        (f"02_spacing_0.5", 0.5, ref_cutoff, ref_omega),
        (f"03_spacing_0.6", 0.6, ref_cutoff, ref_omega),
        (f"04_spacing_0.8", 0.8, ref_cutoff, ref_omega),

        # cutoff sweep (spacing + omega = ref)
        (f"05_cutoff_8",    ref_spacing, 8.0, ref_omega),
        (f"06_cutoff_10",   ref_spacing, 10.0, ref_omega),
        (f"07_cutoff_12",   ref_spacing, 12.0, ref_omega),

        # omega sweep (spacing + cutoff = ref)
        (f"08_omega_0.3",   ref_spacing, ref_cutoff, 0.3),
        (f"09_omega_0.7",   ref_spacing, ref_cutoff, 0.7),
    ]

    results: list[PerfRow] = []
    for label, spacing, cutoff, omega in configs:
        results.append(run_one(label, system, basis,
                               spacing=spacing, cutoff=cutoff, omega=omega))

    # CSV
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "spacing_bohr", "cutoff_bohr", "omega",
                    "converged", "n_iter", "energy_ha", "wall_s", "note"])
        for r in results:
            w.writerow([r.label, r.spacing_bohr, r.cutoff_bohr, r.omega,
                        r.converged, r.n_iter, r.energy_ha,
                        r.wall_s, r.note])

    # Reference for ΔE
    ref = next((r for r in results if r.label.startswith("00_ref")), None)
    e_ref = ref.energy_ha if (ref and ref.energy_ha is not None) else None

    print()
    print("=" * 72)
    print(" Summary — wall vs precision (ΔE vs reference)")
    print("=" * 72)
    print(f"  reference: {ref.label if ref else 'N/A'}  "
          f"E = {e_ref}")
    print()
    print(f"  {'label':32s}  {'spacing':>7s}  {'cutoff':>6s}  "
          f"{'omega':>5s}  {'wall':>6s}  {'ΔE (Ha)':>10s}  flag")
    for r in results:
        flag = "OK" if r.converged else "FAIL"
        de = (r.energy_ha - e_ref
              if (r.energy_ha is not None and e_ref is not None)
              else None)
        de_str = f"{de:+10.4e}" if de is not None else "         -"
        print(f"  {r.label:32s}  {r.spacing_bohr:7.2f}  "
              f"{r.cutoff_bohr:6.1f}  {r.omega:5.2f}  "
              f"{r.wall_s:5.1f}s  {de_str}  {flag}")

    print()
    print("  Reading the table:")
    print("    spacing  smaller → more FFT grid pts → wall scales as 1/spacing³")
    print("    cutoff   larger  → more lattice cells → wall scales linearly")
    print("    omega    higher  → more weight to real-space, less FFT")
    print()
    print(f"  CSV: {CSV_PATH.relative_to(HERE.parent.parent)}")


if __name__ == "__main__":
    main()
