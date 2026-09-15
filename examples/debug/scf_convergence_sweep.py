"""NaCl / STO-3G / RKS-LDA — convergence-aid sweep (debug substrate).

Pinned reproducer for the other chat's "NaCl-LDA SCF diverges"
debugging. Sweeps the convergence-aid axes around a sensible center
and reports for each configuration:

    converged?  |  n_iter  |  final E (Ha)  |  wall (s)  |  notes

Each failing SCF leaves a ``.dump`` file under ``output/`` via
``vibeqc.crash_dump_context`` (v0.6) so the failed state can be
loaded back for post-mortem inspection without re-running. Each
SCF is also wrapped in ``vibeqc.perf_log`` so per-phase timings
are accumulated into a sibling ``.perf`` file.

The sweep axes (one knob at a time around a center, then a small
set of stress combos):

    initial_guess  ∈ {HCORE, SAD}             # SAD is v0.6.x
    damping        ∈ {0.0, 0.3, 0.5, 0.7}     # 0.5 is the default
    level_shift    ∈ {0.0, 0.3, 0.5, 1.0}     # 0.0 is the default
    use_diis       ∈ {True, False}            # True is the default
    smearing_T     ∈ {0.0, 0.001, 0.005}      # 0.0 = no smearing

Total: 4 + 4 + 2 + 3 + 1 (center) = 14 configurations. With
``max_iter=20`` and STO-3G basis, the whole sweep finishes in a
few minutes on a workstation.

Wall: ~2-10 min total (most of it spent on configurations that
diverge and walk all 20 iterations).

Run:
    .venv/bin/python examples/debug/scf_convergence_sweep.py

Outputs (under ``examples/debug/output/``):
    sweep.csv                          — table of all results
    {config_label}.dump                — crash dump per failing SCF
    {config_label}.dump.density.npy    — last-iter density per failure
    {config_label}.perf                — perf-log per configuration

Use the .dump files for the other chat:
    vq.load_dump("output/cfg_damping_0.0.dump")  # → dict with the
                                                  # full failed state
"""

from __future__ import annotations

import csv
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output"
OUT_DIR.mkdir(exist_ok=True)
SWEEP_CSV = OUT_DIR / "sweep.csv"

# ---- 1. Fixed system: NaCl conventional cubic ----------------------
A_BOHR = 5.640 / 0.529177210903
NA_FRAC = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
CL_FRAC = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]


def build_system() -> vq.PeriodicSystem:
    unit_cell = (
        [vq.Atom(11, [f * A_BOHR for f in p]) for p in NA_FRAC] +
        [vq.Atom(17, [f * A_BOHR for f in p]) for p in CL_FRAC]
    )
    return vq.PeriodicSystem(
        dim=3,
        lattice=A_BOHR * np.eye(3),
        unit_cell=unit_cell,
    )


def make_opts(
    *,
    initial_guess: vq.InitialGuess = vq.InitialGuess.HCORE,
    damping: float = 0.5,
    level_shift: float = 0.0,
    use_diis: bool = True,
    smearing_T: float = 0.0,
) -> vq.PeriodicKSOptions:
    """Build a fresh PeriodicKSOptions with the requested knobs."""
    opts = vq.PeriodicKSOptions()
    opts.functional = "LDA"
    opts.lattice_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    opts.lattice_opts.cutoff_bohr = 10.0
    opts.lattice_opts.nuclear_cutoff_bohr = 18.0
    opts.conv_tol_energy = 1e-6
    opts.max_iter = 20            # bound the wall on diverging cases

    opts.initial_guess = initial_guess
    opts.damping = damping
    opts.level_shift = level_shift
    opts.use_diis = use_diis
    opts.smearing_temperature = smearing_T
    return opts


@dataclass
class RunRecord:
    label: str
    initial_guess: str
    damping: float
    level_shift: float
    use_diis: bool
    smearing_T: float
    converged: Optional[bool]
    n_iter: Optional[int]
    energy_ha: Optional[float]
    wall_s: float
    note: str


def run_one(label: str, system: vq.PeriodicSystem, basis: vq.BasisSet,
            opts: vq.PeriodicKSOptions) -> RunRecord:
    """Run one SCF, catch divergence, write crash dump on failure."""
    out_stem = OUT_DIR / label
    print(f"\n[{label}]", flush=True)

    rec = RunRecord(
        label=label,
        initial_guess=opts.initial_guess.name,
        damping=opts.damping,
        level_shift=opts.level_shift,
        use_diis=opts.use_diis,
        smearing_T=opts.smearing_temperature,
        converged=None, n_iter=None, energy_ha=None,
        wall_s=0.0, note="",
    )

    t0 = time.perf_counter()
    try:
        with vq.crash_dump_context(out_stem):
            with vq.perf_log(out_stem.with_suffix(".perf")):
                result = vq.run_rks_periodic_scf(
                    system, basis,
                    vq.KPoints.gamma(system),
                    opts,
                    progress=False,           # quiet; per-config
                    spacing_bohr=0.6,         # debug-friendly FFT
                )
        rec.wall_s = time.perf_counter() - t0
        rec.converged = bool(result.converged)
        rec.n_iter = int(result.n_iter)
        rec.energy_ha = float(result.energy)
        rec.note = "ok" if rec.converged else "max_iter reached, no convergence"
    except Exception as exc:
        rec.wall_s = time.perf_counter() - t0
        rec.converged = False
        rec.note = f"{type(exc).__name__}: {str(exc)[:80]}"
        # The crash_dump_context already wrote the .dump file with
        # the last-iter state; nothing else to do.

    flag = "✓" if rec.converged else "✗"
    print(f"  {flag} iter={rec.n_iter}  E={rec.energy_ha}  "
          f"wall={rec.wall_s:.1f}s  {rec.note}")
    return rec


def main() -> None:
    print("=" * 72)
    print(" NaCl/STO-3G/RKS-LDA convergence-aid sweep")
    print("=" * 72)

    system = build_system()
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    print(f"  cell:  {len(system.unit_cell)} atoms (4 Na + 4 Cl)")
    print(f"  basis: sto-3g, {basis.nbasis} bf per cell")
    print(f"  output dir: {OUT_DIR}")

    # Sweep configurations. Center = vibe-qc defaults.
    configs: list[tuple[str, dict]] = [
        # Center / default
        ("center_defaults", {}),

        # initial_guess sweep
        ("guess_HCORE", dict(initial_guess=vq.InitialGuess.HCORE)),
        ("guess_SAD",   dict(initial_guess=vq.InitialGuess.SAD)),

        # damping sweep
        ("damping_0.0", dict(damping=0.0)),
        ("damping_0.3", dict(damping=0.3)),
        ("damping_0.5", dict(damping=0.5)),
        ("damping_0.7", dict(damping=0.7)),

        # level_shift sweep
        ("ls_0.0",  dict(level_shift=0.0)),
        ("ls_0.3",  dict(level_shift=0.3)),
        ("ls_0.5",  dict(level_shift=0.5)),
        ("ls_1.0",  dict(level_shift=1.0)),

        # DIIS off
        ("diis_off", dict(use_diis=False)),

        # Smearing sweep
        ("smear_0.001", dict(smearing_T=0.001)),
        ("smear_0.005", dict(smearing_T=0.005)),

        # Stress combos: the v0.5.6 hint string suggests this combo.
        ("stress_damping_ls_diisoff",
         dict(damping=0.5, level_shift=0.5, use_diis=False)),

        # Belt-and-suspenders: SAD + level shift + smearing.
        ("sad_ls_smear",
         dict(initial_guess=vq.InitialGuess.SAD,
              level_shift=0.5, smearing_T=0.005)),
    ]

    results: list[RunRecord] = []
    for label, kwargs in configs:
        opts = make_opts(**kwargs)
        rec = run_one(label, system, basis, opts)
        results.append(rec)

    # CSV summary
    with open(SWEEP_CSV, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "label", "initial_guess", "damping", "level_shift",
            "use_diis", "smearing_T",
            "converged", "n_iter", "energy_ha", "wall_s", "note",
        ])
        for r in results:
            writer.writerow([
                r.label, r.initial_guess, r.damping, r.level_shift,
                r.use_diis, r.smearing_T,
                r.converged, r.n_iter, r.energy_ha, r.wall_s, r.note,
            ])

    # Pretty summary
    print()
    print("=" * 72)
    print(" Summary")
    print("=" * 72)
    n_ok = sum(1 for r in results if r.converged)
    print(f"  {n_ok} / {len(results)} configurations converged")
    print()
    print(f"  {'label':30s}  {'flag':4s}  {'iter':4s}  "
          f"{'E (Ha)':>14s}  {'wall':>6s}  note")
    print(f"  {'-'*30}  {'-'*4}  {'-'*4}  {'-'*14}  {'-'*6}  ----")
    for r in results:
        flag = "OK" if r.converged else "FAIL"
        e_str = f"{r.energy_ha:14.6f}" if r.energy_ha is not None else " " * 14
        i_str = f"{r.n_iter:4d}" if r.n_iter is not None else "  - "
        print(f"  {r.label:30s}  {flag:4s}  {i_str}  {e_str}  "
              f"{r.wall_s:5.1f}s  {r.note[:60]}")
    print()
    print(f"  CSV: {SWEEP_CSV.relative_to(HERE.parent.parent)}")
    print(f"  Per-failure dumps: {OUT_DIR.relative_to(HERE.parent.parent)}/*.dump")
    print(f"  Per-config perf logs: {OUT_DIR.relative_to(HERE.parent.parent)}/*.perf")
    print()
    print("  Inspect a failure:")
    print("    import vibeqc as vq")
    print("    dump = vq.load_dump('output/damping_0.0.dump')")
    print("    print(dump['crash']['phase'], dump['hint']['likely_cause'])")
    print("    last = dump['scf.last_iter']  # iter, energy, dE, grad_norm, diis")


if __name__ == "__main__":
    main()
