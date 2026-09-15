"""vibe-qc GDF (Steps 1+2 gauge fix) sweep over the simple-systems baseline.

Companion to ``run_vibeqc_ewald3d_sweep.py``: same 6 systems, same
8-atom conventional cell, same STO-3G basis, but now via the native
gamma GDF driver after the 2026-05-13 gauge fix Steps 1 (V_ne / e_nuc
in Ewald gauge) + 2 (J via composed Ewald-3D, K still via Lpq).

The H2 / 12-bohr test reproduces vibeqc EWALD_3D to ~25 µHa after
Steps 1+2 (no Step 3 yet) — the modrho compensation in Lpq appears
to already handle the K-side G=0 gauge for soft cells.  This sweep
tests whether the same holds for the tighter ionic crystals (LiH at
a=4 Å, MgO at a=4.2 Å, NaCl at a=5.6 Å, etc.) where the modrho
residual matters more.

If GDF matches CRYSTAL14 + vibeqc EWALD_3D within parity tolerance
(target: ~1 mHa / FU at STO-3G), Step 3 (K Madelung correction) is
NOT needed and the v0.8.0 gauge fix ships as just Steps 1+2.

If GDF parity is poor (10+ mHa / FU off), Step 3 (or Option B's
proper slice 3d) is needed for tight cells.

Run via vq:
    vq submit examples/regression/crystal_parity/baseline_sto3g/run_vibeqc_gdf_sweep.py
"""
from __future__ import annotations

import os
import sys
import time

os.environ.setdefault("PYSCF_TMPDIR", "/tmp/pyscf_vibeqc")

# Force line-buffered stdout so vq's stdout.log captures progress live.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import vibeqc as vq


ANG2BOHR = 1.0 / 0.529177210903


def _rocksalt_conv(a_ang: float, z_cation: int, z_anion: int):
    """Build the 8-atom conventional cubic rocksalt cell."""
    a = a_ang * ANG2BOHR
    cation_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    anion_frac  = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    atoms = []
    for fx, fy, fz in cation_frac:
        atoms.append(vq.Atom(z_cation, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in anion_frac:
        atoms.append(vq.Atom(z_anion, [fx * a, fy * a, fz * a]))
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def _diamond_conv(a_ang: float, z: int):
    """Build the 8-atom conventional cubic diamond cell."""
    a = a_ang * ANG2BOHR
    frac = [
        (0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0),
        (0.25, 0.25, 0.25), (0.25, 0.75, 0.75),
        (0.75, 0.25, 0.75), (0.75, 0.75, 0.25),
    ]
    atoms = [vq.Atom(z, [fx * a, fy * a, fz * a]) for fx, fy, fz in frac]
    system = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


SYSTEMS = [
    ("LiH",        lambda: _rocksalt_conv(4.084, 3, 1)),
    ("MgO",        lambda: _rocksalt_conv(4.21,  12, 8)),
    ("NaCl",       lambda: _rocksalt_conv(5.640, 11, 17)),
    ("LiF",        lambda: _rocksalt_conv(4.020, 3, 9)),
    ("C-diamond",  lambda: _diamond_conv(3.567,  6)),
    ("Si-diamond", lambda: _diamond_conv(5.430, 14)),
]


def main() -> int:
    print(f"vibeqc {vq.__version__} — native gamma GDF sweep (gauge Steps 1+2)")
    print(f"  basis: STO-3G (conventional 8-atom cell for all systems)")
    print()
    print(f"{'system':<12s} {'E_total/cell (Ha)':>20s} {'iter':>6s} "
          f"{'converged':>10s} {'wall_s':>8s}  notes")
    print("-" * 80)
    failures = 0
    for name, builder in SYSTEMS:
        t0 = time.time()
        print(f"  [{name}] start ...", flush=True)
        try:
            system, basis = builder()
            opts = vq.PeriodicRHFOptions()
            opts.use_diis = True
            opts.damping = 0.0
            opts.max_iter = 30
            opts.conv_tol_energy = 1e-8
            r = vq.run_rhf_periodic_gamma_gdf(
                system, basis, opts,
                aux_basis="def2-svp-jk",
                progress=False,
            )
            dt = time.time() - t0
            note = ""
            if not r.converged:
                note = "DID NOT CONVERGE"
                failures += 1
            print(f"{name:<12s} {r.energy:>20.8f} {r.n_iter:>6d} "
                  f"{str(r.converged):>10s} {dt:>8.1f}  {note}", flush=True)
        except Exception as exc:
            dt = time.time() - t0
            failures += 1
            print(f"{name:<12s} {'ERROR':>20s} {'-':>6s} {'-':>10s} {dt:>8.1f}  "
                  f"{type(exc).__name__}: {exc}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
