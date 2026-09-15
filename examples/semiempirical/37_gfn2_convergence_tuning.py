#!/usr/bin/env python3
"""GFN2-xTB SCC convergence tuning — mixers, damping, electronic temperature.

Mirrors the xtb workshop convergence troubleshooting. Demonstrates:
  1. Simple damping vs DIIS vs Broyden mixer comparison
  2. Charge-mixing fraction effects
  3. Electronic temperature for metallic/near-degenerate systems
  4. AES faithfulness flag
  5. Convergence diagnostics and timing

Run:
    .venv/bin/python examples/semiempirical/37_gfn2_convergence_tuning.py
"""

from __future__ import annotations

import time
import numpy as np
from vibeqc import Molecule, Atom
from vibeqc._vibeqc_core.semiempirical import xtb as _xtb
from vibeqc._vibeqc_core import semiempirical as _se
from vibeqc.semiempirical.methods.gfn2_params import load_gfn2_params

params = load_gfn2_params()


def run_with_mixer(mol, mixer_name, mixer_enum, charge_mixing=0.1,
                   max_iter=500, mixer_memory=8, mixer_damping=0.0,
                   etemp=0.0, auto_stabilize=False, aes_faithful=False):
    """Run GFN2-xTB with specified SCC options, return (energy, n_iter, converged, dt)."""
    opts = _xtb.XTBSccOptions()
    opts.scc_mixer = mixer_enum
    opts.charge_mixing = charge_mixing
    opts.max_iter = max_iter
    opts.mixer_memory = mixer_memory
    opts.mixer_damping = mixer_damping
    opts.electronic_temperature = etemp
    opts.auto_stabilize = auto_stabilize
    opts.aes_faithful = aes_faithful

    t0 = time.perf_counter()
    result = _xtb.run_gfn2_xtb(mol, params, opts)
    dt = time.perf_counter() - t0
    return result.energy, result.n_iter, result.converged, dt


# ── Test molecules ──────────────────────────────────────────────────────
print("=" * 72)
print("GFN2-xTB Convergence Tuning")

# Water (well-behaved)
mol_h2o = Molecule([
    Atom(8, [ 0.00,  0.00,  0.00]),
    Atom(1, [ 1.43,  0.98,  0.00]),
    Atom(1, [-1.43,  0.98,  0.00]),
])

# CO2 (linear, polar bonds)
mol_co2 = Molecule([
    Atom(8, [-2.20, 0.0, 0.0]),
    Atom(6, [ 0.00, 0.0, 0.0]),
    Atom(8, [ 2.20, 0.0, 0.0]),
])

# ── 1. Mixer comparison ─────────────────────────────────────────────────
print("\n" + "-" * 48)
print("1. Mixer comparison (water)")
print(f"   {'Mixer':>12s}  {'Energy':>14s}  {'n_iter':>6s}  {'conv':>5s}  {'time_ms':>8s}")

for name, enum in [("Simple", _se.SCCMixer.Simple),
                    ("DIIS", _se.SCCMixer.DIIS),
                    ("Broyden", _se.SCCMixer.Broyden)]:
    e, n, c, dt = run_with_mixer(mol_h2o, name, enum, max_iter=500)
    print(f"   {name:>12s}  {e:14.8f}  {n:6d}  {str(c):>5s}  {dt*1000:8.1f}")

# ── 2. Charge-mixing sweep ──────────────────────────────────────────────
print("\n" + "-" * 48)
print("2. Charge mixing fraction sweep (water, Simple mixer)")
print(f"   {'mixing':>8s}  {'Energy':>14s}  {'n_iter':>6s}  {'conv':>5s}")

for cm in [0.05, 0.10, 0.20, 0.30, 0.50]:
    e, n, c, dt = run_with_mixer(mol_h2o, f"cm={cm}", _se.SCCMixer.Simple,
                                  charge_mixing=cm, max_iter=500)
    print(f"   {cm:8.2f}  {e:14.8f}  {n:6d}  {str(c):>5s}")

# ── 3. DIIS memory sweep ────────────────────────────────────────────────
print("\n" + "-" * 48)
print("3. DIIS subspace size sweep (water)")
print(f"   {'memory':>8s}  {'Energy':>14s}  {'n_iter':>6s}  {'conv':>5s}")

for mem in [4, 6, 8, 12, 16]:
    e, n, c, dt = run_with_mixer(mol_h2o, f"DIIS m={mem}", _se.SCCMixer.DIIS,
                                  mixer_memory=mem, max_iter=500)
    print(f"   {mem:8d}  {e:14.8f}  {n:6d}  {str(c):>5s}")

# ── 4. Broyden damping sweep ────────────────────────────────────────────
print("\n" + "-" * 48)
print("4. Broyden damping sweep (CO₂, polar bonds)")
print(f"   {'damping':>8s}  {'Energy':>14s}  {'n_iter':>6s}  {'conv':>5s}")

for damp in [0.0, 0.1, 0.2, 0.3, 0.5]:
    e, n, c, dt = run_with_mixer(mol_co2, f"Broyden d={damp}", _se.SCCMixer.Broyden,
                                  mixer_damping=damp, charge_mixing=0.15,
                                  max_iter=500)
    print(f"   {damp:8.2f}  {e:14.8f}  {n:6d}  {str(c):>5s}")

# ── 5. Electronic temperature ───────────────────────────────────────────
print("\n" + "-" * 48)
print("5. Electronic temperature (CO₂)")
print("   (Useful for metals / near-degenerate HOMO-LUMO)")
print(f"   {'T_el (K)':>10s}  {'Energy':>14s}  {'n_iter':>6s}  {'conv':>5s}")

for etemp_ha in [0.0, 0.0001, 0.0005, 0.001, 0.002]:
    t_k = etemp_ha / 3.1668114e-6 if etemp_ha > 0 else 0.0
    e, n, c, dt = run_with_mixer(mol_co2, f"T={t_k:.0f}K", _se.SCCMixer.Broyden,
                                  etemp=etemp_ha, max_iter=500)
    print(f"   {t_k:10.0f}  {e:14.8f}  {n:6d}  {str(c):>5s}")

# ── 6. AES faithfulness ─────────────────────────────────────────────────
print("\n" + "-" * 48)
print("6. AES dipole vs dipole+quadrupole (water)")

for aes_label, aes_val in [("dipole only", False), ("dipole+quad", True)]:
    e, n, c, dt = run_with_mixer(mol_h2o, aes_label, _se.SCCMixer.Simple,
                                  aes_faithful=aes_val, max_iter=500)
    print(f"   {aes_label:>16s}: E = {e:14.8f} Ha  n_iter={n}")

# ── 7. Auto-stabilization ───────────────────────────────────────────────
print("\n" + "-" * 48)
print("7. Auto-stabilization (water, tight budget)")
print("   (Primary solve with 20 iters cap; auto-stabilize retries with DIIS)")

for auto in [False, True]:
    e, n, c, dt = run_with_mixer(mol_h2o, f"auto={auto}", _se.SCCMixer.Broyden,
                                  max_iter=20, auto_stabilize=auto)
    print(f"   auto_stabilize={auto!s:>5s}: E = {e:14.8f}  n_iter={n:3d}  "
          f"conv={c}  time={dt*1000:.1f} ms")

# ── 8. Water dimer (intermolecular) ─────────────────────────────────────
print("\n" + "-" * 48)
print("8. Water dimer — all mixers")
# Approximate water dimer geometry
mol_dimer = Molecule([
    Atom(8, [ 0.00,  0.00,  0.00]),
    Atom(1, [ 1.43,  0.98,  0.00]),
    Atom(1, [-1.43,  0.98,  0.00]),
    Atom(8, [ 5.50,  0.00,  0.00]),
    Atom(1, [ 5.50,  1.43,  0.98]),
    Atom(1, [ 5.50, -1.43,  0.98]),
])

for name, enum in [("Simple", _se.SCCMixer.Simple),
                    ("DIIS", _se.SCCMixer.DIIS),
                    ("Broyden", _se.SCCMixer.Broyden)]:
    e, n, c, dt = run_with_mixer(mol_dimer, name, enum, max_iter=1000)
    print(f"   {name:>12s}: E = {e:14.8f}  n_iter={n:4d}  conv={c}  "
          f"time={dt*1000:.1f} ms")

# ── Summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 72)
print("Recommendations:")
print("  • Closed-shell organics:  Simple, cm=0.1 (default)")
print("  • Polar molecules:        DIIS, cm=0.15, memory=6–8")
print("  • Periodic / metals:      Broyden, cm=0.15, damping=0.2")
print("  • Near-degenerate:        add electronic_temperature=0.001 Ha")
print("  • Tough convergence:      auto_stabilize=True + generous max_iter")
