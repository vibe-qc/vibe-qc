"""H₂ in a 12-bohr vacuum cubic box — primary smoke test for the
periodic-Coulomb-method zoo.

Five comparison axes:

  1. methods × {EWALD3D, WOLF, PLAIN_EWALD, ADFT} on H₂ / sto-3g
  2. ADFT aux basis quality (sto-6g vs def2-svp-jk)
  3. EWALD3D FFT-grid convergence (PLAIN_EWALD as the analytic limit)
  4. WOLF α sensitivity (truncation parameter)
  5. ω-invariance for EWALD3D and PLAIN_EWALD

The molecular RHF total energy at the same geometry is the
"infinite-cell" limit reference; periodic-in-vacuum-box methods get an
extra Madelung-like leak from the FFT G=0 gauge.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from scf_driver import run_rhf_method  # noqa: E402

ANG2BOHR = 1.0 / 0.529177210903

A = 12.0
r_HH = 0.7414 * ANG2BOHR
center = np.array([A / 2, A / 2, A / 2])
unit_cell = [
    vq.Atom(1, list(center + np.array([-r_HH / 2, 0, 0]))),
    vq.Atom(1, list(center + np.array([+r_HH / 2, 0, 0]))),
]
sysp = vq.PeriodicSystem(3, np.diag([A, A, A]), unit_cell)
basis = vq.BasisSet(sysp.unit_cell_molecule(), "sto-3g")
mol = sysp.unit_cell_molecule()

print(f"H₂ in {A:.1f}-bohr cubic box / sto-3g  ({basis.nbasis} BFs/cell)")
print(f"  H-H bond = {r_HH:.4f} bohr ({r_HH/ANG2BOHR:.4f} Å)")
print()

# Molecular RHF reference (infinite-cell limit).
mol_res = vq.run_rhf(mol, basis)
print(f"  molecular RHF (∞-cell limit): E = {mol_res.energy:.8f} Ha\n")

# ---- (1) Method comparison at fixed parameters -----------------------------

print("=" * 92)
print("(1) Method comparison at fixed ω = 0.5, FFT spacing = 0.3 bohr")
print("=" * 92)

methods = ["EWALD3D", "WOLF", "PLAIN_EWALD", "ADFT"]
aux_basis_adft = vq.BasisSet(mol, "def2-svp-jk")
print(f"  ADFT aux basis: def2-svp-jk  (n_aux = {aux_basis_adft.nbasis})\n")

results = {}
for method in methods:
    kwargs = dict(
        omega=0.5, spacing_bohr=0.3, alpha_wolf=0.5, g_cutoff_factor=8.0,
        verbose=False, max_iter=40, damping=0.3, use_diis=True,
    )
    if method == "ADFT":
        kwargs["aux_basis"] = aux_basis_adft
    try:
        res = run_rhf_method(method, sysp, basis, **kwargs)
        results[method] = res
    except Exception as e:
        print(f"  {method}: FAILED — {type(e).__name__}: {e}")
        results[method] = None

ref = results["EWALD3D"]
print(f"{'method':<14} {'E_total':>14} {'ΔE vs EWALD3D':>16} "
      f"{'ΔE vs molecular':>18} {'iter':>6} {'wall_s':>9}")
print("-" * 92)
for m in methods:
    r = results[m]
    if r is None:
        continue
    de_ewald = r.energy - ref.energy
    de_mol = r.energy - mol_res.energy
    print(f"{m:<14} {r.energy:>14.8f} {de_ewald:>+16.3e} "
          f"{de_mol:>+18.3e} {r.n_iter:>6d} {r.wall_s:>9.2f}")
print()

# ---- (2) ADFT aux basis quality -------------------------------------------

print("=" * 92)
print("(2) ADFT aux basis quality on H₂ / sto-3g")
print("=" * 92)
adft_results = {}
for aux_name in ["sto-6g", "def2-svp-jk", "def2-tzvp-jk", "def2-qzvp-jk"]:
    try:
        aux_basis = vq.BasisSet(mol, aux_name)
        res = run_rhf_method(
            "ADFT", sysp, basis, aux_basis=aux_basis,
            verbose=False, max_iter=40, damping=0.3, use_diis=True,
        )
        adft_results[aux_name] = res
        print(f"  aux={aux_name:<16} n_aux={aux_basis.nbasis:>4d}  "
              f"E = {res.energy:.8f}  Δ vs EWALD3D = {res.energy-ref.energy:+.3e}  "
              f"Δ vs mol = {res.energy-mol_res.energy:+.3e}")
    except Exception as e:
        print(f"  aux={aux_name:<16} FAILED: {type(e).__name__}: {e}")
print()

# ---- (3) EWALD3D FFT-grid convergence vs PLAIN_EWALD analytic limit -------

print("=" * 92)
print("(3) EWALD3D FFT-grid convergence vs PLAIN_EWALD (analytic limit)")
print("=" * 92)
plain_res = results["PLAIN_EWALD"]
print(f"  PLAIN_EWALD reference E = {plain_res.energy:.10f}\n")
for spacing in [0.6, 0.5, 0.4, 0.3, 0.2, 0.15]:
    try:
        res = run_rhf_method(
            "EWALD3D", sysp, basis, spacing_bohr=spacing,
            verbose=False, max_iter=40, damping=0.3, use_diis=True,
        )
        de = res.energy - plain_res.energy
        print(f"  spacing={spacing:.2f} bohr   E = {res.energy:.10f}   "
              f"Δ vs PLAIN_EWALD = {de:+.3e}  iter={res.n_iter:2d}  "
              f"wall={res.wall_s:.2f}s")
    except Exception as e:
        print(f"  spacing={spacing:.2f} FAILED: {type(e).__name__}: {e}")
print()

# ---- (4) WOLF α sensitivity ------------------------------------------------

print("=" * 92)
print("(4) WOLF α sensitivity (truncation)")
print("=" * 92)
wolf_results = {}
for alpha in [0.1, 0.3, 0.5, 0.8, 1.2, 2.0]:
    try:
        res = run_rhf_method(
            "WOLF", sysp, basis, alpha_wolf=alpha,
            verbose=False, max_iter=40, damping=0.3, use_diis=True,
        )
        wolf_results[alpha] = res
        de = res.energy - ref.energy
        print(f"  α={alpha:.2f}   E = {res.energy:.8f}   "
              f"Δ vs EWALD3D = {de:+.3e}   "
              f"Δ vs mol = {res.energy-mol_res.energy:+.3e}   iter={res.n_iter:2d}")
    except Exception as e:
        print(f"  α={alpha:.2f} FAILED: {type(e).__name__}: {e}")
print()

# ---- (5) ω-invariance for EWALD3D and PLAIN_EWALD --------------------------

print("=" * 92)
print("(5) ω-invariance for EWALD3D and PLAIN_EWALD")
print("=" * 92)
omega_results = {"EWALD3D": {}, "PLAIN_EWALD": {}}
for omega in [0.3, 0.5, 0.8, 1.2]:
    for m in ["EWALD3D", "PLAIN_EWALD"]:
        try:
            res = run_rhf_method(
                m, sysp, basis, omega=omega, spacing_bohr=0.25,
                verbose=False, max_iter=40, damping=0.3, use_diis=True,
            )
            omega_results[m][omega] = res
        except Exception as e:
            print(f"  {m} ω={omega:.2f} FAILED: {type(e).__name__}: {e}")
            omega_results[m][omega] = None

print(f"  {'ω':>6}   {'EWALD3D E':>16}   {'PLAIN_EWALD E':>16}   {'ΔE':>14}")
for omega in [0.3, 0.5, 0.8, 1.2]:
    e_ewald = omega_results["EWALD3D"].get(omega)
    e_plain = omega_results["PLAIN_EWALD"].get(omega)
    if e_ewald is None or e_plain is None:
        continue
    print(f"  {omega:>6.2f}   {e_ewald.energy:>16.8f}   "
          f"{e_plain.energy:>16.8f}   "
          f"{e_plain.energy-e_ewald.energy:>+14.3e}")
print()

# ---- Persist ----

out = HERE / "results" / "bench_h2.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as fh:
    json.dump({
        "system": "H2 in 12-bohr cubic box",
        "basis": "sto-3g",
        "molecular_rhf_ref": mol_res.energy,
        "main_results": {
            m: ({"energy": results[m].energy, "n_iter": results[m].n_iter,
                 "wall_s": results[m].wall_s, "converged": results[m].converged}
                if results[m] is not None else None)
            for m in methods
        },
        "adft_aux_sweep": {
            k: {"energy": v.energy, "n_aux": vq.BasisSet(mol, k).nbasis,
                "wall_s": v.wall_s}
            for k, v in adft_results.items()
        },
        "wolf_alpha_sweep": {
            f"{k:.2f}": {"energy": v.energy, "wall_s": v.wall_s}
            for k, v in wolf_results.items()
        },
        "omega_sweep": {
            m: {f"{k:.2f}": (v.energy if v else None)
                for k, v in d.items()}
            for m, d in omega_results.items()
        },
    }, fh, indent=2)
print(f"Saved {out.relative_to(HERE.parent.parent.parent)}")
