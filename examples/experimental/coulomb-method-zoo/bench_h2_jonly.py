"""H₂ in a vacuum box — J-builder accuracy comparison.

The core question of this spike is: how do alternative ways of building
the Coulomb J matrix compare in **accuracy**? The SCF cycle adds gauge
bookkeeping (EWALD/DIRECT_TRUNCATED V_ne pairing, Madelung leak fixes,
Bloch sums, …) that is method-incidental. Strip it away: take the
converged molecular density D for H₂/sto-3g, build J via every method,
compare against the direct molecular ERI ``compute_eri``-contracted
``J_full``.

This is exactly the "J-build math" that the spike was asked to compare,
without the periodic-SCF complications.

Five reports:

  (1) main comparison table at default params
  (2) ADFT aux basis sweep (size vs accuracy)
  (3) EWALD3D FFT-grid sweep (vs PLAIN_EWALD analytic limit)
  (4) WOLF α sensitivity
  (5) ω-invariance for EWALD3D and PLAIN_EWALD

The molecular density D is computed with vibe-qc's ``run_rhf`` first.
Every J-build is timed; we report wall_s and a Frobenius error
‖J - J_full‖_F.
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
from j_builders import (  # noqa: E402
    ADFTContext, build_j_adft, build_j_ewald3d, build_j_plain_ewald,
    build_j_wolf,
)

ANG2BOHR = 1.0 / 0.529177210903

# H₂ in 12-bohr vacuum cubic box, sto-3g.
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

print(f"H₂ in {A:.1f}-bohr cubic box / sto-3g  (n_bf = {basis.nbasis}/cell)")
print(f"  H-H bond = {r_HH:.4f} bohr")

# Converged molecular D from vibe-qc's molecular RHF.
mol_res = vq.run_rhf(mol, basis)
D = np.array(mol_res.density)
print(f"  molecular RHF E = {mol_res.energy:.8f} Ha   D shape = {D.shape}")
print()

# Reference J via direct 4-center ERI (the unambiguous molecular limit).
eri = vq.compute_eri(basis)
J_full = np.einsum("ijkl,kl->ij", eri, D, optimize=True)
e_J_full = 0.5 * np.einsum("ij,ij->", D, J_full, optimize=True)
print(f"reference  J_full[0,:] = {J_full[0]}")
print(f"           ½Tr(D·J_full) = {e_J_full:.8f} Ha")
print()


def report(name: str, J: np.ndarray, t: float):
    err = J - J_full
    fro = float(np.linalg.norm(err))
    e_J = 0.5 * np.einsum("ij,ij->", D, J, optimize=True)
    de_J = e_J - e_J_full
    print(f"  {name:<22} ‖ΔJ‖_F = {fro:.3e}   "
          f"½Tr(D·J) = {float(e_J):.6f}   ΔE_J = {float(de_J):+.3e}   "
          f"wall = {t:.3f} s")
    return {"fro": fro, "e_J": float(e_J), "de_J": float(de_J), "wall_s": t}


# Lattice opts (EWALD_3D for the gauge-aligned methods) — these are
# the V_ne and short-range opts. We're not running SCF here so this
# only affects build_jk_gamma_molecular_limit's lattice sums.
lat_opts_ewald = vq.LatticeSumOptions()
lat_opts_ewald.coulomb_method = vq.CoulombMethod.EWALD_3D
lat_opts_ewald.cutoff_bohr = 12.0
lat_opts_ewald.nuclear_cutoff_bohr = 25.0

lat_opts_direct = vq.LatticeSumOptions()
lat_opts_direct.coulomb_method = vq.CoulombMethod.DIRECT_TRUNCATED
lat_opts_direct.cutoff_bohr = 12.0
lat_opts_direct.nuclear_cutoff_bohr = 25.0

records = {}


# ---- (1) Main comparison ---------------------------------------------------

print("=" * 92)
print("(1) Main J-builder comparison vs J_full at converged molecular D")
print("=" * 92)

t = time.perf_counter()
J_e3d = build_j_ewald3d(basis, sysp, D, lattice_opts=lat_opts_ewald,
                        omega=0.5, spacing_bohr=0.3)
records["EWALD3D"] = report("EWALD3D (ω=0.5, h=0.3)", J_e3d, time.perf_counter()-t)

t = time.perf_counter()
J_pe = build_j_plain_ewald(basis, sysp, D, lattice_opts=lat_opts_ewald,
                           omega=0.5, g_cutoff_factor=8.0)
records["PLAIN_EWALD"] = report("PLAIN_EWALD (ω=0.5, G/2ω≤8)", J_pe,
                                time.perf_counter()-t)

t = time.perf_counter()
J_wolf = build_j_wolf(basis, sysp, D, lattice_opts=lat_opts_direct,
                      alpha_wolf=0.5)
records["WOLF"] = report("WOLF (α=0.5)", J_wolf, time.perf_counter()-t)

# ADFT requires an aux basis. Use 6-31g (small enough to load, max_l=0).
aux = vq.BasisSet(mol, "6-31g")
t = time.perf_counter()
ctx = ADFTContext.build(basis, aux)
J_adft = build_j_adft(basis, sysp, D, ctx=ctx)
records["ADFT_631g"] = report(f"ADFT (aux=6-31g, n_aux={aux.nbasis})", J_adft,
                              time.perf_counter()-t)
print()


# ---- (2) ADFT aux basis sweep ---------------------------------------------

print("=" * 92)
print("(2) ADFT aux basis sweep — fit quality vs aux size")
print("=" * 92)

# NOTE: vibe-qc's vendored libint segfaults during compute_2c_eri on
# any aux basis containing d-shells (l ≥ 2) — the build was configured
# for an AM ceiling that JK-fit auxes routinely exceed. We detect that
# at the Python level and skip rather than catch the SIGSEGV (which we
# can't from Python). See feedback_no_homebrew_libint and the ADFT
# section of POSTMORTEM.md for follow-up.
LIBINT_2C_MAX_L = 0   # empirically: only s-shell aux works in this build;
                      # cc-pvdz (max_l=1) already segfaults compute_2c_eri.

aux_records = {}
for aux_name in ["sto-3g", "sto-6g", "6-31g", "cc-pvdz", "cc-pvtz", "def2-svp",
                 "def2-svp-jk", "def2-tzvp-jk"]:
    try:
        aux = vq.BasisSet(mol, aux_name)
        max_l = max(sh.l for sh in aux.shells())
    except Exception as e:
        print(f"  aux={aux_name:<14s} load failed: {type(e).__name__}: {e}")
        aux_records[aux_name] = None
        continue
    if max_l > LIBINT_2C_MAX_L:
        print(f"  aux={aux_name:<14s} n_aux={aux.nbasis:>3d}  "
              f"SKIPPED: max_l={max_l} exceeds libint 2c-2e ceiling "
              f"(={LIBINT_2C_MAX_L})")
        aux_records[aux_name] = {"skipped": True, "n_aux": aux.nbasis,
                                 "max_l": max_l}
        continue
    t = time.perf_counter()
    ctx = ADFTContext.build(basis, aux)
    J = build_j_adft(basis, sysp, D, ctx=ctx)
    rec = report(f"aux={aux_name:<14s} n_aux={aux.nbasis:>3d}", J,
                 time.perf_counter()-t)
    aux_records[aux_name] = {**rec, "n_aux": aux.nbasis, "max_l": max_l}
records["adft_aux_sweep"] = aux_records
print()


# ---- (3) EWALD3D FFT-grid convergence vs PLAIN_EWALD analytic limit -------

print("=" * 92)
print("(3) EWALD3D FFT-grid convergence vs PLAIN_EWALD (analytic G-vector sum)")
print("=" * 92)

# Reference: PLAIN_EWALD at very tight G-cutoff
J_pe_tight = build_j_plain_ewald(basis, sysp, D, lattice_opts=lat_opts_ewald,
                                 omega=0.5, g_cutoff_factor=12.0)
err_pe = float(np.linalg.norm(J_pe_tight - J_full))
print(f"  PLAIN_EWALD (G/2ω≤12, very tight): ‖ΔJ‖_F = {err_pe:.3e}")
print()

grid_records = {}
for spacing in [0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15]:
    t = time.perf_counter()
    J = build_j_ewald3d(basis, sysp, D, lattice_opts=lat_opts_ewald,
                        omega=0.5, spacing_bohr=spacing)
    err_full = float(np.linalg.norm(J - J_full))
    err_pe = float(np.linalg.norm(J - J_pe_tight))
    e_J = float(0.5 * np.einsum("ij,ij->", D, J))
    print(f"  h={spacing:.2f} bohr   ‖J-J_full‖_F = {err_full:.3e}   "
          f"‖J-J_pe‖_F = {err_pe:.3e}   ½Tr(D·J) = {e_J:.6f}   "
          f"wall = {time.perf_counter()-t:.3f}s")
    grid_records[f"{spacing:.2f}"] = {
        "err_full": err_full, "err_pe": err_pe, "e_J": e_J,
        "wall_s": time.perf_counter()-t,
    }
records["grid_sweep"] = grid_records
print()


# ---- (4) WOLF α sensitivity ------------------------------------------------

print("=" * 92)
print("(4) WOLF α sensitivity (smaller α → harder cutoff, more error)")
print("=" * 92)

wolf_records = {}
for alpha in [0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0]:
    t = time.perf_counter()
    J = build_j_wolf(basis, sysp, D, lattice_opts=lat_opts_direct,
                     alpha_wolf=alpha)
    err_full = float(np.linalg.norm(J - J_full))
    e_J = float(0.5 * np.einsum("ij,ij->", D, J))
    de_J = e_J - e_J_full
    print(f"  α={alpha:>4.1f}   ‖ΔJ‖_F = {err_full:.3e}   "
          f"½Tr(D·J) = {e_J:.6f}   ΔE_J = {de_J:+.3e}   "
          f"wall = {time.perf_counter()-t:.3f}s")
    wolf_records[f"{alpha:.1f}"] = {
        "err_full": err_full, "e_J": e_J, "de_J": de_J,
        "wall_s": time.perf_counter()-t,
    }
records["wolf_sweep"] = wolf_records
print()


# ---- (5) ω-invariance ------------------------------------------------------

print("=" * 92)
print("(5) ω-invariance: J should be independent of the splitting parameter")
print("=" * 92)

omega_records = {"EWALD3D": {}, "PLAIN_EWALD": {}}
for omega in [0.2, 0.3, 0.5, 0.8, 1.2, 2.0]:
    J_e = build_j_ewald3d(basis, sysp, D, lattice_opts=lat_opts_ewald,
                          omega=omega, spacing_bohr=0.2)
    J_p = build_j_plain_ewald(basis, sysp, D, lattice_opts=lat_opts_ewald,
                              omega=omega, g_cutoff_factor=10.0)
    e_J_e = float(0.5 * np.einsum("ij,ij->", D, J_e))
    e_J_p = float(0.5 * np.einsum("ij,ij->", D, J_p))
    err_e_full = float(np.linalg.norm(J_e - J_full))
    err_p_full = float(np.linalg.norm(J_p - J_full))
    err_pe_e3d = float(np.linalg.norm(J_e - J_p))
    print(f"  ω={omega:.2f}   ½Tr(D·J_E3D)  = {e_J_e:.6f}   "
          f"½Tr(D·J_PE) = {e_J_p:.6f}   "
          f"‖J_PE-J_E3D‖_F = {err_pe_e3d:.3e}")
    omega_records["EWALD3D"][f"{omega:.2f}"] = {
        "e_J": e_J_e, "err_full": err_e_full,
    }
    omega_records["PLAIN_EWALD"][f"{omega:.2f}"] = {
        "e_J": e_J_p, "err_full": err_p_full,
    }
records["omega_sweep"] = omega_records
print()


# ---- Persist ---------------------------------------------------------------

records["meta"] = {
    "system": "H2 in 12-bohr cubic box",
    "basis": "sto-3g",
    "n_bf": basis.nbasis,
    "molecular_RHF_E": float(mol_res.energy),
    "J_full_e_J": float(e_J_full),
    "J_full_diag": [float(x) for x in np.diag(J_full)],
}
out = HERE / "results" / "bench_h2_jonly.json"
out.parent.mkdir(parents=True, exist_ok=True)
with open(out, "w") as fh:
    json.dump(records, fh, indent=2)
print(f"Saved {out.relative_to(HERE.parent.parent.parent)}")
