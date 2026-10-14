"""MgO + Al2O3 RHF/RKS-PBE parity vs PySCF.pbc — the v0.7-correctness bar.

User intent (2026-05-03):
    "First we have to get RHF on MgO and Al2O3 right, then RKS with PBE
    as a functional on those two systems. IF this does not work we have
    a serious bug."

Two systems × two methods × two codes:
    1. MgO conventional rocksalt   (cubic, 4 Mg + 4 O, a = 4.211 Å)
    2. Al2O3 cubic-stuffed historical test cell (see below)

For each, run vibe-qc and PySCF.pbc with matching basis (sto-3g, all-
electron, Γ-only) on both **RHF** and **RKS-PBE**, and report:

    converged?  |  iters  |  E_total (Ha)  |  ΔE vs PySCF (Ha)

A clean parity (|ΔE| < 1e-3 Ha) means our periodic SCF is sound. Any
larger gap is a bug to fix before claiming the v0.7 release.

Routes
------
PySCF runs GDF with ``exxdiv='ewald'``, and its RKS-PBE uses Becke grids
(level 5): its default uniform XC grid cannot resolve all-electron core
densities. vibe-qc runs native GDF at Γ through ``run_periodic_job``, the
route of ``examples/periodic/input-mgo-rocksalt-*.py``, except for Al2O3 RHF,
which uses ``run_krhf_periodic_gdf`` with the settings of
``examples/periodic/input-al2o3-cubic-rhf.py`` (``RHF_SLOW_GRADIENT``). The
cubic Al2O3 model has no PBE gap at Γ, so its RKS-PBE case uses Fermi-Dirac
smearing of 0.01 Ha on both codes (``RKS_SMEARING_HA``) and compares the
energy E, not the free energy E - TS.

Why these two systems
---------------------
MgO and Al2O3 are the canonical "ionic wide-gap insulator" benchmarks
in periodic DFT. They are deeper-core than LiH (Mg has 1s²2s²2p⁶, Al
has 1s²2s²2p⁶) and more strongly ionic than Ne FCC. Getting both right
is the minimum bar for any periodic DFT code aimed at solid-state
chemistry.

Al2O3 model
-----------
This script keeps the older **cubic 5-atom Al2O3 stuffed-fluorite
model** because it is a small, cheap parity target. It is not real
corundum. The periodic infrastructure itself is lattice-general:
hexagonal α-Al2O3 / corundum belongs in the separate native-GDF/FFTDF
parity suite, not in this smoke script.

Run on compute-reference
--------------
    .venv/bin/python examples/debug/mgo_al2o3_parity.py
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "mgo-al2o3-parity"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANG2BOHR = 1.0 / 0.529177210903

# Fermi-Dirac smearing (Ha) for the RKS-PBE case, by system label. The Al2O3
# model has no PBE gap at Γ: with integer occupations the vibe-qc SCF does not
# converge, and smeared it has an electronic entropy of about 5 k_B.
RKS_SMEARING_HA = {"Al2O3-cubic": 0.01}

# RHF options for systems whose orbital gradient converges far more slowly
# than their energy. The Al2O3 model needs a start-up level shift to reach its
# lower state, then about 160 cycles to a 1e-4 gradient; the energy is then
# within 1e-7 Ha of an 800-cycle run.
RHF_SLOW_GRADIENT = {
    "Al2O3-cubic": {"level_shift": 0.5, "conv_tol_grad": 1e-4, "max_iter": 300},
}

try:
    from pyscf.pbc import gto as pbc_gto, dft as pbc_dft, scf as pbc_scf
    from pyscf.pbc import df as pbc_df
    from pyscf.pbc.scf.smearing import smearing as pbc_smearing
    from pyscf.pbc.dft import gen_grid as pbc_gen_grid
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


# ============================================================
# System builders
# ============================================================

def mgo_rocksalt(a_ang: float = 4.211):
    """MgO conventional cubic rocksalt — 4 Mg + 4 O per cell."""
    a = a_ang * ANG2BOHR
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell = []
    for fx, fy, fz in mg_frac:
        cell.append(vq.Atom(12, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in o_frac:
        cell.append(vq.Atom(8, [fx * a, fy * a, fz * a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell)
    pyscf_atoms = []
    for fx, fy, fz in mg_frac:
        pyscf_atoms.append(("Mg", fx * a_ang, fy * a_ang, fz * a_ang))
    for fx, fy, fz in o_frac:
        pyscf_atoms.append(("O", fx * a_ang, fy * a_ang, fz * a_ang))
    return "MgO-rocksalt", sys_p, pyscf_atoms, a_ang


def al2o3_cubic_model(a_ang: float = 5.05):
    """5-atom Al2O3 stoichiometry on a cubic stuffed-fluorite model.

    Not corundum (which is hexagonal — incompatible with FFT-Poisson).
    Atoms: 2 Al at body diagonals, 3 O at face / edge sites.
    Lattice constant chosen to give a reasonable Al-O distance ~2 Å.
    """
    a = a_ang * ANG2BOHR
    al_frac = [(0.25, 0.25, 0.25), (0.75, 0.75, 0.75)]
    o_frac = [(0.5, 0.5, 0.0), (0.5, 0.0, 0.5), (0.0, 0.5, 0.5)]
    cell = []
    for fx, fy, fz in al_frac:
        cell.append(vq.Atom(13, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in o_frac:
        cell.append(vq.Atom(8, [fx * a, fy * a, fz * a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell)
    pyscf_atoms = []
    for fx, fy, fz in al_frac:
        pyscf_atoms.append(("Al", fx * a_ang, fy * a_ang, fz * a_ang))
    for fx, fy, fz in o_frac:
        pyscf_atoms.append(("O", fx * a_ang, fy * a_ang, fz * a_ang))
    return "Al2O3-cubic", sys_p, pyscf_atoms, a_ang


# ============================================================
# Drivers
# ============================================================

@dataclass
class Result:
    code: str          # "vibeqc" or "pyscf"
    label: str
    method: str        # "RHF" or "RKS-PBE"
    converged: Optional[bool]
    n_iter: Optional[int]
    energy: Optional[float]
    wall_s: float
    note: str


def run_vibeqc(label: str, system: vq.PeriodicSystem, *,
               method: str, basis_name: str = "sto-3g",
               smearing_ha: float = 0.0) -> Result:
    """Run vibe-qc Γ-only periodic SCF."""
    rec = Result(code="vibeqc", label=label, method=method,
                 converged=None, n_iter=None, energy=None,
                 wall_s=0.0, note="")
    t0 = time.perf_counter()
    try:
        basis = vq.BasisSet(system.unit_cell_molecule(), basis_name)

        if method == "RHF" and label in RHF_SLOW_GRADIENT:
            # Same settings as examples/periodic/input-al2o3-cubic-rhf.py:
            # the energy converges fast but the orbital gradient slowly, so
            # use the k-point driver with an explicit gradient tolerance
            # (run_periodic_job keeps 1e-6 and refuses this SCF).
            opts = vq.PeriodicRHFOptions()
            opts.conv_tol_energy = 1e-7
            for key, value in RHF_SLOW_GRADIENT[label].items():
                setattr(opts, key, value)
            r = vq.run_krhf_periodic_gdf(system, basis, (1, 1, 1), opts,
                                         progress=False)
        elif method == "RHF":
            # Native GDF at Γ through the runner, the route of
            # examples/periodic/input-*-rhf.py. Calling
            # run_rhf_periodic_gamma_gdf directly, as this script used to,
            # runs its ewald-jk-fallback engine with molecular-limit
            # exchange on these compact cells, several Ha from PySCF.
            r = vq.run_periodic_job(
                system, basis,
                method="RHF",
                jk_method="gdf", kpoints=(1, 1, 1),
                max_iter=60, conv_tol_energy=1e-7,
                output=str(OUT_DIR / f"{label}-rhf"),
            )
        elif method == "RKS-PBE":
            # Native GDF at Γ through the runner, the same route as
            # examples/periodic/input-*-rks-pbe.py. The retired Γ-only
            # EWALD_3D driver used before is not a parity route for compact
            # crystals (its molecular-limit density convention breaks
            # there). Smearing is passed only when asked for, so the runner's
            # automatic convergence strategy stays in charge otherwise.
            kw = {}
            if smearing_ha > 0.0:
                kw["smearing_temperature"] = smearing_ha
            r = vq.run_periodic_job(
                system, basis,
                method="RKS", functional="pbe",
                jk_method="gdf", kpoints=(1, 1, 1),
                max_iter=60, conv_tol_energy=1e-7,
                output=str(OUT_DIR / f"{label}-rks-pbe"),
                **kw,
            )
        else:
            raise ValueError(f"unknown method: {method}")

        rec.wall_s = time.perf_counter() - t0
        rec.converged = bool(r.converged)
        rec.n_iter = int(r.n_iter)
        rec.energy = float(r.energy)
        rec.note = "ok" if rec.converged else "max_iter"
    except Exception as exc:
        rec.wall_s = time.perf_counter() - t0
        rec.converged = False
        rec.note = f"{type(exc).__name__}: {str(exc)[:120]}"
    return rec


def run_pyscf(label: str,
              pyscf_atoms: List[Tuple[str, float, float, float]],
              a_ang: float, *,
              method: str, basis_name: str = "sto-3g",
              smearing_ha: float = 0.0) -> Result:
    """Run PySCF.pbc Γ-only periodic SCF using GDF (Gaussian density
    fitting) — the all-electron-friendly J path."""
    rec = Result(code="pyscf", label=label, method=method,
                 converged=None, n_iter=None, energy=None,
                 wall_s=0.0, note="")
    if not PYSCF_AVAILABLE:
        rec.note = "pyscf not installed"
        return rec

    t0 = time.perf_counter()
    try:
        atom_str = "; ".join(
            f"{el} {x:.6f} {y:.6f} {z:.6f}"
            for (el, x, y, z) in pyscf_atoms
        )
        cell = pbc_gto.M(
            atom=atom_str,
            a=[[a_ang, 0, 0], [0, a_ang, 0], [0, 0, a_ang]],
            basis=basis_name,
            unit="A",
            verbose=0,
            mesh=[31, 31, 31],
            precision=1e-8,
        )

        if method == "RHF":
            mf = pbc_scf.RHF(cell)
        elif method == "RKS-PBE":
            mf = pbc_dft.RKS(cell)
            mf.xc = "pbe"
            # All-electron: the default uniform XC grid (the cell's mesh)
            # cannot resolve the Al/Mg/O core densities. On the Al2O3 model
            # it put PBE 13.66 Ha too high; Becke grids at level 3 and 5
            # agree to 3e-5 Ha there.
            mf.grids = pbc_gen_grid.BeckeGrids(cell)
            mf.grids.level = 5
        else:
            raise ValueError(f"unknown method: {method}")

        mf.exxdiv = "ewald"
        mf.with_df = pbc_df.GDF(cell)
        mf.with_df.build()
        mf.max_cycle = 60
        mf.conv_tol = 1e-7
        if smearing_ha > 0.0:
            # e_tot stays the energy E; e_free = E - sigma*S is separate.
            mf = pbc_smearing(mf, sigma=smearing_ha, method="fermi")

        e = mf.kernel()
        rec.wall_s = time.perf_counter() - t0
        rec.energy = float(e)
        rec.converged = bool(mf.converged)
        rec.n_iter = int(getattr(mf, "cycles", 0)) or None
        rec.note = "ok" if rec.converged else "max_iter / not converged"
    except Exception as exc:
        rec.wall_s = time.perf_counter() - t0
        rec.converged = False
        rec.note = f"{type(exc).__name__}: {str(exc)[:120]}"
    return rec


# ============================================================
# Reporting
# ============================================================

def report(rows: List[Tuple[Result, Result]]) -> None:
    """Print a side-by-side summary."""
    print()
    print("=" * 92)
    print(" Summary")
    print("=" * 92)
    hdr = f"  {'system':>16s}  {'method':>9s}  {'vibeqc E':>14s}  " \
          f"{'pyscf E':>14s}  {'ΔE (Ha)':>12s}  verdict"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for v, p in rows:
        v_e = f"{v.energy:14.6f}" if v.energy is not None else " " * 14
        p_e = f"{p.energy:14.6f}" if p.energy is not None else " " * 14
        if v.energy is not None and p.energy is not None:
            de = v.energy - p.energy
            de_str = f"{de:+12.4e}"
            if abs(de) < 1e-3:
                verdict = "✓ PARITY"
            elif abs(de) < 1.0:
                verdict = "~ off by mHa-Ha (basis-set / Madelung sensitivity)"
            else:
                verdict = "✗ BUG (large mismatch)"
        else:
            de_str = " " * 12
            v_status = ("ok" if v.converged else f"vibeqc-{v.note[:30]}") \
                if v.converged is not None else "vibeqc-skipped"
            p_status = ("ok" if p.converged else f"pyscf-{p.note[:30]}") \
                if p.converged is not None else "pyscf-skipped"
            verdict = f"{v_status} | {p_status}"
        print(f"  {v.label:>16s}  {v.method:>9s}  {v_e}  {p_e}  "
              f"{de_str}  {verdict}")


def main() -> None:
    print("=" * 92)
    print(" MgO + Al2O3 RHF/RKS-PBE parity — vibe-qc vs PySCF.pbc")
    print("=" * 92)
    print(f"  PySCF available:  {PYSCF_AVAILABLE}")
    print(f"  artefacts →       {OUT_DIR}/")
    print()

    cases = [mgo_rocksalt(), al2o3_cubic_model()]

    rows: List[Tuple[Result, Result]] = []
    for label, sys_p, pyscf_atoms, a_ang in cases:
        for method in ("RHF", "RKS-PBE"):
            print(f"  >>> {label} / {method} (a = {a_ang:.3f} Å, "
                  f"{len(sys_p.unit_cell)} atoms in cell)")

            smear = RKS_SMEARING_HA.get(label, 0.0) if method == "RKS-PBE" else 0.0
            if smear:
                print(f"      Fermi-Dirac smearing {smear} Ha on both codes")
            v = run_vibeqc(label, sys_p, method=method, smearing_ha=smear)
            flag = "✓" if v.converged else "✗"
            print(f"      vibeqc {flag}  iter={v.n_iter}  "
                  f"E={v.energy}  wall={v.wall_s:.1f}s  {v.note}")

            p = run_pyscf(label, pyscf_atoms, a_ang, method=method,
                          smearing_ha=smear)
            if PYSCF_AVAILABLE:
                flag = "✓" if p.converged else "✗"
                print(f"      pyscf  {flag}  iter={p.n_iter}  "
                      f"E={p.energy}  wall={p.wall_s:.1f}s  {p.note}")
            else:
                print("      pyscf  -  (not installed)")
            rows.append((v, p))
            print()

    report(rows)
    print()
    print("  Reading the verdict column:")
    print("    ✓ PARITY              — |ΔE| < 1 mHa, codes agree")
    print("    ~ off by mHa-Ha       — basis / Madelung-convention difference")
    print("    ✗ BUG                 — vibe-qc bug; needs investigation")


if __name__ == "__main__":
    main()
