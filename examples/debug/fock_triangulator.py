"""Fock-component triangulator — vibe-qc vs PySCF.pbc on iter-1 SAD density.

Goal
----
Localise WHICH Fock component (V_ne, J, V_xc) drives the LiH +2.4 Ha
disagreement and the NaCl ~kHa Fock oscillation observed in the
v0.7.0 regression suite (compute-reference, 2026-05-03).

The bug fingerprint scales with (deep cores) × (cell-spanning charge
transfer):

    Ne  FCC                 ✓ converges to numerical precision
    LiH rocksalt            ⚠ converges cleanly to E_total +2.4 Ha
                              vs PySCF.pbc (sto-3g, EWALD_3D, LDA)
    NaCl rocksalt           ✗ Fock oscillates between -863 and
                              +5794 Ha across iter 19+

Severity rules out integrators (Ne is fine) and rules out SCF
machinery (LiH converges cleanly, just to the wrong value). The
component must produce a finite-but-shifted contribution at iter-1
SAD density on LiH and a divergent one on NaCl.

What this script does
---------------------
1. Build LiH (and NaCl) periodic system + sto-3g basis.
2. Both vibe-qc and PySCF.pbc independently compute their own
   iter-1 SAD density on the same lattice + basis.
3. Each code computes its own iter-1 Fock components on its own
   density:  T, V_ne, J, V_xc, K (with α=0 for LDA).
4. Report per-code component traces tr(D · M) and compare:
     E_kin   = tr(D · T)
     E_ne    = tr(D · V_ne)
     E_J     = ½ tr(D · J)
     E_xc    (libxc on Becke grid)
     E_nuc   (Ewald nuclear repulsion per cell)
     E_iter1 = E_kin + E_ne + E_J + E_xc + E_nuc
   Plus the basis-independent **eigenvalue spectrum** of S, T, V_ne
   and the iter-1 Fock — these are immune to AO ordering / sign
   conventions and tell us whether the operators are equal as
   linear maps, not just as matrix elements.

Output: a wide ASCII table per system; the row with the largest |Δ|
between vibe-qc and PySCF is the offending Fock component.

Run on compute-reference
--------------
    .venv/bin/python examples/debug/fock_triangulator.py
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import vibeqc as vq

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "output" / "fock-triangulator"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANG2BOHR = 1.0 / 0.529177210903

# --- PySCF availability -----------------------------------------------
try:
    from pyscf.pbc import gto as pbc_gto, dft as pbc_dft, scf as pbc_scf
    from pyscf import dft as mol_dft
    PYSCF_AVAILABLE = True
except ImportError:
    PYSCF_AVAILABLE = False


# ============================================================
# System builders
# ============================================================

def lih_rocksalt(a_ang: float = 4.084):
    """LiH conventional rocksalt, 4 LiH per cell."""
    a = a_ang * ANG2BOHR
    li_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    h_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell = []
    for fx, fy, fz in li_frac:
        cell.append(vq.Atom(3, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in h_frac:
        cell.append(vq.Atom(1, [fx * a, fy * a, fz * a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell)
    pyscf_atoms = []
    for fx, fy, fz in li_frac:
        pyscf_atoms.append(("Li", fx * a_ang, fy * a_ang, fz * a_ang))
    for fx, fy, fz in h_frac:
        pyscf_atoms.append(("H", fx * a_ang, fy * a_ang, fz * a_ang))
    return "LiH-rocksalt", sys_p, pyscf_atoms, a_ang


def nacl_rocksalt(a_ang: float = 5.640):
    """NaCl conventional rocksalt, 4 NaCl per cell."""
    a = a_ang * ANG2BOHR
    na_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    cl_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell = []
    for fx, fy, fz in na_frac:
        cell.append(vq.Atom(11, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in cl_frac:
        cell.append(vq.Atom(17, [fx * a, fy * a, fz * a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell)
    pyscf_atoms = []
    for fx, fy, fz in na_frac:
        pyscf_atoms.append(("Na", fx * a_ang, fy * a_ang, fz * a_ang))
    for fx, fy, fz in cl_frac:
        pyscf_atoms.append(("Cl", fx * a_ang, fy * a_ang, fz * a_ang))
    return "NaCl-rocksalt", sys_p, pyscf_atoms, a_ang


def neon_fcc(a_ang: float = 4.43):
    """Ne FCC — control: deep cores, no charge transfer, expected ✓."""
    a = a_ang * ANG2BOHR
    fcc = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    cell = [vq.Atom(10, [fx * a, fy * a, fz * a]) for fx, fy, fz in fcc]
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell)
    pyscf_atoms = [("Ne", fx * a_ang, fy * a_ang, fz * a_ang)
                   for fx, fy, fz in fcc]
    return "Ne-FCC", sys_p, pyscf_atoms, a_ang


# ============================================================
# vibe-qc Fock-component builder
# ============================================================

def vibeqc_components(
    sys_p: vq.PeriodicSystem,
    basis_name: str,
    *,
    omega: float = 0.5,
    cutoff_bohr: float = 12.0,
    nuclear_cutoff_bohr: float = 25.0,
    spacing_bohr: float = 0.5,
    functional: str = "lda",
) -> Dict[str, Any]:
    """Build vibe-qc's iter-1 Fock components at SAD density.

    Returns Frobenius norms, traces tr(D·M), and the iter-1 Fock
    eigenvalue spectrum.
    """
    from vibeqc import (
        BasisSet, CoulombMethod, Functional, GridOptions,
        LatticeMatrixSet, LatticeSumOptions, build_grid,
        bloch_sum, build_xc_periodic, build_j_ewald_3d,
        compute_kinetic_lattice, compute_overlap_lattice,
        nuclear_repulsion_per_cell, sad_density,
    )
    from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch
    from vibeqc.periodic_grid import build_periodic_becke_grid
    from vibeqc.ewald_j import auto_grid

    basis = BasisSet(sys_p.unit_cell_molecule(), basis_name)
    nbf = basis.nbasis

    lat_opts = LatticeSumOptions()
    lat_opts.coulomb_method = CoulombMethod.EWALD_3D
    lat_opts.cutoff_bohr = cutoff_bohr
    lat_opts.nuclear_cutoff_bohr = nuclear_cutoff_bohr

    k_gamma = np.zeros(3)

    # --- One-electron operators ---------------------------------------
    S_lat = compute_overlap_lattice(basis, sys_p, lat_opts)
    T_lat = compute_kinetic_lattice(basis, sys_p, lat_opts)
    # V_ne via dispatch (Ewald path for EWALD_3D)
    V_lat_dispatch = compute_nuclear_lattice_dispatch(basis, sys_p, lat_opts)
    # V_ne via bare libint (legacy) for comparison
    from vibeqc import compute_nuclear_lattice as _bare_v_ne
    V_lat_bare = _bare_v_ne(basis, sys_p, lat_opts)

    S = np.real(bloch_sum(S_lat, k_gamma))
    T = np.real(bloch_sum(T_lat, k_gamma))
    V_ne = np.real(bloch_sum(V_lat_dispatch, k_gamma))
    V_ne_bare = np.real(bloch_sum(V_lat_bare, k_gamma))
    S = 0.5 * (S + S.T)
    T = 0.5 * (T + T.T)
    V_ne = 0.5 * (V_ne + V_ne.T)
    V_ne_bare = 0.5 * (V_ne_bare + V_ne_bare.T)
    Hcore = T + V_ne

    # --- SAD density --------------------------------------------------
    D = np.asarray(sad_density(sys_p.unit_cell_molecule(), basis))
    D = 0.5 * (D + D.T)
    n_elec_observed = float(np.einsum("ij,ij->", D, S))
    n_elec_expected = sys_p.n_electrons()

    # --- J via FFT-Poisson Ewald -------------------------------------
    lat = np.asarray(sys_p.lattice, dtype=float)
    grid_shape = auto_grid(lat, spacing_bohr)
    J = build_j_ewald_3d(
        basis, sys_p, D, omega=float(omega),
        lattice_opts=lat_opts,
        grid_shape=grid_shape, origin=None,
        spacing_bohr=spacing_bohr,
    )
    J = 0.5 * (J + J.T)

    # --- V_xc via libxc on periodic Becke grid -----------------------
    func = Functional(functional, 1)
    grid = build_periodic_becke_grid(sys_p, grid_options=GridOptions())

    # Wrap D in a degenerate LatticeMatrixSet (block 0 = D, others zero).
    D_set = compute_overlap_lattice(basis, sys_p, lat_opts)
    zero = np.zeros_like(D)
    for i in range(len(D_set)):
        D_set.set_block(i, D if i == 0 else zero)
    xc_contrib = build_xc_periodic(basis, sys_p, grid, func, D_set, lat_opts)
    V_xc = np.real(bloch_sum(xc_contrib.V_xc, k_gamma))
    V_xc = 0.5 * (V_xc + V_xc.T)
    E_xc = float(xc_contrib.e_xc)

    # --- Nuclear repulsion --------------------------------------------
    E_nuc = float(nuclear_repulsion_per_cell(sys_p, lat_opts))

    # --- Iter-1 Fock + eigenvalues -----------------------------------
    F = Hcore + J + V_xc

    # Component traces (energy decomposition of iter-1)
    E_kin = float(np.einsum("ij,ij->", D, T))
    E_ne = float(np.einsum("ij,ij->", D, V_ne))
    E_ne_bare = float(np.einsum("ij,ij->", D, V_ne_bare))
    E_J = 0.5 * float(np.einsum("ij,ij->", D, J))
    E_iter1 = E_kin + E_ne + E_J + E_xc + E_nuc

    # Generalised eigenvalue spectrum F·C = εS·C  (basis-independent
    # iter-1 KS energy levels — directly comparable to PySCF.pbc).
    try:
        eps_F = np.sort(np.real(_gen_eigvalsh(F, S)))
    except Exception:
        eps_F = np.full(nbf, np.nan)
    eps_S = np.sort(np.linalg.eigvalsh(S))
    eps_T = np.sort(np.linalg.eigvalsh(T))
    eps_V_ne = np.sort(np.linalg.eigvalsh(V_ne))
    eps_V_ne_bare = np.sort(np.linalg.eigvalsh(V_ne_bare))
    eps_J = np.sort(np.linalg.eigvalsh(J))
    eps_V_xc = np.sort(np.linalg.eigvalsh(V_xc))

    return dict(
        code="vibeqc",
        nbf=nbf,
        n_elec_observed=n_elec_observed,
        n_elec_expected=n_elec_expected,
        E_kin=E_kin, E_ne=E_ne, E_ne_bare=E_ne_bare, E_J=E_J,
        E_xc=E_xc, E_nuc=E_nuc, E_iter1=E_iter1,
        eps_S=eps_S, eps_T=eps_T,
        eps_V_ne=eps_V_ne, eps_V_ne_bare=eps_V_ne_bare,
        eps_J=eps_J, eps_V_xc=eps_V_xc,
        eps_F=eps_F,
        # Frobenius norms (also basis-independent)
        fro_S=np.linalg.norm(S),
        fro_T=np.linalg.norm(T),
        fro_V_ne=np.linalg.norm(V_ne),
        fro_V_ne_bare=np.linalg.norm(V_ne_bare),
        fro_J=np.linalg.norm(J),
        fro_V_xc=np.linalg.norm(V_xc),
    )


def _gen_eigvalsh(F: np.ndarray, S: np.ndarray) -> np.ndarray:
    """F·C = εS·C via canonical orthogonalisation, dropping
    near-singular S directions (linear-dep robust)."""
    s_eig, U = np.linalg.eigh(S)
    keep = s_eig > 1e-7
    X = U[:, keep] / np.sqrt(s_eig[keep])
    Fp = X.T @ F @ X
    Fp = 0.5 * (Fp + Fp.T)
    return np.linalg.eigvalsh(Fp)


# ============================================================
# PySCF.pbc Fock-component builder
# ============================================================

def pyscf_components(
    pyscf_atoms: List[Tuple[str, float, float, float]],
    a_ang: float,
    *,
    basis_name: str = "sto-3g",
    functional: str = "lda",
) -> Dict[str, Any]:
    """Build PySCF.pbc's iter-1 Fock components at SAD density.

    Uses Γ-only RKS-LDA at sto-3g, which is the matched setup.
    """
    if not PYSCF_AVAILABLE:
        return dict(code="pyscf", available=False)

    atom_str = "; ".join(
        f"{el} {x:.6f} {y:.6f} {z:.6f}" for (el, x, y, z) in pyscf_atoms
    )
    # All-electron PBC: PySCF's default FFTDF auto-sizes ``ke_cutoff``
    # off the tightest Gaussian primitive (sto-3g Ne is ~200 Ha tight
    # → 513³ PW mesh = 135M points → OOM). Force a sane mesh that
    # matches vibe-qc's spacing (~0.5 bohr → ~17³ grid for an 8.4 bohr
    # cell). For Coulomb J we use Gaussian density fitting (GDF), which
    # is the all-electron-friendly PBC integral path in PySCF.
    from pyscf.pbc import df as pbc_df
    cell = pbc_gto.M(
        atom=atom_str,
        a=[[a_ang, 0, 0], [0, a_ang, 0], [0, 0, a_ang]],
        basis=basis_name,
        unit="A",
        verbose=0,
        mesh=[31, 31, 31],     # ~0.27 bohr spacing (better than vibe-qc's)
        precision=1e-8,
    )

    mf = pbc_dft.RKS(cell)
    mf.xc = functional
    mf.exxdiv = "ewald"           # match vibe-qc's gauge
    mf.with_df = pbc_df.GDF(cell)  # Gaussian density fitting for J
    mf.with_df.build()

    # --- Operators independent of density ---------------------------
    S = mf.get_ovlp()
    T = cell.pbc_intor("int1e_kin", hermi=1)
    V_ne = mf.get_hcore() - T
    Hcore = mf.get_hcore()

    # --- SAD-equivalent guess: PySCF's atom guess --------------------
    # PySCF's "atom" SCF guess = SAD (sums of atomic-AO densities).
    D = mf.get_init_guess(cell, key="atom")

    n_elec_observed = float(np.einsum("ij,ij->", D, S))
    n_elec_expected = cell.nelectron

    # --- J + V_xc (no exchange for LDA) ------------------------------
    veff = mf.get_veff(cell, D)
    # veff is J + V_xc for pure LDA on closed-shell.
    # Decompose into J and V_xc:
    J = mf.get_j(cell, D)
    V_xc = veff - J  # = V_xc since α = 0
    E_xc = float(veff.exc)

    # --- Nuclear repulsion -------------------------------------------
    E_nuc = float(cell.energy_nuc())

    # --- Iter-1 Fock + spectra --------------------------------------
    F = Hcore + J + V_xc

    E_kin = float(np.einsum("ij,ij->", D, T))
    E_ne = float(np.einsum("ij,ij->", D, V_ne))
    E_J = 0.5 * float(np.einsum("ij,ij->", D, J))
    E_iter1 = E_kin + E_ne + E_J + E_xc + E_nuc

    eps_F = np.sort(np.real(_gen_eigvalsh(F, S)))
    eps_S = np.sort(np.linalg.eigvalsh(S))
    eps_T = np.sort(np.linalg.eigvalsh(T))
    eps_V_ne = np.sort(np.linalg.eigvalsh(V_ne))
    eps_J = np.sort(np.linalg.eigvalsh(J))
    eps_V_xc = np.sort(np.linalg.eigvalsh(V_xc))

    return dict(
        code="pyscf",
        available=True,
        nbf=cell.nao,
        n_elec_observed=n_elec_observed,
        n_elec_expected=n_elec_expected,
        E_kin=E_kin, E_ne=E_ne, E_J=E_J,
        E_xc=E_xc, E_nuc=E_nuc, E_iter1=E_iter1,
        eps_S=eps_S, eps_T=eps_T,
        eps_V_ne=eps_V_ne, eps_J=eps_J, eps_V_xc=eps_V_xc,
        eps_F=eps_F,
        fro_S=np.linalg.norm(S),
        fro_T=np.linalg.norm(T),
        fro_V_ne=np.linalg.norm(V_ne),
        fro_J=np.linalg.norm(J),
        fro_V_xc=np.linalg.norm(V_xc),
    )


# ============================================================
# Reporting
# ============================================================

def _fmt_eps_summary(eps: np.ndarray, n: int = 5) -> str:
    """First/last n eigenvalues, compact."""
    if eps is None or np.any(np.isnan(eps)):
        return "[NaN]"
    if len(eps) <= 2 * n:
        return "  ".join(f"{e:+9.4f}" for e in eps)
    head = "  ".join(f"{e:+9.4f}" for e in eps[:n])
    tail = "  ".join(f"{e:+9.4f}" for e in eps[-n:])
    return f"{head}  ...  {tail}"


def report(label: str, vqr: Dict[str, Any], psr: Dict[str, Any]) -> None:
    """Print a side-by-side comparison + delta column."""
    print()
    print("=" * 80)
    print(f"  {label}")
    print("=" * 80)

    if not psr.get("available", False):
        print("  PySCF not installed — vibe-qc only:")
        for k in ("nbf", "n_elec_observed", "n_elec_expected",
                  "E_kin", "E_ne", "E_ne_bare", "E_J", "E_xc",
                  "E_nuc", "E_iter1"):
            v = vqr.get(k)
            if v is None:
                continue
            print(f"  {k:>20s}  {v}")
        return

    # Sanity check: same # of basis functions, same electron count
    same_nbf = vqr["nbf"] == psr["nbf"]
    same_nel = abs(vqr["n_elec_expected"] - psr["n_elec_expected"]) < 1e-6
    print(f"  nbf:   vibeqc {vqr['nbf']:>3d}   pyscf {psr['nbf']:>3d}   "
          f"{'✓' if same_nbf else '✗ MISMATCH'}")
    print(f"  N_e:   vibeqc {vqr['n_elec_expected']:>3d}   pyscf "
          f"{psr['n_elec_expected']:>3d}   "
          f"{'✓' if same_nel else '✗ MISMATCH'}")
    print(f"         (SAD-projected ⟨D|S⟩: vibeqc "
          f"{vqr['n_elec_observed']:.3f}   pyscf "
          f"{psr['n_elec_observed']:.3f})")

    # --- Energy decomposition table -----------------------------------
    print()
    print(f"  {'component':>12s}  {'vibeqc (Ha)':>14s}  "
          f"{'pyscf (Ha)':>14s}  {'Δ (Ha)':>14s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}")
    rows = [
        ("E_kin",   vqr["E_kin"],   psr["E_kin"]),
        ("E_ne",    vqr["E_ne"],    psr["E_ne"]),
        ("E_ne_bare", vqr.get("E_ne_bare"), None),
        ("E_J",     vqr["E_J"],     psr["E_J"]),
        ("E_xc",    vqr["E_xc"],    psr["E_xc"]),
        ("E_nuc",   vqr["E_nuc"],   psr["E_nuc"]),
        ("E_iter1", vqr["E_iter1"], psr["E_iter1"]),
    ]
    for name, vv, pv in rows:
        if vv is None:
            continue
        if pv is None:
            print(f"  {name:>12s}  {vv:>+14.6f}  {'-':>14s}  {'-':>14s}")
        else:
            d = vv - pv
            mark = " "
            if name != "E_iter1" and abs(d) > 0.5:
                mark = "  ← LARGE"
            print(f"  {name:>12s}  {vv:>+14.6f}  {pv:>+14.6f}  "
                  f"{d:>+14.6e}{mark}")

    # --- Frobenius norms (basis-independent) -------------------------
    print()
    print(f"  {'||M||_F':>12s}  {'vibeqc':>14s}  {'pyscf':>14s}  "
          f"{'Δ rel':>14s}")
    print(f"  {'-'*12}  {'-'*14}  {'-'*14}  {'-'*14}")
    for k in ("fro_S", "fro_T", "fro_V_ne", "fro_J", "fro_V_xc"):
        if k in vqr and k in psr:
            v, p = vqr[k], psr[k]
            rel = (v - p) / max(abs(p), 1e-12)
            mark = "  ← LARGE" if abs(rel) > 0.05 else ""
            print(f"  {k:>12s}  {v:>14.4f}  {p:>14.4f}  {rel:>+14.3e}{mark}")
    if "fro_V_ne_bare" in vqr:
        v = vqr["fro_V_ne_bare"]
        print(f"  {'fro_V_ne_bare':>12s}  {v:>14.4f}  {'-':>14s}  "
              f"{'-':>14s}  (legacy non-Ewald V_ne for reference)")

    # --- Eigenvalue spectra (basis-independent) ----------------------
    print()
    print("  Eigenvalue spectra (basis-independent — bulk extrema):")
    for k, eps_v_key, eps_p_key in [
        ("S",     "eps_S",      "eps_S"),
        ("T",     "eps_T",      "eps_T"),
        ("V_ne",  "eps_V_ne",   "eps_V_ne"),
        ("J",     "eps_J",      "eps_J"),
        ("V_xc",  "eps_V_xc",   "eps_V_xc"),
        ("F",     "eps_F",      "eps_F"),
    ]:
        ev = vqr.get(eps_v_key)
        ep = psr.get(eps_p_key)
        if ev is None or ep is None:
            continue
        if len(ev) != len(ep):
            print(f"    {k:>5s}  shape mismatch: {len(ev)} vs {len(ep)}")
            continue
        # Spectra are basis-invariant → if equal as linear maps, sorted
        # eigenvalues match exactly (up to numerical ε).
        diff = ev - ep
        max_abs = float(np.max(np.abs(diff)))
        mark = ""
        if k in ("V_ne", "J", "V_xc") and max_abs > 0.5:
            mark = "  ← LARGE"
        elif k in ("S", "T") and max_abs > 1e-3:
            mark = "  ← LARGE"
        print(f"    {k:>5s}  max|Δλ| = {max_abs:.3e}{mark}")
        # First / last few eigenvalues
        print(f"          vibeqc:  {_fmt_eps_summary(ev)}")
        print(f"          pyscf :  {_fmt_eps_summary(ep)}")

    # --- Verdict ------------------------------------------------------
    print()
    print("  --- Verdict ---")
    deltas: List[Tuple[str, float]] = []
    for name, vv, pv in [
        ("V_ne (E_ne)",          vqr["E_ne"],    psr["E_ne"]),
        ("J (E_J)",               vqr["E_J"],    psr["E_J"]),
        ("V_xc (E_xc)",           vqr["E_xc"],   psr["E_xc"]),
        ("E_nuc",                 vqr["E_nuc"],  psr["E_nuc"]),
    ]:
        deltas.append((name, abs(vv - pv)))
    deltas.sort(key=lambda x: -x[1])
    print(f"    Largest |Δ| component: {deltas[0][0]}  "
          f"({deltas[0][1]:+.4e} Ha)")
    print(f"    Ranked: " + ", ".join(
        f"{n} ({d:+.2e})" for n, d in deltas
    ))


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("=" * 80)
    print(" Fock-component triangulator — vibe-qc vs PySCF.pbc on iter-1 SAD")
    print("=" * 80)
    print(f"  PySCF available:  {PYSCF_AVAILABLE}")
    print(f"  artefacts →       {OUT_DIR}/")

    cases = [
        neon_fcc(),         # control (expected: ✓ everywhere)
        lih_rocksalt(),     # converges-wrong: ~+2.4 Ha vs PySCF
        nacl_rocksalt(),    # diverges:        Fock oscillates ~kHa
    ]

    for label, sys_p, pyscf_atoms, a_ang in cases:
        print()
        print(f"  >>> {label} (a = {a_ang:.3f} Å, "
              f"{len(sys_p.unit_cell)} atoms in cell)")
        t0 = time.perf_counter()
        try:
            vqr = vibeqc_components(sys_p, "sto-3g")
        except Exception as exc:
            print(f"    vibe-qc FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:120]}")
            continue
        t_vibeqc = time.perf_counter() - t0
        print(f"    vibe-qc components built in {t_vibeqc:.1f}s")

        t0 = time.perf_counter()
        try:
            psr = pyscf_components(pyscf_atoms, a_ang, basis_name="sto-3g")
        except Exception as exc:
            print(f"    pyscf  FAILED: {type(exc).__name__}: "
                  f"{str(exc)[:120]}")
            psr = dict(code="pyscf", available=False)
        t_pyscf = time.perf_counter() - t0
        if psr.get("available", False):
            print(f"    pyscf   components built in {t_pyscf:.1f}s")

        report(label, vqr, psr)

    print()
    print("=" * 80)
    print(" Done.  Look for the row marked '← LARGE' in the energy or")
    print(" Frobenius / eigenvalue tables: that component is the bug.")
    print("=" * 80)


if __name__ == "__main__":
    main()
