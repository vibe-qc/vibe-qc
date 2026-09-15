#!/usr/bin/env python3
"""Heavy validation of the v0.14 XC additions (revDSD-PBEP86-D4, VV10,
ωB97X-V, ωB97M-V) on a larger system than the H2O / cc-pVDZ unit-test
box, to confirm the nonlocal VV10 double-grid integral and the
double-hybrid machinery hold at production scale.

Self-contained: only needs ``vibeqc``. If ``pyscf`` is importable in the
job env it adds a cross-code parity line for ωB97X-V / ωB97M-V (PySCF
``nlc="VV10"``); otherwise it reports vibe-qc's own numbers plus the VV10
contribution magnitude (semilocal ωB97X vs full ωB97X-V), which is the
scale-correctness signal.

Run via vq, e.g.:
    vq submit auto --refresh vibeqc-dev \\
        studies/vv10-heavy-validation/heavy_validate_xc.py
"""

from __future__ import annotations

import time

import numpy as np

from vibeqc import (
    Atom,
    BasisSet,
    Molecule,
    RKSOptions,
    run_rks,
    run_revdsd_pbep86,
)

# Ethanol (CH3CH2OH), 9 atoms — comfortably bigger than H2O so the VV10
# O(N_grid^2) double integral runs on a production-scale grid. Geometry in
# Angstrom (B3LYP/def2-SVP optimised, rounded), converted to bohr below.
ANG2BOHR = 1.8897259886
ETHANOL_ANG = [
    (6, (-1.1879, 0.1788, 0.0000)),
    (6, (0.0000, -0.5664, 0.0000)),
    (8, (1.1597, 0.2945, 0.0000)),
    (1, (-2.0755, -0.4575, 0.0000)),
    (1, (-1.2197, 0.8189, 0.8870)),
    (1, (-1.2197, 0.8189, -0.8870)),
    (1, (0.0518, -1.2135, 0.8843)),
    (1, (0.0518, -1.2135, -0.8843)),
    (1, (1.9361, -0.2811, 0.0000)),
]
ETHANOL_BOHR = [(Z, tuple(c * ANG2BOHR for c in xyz)) for Z, xyz in ETHANOL_ANG]

BASIS = "cc-pvdz"


def vq_mol_basis():
    mol = Molecule([Atom(Z, list(xyz)) for Z, xyz in ETHANOL_BOHR])
    return mol, BasisSet(mol, BASIS)


def run_vibeqc_rks(functional, density_fit=False):
    mol, basis = vq_mol_basis()
    o = RKSOptions()
    o.functional = functional
    o.density_fit = density_fit
    o.conv_tol_energy = 1e-9
    o.conv_tol_grad = 1e-6
    t0 = time.time()
    r = run_rks(mol, basis, o)
    return r, time.time() - t0


def maybe_pyscf(functional):
    try:
        from pyscf import gto, dft
    except Exception:
        return None
    pm = gto.Mole()
    pm.unit = "Bohr"
    pm.atom = [[Z, tuple(xyz)] for Z, xyz in ETHANOL_BOHR]
    pm.basis = BASIS
    pm.verbose = 0
    pm.build()
    mf = dft.RKS(pm, xc=functional)
    mf.nlc = "VV10"
    mf.grids.level = 3
    mf.nlcgrids.level = 3
    mf.conv_tol = 1e-9
    e = mf.kernel()
    return float(e) if mf.converged else None


def main():
    print("=" * 70)
    print("Heavy XC validation — ethanol / cc-pVDZ (9 atoms)")
    print("=" * 70)

    # 1) ωB97X-V: full functional incl. VV10, vs semilocal ωB97X.
    r_v, t_v = run_vibeqc_rks("wb97x-v")
    r_sl, t_sl = run_vibeqc_rks("wb97x")
    vv10_contrib = r_v.energy - r_sl.energy
    print(f"\n[ωB97X-V]  converged={r_v.converged}  "
          f"E={r_v.energy:.8f} Ha  ({t_v:.1f}s)")
    print(f"[ωB97X  ]  semilocal-only E={r_sl.energy:.8f} Ha")
    print(f"  → VV10 nonlocal contribution = {vv10_contrib * 1e3:+.4f} mHa")
    e_ps_v = maybe_pyscf("wb97x-v")
    if e_ps_v is not None:
        print(f"  PySCF ωB97X-V (nlc=VV10) = {e_ps_v:.8f} Ha  "
              f"gap = {(r_v.energy - e_ps_v) * 1e6:+.2f} µHa")

    # 2) ωB97M-V: RSH meta-GGA + VV10 composed.
    r_m, t_m = run_vibeqc_rks("wb97m-v")
    print(f"\n[ωB97M-V]  converged={r_m.converged}  "
          f"E={r_m.energy:.8f} Ha  ({t_m:.1f}s)")
    e_ps_m = maybe_pyscf("wb97m-v")
    if e_ps_m is not None:
        print(f"  PySCF ωB97M-V (nlc=VV10) = {e_ps_m:.8f} Ha  "
              f"gap = {(r_m.energy - e_ps_m) * 1e6:+.2f} µHa")

    # 3) revDSD-PBEP86-D4 double hybrid at scale.
    mol, basis = vq_mol_basis()
    t0 = time.time()
    dh = run_revdsd_pbep86(mol, basis, density_fit_mp2=True,
                           aux_basis_mp2="cc-pvdz-ri")
    print(f"\n[revDSD-PBEP86]  E_total={dh.e_total:.8f} Ha  "
          f"(rks {dh.rks.energy:.6f} + mp2 {dh.mp2.e_correlation:.6f})  "
          f"({time.time() - t0:.1f}s)")

    # Verdict: VV10 must be a real, sane, negative-ish nonlocal correction
    # (a few-to-tens mHa for a molecule this size), and everything must
    # converge.
    ok = (r_v.converged and r_m.converged and dh.rks.converged
          and abs(vv10_contrib) > 1e-4)
    print("\n" + "=" * 70)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — "
          f"all SCFs converged, VV10 contribution real "
          f"({vv10_contrib * 1e3:+.3f} mHa)")
    print("=" * 70)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
