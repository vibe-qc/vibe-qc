"""LiH primitive cell multi-k compcell SCF parity vs PySCF.

After the multi-k SCF fix in c74527b, H2 multi-k compcell matches
PySCF at sub-mHa on a flat-band vacuum-box system. The next question:
does it work on an actual ionic crystal with BZ dispersion?

LiH primitive cell (FCC): 2 atoms (1 Li + 1 H) per primitive cell.
PySCF kmesh=(2,2,2) is the canonical small-mesh validation point;
kmesh=(4,4,4) would be tighter BZ but takes longer.

The Γ-only LiH 8-atom conv cell exposes the pair-FT convention issue
(divergent compcell with L>0 chg shells on the Li 2p side). Multi-k
sampling MIGHT improve things (because the per-k phase rotation could
smooth out the per-shell convention residue), but no guarantees —
this script gets us the actual data point.

PySCF subprocess per CLAUDE.md §10.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import vibeqc as vq


ANG2BOHR = 1.0 / 0.529177210903
A_LIH_ANG = 4.084  # FCC conventional-cell parameter; primitive a = a/√2


PYSCF_DRIVER = r'''
import json, sys
import numpy as np
from pyscf.pbc import gto as pbc_gto, scf as pbc_scf

# LiH primitive FCC: a_conv = 4.084 Å; primitive lattice vectors are
# the FCC face-centred basis.
A_CONV = {a_conv}
a = A_CONV
# Primitive cell: rhombohedral FCC basis
lattice = np.array([
    [0.0, 0.5, 0.5],
    [0.5, 0.0, 0.5],
    [0.5, 0.5, 0.0],
]) * a

cell = pbc_gto.M(
    atom=[("Li", (0.0, 0.0, 0.0)),
          ("H",  (0.5*a, 0.5*a, 0.5*a))],
    a=lattice,
    unit="A",
    basis="sto-3g",
    precision=1e-10,
    verbose=0,
)
KMESH = {kmesh}
kpts = cell.make_kpts(KMESH)

mf = pbc_scf.KRHF(cell, kpts).density_fit(auxbasis="def2-svp-jkfit")
mf.exxdiv = "ewald"
mf.conv_tol = 1e-8
mf.max_cycle = 100
E = mf.kernel()

print("VIBEQC-PYSCF-RESULT:" + json.dumps({{
    "E_total": float(E),
    "converged": bool(mf.converged),
    "n_iter": int(getattr(mf, "cycles", -1)),
    "n_kpts": int(len(kpts)),
    "kmesh": KMESH,
    "n_ao": int(cell.nao_nr()),
    "n_elec": int(cell.nelectron),
}}, sort_keys=True))
'''


def run_pyscf(kmesh) -> dict:
    py = os.environ.get("VIBEQC_PYSCF_PYTHON", sys.executable)
    script = PYSCF_DRIVER.format(a_conv=A_LIH_ANG, kmesh=list(kmesh))
    proc = subprocess.run([py, "-c", script], capture_output=True,
                          text=True, check=False)
    if proc.returncode != 0:
        print("PySCF subprocess failed:", proc.returncode, file=sys.stderr)
        print(proc.stderr[-2000:], file=sys.stderr)
        sys.exit(2)
    for line in proc.stdout.splitlines():
        if line.startswith("VIBEQC-PYSCF-RESULT:"):
            return json.loads(line[len("VIBEQC-PYSCF-RESULT:"):])
    sys.exit(2)


def _lih_primitive_system():
    a = A_LIH_ANG * ANG2BOHR
    lattice = np.array([
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ]) * a
    atoms = [
        vq.Atom(3, [0.0, 0.0, 0.0]),
        vq.Atom(1, [0.5 * a, 0.5 * a, 0.5 * a]),
    ]
    system = vq.PeriodicSystem(3, lattice, atoms)
    basis = vq.BasisSet(system.unit_cell_molecule(), "sto-3g")
    return system, basis


def vibeqc_compcell(system, basis, kmesh, *, aft=False, eta=0.25,
                    rcut_strategy=None, rcut_precision=1e-8,
                    aft_precision=1e-10, aft_ft_convention="libcint"):
    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 50
    opts.conv_tol_energy = 1e-7
    opts.lattice_opts.cutoff_bohr = 30.0
    opts.lattice_opts.nuclear_cutoff_bohr = 30.0
    t0 = time.perf_counter()
    r = vq.run_krhf_periodic_gdf(
        system, basis, kmesh=tuple(kmesh), options=opts,
        aux_basis="def2-svp-jk",
        use_compcell=True,
        compcell_eta=eta,
        apply_aft_correction=aft,
        aft_precision=aft_precision,
        aft_ft_convention=aft_ft_convention,
        rcut_strategy=rcut_strategy,
        rcut_precision=rcut_precision,
        progress=False,
    )
    return r, time.perf_counter() - t0


def main() -> int:
    system, basis = _lih_primitive_system()
    print(f"vibeqc {vq.__version__}: LiH primitive multi-k compcell vs PySCF")
    print(f"  Li-H FCC, a_conv={A_LIH_ANG} Å (primitive 2 atoms)")
    print(f"  basis=STO-3G, aux=def2-svp-jk, exxdiv='ewald' (auto via Ewald-3D gauge)")
    print()
    print(f"  {'kmesh':>10s}  {'mode':>28s}  {'PySCF E':>16s}  "
          f"{'vibe-qc E':>16s}  {'Δ (Ha)':>14s}  {'cnv':>5s}  iter  wall(s)")
    # Start small. (2,2,2) is the canonical small test.
    for kmesh in [(2, 2, 2)]:
        py = run_pyscf(kmesh)
        # Three modes: bare-compcell, AFT-flat-rcut, AFT-pyscf-auto-rcut.
        # The 3rd isolates whether the residue is in the bare 3c lattice
        # cutoff (PySCF's per-shell rcut is much tighter for diffuse
        # primitives than vibe-qc's flat 30-bohr cut).
        # The H2 sub-mHa path uses aft_ft_convention="libint" (matched
        # to the libint bare lattice sum); the multi-k driver defaults
        # to "libcint". Test both conventions at converged precision to
        # see which one the LiH bare lattice sum is actually in.
        modes = [
            ("bare",                       dict(aft=False, rcut_strategy=None)),
            ("AFT libcint/auto ε=1e-12",   dict(aft=True, rcut_strategy="pyscf_auto",
                                                rcut_precision=1e-10,
                                                aft_precision=1e-12,
                                                aft_ft_convention="libcint")),
            ("AFT libint/auto ε=1e-12",    dict(aft=True, rcut_strategy="pyscf_auto",
                                                rcut_precision=1e-10,
                                                aft_precision=1e-12,
                                                aft_ft_convention="libint")),
            ("AFT libint/auto ε=1e-14",    dict(aft=True, rcut_strategy="pyscf_auto",
                                                rcut_precision=1e-12,
                                                aft_precision=1e-14,
                                                aft_ft_convention="libint")),
        ]
        for mode, kw in modes:
            try:
                vqc, dt = vibeqc_compcell(system, basis, kmesh, **kw)
                delta = vqc.energy - py["E_total"]
                print(f"  {str(kmesh):>10s}  {mode:>28s}"
                      f"{py['E_total']:>16.8f}  {vqc.energy:>16.8f}  "
                      f"{delta:>+14.4e}  {str(vqc.converged):>5s}  "
                      f"{vqc.n_iter:>4d}  {dt:>5.1f}")
            except Exception as exc:
                print(f"  {str(kmesh):>10s}  {mode:>28s}"
                      f"{py['E_total']:>16.8f}  "
                      f"{'ERR':>16s}  -  -    -  -")
                print(f"    {type(exc).__name__}: {str(exc)[:200]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
