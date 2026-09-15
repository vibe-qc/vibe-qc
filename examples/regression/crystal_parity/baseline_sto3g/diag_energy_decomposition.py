"""Energy-contribution decomposition: vibe-qc periodic RHF vs CRYSTAL14.

The vibe-qc Γ-only conventional-cell RHF energy is +33 Ha/FU
*under-bound* vs CRYSTAL14 on MgO (and +0.6 Ha/FU on LiH).
e_nuclear is already confirmed bit-exact, and molecular-limit
single-atom RHF matches textbook STO-3G to ~1e-5 — so the bug is
periodic and electronic.  This script splits the periodic
electronic energy into kinetic / electron-nuclear / electron-
electron and prints each term against the CRYSTAL14 reference,
to see *which* contribution is wrong.

LiH first (fast — also carries a smaller version of the gap),
then MgO.  progress=True for a live SCF trace.

Run via vq:
    vq submit examples/regression/crystal_parity/baseline_sto3g/diag_energy_decomposition.py
"""
import os
import sys

os.environ.setdefault("PYSCF_TMPDIR", "/tmp/pyscf_vibeqc")
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import numpy as np
import vibeqc as vq
from vibeqc._vibeqc_core import (
    LatticeSumOptions, bloch_sum,
    compute_kinetic_lattice,
)
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

ANG2BOHR = 1.0 / 0.529177210903


def rocksalt(a_ang, zc, za):
    a = a_ang * ANG2BOHR
    cf = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    af = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    atoms = [vq.Atom(zc, [x * a, y * a, z * a]) for x, y, z in cf]
    atoms += [vq.Atom(za, [x * a, y * a, z * a]) for x, y, z in af]
    sysm = vq.PeriodicSystem(3, np.diag([a, a, a]), atoms)
    basis = vq.BasisSet(sysm.unit_cell_molecule(), "sto-3g")
    return sysm, basis


# CRYSTAL14 per-FU references (from the sealed STO-3G baseline .out files)
C14 = {
    "LiH": dict(a=4.084, zc=3, za=1,
                E_NN=-3.39397848, E_kin=7.87893408,
                E_ne=-12.27304215, E_ee=-0.15008156, E_total=-7.93816810),
    "MgO": dict(a=4.21, zc=12, za=8,
                E_NN=-73.08427668, E_kin=267.12905642,
                E_ne=-511.75571619, E_ee=46.49278937, E_total=-271.21814708),
}

NFU = 4  # rocksalt conventional cell = 4 formula units


def decompose(name):
    r = C14[name]
    sysm, basis = rocksalt(r["a"], r["zc"], r["za"])
    print(f"\n{'='*70}", flush=True)
    print(f"=== {name}: {basis.nbasis} BFs, {sysm.n_electrons()} electrons, "
          f"8-atom conv cell ({NFU} FU) ===", flush=True)
    print(f"{'='*70}", flush=True)

    opts = vq.PeriodicRHFOptions()
    opts.use_diis = True
    opts.damping = 0.0
    opts.max_iter = 30
    opts.conv_tol_energy = 1e-8

    res = vq.run_rhf_periodic_gamma_gdf(sysm, basis, opts, progress=True, verbose=1)

    # Rebuild T and V(Ewald gauge) at Gamma to split the one-electron energy.
    lat = LatticeSumOptions()
    gl = LatticeSumOptions()
    gl.coulomb_method = vq.CoulombMethod.EWALD_3D
    kg = np.zeros(3)
    T = np.real(bloch_sum(compute_kinetic_lattice(basis, sysm, lat), kg))
    T = 0.5 * (T + T.T)
    V = np.real(bloch_sum(compute_nuclear_lattice_dispatch(basis, sysm, gl), kg))
    V = 0.5 * (V + V.T)

    D = np.asarray(res.density)
    E_kin = float(np.einsum("ij,ij->", D, T))
    E_ne = float(np.einsum("ij,ij->", D, V))
    E_J = float(res.e_coulomb)        # Coulomb (J) energy
    E_K = float(res.e_hf_exchange)    # HF exchange (K) energy
    E_ee = E_J + E_K

    print(f"\n  converged={res.converged}  n_iter={res.n_iter}", flush=True)
    print(f"\n  {'term':<14}{'vibe-qc /cell':>16}{'vibe-qc /FU':>15}"
          f"{'CRYSTAL14 /FU':>15}{'Δ/FU':>13}", flush=True)
    rows = [
        ("E_kinetic", E_kin, r["E_kin"]),
        ("E_ne",      E_ne,  r["E_ne"]),
        ("E_ee",      E_ee,  r["E_ee"]),
        ("E_nuclear", float(res.e_nuclear), r["E_NN"]),
        ("E_total",   float(res.energy),    r["E_total"]),
    ]
    for lbl, cell_v, c14_fu in rows:
        fu_v = cell_v / NFU
        print(f"  {lbl:<14}{cell_v:>16.6f}{fu_v:>15.6f}"
              f"{c14_fu:>15.6f}{fu_v - c14_fu:>+13.6f}", flush=True)
    # E_ee split — CRYSTAL14 prints only the combined TOTAL E-E, so no
    # per-FU CRYSTAL column here; the point is to see whether vibe-qc's
    # J or K (or both) carries the E_ee discrepancy.
    print(f"\n  E_ee split (vibe-qc):  E_J(Coulomb) /cell = {E_J:.6f}  "
          f"/FU = {E_J / NFU:.6f}", flush=True)
    print(f"                         E_K(exchange)/cell = {E_K:.6f}  "
          f"/FU = {E_K / NFU:.6f}", flush=True)
    # Cross-check: e_electronic vs sum of its parts
    e_elec_parts = E_kin + E_ne + E_ee
    print(f"\n  cross-check: E_kin+E_ne+E_ee = {e_elec_parts:.6f}  "
          f"res.e_electronic = {float(res.e_electronic):.6f}  "
          f"Δ = {e_elec_parts - float(res.e_electronic):+.2e}", flush=True)


def main():
    print(f"vibeqc {vq.__version__} — periodic RHF energy decomposition", flush=True)
    for name in ("LiH", "MgO"):
        try:
            decompose(name)
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
