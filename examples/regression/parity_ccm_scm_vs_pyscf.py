#!/usr/bin/env python3
"""CCM ≡ SCM-Γ: the Cyclic Cluster Model HF energy is periodic HF (out-of-process
PySCF validation per CLAUDE.md §10).

The thesis result behind the AICCM (Peintinger, PhD Bonn 2013, Kap. 8, §8.6.1;
Peintinger & Bredow, J. Comput. Chem. 35, 839 (2014), doi:10.1002/jcc.23550)
is that the **Cyclic Cluster Model equals the Supercell Model evaluated at the
Γ-point, exactly**:

    E_CCM(N1,N2,N3 cluster)  ==  E_SCM-Γ(N1·N2·N3 supercell)
                             ==  E_periodic-HF(unit cell, k-mesh = (N1,N2,N3))

The first equality is the definition of the CCM (a finite supercell with cyclic
Born–von Kármán boundary conditions, i.e. a Γ-point calculation on the
supercell). The second is the standard band-folding identity: a Γ-point energy
of the (N1,N2,N3) supercell equals the BZ-averaged energy of the unit cell on a
(N1,N2,N3) Monkhorst–Pack mesh.

This script demonstrates both, on a dilute 3-D molecular crystal where vibe-qc's
Γ-only Ewald-3D periodic HF is reliable (the molecular-limit density convention
D(g≠0)=0 holds; cf. the dense-ionic refusal in ``run_rhf_periodic_gamma_ewald3d``
for space-filling crystals such as LiH/STO-3G):

  PART A — band-folding identity *within vibe-qc* (the CCM≡SCM-Γ proof):
    E_Γ(supercell)/N  ==  E_multi-k(unit cell, k-mesh=N)   to ~µHa, for cubic N.

  PART B — external validation: both equal the PySCF KRHF periodic-HF energy in
    the thermodynamic limit, to the cross-method Coulomb gap (Ewald-3D FFT vs
    PySCF GDF + exxdiv finite-size; a few mHa at finite mesh, shrinking with N),
    converging to a common TDL.

  PART C — the exact-exchange q→0 gauge table both parts stand on (#560).

**The identity holds within one exchange gauge, not across two.** The corrected
Ewald split (PySCF's ``exxdiv='ewald'``) and the molecular-limit kernel are
different finite-size conventions with the same infinite-k limit — Carrier,
Rohra & Görling, Phys. Rev. B 75, 205126 (2007), doi:10.1103/PhysRevB.75.205126,
Sec. IV. vibe-qc's Γ-only drivers use the molecular-limit one and its multi-k
drivers the corrected one, so this script names ``exchange_exxdiv='ewald'`` on
both sides of PART A; PART C is the measurement of what that choice is worth.

PySCF runs in a child interpreter and is parsed back; vibe-qc never imports it.

Usage:
    python examples/regression/parity_ccm_scm_vs_pyscf.py
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np

# Dilute H2 molecular crystal: H2 (d = 1.4 bohr) on a simple-cubic lattice
# (a = 8 bohr). Closed-shell, genuinely 3-D periodic, intermolecular spacing
# large enough that the molecular-limit Γ-Ewald-3D driver is reliable.
A_BOHR = 8.0
D_BOHR = 1.4

# PySCF child: KRHF/GDF on the unit cell for a list of k-meshes, exxdiv='ewald'
# (its default Madelung exchange treatment). Reports e_tot per primitive cell.
_PYSCF_CHILD = r"""
import json, sys
import numpy as np
from pyscf.pbc import gto, scf, tools
p = json.loads(sys.argv[1])
cell = gto.Cell()
cell.a = np.eye(3) * p["a"]
cell.atom = [["H", [0.0, 0.0, 0.0]], ["H", [0.0, 0.0, p["d"]]]]
cell.unit = "Bohr"; cell.basis = "sto3g"; cell.verbose = 0
cell.precision = 1e-10
cell.build()
out = {}
for km in p["kmeshes"]:
    kpts = cell.make_kpts(km)
    for exxdiv in ("ewald", None):
        mf = scf.KRHF(cell, kpts).density_fit()
        mf.exxdiv = exxdiv
        mf.conv_tol = 1e-10
        # e_tot is already per unit cell.
        out[f"{tuple(km)}|{exxdiv}"] = float(mf.kernel())
    out[f"madelung|{tuple(km)}"] = float(tools.pbc.madelung(cell, kpts))
    # The same identity PART A checks, inside PySCF: a Gamma-point run on the
    # supercell the mesh spans. Both gauges, so the table shows band folding
    # holding in each and the gauge step between them.
    if p.get("supercell") and tuple(km) != (1, 1, 1):
        n = int(np.prod(km))
        scell = tools.super_cell(cell, list(km))
        scell.precision = 1e-10
        scell.build()
        for exxdiv in ("ewald", None):
            mf = scf.RHF(scell).density_fit()
            mf.exxdiv = exxdiv
            mf.conv_tol = 1e-10
            out[f"supercell{tuple(km)}|{exxdiv}"] = float(mf.kernel()) / n
print("PYSCF-RESULT:" + json.dumps(out), flush=True)
"""

# Ewald split parameter: both sides of PART A take the driver default. Before
# GitLab #651 this script pinned omega = 0.35 as a workaround, because the
# 16-bohr supercell's volume-scaled default (0.175 bohr^-1) had not decayed by
# the 15-bohr real-space cutoff and cost ~6e-5 Ha/cell. The default is now
# bounded below by sqrt(-ln tol) / cutoff_bohr = 0.3504 on both cells
# (pbc_bipole_common.default_ewald_alpha), so no pin is needed.


def _unit_cell():
    import vibeqc as vq

    return vq.PeriodicSystem(
        3, np.diag([A_BOHR, A_BOHR, A_BOHR]),
        [vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, D_BOHR])],
        charge=0, multiplicity=1,
    )


def ccm_gamma_supercell(nrep, exchange_exxdiv="ewald"):
    """CCM energy/cell via a Γ-point calculation on the (nrep) supercell.

    A (1,1,1) mesh *is* the Γ-point calculation of the supercell.
    ``exchange_exxdiv='ewald'`` is what stops the driver short-circuiting to
    the molecular-limit exchange kernel, which is the other gauge (#560).
    ``run_rhf_periodic_gamma_ewald3d`` is that other gauge and is reported
    alongside in PART C.
    """
    import vibeqc as vq
    from vibeqc.periodic.ccm import CCMSystem

    ccm = CCMSystem(_unit_cell(), nrep, "sto-3g")
    res = vq.run_rhf_periodic_multi_k_ewald3d(
        ccm.cluster_system,
        ccm.basis,
        vq.monkhorst_pack(ccm.cluster_system, [1, 1, 1]),
        exchange_exxdiv=exchange_exxdiv,
    )
    n = nrep[0] * nrep[1] * nrep[2]
    return res.energy / n, bool(res.converged)


def gamma_driver_supercell(nrep):
    """The Γ-only driver's own (molecular-limit gauge) energy/cell."""
    import vibeqc as vq
    from vibeqc.periodic.ccm import CCMSystem

    ccm = CCMSystem(_unit_cell(), nrep, "sto-3g")
    res = vq.run_rhf_periodic_gamma_ewald3d(ccm.cluster_system, ccm.basis)
    n = nrep[0] * nrep[1] * nrep[2]
    return res.energy / n, bool(res.converged)


def vibeqc_multik(kmesh, exchange_exxdiv="ewald"):
    """vibe-qc multi-k periodic HF energy/cell on the unit cell."""
    import vibeqc as vq

    unit = _unit_cell()
    basis = vq.make_basis(unit.unit_cell_molecule(), "sto-3g")
    res = vq.run_rhf_periodic_multi_k_ewald3d(
        unit,
        basis,
        vq.monkhorst_pack(unit, list(kmesh)),
        exchange_exxdiv=exchange_exxdiv,
    )
    return float(res.energy), bool(res.converged)


def pyscf_krhf(kmeshes, supercell=False):
    payload = {
        "a": A_BOHR,
        "d": D_BOHR,
        "kmeshes": [list(k) for k in kmeshes],
        "supercell": bool(supercell),
    }
    out = subprocess.run(
        [sys.executable, "-c", _PYSCF_CHILD, json.dumps(payload)],
        capture_output=True, text=True,
    )
    for line in out.stdout.splitlines():
        if line.startswith("PYSCF-RESULT:"):
            return json.loads(line[len("PYSCF-RESULT:"):])
    raise RuntimeError("no PySCF result\nSTDERR:\n" + out.stderr[-2000:])


def main() -> int:
    print("=== CCM ≡ SCM-Γ ≡ periodic HF (H2 crystal, a=8 bohr, STO-3G) ===\n")

    # Cubic clusters only: the Γ-supercell FFT-Poisson cost grows fast, and the
    # multi-k energy/cell accessor is consistent for symmetric meshes.
    meshes = [(1, 1, 1), (2, 2, 2)]
    e_ccm = {n: ccm_gamma_supercell(n) for n in meshes}   # cache (computed once)

    # PART A — band-folding identity within vibe-qc (the CCM≡SCM-Γ proof).
    print("PART A — CCM(Γ-supercell)/N  vs  vibe-qc multi-k(unit, k=N), both on "
          "exchange_exxdiv='ewald':")
    a_ok = True
    for nrep in meshes:
        ec, c1 = e_ccm[nrep]
        em, c2 = vibeqc_multik(nrep)
        d_uha = abs(ec - em) * 1e6
        flag = "OK" if d_uha < 50.0 else "FAIL"
        a_ok = a_ok and d_uha < 50.0 and c1 and c2
        print(f"  N={nrep}: CCM/N={ec:.8f}  multi-k={em:.8f}  "
              f"Δ={d_uha:7.2f} µHa  [{flag}]")
    print(f"  → band-folding identity holds: {a_ok} "
          "(CCM is the Γ-point of the supercell ≡ k-folded periodic HF)\n")

    # PART B — external TDL validation against PySCF KRHF.
    print("PART B — TDL convergence vs PySCF KRHF/GDF (exxdiv='ewald'):")
    py = pyscf_krhf(meshes, supercell=True)
    last_delta = None
    for nrep in meshes:
        ec, _ = e_ccm[nrep]
        e_py = py[f"{nrep}|ewald"]
        d_mha = (ec - e_py) * 1e3
        last_delta = abs(d_mha)
        print(f"  N={nrep}: CCM/N={ec:.6f}  PySCF/cell={e_py:.6f}  "
              f"Δ={d_mha:+.3f} mHa")
    print(f"\n  Δ at largest mesh = {last_delta:.3f} mHa "
          "(Ewald-3D FFT vs GDF Coulomb + exxdiv finite-size; shrinks with N)")

    # PART C — the gauge table (#560). PySCF band-folds in EITHER gauge, so the
    # identity is never in doubt; what breaks it is comparing across the two.
    print("\nPART C — exact-exchange q→0 gauge (#560):")
    print("  PySCF, per unit cell:")
    for nrep in meshes:
        e_ew = py[f"{nrep}|ewald"]
        e_no = py[f"{nrep}|None"]
        line = (f"    k={nrep}: exxdiv='ewald' {e_ew:.10f}   "
                f"exxdiv=None {e_no:.10f}   "
                f"gauge step {abs(e_ew - e_no) * 1e3:.3f} mHa")
        sc_key = f"supercell{nrep}|ewald"
        if sc_key in py:
            line += (f"\n      PySCF Γ-supercell/N, same gauges: "
                     f"{py[sc_key]:.10f} / {py[f'supercell{nrep}|None']:.10f} "
                     "(band folding, both)")
        print(line)
    print("  vibe-qc, per unit cell:")
    for nrep in meshes:
        ec, _ = e_ccm[nrep]
        eg, _ = gamma_driver_supercell(nrep)
        print(f"    N={nrep}: Γ-supercell/N corrected {ec:.10f}   "
              f"run_rhf_periodic_gamma_ewald3d/N (molecular-limit) "
              f"{eg:.10f}   gauge step {abs(ec - eg) * 1e3:.3f} mHa")

    # The cross-method Coulomb gap puts the realistic bar at a few mHa for this
    # min-basis system (cf. examples/regression/parity_mgo_rks_bipole_vs_pyscf.py
    # and parity_hexagonal_lattice_vs_pyscf.py, both ≤5 mHa PASS).
    ok = a_ok and last_delta < 5.0
    print(f"\n{'PASS' if ok else 'FAIL'}: CCM(Γ-supercell) ≡ k-folded periodic HF "
          "(vibe-qc, µHa) and ≡ PySCF periodic HF in the TDL (few-mHa method gap).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
