"""Multi-k Ewald-exchange-split reference generator vs PySCF (out-of-process).

The Phase-3 companion of the option (b) energy-assembly redesign
(2026-06-11): generates the PySCF 2.13 references pinned in
``tests/test_bipole_fock_ewald_exchange.py`` for the multi-k q ≠ 0
LR-exchange channels —

* ``tools.pbc.madelung(cell, kpts)`` — the BvK-supercell probe-charge
  Madelung constants (vibe-qc: ``probe_charge_madelung_supercell``)
  for the H₂ 12-bohr box at [2,1,1] / [2,2,2] and MgO fcc primitive
  at [2,2,2]. PySCF applies this single constant at every k in
  ``_ewald_exxdiv_for_G0`` — same convention as the BIPOLE multi-k
  correction ``(ξ_M^sc − π/(V_sc·ω²))·S(k)D(k)S(k)``.
* ``KRHF(cell, kpts).density_fit()`` with ``exxdiv='ewald'`` /
  conv_tol 1e-10 — the H₂-box multi-k total-energy pins.

Recorded values (PySCF 2.13, 2026-06-11):

  h2_box12 [2,1,1]: madelung 0.150486817543, E_KRHF −1.1201398824
  h2_box12 [2,2,2]: madelung 0.118220728315, E_KRHF −1.1174569467
  MgO      [2,2,2]: madelung 0.288147805800 (= ξ_Γ/2: cubic N^{-1/3})
  MgO       Γ     : madelung 0.576295611599 (the Phase-1/2 pin)
  h2 KRKS  [2,1,1]: slater,vwn5 −1.1212703585, pbe0 −1.1552192298
  h2 KUHF triplet : Γ −0.5358505084 (= the Phase-4b pin, exact),
                    [2,1,1] −0.5337291457
  h2 KUKS pbe0 trip: Γ −0.5420559972, [2,1,1] −0.5415218581

OPEN-SHELL TRAP (KUHF/KUKS): PySCF interprets ``cell.spin`` at
multi-k as the TOTAL spin imbalance over the BvK supercell — with
spin=2 at [2,1,1] it fills nelec=(3,1), a mixed state ~141 mHa below
the uniform triplet lattice. The references above pin the per-cell
triplet (vibe-qc multiplicity=3 semantics) via ``mf.nelec = (4, 0)``.

BIPOLE cross-validation at these references (2026-06-11/12, all at
cutoff 7 unless noted): [2,1,1] H₂ box RHF +0.121 mHa (−0.0022 mHa
at cutoff 12); RKS SVWN +0.043 / PBE0 +0.062 mHa; UHF triplet
+0.034 mHa with k-sampling shift agreement 0.038 mHa; UKS PBE0
triplet +0.77 mHa (the documented Γ grid/truncation offset, mesh-
independent) with shift agreement 0.013 mHa; [2,1,1] RHF vs the
explicitly-doubled supercell at Γ (matched ω): +0.0001 mHa/cell.

Run from the dev venv (PySCF is a dev-time dependency only — this is
the CLAUDE.md §10 subprocess pattern; vibeqc never imports pyscf):

    .venv/bin/python examples/regression/bipole_parity/multik_exchange_split_refs_pyscf.py
"""

import json

import numpy as np
from pyscf.pbc import gto, scf, tools

results = {}

# ---- H2 in a 12-bohr cubic box, STO-3G (the Phase 2 validation cell).
# GEOMETRY MATTERS: exactly 1.4 BOHR separation — the vibe-qc test
# fixtures' geometry. An earlier draft used 0.74 Å (= 1.39839 bohr);
# on the steep repulsive triplet curve that 0.0016-bohr offset shifts
# E by ~1 mHa and masquerades as a truncation error.
h2 = gto.Cell()
h2.atom = [["H", (0.0, 0.0, -0.7)], ["H", (0.0, 0.0, 0.7)]]
h2.a = np.eye(3) * 12.0
h2.basis = "sto-3g"
h2.unit = "B"
h2.verbose = 0
h2.build()

for kmesh in ([2, 1, 1], [2, 2, 2]):
    kpts = h2.make_kpts(kmesh)
    xi = tools.pbc.madelung(h2, kpts)
    mf = scf.KRHF(h2, kpts).density_fit()
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-10
    e = mf.kernel()
    results[f"h2_box12_{'x'.join(map(str, kmesh))}"] = {
        "madelung": float(xi),
        "e_krhf_per_cell": float(e),
        "converged": bool(mf.converged),
    }

# ---- MgO fcc primitive, STO-3G ---------------------------------------
a_ang = 4.21
mgo = gto.Cell()
mgo.atom = f"Mg 0 0 0; O {a_ang / 2} {a_ang / 2} {a_ang / 2}"
mgo.a = (a_ang / 2.0) * (np.ones((3, 3)) - np.eye(3))
mgo.basis = "sto-3g"
mgo.unit = "A"
mgo.verbose = 0
mgo.build()

kpts = mgo.make_kpts([2, 2, 2])
results["mgo_222"] = {"madelung": float(tools.pbc.madelung(mgo, kpts))}
results["mgo_gamma"] = {
    "madelung": float(tools.pbc.madelung(mgo, np.zeros((1, 3))))
}

# ---- KRKS (closed shell) + KUHF/KUKS (uniform triplet lattice) --------
from pyscf.pbc import dft  # noqa: E402

kpts_211 = h2.make_kpts([2, 1, 1])
for xc in ("slater,vwn5", "pbe0"):
    mf = dft.KRKS(h2, kpts_211).density_fit()
    mf.xc = xc
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-10
    e = mf.kernel()
    results[f"h2_krks_{xc.replace(',', '_')}_2x1x1"] = {
        "e_per_cell": float(e),
        "converged": bool(mf.converged),
    }

h2t = h2.copy()
h2t.spin = 2  # placeholder; the REAL per-cell triplet is set via nelec
h2t.build()
for kmesh in ([1, 1, 1], [2, 1, 1]):
    kpts = h2t.make_kpts(kmesh)
    n_k = len(kpts)
    mf = scf.KUHF(h2t, kpts).density_fit()
    mf.exxdiv = "ewald"
    mf.conv_tol = 1e-10
    # Uniform triplet lattice: 2 alpha / 0 beta PER CELL (see the
    # open-shell trap note in the module docstring).
    mf.nelec = (2 * n_k, 0)
    e = mf.kernel()
    results[f"h2_kuhf_triplet_{'x'.join(map(str, kmesh))}"] = {
        "e_per_cell": float(e),
        "converged": bool(mf.converged),
    }
    mk = dft.KUKS(h2t, kpts).density_fit()
    mk.xc = "pbe0"
    mk.exxdiv = "ewald"
    mk.conv_tol = 1e-10
    mk.nelec = (2 * n_k, 0)
    e = mk.kernel()
    results[f"h2_kuks_pbe0_triplet_{'x'.join(map(str, kmesh))}"] = {
        "e_per_cell": float(e),
        "converged": bool(mk.converged),
    }

print(json.dumps(results, indent=2))
