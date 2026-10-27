"""Multi-k Ewald-exchange-split reference generator vs PySCF (out-of-process).

The Phase-3 companion of the option (b) energy-assembly redesign
(2026-06-11): generates external references pinned in
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

OPEN-SHELL TRAP (KUHF/KUKS): PySCF interprets ``cell.spin`` at
multi-k as the TOTAL spin imbalance over the BvK supercell — with
spin=2 at [2,1,1] it fills nelec=(3,1), a mixed state ~141 mHa below
the uniform triplet lattice. The references above pin the per-cell
triplet (vibe-qc multiplicity=3 semantics) via ``mf.nelec = (4, 0)``.

BIPOLE cross-validation at these references (2026-06-11/12, all at
cutoff 7 unless noted): [2,1,1] H₂ box RHF +0.121 mHa (−0.0022 mHa
at cutoff 12); RKS SVWN +0.043 / PBE0 +0.062 mHa; UHF triplet
+0.034 mHa with k-sampling shift agreement 0.038 mHa; [2,1,1] RHF vs the
explicitly-doubled supercell at Γ (matched ω): +0.0001 mHa/cell.

The UKS references use PySCF AFTDF (analytic AO-pair Fourier transforms)
instead of the historical default Gaussian fitting basis. At [2,1,1],
that GDF approximation lowers the PBE0 triplet energy by 0.837 mHa;
it is not a BIPOLE grid/truncation offset. AFTDF allows an explicit
plane-wave convergence check. See Sun et al., JCP 147, 164119 (2017),
https://doi.org/10.1063/1.4998644, Sec. II A (Eq. 16) and Sec. III/Fig. 1.
This uses pure plane waves, not the paper's mixed Gaussian/PW expansion.

Regenerate just the UKS pins with ``--uks-aft-only``; compare against
``--uks-aft-only --pw-mesh 121 --radial 220 --angular 974`` to check
reciprocal-space and XC quadrature convergence. Energies and components
are per cell in hartree; exchange includes the PBE0 factor 0.25.
These secondary references do not certify CRYSTAL timing/accuracy parity.

Run from the dev venv (PySCF is a dev-time dependency only — this is
the CLAUDE.md §10 subprocess pattern; vibeqc never imports pyscf):

    .venv/bin/python examples/regression/bipole_parity/multik_exchange_split_refs_pyscf.py
"""

import argparse
import json

import numpy as np
import pyscf
from pyscf.dft import libxc
from pyscf.pbc import df, dft, gto, scf, tools


def uks_aft_references(pw_mesh=101, radial=150, angular=590):
    """Converge the fully polarized H2 lattice without Gaussian fit error."""
    cell = gto.Cell()
    cell.atom = [["H", (0.0, 0.0, -0.7)], ["H", (0.0, 0.0, 0.7)]]
    cell.a = np.eye(3) * 12.0
    cell.basis = "sto-3g"
    cell.unit = "B"
    cell.spin = 2
    cell.precision = 1e-10
    cell.verbose = 0
    cell.max_memory = 2000
    cell.build()
    cell.mesh = [pw_mesh] * 3
    results = {}
    for kmesh in ([1, 1, 1], [2, 1, 1]):
        kpts = cell.make_kpts(kmesh)
        nk = len(kpts)
        mf = dft.KUKS(cell, kpts)
        mf.with_df = df.AFTDF(cell, kpts)
        mf.with_df.mesh = cell.mesh
        mf.xc = "pbe0"
        mf.exxdiv = "ewald"
        mf.nelec = (2 * nk, 0)
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-8
        mf.max_cycle = 40
        mf.grids = dft.gen_grid.BeckeGrids(cell)
        mf.grids.atom_grid = {"H": (radial, angular)}
        mf.grids.prune = None
        energy = float(mf.kernel())
        density = np.asarray(mf.make_rdm1())
        overlap = np.asarray(mf.get_ovlp())
        hcore = np.asarray(mf.get_hcore())
        vj, vk = mf.get_jk(cell, density, kpts=kpts)
        veff = mf.get_veff(cell, density)
        total_density = density.sum(axis=0)
        e_one = np.einsum("kij,kji->", total_density, hcore).real / nk
        e_j = 0.5 * np.einsum("kij,kji->", total_density, vj.sum(axis=0)).real / nk
        e_k = -0.5 * 0.25 * np.einsum("skij,skji->", density, vk).real / nk
        e_xc = float(veff.exc) - e_k
        e_nuc = float(cell.energy_nuc())
        spin_counts = np.einsum("skij,kji->s", density, overlap).real / nk
        if not mf.converged:
            raise RuntimeError(f"UKS AFT reference did not converge: {kmesh}")
        np.testing.assert_allclose(spin_counts, [2.0, 0.0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(
            energy, e_one + e_j + e_k + e_xc + e_nuc, rtol=0, atol=1e-9
        )
        results[f"h2_kuks_pbe0_triplet_{'x'.join(map(str, kmesh))}"] = {
            "e_per_cell": energy,
            "e_one": float(e_one),
            "e_coulomb": float(e_j),
            "e_hf_exchange": float(e_k),
            "e_xc": float(e_xc),
            "e_nuclear": e_nuc,
            "spin_electrons_per_cell": spin_counts.tolist(),
            "converged": bool(mf.converged),
            "backend": "AFTDF",
            "pyscf_version": pyscf.__version__,
            "libxc_version": libxc.__version__,
            "cell_precision": cell.precision,
            "pw_mesh": list(map(int, cell.mesh)),
            "xc_atom_grid": [radial, angular],
            "xc_prune": None,
            "xc_grid_points": len(mf.grids.weights),
            "exxdiv": mf.exxdiv,
            "conv_tol": mf.conv_tol,
            "conv_tol_grad": mf.conv_tol_grad,
        }
    return results


def all_references(pw_mesh=101, radial=150, angular=590):
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

    results.update(uks_aft_references(pw_mesh, radial, angular))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uks-aft-only", action="store_true")
    parser.add_argument("--pw-mesh", type=int, default=101)
    parser.add_argument("--radial", type=int, default=150)
    parser.add_argument("--angular", type=int, default=590)
    args = parser.parse_args()
    if min(args.pw_mesh, args.radial, args.angular) <= 0:
        parser.error("mesh and quadrature sizes must be positive")
    generate = uks_aft_references if args.uks_aft_only else all_references
    print(json.dumps(generate(args.pw_mesh, args.radial, args.angular), indent=2))
