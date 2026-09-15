"""Probe vibe-qc vs PySCF per-intermediate energy parity.

Prints, for each cell of the parity matrix, the per-piece difference
(E_nuc, E_1e, E_coulomb, E_exchange, E_xc, E_total, max MO-eigenvalue)
between vibe-qc's decomposition (vibeqc.parity) and PySCF's. Used to
set data-driven tolerances for tests/test_parity_hf_dft.py and to
re-calibrate them when new cells are added to the matrix.

Run modes:

* ``python -m scripts.parity_pyscf_probe`` — closed-shell diagonal
  (H2O / H2CO × def2-svp/tzvp × RHF/RKS-PBE/RKS-B3LYP), the
  original surface. Fast (~1 min).
* ``python -m scripts.parity_pyscf_probe --full`` — the full
  ``tests/test_parity_hf_dft.py::_PARITY_CASES`` matrix, including
  open-shell def2-tzvp radicals (NO·, CN·, CH3OO·, OH·). Slow
  (~15-20 min). Emits a "worst observed by class" summary at the
  end suitable for tightening the per-piece tolerances.

PySCF is imported directly here — this is a dev-time probe script, not
runtime code (same status as tests/, per CLAUDE.md § 10).
"""
from __future__ import annotations
import argparse
import numpy as np
import vibeqc as vq
from vibeqc.parity import (decompose_energy_rhf, decompose_energy_rks,
                           decompose_energy_uhf, decompose_energy_uks)

ANG = 1.0 / 0.529177210903
H2O = [(8, [0.0, 0.0, 0.0]),
        (1, [0.0, 0.793353 * ANG, -0.613510 * ANG]),
        (1, [0.0, -0.793353 * ANG, -0.613510 * ANG])]
H2CO = [(6, [0.0, 0.0, 0.0]),
        (8, [0.0, 0.0, 1.205 * ANG]),
        (1, [0.0, 0.943 * ANG, -0.587 * ANG]),
        (1, [0.0, -0.943 * ANG, -0.587 * ANG])]
GEOMS = {"H2O": H2O, "H2CO": H2CO}

# vibe-qc functional name -> pyscf xc string
# vibe-qc's "B3LYP" = VWN5 variant (ORCA convention) → PySCF "b3lyp5".
PYSCF_XC = {"PBE": "pbe,pbe", "B3LYP": "b3lyp5", "PBE0": "pbe0"}


def vibeqc_decompose(atoms, basis_name, method):
    mol = vq.Molecule([vq.Atom(z, list(xyz)) for z, xyz in atoms])
    basis = vq.BasisSet(mol, basis_name)
    if method == "RHF":
        o = vq.RHFOptions()
        o.conv_tol_energy = 1e-12
        o.conv_tol_grad = 1e-9
        r = vq.run_rhf(mol, basis, o)
        return decompose_energy_rhf(mol, basis, r)
    else:  # RKS
        o = vq.RKSOptions()
        o.functional = method.split("-", 1)[1]
        o.conv_tol_energy = 1e-12
        o.conv_tol_grad = 1e-9
        r = vq.run_rks(mol, basis, o)
        return decompose_energy_rks(mol, basis, r, o.grid)


def pyscf_decompose(atoms, basis_name, method):
    from pyscf import gto, scf, dft
    mol = gto.M(
        atom=[[z, tuple(xyz)] for z, xyz in atoms],
        basis=basis_name, unit="Bohr", verbose=0)
    if method == "RHF":
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()
        hcore = mf.get_hcore()
        j, k = mf.get_jk(mol, dm)
        e_nuc = mol.energy_nuc()
        e_1e = np.einsum("ij,ji->", hcore, dm)
        e_coulomb = 0.5 * np.einsum("ij,ji->", j, dm)
        e_exchange_hf = -0.25 * np.einsum("ij,ji->", k, dm)
        alpha_hf, e_xc = 1.0, 0.0
        e_exchange = e_exchange_hf
        mo = mf.mo_energy
    else:
        mf = dft.RKS(mol)
        mf.xc = PYSCF_XC[method.split("-", 1)[1]]
        mf.grids.level = 5
        mf.conv_tol = 1e-12
        mf.kernel()
        dm = mf.make_rdm1()
        hcore = mf.get_hcore()
        e_nuc = mol.energy_nuc()
        e_1e = np.einsum("ij,ji->", hcore, dm)
        j = mf.get_j(mol, dm)
        e_coulomb = 0.5 * np.einsum("ij,ji->", j, dm)
        ni = mf._numint
        # Pure libxc XC energy (NOT scf_summary['exc'], which folds the
        # hybrid HF-exchange into exc for hybrids).
        alpha_hf = float(ni.hybrid_coeff(mf.xc))
        e_xc = float(ni.nr_rks(mol, mf.grids, mf.xc, dm)[1])
        if alpha_hf != 0.0:
            k = mf.get_k(mol, dm)
            e_exchange_hf = -0.25 * np.einsum("ij,ji->", k, dm)
        else:
            e_exchange_hf = 0.0
        e_exchange = alpha_hf * e_exchange_hf
        mo = mf.mo_energy
    e_total = e_nuc + e_1e + e_coulomb + e_exchange + e_xc
    return {
        "e_nuc": float(e_nuc), "e_1e": float(e_1e),
        "e_coulomb": float(e_coulomb), "e_exchange": float(e_exchange),
        "e_xc": float(e_xc), "e_total": float(e_total),
        "e_total_reported": float(mf.e_tot),
        "mo_energies": np.asarray(mo, dtype=float),
        "alpha_hf": float(alpha_hf),
    }


def _classify(method: str, aux: str, sysname: str, basis: str) -> str:
    """Return the (method_class, path, hard?) bucket id for the worst-observed table."""
    is_dft = method.startswith(("RKS-", "UKS-"))
    klass = "dft" if is_dft else "hf"
    path = "df" if aux else "direct"
    is_hard = (sysname, basis) in {
        ("NO", "def2-tzvp"), ("CN", "def2-tzvp"),
        ("CH3OO", "def2-tzvp"), ("OH", "def2-tzvp"),
    }
    return f"{klass}_{path}{'_hard' if is_hard else ''}"


def main():
    ap = argparse.ArgumentParser(description="vibe-qc vs PySCF parity probe")
    ap.add_argument("--full", action="store_true",
                    help="probe the full tests/test_parity_hf_dft.py::_PARITY_CASES "
                         "matrix (open-shell radicals included). Default: only "
                         "the closed-shell H2O/H2CO diagonal.")
    args = ap.parse_args()

    if args.full:
        # Delegate to the test module's helpers + case list — keeps the
        # probe's coverage in lock-step with the actual parity tests.
        from tests.test_parity_hf_dft import (
            _GEOMS, _PARITY_CASES, _vibeqc_decompose, _pyscf_decompose,
        )
        cells = [(s, b, m, ch, sp, aux) for s, b, m, ch, sp, aux in _PARITY_CASES
                 if s in _GEOMS]
    else:
        cells = [(s, b, m, 0, 0, "")
                 for s in ("H2O", "H2CO")
                 for b in ("def2-svp", "def2-tzvp")
                 for m in ("RHF", "RKS-PBE", "RKS-B3LYP")]

    hdr = (f"{'cell':40s} {'dE_nuc':>10s} {'dE_1e':>10s} "
           f"{'dE_coul':>10s} {'dE_exch':>10s} {'dE_xc':>10s} "
           f"{'dE_tot':>10s} {'dMO_max':>10s}")
    print(hdr)
    print("-" * len(hdr))

    worst: dict = {}
    def _upd(key, val):
        worst[key] = max(worst.get(key, 0.0), abs(float(val)))

    for cell in cells:
        if len(cell) == 3:
            sysname, basis, method = cell
            charge, spin, aux = 0, 0, ""
        else:
            sysname, basis, method, charge, spin, aux = cell
        atoms = (GEOMS.get(sysname)
                 or __import__("tests.test_parity_hf_dft", fromlist=["_GEOMS"])._GEOMS[sysname])

        if args.full:
            from tests.test_parity_hf_dft import _vibeqc_decompose, _pyscf_decompose
            mult = spin + 1
            try:
                vqd = _vibeqc_decompose(atoms, basis, method, mult=mult,
                                        aux_basis=aux, sysname=sysname)
                psd = _pyscf_decompose(atoms, basis, method,
                                       charge=charge, spin=spin, aux_basis=aux)
            except Exception as exc:  # noqa: BLE001
                print(f"{sysname}/{basis}/{method}  [ERR: {type(exc).__name__}: {exc}]")
                continue
        else:
            vqd = vibeqc_decompose(atoms, basis, method)
            psd = pyscf_decompose(atoms, basis, method)
            ps_selfres = psd["e_total"] - psd["e_total_reported"]
            if abs(ps_selfres) > 1e-6:
                print(f"  WARN PySCF self-res {ps_selfres:.1e} for {sysname}/{basis}/{method}")

        d_nuc = vqd["e_nuc"] - psd["e_nuc"]
        d_1e = vqd["e_1e"] - psd["e_1e"]
        d_coul = vqd["e_coulomb"] - psd["e_coulomb"]
        d_exch = vqd["e_exchange"] - psd["e_exchange"]
        d_xc = vqd["e_xc"] - psd["e_xc"]
        d_tot = vqd["e_total"] - psd["e_total"]
        if "mo_energies" in vqd:
            nmo = min(len(vqd["mo_energies"]), len(psd["mo_energies"]))
            d_mo = float(np.abs(vqd["mo_energies"][:nmo]
                                - psd["mo_energies"][:nmo]).max())
        else:
            d_mo = 0.0
            for k in ("mo_energies_alpha", "mo_energies_beta"):
                nmo = min(len(vqd[k]), len(psd[k]))
                if nmo:
                    d_mo = max(d_mo, float(np.abs(
                        np.asarray(vqd[k][:nmo]) - np.asarray(psd[k][:nmo])
                    ).max()))

        klass = _classify(method, aux, sysname, basis)
        _upd(f"{klass}:e_nuc", d_nuc)
        _upd(f"{klass}:e_1e", d_1e)
        _upd(f"{klass}:e_coulomb", d_coul)
        _upd(f"{klass}:e_exchange", d_exch)
        _upd(f"{klass}:e_xc", d_xc)
        _upd(f"{klass}:e_total", d_tot)
        _upd(f"{klass}:mo", d_mo)

        label = f"{sysname}/{basis}/{method}" + ("/DF" if aux else "")
        print(f"{label:40s} {d_nuc:>10.2e} {d_1e:>10.2e} {d_coul:>10.2e} "
              f"{d_exch:>10.2e} {d_xc:>10.2e} {d_tot:>10.2e} {d_mo:>10.2e}")

    if args.full and worst:
        print()
        print("Worst observed by (method, path, hard) bucket:")
        for key in sorted(worst):
            print(f"  {key:40s}  {worst[key]:.3e}")
        print()
        print("# Suggested tolerances = ~2x worst observed:")
        print("#  e_nuc:      1e-12  (analytic; worst observed ~1e-14)")
        print("#  e_1e:       1e-5   (hard-cell floor ~1e-5)")
        print("#  e_coulomb:  1e-5   (hard-cell floor ~1e-5)")
        print("#  e_exchange: 2e-6   (hard-cell floor ~1.2e-6)")
        print("#  e_xc:       2e-6   (hard-cell floor ~1.4e-6)")
        print("#  e_total:    2e-6   (hard-cell floor ~8e-7)")
        print("#  mo:         2e-5   (hard-cell floor ~1.5e-5)")
        print("# Further tightening of e_xc is gated on Lebedev angular grids in vibe-qc.")


if __name__ == "__main__":
    main()
