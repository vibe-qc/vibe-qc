"""Print sto-3g basis exponents/contractions for Mg and O in both
vibe-qc and PySCF, to confirm they're using the *same* basis.

Also: re-do MgO S eigenvalue compare with a much larger cutoff_bohr
for vibe-qc (24, 36, 48) to see whether the 7e-3 eigenvalue mismatch
shrinks — if it does, the bug is just lattice truncation.
"""
from __future__ import annotations

import numpy as np

import vibeqc as vq
from vibeqc.periodic_v_ne import compute_nuclear_lattice_dispatch

from pyscf.pbc import gto as pbc_gto


ANG2BOHR = 1.0 / 0.529177210903
A_ANG = 4.211


def dump_pyscf_basis():
    mol = pbc_gto.M(
        atom="Mg 0 0 0; O 4 0 0",
        a=[[10, 0, 0], [0, 10, 0], [0, 0, 10]],
        basis="sto-3g",
        unit="A",
        verbose=0,
    )
    print("PySCF sto-3g raw bas data:")
    print("  Mg ENV (bas data):")
    for ish in range(mol.nbas):
        atom = mol.bas_atom(ish)
        if atom != 0:
            continue
        l = mol.bas_angular(ish)
        nprim = mol.bas_nprim(ish)
        nctr = mol.bas_nctr(ish)
        exps = mol.bas_exp(ish)
        ctrs = mol._libcint_ctr_coeff(ish)
        print(f"    sh{ish}: l={l} nprim={nprim} nctr={nctr}")
        print(f"      exps:  {exps}")
        print(f"      ctrs:  {np.asarray(ctrs).flatten()}")
    print("  O ENV (bas data):")
    for ish in range(mol.nbas):
        atom = mol.bas_atom(ish)
        if atom != 1:
            continue
        l = mol.bas_angular(ish)
        nprim = mol.bas_nprim(ish)
        nctr = mol.bas_nctr(ish)
        exps = mol.bas_exp(ish)
        ctrs = mol._libcint_ctr_coeff(ish)
        print(f"    sh{ish}: l={l} nprim={nprim} nctr={nctr}")
        print(f"      exps:  {exps}")
        print(f"      ctrs:  {np.asarray(ctrs).flatten()}")


def dump_vibeqc_basis():
    """vibe-qc reads basis from libint .g94 files."""
    print("vibe-qc sto-3g via libint .g94:")
    import os
    libint_path = os.environ.get("LIBINT_DATA_PATH", "")
    g94 = os.path.join(libint_path, "basis", "sto-3g.g94")
    if not os.path.exists(g94):
        print(f"  not found at {g94}")
        return
    print(f"  source: {g94}")
    in_mg = in_o = False
    with open(g94) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("****"):
                in_mg = in_o = False
                continue
            tok = line.split()
            if len(tok) >= 1 and tok[0] in ("MG", "Mg"):
                in_mg = True
                in_o = False
                print("  Mg:")
                continue
            if len(tok) >= 1 and tok[0] in ("O",):
                in_mg = False
                in_o = True
                print("  O:")
                continue
            if in_mg or in_o:
                print(f"    {line}")


def compare_S_at_cutoff(cutoff_bohr):
    a = A_ANG * ANG2BOHR
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    cell_atoms = []
    for fx, fy, fz in mg_frac:
        cell_atoms.append(vq.Atom(12, [fx * a, fy * a, fz * a]))
    for fx, fy, fz in o_frac:
        cell_atoms.append(vq.Atom(8, [fx * a, fy * a, fz * a]))
    sys_p = vq.PeriodicSystem(3, np.diag([a, a, a]), cell_atoms)
    basis = vq.BasisSet(sys_p.unit_cell_molecule(), "sto-3g")

    lat_opts = vq.LatticeSumOptions()
    lat_opts.coulomb_method = vq.CoulombMethod.EWALD_3D
    lat_opts.cutoff_bohr = cutoff_bohr
    lat_opts.nuclear_cutoff_bohr = max(25.0, 2.0 * cutoff_bohr)
    k_gamma = np.zeros(3)
    S_lat = vq.compute_overlap_lattice(basis, sys_p, lat_opts)
    S = np.real(vq.bloch_sum(S_lat, k_gamma))
    S = 0.5 * (S + S.T)
    return np.sort(np.linalg.eigvalsh(S))


def main():
    dump_pyscf_basis()
    print()
    dump_vibeqc_basis()

    # Reference PySCF eigenvalues
    atom_lines = []
    a_ang = A_ANG
    mg_frac = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    o_frac = [(0.5, 0.5, 0.5), (0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    for fx, fy, fz in mg_frac:
        atom_lines.append(f"Mg {fx*a_ang:.6f} {fy*a_ang:.6f} {fz*a_ang:.6f}")
    for fx, fy, fz in o_frac:
        atom_lines.append(f"O  {fx*a_ang:.6f} {fy*a_ang:.6f} {fz*a_ang:.6f}")
    cell = pbc_gto.M(atom="; ".join(atom_lines),
                     a=[[a_ang, 0, 0], [0, a_ang, 0], [0, 0, a_ang]],
                     basis="sto-3g", unit="A", verbose=0,
                     mesh=[31, 31, 31], precision=1e-8)
    Sp = cell.pbc_intor("int1e_ovlp", hermi=1)
    ev_p = np.sort(np.linalg.eigvalsh(Sp))
    print()
    print(f"PySCF S eigenvalues (selected):")
    print(f"   first 6: {ev_p[:6]}")
    print(f"   last  6: {ev_p[-6:]}")

    print()
    print("vibe-qc S eigenvalues vs cutoff_bohr:")
    for c in (12.0, 18.0, 24.0, 36.0):
        ev_v = compare_S_at_cutoff(c)
        diff = ev_v - ev_p
        max_ad = float(np.max(np.abs(diff)))
        print(f"   cutoff={c:>5.1f}  max|Δλ|={max_ad:.3e}  "
              f"first6={ev_v[:6]}  last6={ev_v[-6:]}")


if __name__ == "__main__":
    main()
