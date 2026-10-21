#!/usr/bin/env python3
"""Reproduce the IAO guide's literature comparison and open-shell examples.

Knizia, JCTC 9, 4834 (2013), doi:10.1021/ct400687b, Table 1.
The cc-pVTZ comparison uses the paper's Huzinaga MINI row (footnote c).
The other orbital bases use its MINAO rows (footnote a) for context only.
Geometries below are explicit fixed test geometries, not the paper's
unavailable optimized coordinates. Do not interpret differences as exact
same-input implementation errors. All calculations use vibe-qc itself.

Run with --output-dir pointing outside the source checkout. The ordinary
output writer preserves .out, .system, population and citation files; pass
--qvf to include the compatible charge field and full IAO vendor payload.
With --pyscf, optionally repeat the SCF, integrals and IAO construction in
PySCF using the same Gaussian basis files and symmetric orthogonalization.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import vibeqc as vq

ANGSTROM_TO_BOHR = 1.8897261254578281


def molecules():
    """Return atom order, charge, multiplicity and selected displayed bonds."""
    d = 1.087 * ANGSTROM_TO_BOHR / np.sqrt(3.0)
    return {
        "ch4": ([(6, [0., 0., 0.]), (1, [d, d, d]), (1, [d, -d, -d]),
                 (1, [-d, d, -d]), (1, [-d, -d, d])], 0, 1, [(0, 1)]),
        "hcn": ([(1, [0., 0., -1.0655 * ANGSTROM_TO_BOHR]),
                 (6, [0., 0., 0.]),
                 (7, [0., 0., 1.153 * ANGSTROM_TO_BOHR])], 0, 1, [(0, 1), (1, 2)]),
        "h2": ([(1, [0., 0., 0.]), (1, [0., 0., 1.4])], 0, 1, [(0, 1)]),
        "h2plus": ([(1, [0., 0., 0.]), (1, [0., 0., 2.])], 1, 2, [(0, 1)]),
        "oh": ([(8, [0., 0., 0.]), (1, [0., 0., 1.83])], 0, 2, [(0, 1)]),
    }


# Printed charges in e, in the atom order above. These are rounded paper
# values; e.g. the printed CH4 row need not sum to exactly zero.
PUBLISHED = {
    ("ch4", "def2-svp"): ("MINAO/a", [-0.49, 0.12, 0.12, 0.12, 0.12]),
    ("ch4", "def2-tzvpp"): ("MINAO/a", [-0.52, 0.13, 0.13, 0.13, 0.13]),
    ("ch4", "cc-pvtz"): ("MINI/c", [-0.49, 0.12, 0.12, 0.12, 0.12]),
    ("hcn", "def2-svp"): ("MINAO/a", [0.21, -0.01, -0.20]),
    ("hcn", "def2-tzvpp"): ("MINAO/a", [0.22, -0.01, -0.21]),
    ("hcn", "cc-pvtz"): ("MINI/c", [0.22, -0.03, -0.19]),
}


def pyscf_reference(atoms, charge, multiplicity, basis, output):
    """Independent optional SCF/integral/IAO oracle, with matched basis data."""
    from pyscf import gto, scf
    from pyscf.gto.basis import parse_gaussian
    from pyscf.lo import iao, orth
    from pyscf.data.elements import ELEMENTS

    library = Path(vq.__file__).parent / "basis_library" / "basis"
    elements = {ELEMENTS[z] for z, _ in atoms}
    orbital_basis = {e: parse_gaussian.load(str(library / f"{basis}.g94"), e)
                     for e in elements}
    minimal_basis = {e: parse_gaussian.load(str(library / "mini.g94"), e)
                     for e in elements}
    mol = gto.M(atom=atoms, unit="Bohr", basis=orbital_basis, cart=False,
                charge=charge, spin=multiplicity-1, verbose=4, output=str(output))
    restricted = multiplicity == 1
    mf = scf.RHF(mol) if restricted else scf.UHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 100
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("Independent PySCF SCF did not converge")
    overlap = mol.intor_symmetric("int1e_ovlp")
    ref = iao.reference_mol(mol, minimal_basis)
    labels = np.array([row[0] for row in ref.ao_labels(fmt=False)])
    occupied = ([mf.mo_coeff[:, mf.mo_occ > 0]] * 2 if restricted else
                [c[:, f > 0] for c, f in zip(mf.mo_coeff, mf.mo_occ)])
    densities = []
    for c in occupied:
        raw = iao.iao(mol, c, minao=minimal_basis)
        a = raw @ orth.lowdin(raw.conj().T @ overlap @ raw)
        x = a.conj().T @ overlap @ c
        densities.append(x @ x.conj().T)
    populations = [np.bincount(labels, weights=d.diagonal().real,
                               minlength=len(atoms)) for d in densities]
    bonds = np.zeros((len(atoms), len(atoms)))
    for i in range(len(atoms)):
        for j in range(i):
            block = np.ix_(labels == i, labels == j)
            bonds[i, j] = bonds[j, i] = 2 * sum(
                np.sum(np.abs(d[block]) ** 2) for d in densities)
    return (float(mf.e_tot), mol.atom_charges() - sum(populations),
            populations[0] - populations[1], bonds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qvf", action="store_true", help="Also write QVF archives")
    parser.add_argument("--pyscf", action="store_true", help="Run the optional independent oracle")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = molecules()
    cases = list(PUBLISHED) + [(name, "def2-svp") for name in ("h2", "h2plus", "oh")]
    reports = []
    for name, basis in cases:
        atoms, charge, multiplicity, pairs = inputs[name]
        mol = vq.Molecule([vq.Atom(z, xyz) for z, xyz in atoms],
                          charge=charge, multiplicity=multiplicity)
        restricted = multiplicity == 1
        options = vq.RHFOptions() if restricted else vq.UHFOptions()
        options.conv_tol_energy = 1e-10
        options.conv_tol_grad = 1e-8
        options.max_iter = 100
        result = vq.run_job(
            mol, basis=basis, method="rhf" if restricted else "uhf",
            output=args.output_dir / f"{name}-{basis}", name_molecule=False,
            rhf_options=options if restricted else None,
            uhf_options=None if restricted else options,
            density_fit=False, iao_analysis=True, localize=False,
            output_qvf=args.qvf, write_population_file=True,
            write_molden_file=False, structured_log=True, verbose=0,
        )
        analysis = result.iao_analysis
        if not analysis.available:
            raise RuntimeError(f"{name}/{basis}: {analysis.unavailable_reason}")
        if args.pyscf:
            energy, charges, spins, bonds = pyscf_reference(
                atoms, charge, multiplicity, basis,
                args.output_dir / f"{name}-{basis}.pyscf.out")
            spin_error = (0. if restricted else
                          float(np.max(np.abs(analysis.spin_populations - spins))))
            print(f"ORACLE {name}/{basis}: energy={abs(result.energy-energy):.3e} Ha "
                  f"charge={np.max(np.abs(analysis.charges-charges)):.3e} e "
                  f"spin={spin_error:.3e} e "
                  f"bond={np.max(np.abs(analysis.bond_orders-bonds)):.3e}", flush=True)
        reports.append((name, basis, analysis, pairs))

    print("\nTable 1 charge comparisons (e); delta = vibe-qc MINI minus printed paper value")
    print("molecule/basis atom paper-reference paper vibe-qc delta")
    for name, basis, analysis, pairs in reports:
        if (name, basis) in PUBLISHED:
            reference, values = PUBLISHED[name, basis]
            for atom, (computed, published) in enumerate(zip(analysis.charges, values)):
                print(f"{name}/{basis} {atom} {reference} {published:+.2f} "
                      f"{computed:+.8f} {computed-published:+.8f}")
        print(f"RESULT {name}/{basis}: charges={analysis.charges.tolist()} "
              f"spins={None if analysis.spin_populations is None else analysis.spin_populations.tolist()}")
        print(f"BONDS {name}/{basis}: " + ", ".join(
            f"{i}-{j}={analysis.bond_orders[i, j]:.8f}" for i, j in pairs))
        print(f"RESIDUAL {name}/{basis}: " + ", ".join(
            f"{key}={value:.3e}" for key, value in analysis.diagnostics.items()
            if key.endswith("residual")))


if __name__ == "__main__":
    main()
