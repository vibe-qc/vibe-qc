"""Water, one SCF and every charge, bond-order and response property side by side.

System: H2O/6-31G* at the fixed geometry specified below (bohr).
Method: ``run_job(method="rhf", basis="6-31g*", iao_analysis=True,
localize="ibo")``. ``iao_analysis=True`` attaches ``result.iao_analysis``
(intrinsic-atomic-orbital charges and IAO-Wiberg bond orders, MINAO
reference ``mini``); ``localize="ibo"`` writes intrinsic bond orbitals into
the QVF for vibe-view. The script then builds a ``BasisSet`` and calls the
library functions on the same converged result: ``mulliken_charges``,
``loewdin_charges``, ``hirshfeld_charges``, ``mayer_bond_orders``,
``vibeqc.bond_analysis.wiberg_bond_orders``, ``dipole_moment`` and the
coupled-perturbed-HF static polarizability ``dipole_polarizability_rhf``.

Files (next to this script):
    input-h2o-population-analysis.out             SCF, charges, Mayer bond orders, dipole
    input-h2o-population-analysis.population.txt  the population dump (and .json)
    input-h2o-population-analysis.qvf             wavefunction with IBO localised orbitals
    plus .molden / .system / .xyz sidecars

Historical output (vibe-qc 0.17.x development build): oxygen charges Mulliken -0.888, Loewdin -0.657, Hirshfeld -0.360,
IAO -0.642 e (the spread between schemes is the lesson, the sum is zero
for each); O-H bond orders Mayer 0.774, Wiberg 0.903, IAO-Wiberg 0.897;
dipole moment 0.8871 e bohr = 2.255 D; static polarizability diagonal
(2.631, 7.838, 5.352) bohr^3 in the molecular frame, isotropic 5.274
bohr^3. These numbers illustrate the model differences; they are not
current-build acceptance thresholds. Check SCF convergence and electron
bookkeeping before interpreting the properties. This geometry differs from
the introductory properties tutorial, so their numerical values need not match.
"""

from pathlib import Path

import numpy as np

import vibeqc as vq
from vibeqc.bond_analysis import wiberg_bond_orders

HERE = Path(__file__).parent
STEM = Path(__file__).stem

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])
BASIS = "6-31g*"
SYMBOLS = ["O", "H", "H"]

result = vq.run_job(mol, basis=BASIS, method="rhf", iao_analysis=True, localize="ibo",
                    output=str(HERE / STEM), write_molden_file=True)
if not result.converged:
    raise RuntimeError("Water RHF did not converge; inspect the job's .out file")
basis = vq.BasisSet(mol, BASIS)

mulliken = vq.mulliken_charges(result, basis, mol)
loewdin = vq.loewdin_charges(result, basis, mol)
hirshfeld = vq.hirshfeld_charges(result, basis, mol)
iao = result.iao_analysis
mayer = vq.mayer_bond_orders(result, basis, mol)
wiberg = wiberg_bond_orders(result, basis, mol)
dipole = vq.dipole_moment(result, basis, mol)
alpha = np.asarray(vq.dipole_polarizability_rhf(result, basis, mol))

print(f"\n  RHF/{BASIS} water, E = {result.energy:.8f} Ha")
print("\n  atomic charges (e)")
print(f"  {'atom':5s} {'Mulliken':>9s} {'Loewdin':>9s} {'Hirshfeld':>10s} {'IAO':>8s}")
for i, sym in enumerate(SYMBOLS):
    print(f"  {sym + str(i + 1):5s} {mulliken[i]:9.4f} {loewdin[i]:9.4f} "
          f"{hirshfeld.charges[i]:10.4f} {iao.charges[i]:8.4f}")
print(f"  {'sum':5s} {mulliken.sum():9.4f} {loewdin.sum():9.4f} "
      f"{hirshfeld.charges.sum():10.4f} {np.sum(iao.charges):8.4f}")
print(f"  (Hirshfeld promolecule norm {hirshfeld.promolecule_norm:.4f}, "
      f"molecule norm {hirshfeld.molecule_norm:.4f}; IAO reference basis '{iao.reference_basis}')")

print("\n  bond orders")
print(f"  {'pair':6s} {'Mayer':>7s} {'Wiberg':>8s} {'IAO-Wiberg':>11s}")
for i, j in ((0, 1), (0, 2), (1, 2)):
    print(f"  {SYMBOLS[i] + str(i + 1) + '-' + SYMBOLS[j] + str(j + 1):6s} "
          f"{mayer[i, j]:7.4f} {wiberg[i, j]:8.4f} {iao.bond_orders[i][j]:11.4f}")

print("\n  dipole moment (origin at the centre of mass)")
print(f"  mu = ({dipole.x:.5f}, {dipole.y:.5f}, {dipole.z:.5f}) e bohr, "
      f"|mu| = {dipole.total:.5f} e bohr = {dipole.total_debye:.4f} D")

print("\n  static dipole polarizability (CPHF, bohr^3)")
for row in alpha:
    print("   " + " ".join(f"{v:10.4f}" for v in row))
print(f"  isotropic alpha = {np.trace(alpha) / 3.0:.4f} bohr^3, "
      f"anisotropy (max - min eigenvalue) = "
      f"{np.linalg.eigvalsh(alpha).max() - np.linalg.eigvalsh(alpha).min():.4f} bohr^3")
