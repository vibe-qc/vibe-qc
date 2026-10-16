"""Doubly stretched water, UHF/6-31G with four SCF accelerators side by side.

System: H2O with both O-H bonds at twice the length of the shared example
geometry (the two hydrogens are scaled by 2 about the oxygen, so
R(O-H) = 2.006 angstrom against 1.003 angstrom). At this geometry the
closed-shell solution is far from the ground state: UHF breaks spin
symmetry and lies 224 mHa below RHF, which makes it a standard stress test
for SCF convergence. The script prints both energies to show that.
Method: ``run_job(method="uhf", basis="6-31g", uhf_options=UHFOptions(...))``
four times, setting ``uhf_options.scf_accelerator`` to ``SCFAccelerator.DIIS``
(Pulay's commutator DIIS), ``EDIIS_DIIS`` (energy-DIIS far from convergence,
handing over to DIIS; this is also the default of ``UHFOptions``, so it is
set explicitly here only for clarity), ``ADIIS_DIIS`` (the
augmented-Roothaan-Hall variant with the same hand-over) and ``KDIIS``
(Kollmar's DIIS). The other UHF settings stay at their defaults, including
the post-convergence stability analysis (``stability_check=True``);
``SCFAccelerator.from_string("ediis_diis")`` accepts the same names as text.

Files (next to this script): one set per accelerator, plus the RHF run,
    input-h2o-stretched-scf-accelerators-<accelerator>.out / .molden / ...
    input-h2o-stretched-scf-accelerators-rhf.*

Success: all four land on the same UHF energy, -75.779610 Ha (spread
below 1e-9 Ha), in 19 (DIIS), 19 (EDIIS_DIIS), 19 (ADIIS_DIIS) and 20
(KDIIS) iterations, and E(UHF) - E(RHF) = -224.0 mHa (E(RHF) = -75.555614
Ha). On this system the default guess is already inside the DIIS basin, so
the hybrid schemes cost nothing and buy nothing: iteration counts are
equal within one. docs/tutorial/ediis_diis_hybrid.md explains where they
do differ. Tolerance: 1e-6 Ha on the energies, and every run must report
``converged=True`` (vibe-qc 0.17.x development build, Apple M5 Max, 20 to
80 s depending on machine load).
"""

from pathlib import Path

import numpy as np

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem
ANG = 1.8897259886

equilibrium = np.array([[0.0, 0.0, 0.0],
                        [0.0, 1.499, -1.160],
                        [0.0, -1.499, -1.160]])
stretched = equilibrium.copy()
stretched[1:] *= 2.0                                # both O-H at 2 x the example geometry
mol = vq.Molecule([vq.Atom(8, list(stretched[0])),
                   vq.Atom(1, list(stretched[1])),
                   vq.Atom(1, list(stretched[2]))])
print(f"\n  R(O-H) = {np.linalg.norm(stretched[1]) / ANG:.3f} angstrom "
      f"(example geometry: {np.linalg.norm(equilibrium[1]) / ANG:.3f})")

rhf = vq.run_job(mol, basis="6-31g", method="rhf", output=str(HERE / f"{STEM}-rhf"))

ACCELERATORS = ["DIIS", "EDIIS_DIIS", "ADIIS_DIIS", "KDIIS"]

rows = []
for name in ACCELERATORS:
    options = vq.UHFOptions()
    options.scf_accelerator = getattr(vq.SCFAccelerator, name)
    options.max_iter = 200
    result = vq.run_job(mol, basis="6-31g", method="uhf", uhf_options=options,
                        output=str(HERE / f"{STEM}-{name.lower()}"))
    rows.append((name, result.n_iter, float(result.energy), bool(result.converged)))

print("\n  UHF/6-31G on doubly stretched water")
print(f"  {'accelerator':12s} {'iterations':>10s} {'E_UHF (Ha)':>16s}  converged")
for name, n_iter, energy, converged in rows:
    print(f"  {name:12s} {n_iter:10d} {energy:16.8f}  {converged}")
e_ref = rows[0][2]
print(f"  spread of the final energies: {max(abs(r[2] - e_ref) for r in rows):.1e} Ha")
print(f"  E(RHF) = {float(rhf.energy):.8f} Ha; E(UHF) - E(RHF) = "
      f"{(e_ref - float(rhf.energy)) * 1000:.1f} mHa (broken-symmetry UHF is far lower)")
