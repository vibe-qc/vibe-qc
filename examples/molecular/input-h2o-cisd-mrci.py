"""Water, truncated CI against the exact answer: CISD, MRCI on a small CAS, and FCI.

System: H2O/STO-3G at the fixed geometry shared by these examples (bohr);
7 spatial orbitals and 10 electrons, so all-electron FCI is 441
determinants and runs in a few seconds. A second, tiny test uses two H2
molecules 40 bohr apart to measure size-extensivity directly.
Methods: ``run_job(method="cisd")``, ``run_job(method="mrci",
active_space=(4, 4))`` (four active orbitals, four active electrons, the
argument order is ``(n_orb, n_elec)``), and ``run_job(method="fci")``,
plus a plain RHF for the correlation energies. All three CI routes return a
``SolverResult`` whose ``.energy`` is the total energy and whose ``.method``
string records the determinant count or the active space.

Two different quantities are printed and they should not be confused. The
gap to FCI on water is the truncation error of each method on one molecule.
The size-extensivity error is the deviation of E(A...B, far apart) from
E(A) + E(B) for the same method; it is zero for FCI and for coupled
cluster and nonzero for any truncated CI, and it is what makes CISD
degrade as molecules get bigger. Two H2 molecules are the smallest system
that shows it, because CISD is exact for each H2 alone (two electrons).

Files (next to this script):
    input-h2o-cisd-mrci-rhf.*       RHF reference
    input-h2o-cisd-mrci-cisd.*      CISD
    input-h2o-cisd-mrci-mrci.*      MRCI(4e,4o)
    input-h2o-cisd-mrci-fci.*       FCI
    input-h2o-cisd-mrci-h2-*.*      the H2 and (H2)2 size-extensivity runs
(each with .out / .molden / .system / .xyz / .population.* sidecars)

Success: E(FCI) = -75.020073 Ha for water; CISD misses it by 0.945 mHa and
MRCI(4e,4o) by 0.366 mHa, so the ordering FCI < MRCI < CISD holds and the
(4e,4o) reference space recovers most of what CISD leaves out. For the
H2...H2 pair, CISD is 0.510 mHa above twice the CISD energy of one H2
(-2.274042 versus -2.274552 Ha), while FCI matches 2 E(H2) to better than
1e-8 Ha. Tolerance: 0.01 mHa on each water energy and 0.005 mHa on the
size-extensivity error (vibe-qc 0.17.x development build, Apple M5 Max, 20 to
90 s in total depending on machine load).
"""

from pathlib import Path

import vibeqc as vq

HERE = Path(__file__).parent
STEM = Path(__file__).stem

mol = vq.Molecule([
    vq.Atom(8, [0.000,  0.000,  0.000]),
    vq.Atom(1, [0.000,  1.499, -1.160]),
    vq.Atom(1, [0.000, -1.499, -1.160]),
])
BASIS = "sto-3g"

rhf = vq.run_job(mol, basis=BASIS, method="rhf", output=str(HERE / f"{STEM}-rhf"))
cisd = vq.run_job(mol, basis=BASIS, method="cisd", output=str(HERE / f"{STEM}-cisd"))
mrci = vq.run_job(mol, basis=BASIS, method="mrci", active_space=(4, 4),
                  output=str(HERE / f"{STEM}-mrci"))
fci = vq.run_job(mol, basis=BASIS, method="fci", output=str(HERE / f"{STEM}-fci"))

e_rhf = float(rhf.energy)
e_fci = float(fci.energy)
print("\n  H2O/STO-3G configuration-interaction ladder (truncation error on one molecule)")
print(f"  {'method':22s} {'E_total (Ha)':>16s} {'E_corr (mHa)':>13s} {'E - E(FCI) (mHa)':>18s}")
print(f"  {'RHF':22s} {e_rhf:16.8f} {0.0:13.3f} {(e_rhf - e_fci) * 1000:18.3f}")
for res in (cisd, mrci, fci):
    e = float(res.energy)
    print(f"  {res.method:22s} {e:16.8f} {(e - e_rhf) * 1000:13.3f} {(e - e_fci) * 1000:18.3f}")

# Size-extensivity: E(H2...H2) against 2 E(H2), with the pair 40 bohr apart.
h2 = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4])])
pair = vq.Molecule([vq.Atom(1, [0.0, 0.0, 0.0]), vq.Atom(1, [0.0, 0.0, 1.4]),
                    vq.Atom(1, [40.0, 0.0, 0.0]), vq.Atom(1, [40.0, 0.0, 1.4])])
print("\n  size-extensivity test: two H2 molecules 40 bohr apart, STO-3G")
print(f"  {'method':8s} {'E(H2) (Ha)':>14s} {'2 E(H2) (Ha)':>14s} {'E(H2...H2) (Ha)':>16s} "
      f"{'error (mHa)':>12s}")
errors = {}
for method in ("cisd", "fci"):
    e_one = float(vq.run_job(h2, basis=BASIS, method=method,
                             output=str(HERE / f"{STEM}-h2-{method}")).energy)
    e_two = float(vq.run_job(pair, basis=BASIS, method=method,
                             output=str(HERE / f"{STEM}-h2h2-{method}")).energy)
    errors[method] = (e_two - 2.0 * e_one) * 1000.0
    print(f"  {method:8s} {e_one:14.8f} {2.0 * e_one:14.8f} {e_two:16.8f} {errors[method]:12.4f}")

print(f"\n  truncation gap on water: CISD - FCI = {(float(cisd.energy) - e_fci) * 1000:.3f} mHa, "
      f"MRCI(4e,4o) - FCI = {(float(mrci.energy) - e_fci) * 1000:.3f} mHa")
print(f"  size-extensivity error of two H2: CISD {errors['cisd']:+.4f} mHa, "
      f"FCI {errors['fci']:+.4f} mHa")
