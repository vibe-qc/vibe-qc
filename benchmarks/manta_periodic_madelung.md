# Corrected periodic RHF evidence

Run `.venv/bin/python benchmarks/manta_periodic_rhf.py --exchange-q0 BvK-ewald`
from the rebuilt editable checkout. The six fresh single-thread workers use
the same finite source as the [uncorrected benchmark](manta_periodic_rhf.md),
with a separately admitted BvK probe-charge Ewald correction. Each warms up
once and measures five complete SCF runs and owned density/Fock/orbital
exports. Probe enumeration and energy, all integral generation, SCF iteration
and final corrected Fock/energy confirmation are inside build timing.
Reference validation is outside build timing but inside worker time and RSS.

The independent reference uses quadrature/libint integral components, a
separate generalized-eigenvalue SCF loop and the existing independently
implemented Python probe-charge Ewald sum at tolerance 1e-14. The MANTA
path uses the shared native point-charge Ewald source, with its own resource
admission and tolerance 1e-12. Both evaluate the full reciprocal Coulomb
convention; no erfc-split zero-mode subtraction belongs in this comparison.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`ea9d5fe5dae4b92e36376c9803288cc1ff113c34d019304af8d0f7775039d4d0`.

| Mesh | Batch | Median SCF ms | Median CPU ms | Worker s | Peak process MiB | Iterations | Fock calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| gamma | 4 | 0.968 | 0.968 | 3.199 | 155.75 | 2 | 3 |
| gamma | 32 | 0.596 | 0.599 | 1.660 | 156.20 | 2 | 3 |
| multi-k | 4 | 6.927 | 6.932 | 1.691 | 155.67 | 11 | 12 |
| multi-k | 32 | 2.593 | 2.595 | 1.774 | 155.59 | 11 | 12 |
| shifted | 4 | 5.848 | 5.851 | 1.772 | 155.84 | 11 | 12 |
| shifted | 32 | 2.559 | 2.560 | 1.727 | 155.64 | 11 | 12 |

Maximum independent errors are `1.31e-13` hartree in total energy,
`4.85e-11` in density and `9.58e-14` in Fock elements. Probe constants agree
within `1.32e-13` hartree. The measured aggregate storage model is 80.63 to
80.71 MiB. Whole-run shell/primitive-pair calls are 296/56 at Gamma and
2060/308 at two points for batches 4/32. Gamma consumes 98 real and 98
reciprocal probe vectors; each two-point run consumes 106 and 116. These
counts are distinct from the explicit nuclear and electronic integral domains.
Process RSS includes Python and reference libraries. These timings do not
establish a speedup against another SCF implementation.

## Independent periodic Hartree-Fock and refinement

`tests/test_manta_periodic_madelung.py` includes optional PySCF AFT KRHF
comparisons at Gamma, a full two-point mesh and a shifted two-point mesh.
They use the same skew cell and two s orbitals, 27 explicit AO translations,
a 9 bohr AO-pair cutoff, a 12 bohr nuclear radius and reciprocal integer
supports from -12 through 12 on each axis. The reference uses a 25-cubed AFT
grid and the same BvK Ewald exchange convention. S, Hcore, density, Fock,
nuclear energy and total energy are checked separately; the total-energy
gate is 3e-10 hartree.

Measured unshifted complete-SCF energies, in hartree:

| Mesh | MANTA | Independent AFT | Difference |
|---|---:|---:|---:|
| Gamma | -0.7707034732639442 | -0.7707034732637525 | -1.92e-13 |
| 2 x 1 x 1 | -0.7505104402906484 | -0.7505104402910026 | 3.54e-13 |

On the coarser 17-cubed reference grid the Gamma energy difference was
4.51e-7 hartree, with a 2.31e-7 Hcore error. Refining the reference grid
removed that difference. MANTA's Ewald-split nuclear source is already
converged more tightly there; the two nuclear reciprocal constructions need
separate convergence checks. The tests also refine MANTA's reciprocal
integer radius from 8 to 12 and change nuclear alpha from 0.6 to 0.9,
requiring consistent energies within 3e-10 hartree. These are small-system
acceptance cases, not a blanket certificate for arbitrary cutoffs or k meshes.

Other tests check the restricted energy and occupied-spectrum shifts, an
independent same-source SCF, probe alpha invariance, lattice/mesh scaling,
atom/shift independence, final-density confirmation, owned exports and
aggregate resource refusal before vector payload copies.
