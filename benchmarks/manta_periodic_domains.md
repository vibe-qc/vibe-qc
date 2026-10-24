# Periodic RHF cutoff-domain evidence

Run `.venv/bin/python benchmarks/manta_periodic_rhf.py --exchange-q0 BvK-ewald --reciprocal-energy-cutoff 2 --repeats 5`
from the rebuilt editable checkout. Six fresh single-thread workers exercise
Gamma, two-point and shifted two-point meshes with reciprocal batches 4 and
32. The skew cell and two normalized s orbitals are shared with the
[explicit-source RHF benchmark](manta_periodic_rhf.md). The AO-pair cutoff
is 3 bohr, nuclear radius 4 bohr and reciprocal kinetic cutoff 2 hartree.
These deliberately small finite sources test the complete execution path;
they are not converged physical cutoffs.

Every timed call plans and generates the domains, audits conjugacy, computes
all integral and Ewald terms, runs RHF, confirms the final Fock and energy,
and exports owned density/Fock/orbital arrays. Each worker warms up once and
measures five calls. The domain column is a separate single construction
measurement, including complete-run admission. Validation uses independent
quadrature/libint components, a separate generalized-eigenvalue SCF and the
independent Python probe-charge Ewald sum. Worker time and peak process RSS
include imports and reference validation; build timing does not include the
reference calculation.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`0a3cb4e86d1dda05688747523fec057299fe1d5f86c6d98c5b4422c42b2e1584`.

| Mesh | Batch | Domain ms | Median run ms | Median CPU ms | Worker s | Peak process MiB | Shell/pair calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| gamma | 4 | 0.074 | 1.220 | 1.141 | 1.845 | 155.61 | 296/296 |
| gamma | 32 | 0.094 | 0.490 | 0.491 | 1.617 | 155.47 | 56/56 |
| multi-k | 4 | 0.100 | 1.842 | 1.841 | 1.663 | 156.25 | 548/548 |
| multi-k | 32 | 0.110 | 0.802 | 0.804 | 1.504 | 156.67 | 92/92 |
| shifted | 4 | 0.177 | 1.533 | 1.534 | 1.563 | 155.69 | 548/548 |
| shifted | 32 | 0.108 | 0.972 | 0.962 | 1.503 | 156.08 | 92/92 |

Maximum independent errors are 1.30e-13 hartree in total energy,
9.54e-14 in Fock elements and 5.81e-35 in density.
The density comparison is especially simple for this symmetric two-orbital
fixture. All cases converge in two iterations plus one confirmation call.
Each source contains one AO translation and 28 Hartree vectors; exchange
contains 28 vectors at Gamma or 112 across all four ordered two-point pairs.
The native plan reserves 50,432 bytes for domain construction at Gamma and
68,416 bytes for two points. Total modeled storage is 80.67 to 80.77 MiB,
including conservative overlapping reservations. It is not process RSS.
These timings establish no speedup against a different SCF implementation.

## Numerical and resource qualification

`tests/test_manta_periodic_domains.py` compares reciprocal membership with
independent integer-box enumeration for skew cells, odd/even divisions,
shifted meshes and mixed Nyquist transfers. It checks reverse-transfer
closure and exact spherical boundary membership. Physical-pair tests move
one atom and its basis by a lattice vector, verify coverage of the required
images and check the complex Bloch gauge of S, H, F and D and invariant
energy. Owned exports, identity changes, large-mesh refusal and all five
aggregate budget boundaries are tested. Admission must precede even the
empty-Hartree-source check.

Refined optional PySCF AFT KRHF comparisons use a 60 hartree reciprocal
sphere, 9 bohr AO-pair radius, 12 bohr nuclear radius and BvK Ewald correction
at Gamma and on a shifted two-point mesh. The independent reference uses a
25-cubed AFT mesh. Tests pass separate gates for S (3e-11), H/F (3e-10),
density (2e-8), nuclear energy (3e-11 hartree) and total energy (3e-10
hartree). The earlier [Ewald qualification](manta_periodic_madelung.md)
describes why refining the independent nuclear reciprocal source matters.
These cases qualify the implemented source construction, not arbitrary
cutoffs, metallic states, larger bases or other dimensionalities.
