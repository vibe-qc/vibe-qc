# Complete finite periodic RHF benchmark

Run `.venv/bin/python benchmarks/manta_periodic_rhf.py` from the rebuilt
editable checkout. Six fresh single-thread workers exercise Gamma, a full
two-point mesh and a shifted two-point mesh with reciprocal batches 4/32.
Each warms up once and measures five complete native SCF runs, including
S/T/V, Ewald nuclear energy, every J/K build, numerical iteration, terminal
confirmation and density/Fock/orbital exports. This is a whole-run benchmark,
not a comparison with a production periodic SCF implementation.

The neutral H2 fixture has two pure s AOs, one primitive per shell, a skew
three-dimensional lattice, the zero AO image, 26 nonzero nuclear/Hartree G
vectors and explicit exchange supports over every target/source transfer.
The Ewald alpha is 0.6 inverse bohr, nuclear radius 4 bohr and pair cutoff
3 bohr. Full Coulomb exchange omits Q=0 without a Madelung correction. These
coarse finite supports define a verification problem, not converged physical
predictions.

The independent reference uses libint overlap, kinetic and short-range
nuclear attraction, Gauss-Hermite quadrature for Fourier factors, explicit
nuclear Ewald sums and a separate NumPy/SciPy generalized-eigenvalue SCF loop.
The benchmark reuses these scientific fixtures from the test suite. Reference
validation is outside build timing but inside total worker time and RSS.
Imports, fixture construction and validation explain the larger worker time.
The JSON also records min/max build times, fixture/validation times and all
work counters. Wall/CPU differences show contention during this measurement;
no general speed claim follows from these timings.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`cc25c8d32cf8b0988939ad565542bb315d795714f214fe5d8220461c9368fcfb`.

| Mesh | Batch | Median SCF ms | Median CPU ms | Total worker s | Peak process MiB | Iterations | Fock calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| gamma | 4 | 6.215 | 1.885 | 16.810 | 155.61 | 2 | 3 |
| gamma | 32 | 0.779 | 0.768 | 8.299 | 154.48 | 2 | 3 |
| multi-k | 4 | 44.075 | 10.652 | 7.805 | 155.73 | 10 | 11 |
| multi-k | 32 | 11.460 | 4.196 | 12.473 | 154.52 | 10 | 11 |
| shifted | 4 | 23.784 | 9.613 | 6.211 | 154.69 | 10 | 11 |
| shifted | 32 | 4.771 | 3.927 | 16.820 | 154.52 | 10 | 11 |

All runs converged and passed an independent final-density Fock/energy
confirmation. Maximum reference errors are `9.99e-16` hartree in total energy,
`3.28e-11` in density and `6.67e-16` in Fock elements. Maximum terminal
confirmation differences are `3.29e-18` in Fock elements and `1.12e-16`
hartree in energy. The Gamma energy is `-0.38898148916575637` hartree;
both two-point fixtures give `-0.519398091413751` hartree.

Whole-run shell/primitive-pair calls fall from 296 to 56 at Gamma and from
1892 to 284 at two points when the batch grows from 4 to 32. Each run
consumes two directed nuclear pair images. Modeled peak numerical storage
is 80.43 to 80.50 MiB; the aggregate plan conservatively reserves all
component workspaces plus the full-rank solver simultaneously. Process RSS
includes Python and reference libraries and is reported separately. This
small fixture verifies complete finite-source composition and work sharing;
it does not establish large-system scaling or cutoff convergence.
