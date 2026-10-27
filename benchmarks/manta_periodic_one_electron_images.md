# Periodic overlap and kinetic image bounds

Native full AO-image overlap and kinetic enclosures now accompany the
periodic BUPO HF electrostatic ledger. They cover omitted translations and
physical-cutoff exclusions at a fixed evaluated density. They do not bound
exchange, its overlap-dependent BvK correction, basis incompleteness,
changes to the relaxed state or overlap rank, or roundoff.

## Timing and reproduction

Run `.venv/bin/python benchmarks/manta_periodic_one_electron_images.py --repeats 5`.
Three isolated Darwin arm64 workers use Python 3.14.8, NumPy 2.5.3 and one
numerical thread. Each receives a warm-up and five repetitions. Bound
construction is timed with the immutable pair source already available.
Models include retained source and numerical exports; process RSS also
includes Python and imported test/reference libraries. No scaling or
speedup claim follows from these small cases.

| Case | AO | Bound construction microseconds | Model MB | RSS MiB | Maximum S bound | Maximum T bound |
|---|---:|---:|---:|---:|---:|---:|
| sparse-sp | 4 | 14.50 | 8.427 | 167.61 | 18.7 | 41.4 |
| central-cutoff-p | 4 | 16.54 | 8.427 | 169.08 | 23.5 | 52.6 |
| tight-hf | 2 | 17.42 | 8.503 | 168.45 | 2.94e-18 | 2.59e-18 |

The sparse and central-cutoff cases use signed two-primitive s/p contractions,
a skew cell near side 4 bohr and the original three k points. The former
retains origin and two x images; the latter additionally applies cutoff
0.3 bohr, removing the cross-center central pair. Their large enclosures
are deliberately not production-accepted targets.

The tight HF case is the two-s He fixture from the BUPO HF benchmark, with
side 12 bohr, exponents 4 and 1.2, original image cube 1 and cutoff 3 bohr.
Complete HF takes 59.596 ms and its combined numerical model is 180.141 MB.
The maximum observed kinetic-energy image bound is 4.19e-18 hartree.
The default targets are 1e-10 for overlap, 1e-8 for kinetic
matrix entries and 1e-8 hartree for kinetic energy.

## Scientific checks

Independent separable Gauss-Hermite quadrature directly integrates Cartesian
polynomials and their gradients, then assembles every ordered shell pair
and image with its original complex Bloch phase. It uses no native
integral, canonical-inventory or bound routines. Eight-point one-dimensional
quadrature is exact for the tested polynomial degrees through l=6.
The source's established pure transformation supplies the representation
convention. Separate image refinement checks the finite full reference.

Checks cover signed s/p/d shells in cubic/skew cells, Cartesian g and pure i
shells, sparse holes, missing central pairs, physical cutoffs, changed k
order and zero weights, positive subnormal bounds, owned arrays and all five
exact/one-short resource caps. Complete BUPO HF checks retain both-spin and
empty-beta states, BvK correction, evaluated capped density and shared Fock
semantics. The new targets must refuse before a result is returned.

A first subnormal test incorrectly rounded each product before summation.
A 90-digit Decimal reference confirms the native positive logarithmic sum
is conservative. No analytical bound or acceptance target was weakened.
Forty-two focused checks and 166 related regressions pass. The nine independent
matrix cases additionally compare native finite S/T against the quadrature.

Actual local `vq` calculations exercise eleven S/T image cases and six complete
BUPO HF cases. The latter cover RHF, empty-beta UHF, an open-shell doublet,
mixed near/far RHF and UHF with BvK correction, and one deliberately capped
evaluated state. Five converge; the capped result remains unconverged.
All six density, Fock, overlap, core Hamiltonian and energy results are
bitwise unchanged from the previously accepted finite-HF calculation data.
The shared Fock energy and all three new accuracy refusals are also checked.

Nine native component calculations completed before the first batch stopped
at a reference-only Cartesian/pure p mismatch. Their fetched results were
preserved and reused. Reference refinement also showed that the g/i cases
need image cube 3 rather than 2: the changes were 1.58e-9 and 2.53e-7,
respectively. Cube 3 versus 4 then agrees within 1.12e-16; the s/p/d cube 2
versus 3 comparisons also pass. Only independent references were enlarged;
no native result, error target or comparison tolerance was changed.
The completed batch reused those nine cases and ran the eight remaining
native calculations, taking 7.077 seconds with one CPU and a 512 MiB limit.

Fetched artifacts match their original queue workspaces. A separate
NumPy/Decimal assessment recomputes matrix errors, weighted kinetic bounds,
native Fock energy traces and the saved independent finite-HF references.
Maximum native finite S/T disagreement is 1.21e-13. The largest HF energy
and Fock discrepancies are 9.10e-13 hartree, and the native energy trace
differs by at most 3.56e-15 hartree. The largest deliberately poor image
inventory error is 44.61 and is enclosed by its reported bound; this does
not meet a production target. Numerical comparison allowances are separate
from analytical bounds that exclude roundoff.

The measured source is `76471ca7`, with native implementation `bdf74bfd` and
native SHA-256
`a5da6cc94419b5aa26d1c8d7eb434c8aaeac65214a8631472a9a7405002c3051`.
All 270 implementation/reference file hashes and the native bytes agree
before and after the accepted calculations. No matched ORCA/CRYSTAL
omitted-image certificate is available; numerical quadrature, image refinement
and reuse of qualified finite-HF references provide the independent comparison.
Whole-method production profiles and the remaining MANTA matrix stay open.
