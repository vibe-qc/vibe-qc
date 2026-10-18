# Finite Ewald nuclear benchmark

Run `.venv/bin/python benchmarks/manta_periodic_nuclear.py` from the rebuilt
editable checkout. Six fresh single-thread workers write portable JSON to
stdout: the existing Ewald reference and MANTA with reciprocal batches 4/32,
at Gamma and at three arbitrary momenta. Each warms up once and measures five
complete builds, including cell and Bloch exports.

The fixture has nine pure s/p/d AOs, two primitives per shell, seven explicit
AO cells, 14 reciprocal vectors at kinetic cutoff 6 Hartree, alpha 0.7 inverse
bohr, physical AO-pair radius 2.4 bohr and nuclear radius 4.5 bohr. Both use
the pair-complete nuclear midpoint sphere. The reference is the existing
libint/analytical Fourier Ewald lattice operator with amplitude/domain screening
disabled, followed by a NumPy Bloch sum. No reciprocal damping screen removes
vectors at these parameters. MANTA's conservative radial work budget is raised
to `10^14` units; its default storage and other caps are retained.

Build timing covers all three Ewald components. The JSON additionally reports
fixture, validation and total worker times, CPU times, actual MANTA work,
modeled storage and peak process RSS. Validation is outside build timing but
inside total time and RSS. This does not time SCF, exchange, BUPO, symmetry or
nuclear derivatives.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`733a94dfd11371dd52dd0ede449daa6113a1e14e6d7a84d7bf6c85a58f7679db`.

| Mesh | Provider | Batch | Median build ms | Median CPU ms | Total worker s | Peak process MiB |
|---|---|---:|---:|---:|---:|---:|
| gamma | reference | 32 | 2.335 | 2.300 | 1.793 | 145.91 |
| gamma | manta | 4 | 6.201 | 6.202 | 1.336 | 147.59 |
| gamma | manta | 32 | 5.611 | 5.612 | 1.292 | 146.95 |
| multi-k | reference | 32 | 1.945 | 1.946 | 1.230 | 147.00 |
| multi-k | manta | 4 | 7.190 | 6.629 | 1.375 | 146.55 |
| multi-k | manta | 32 | 6.265 | 6.199 | 1.280 | 147.47 |

Cell and Bloch matrices agree within `3.78e-15`. MANTA visits 1281 retained
nuclear images and 5124 primitive-charge products, independent of k count or
reciprocal batch. Increasing the batch cuts Fourier shell calls from 84 to 21
and all primitive-pair evaluations from 504 to 252; vector-primitive products
remain 1176. The modeled native peak is 16.33 to 16.49 MiB. It includes bounded
backend workspace and is separate from process RSS, which includes Python,
libraries, the reference and caches.

MANTA is slower on this small fixture. These measurements establish finite-source
parity, bounded execution and measured work sharing, not a speedup claim.
They do not imply that the finite radii/grid are converged physical parameters.
The separate scalar Gaussian convergence test qualifies the Ewald splitting
and its zero-average limit.
