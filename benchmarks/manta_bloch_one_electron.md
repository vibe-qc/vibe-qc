# Finite Bloch S/T benchmark

Run `.venv/bin/python benchmarks/manta_bloch_one_electron.py` in the rebuilt
editable checkout. The script writes portable JSON to stdout and starts each
provider/case in a fresh process with BLAS and OpenMP fixed to one thread.

The fixture has nine pure AOs (s/p/d), two primitives per shell, seven explicit
images of a skew three-dimensional lattice, and either Gamma or three arbitrary
momenta. No pair cutoff or amplitude screening is applied. Each worker warms
up once and measures seven complete builds, including real-cell and Bloch
matrix export. The reference is the existing libint explicit-cell producer
followed by a NumPy Bloch sum. A separate same-source reference validation is
included in total worker time and process RSS, outside the build timer.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`229fa9645395b48f845be1caa795ee25015897e83f97f6811a57180e8f135a63`.

| Operator | Mesh | Provider | Median build ms | Median CPU ms | Total worker s | Peak process MiB |
|---|---|---|---:|---:|---:|---:|
| overlap | gamma | reference | 0.195 | 0.151 | 5.829 | 144.22 |
| overlap | gamma | manta | 0.353 | 0.353 | 5.370 | 144.39 |
| overlap | multi-k | reference | 1.004 | 0.226 | 4.403 | 144.45 |
| overlap | multi-k | manta | 0.644 | 0.449 | 3.558 | 144.19 |
| kinetic | gamma | reference | 0.056 | 0.058 | 3.294 | 144.22 |
| kinetic | gamma | manta | 0.239 | 0.239 | 1.192 | 145.42 |
| kinetic | multi-k | reference | 0.047 | 0.047 | 1.200 | 145.34 |
| kinetic | multi-k | manta | 0.240 | 0.240 | 1.161 | 144.98 |

All cell matrices agree within `2.23e-15` and Bloch matrices within `4.45e-15`.
MANTA executes 63 shell calls and 252 primitive pairs for either mesh.
Only folding grows: 7 to 21 phase evaluations and 567 to 1701 phase
accumulations. The modeled native peak is 16.09 to 16.11 MiB; process RSS
includes Python, imported numerical libraries, the reference and their caches.
The model is not a measured process-memory guarantee.

The submillisecond timings are noisy and are mostly slower than libint here.
They establish numerical parity and reuse across the mesh, not a general
speedup. The JSON retains min/max build times, setup/validation/total times,
CPU times, actual MANTA work, modeled storage and measured RSS. This fixture
measures neither nuclear/J/K composition nor SCF, BUPO, symmetry or derivatives.
