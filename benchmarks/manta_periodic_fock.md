# Finite periodic spin Fock benchmark

Run `.venv/bin/python benchmarks/manta_periodic_fock.py` from the rebuilt
editable checkout. Six fresh single-thread workers compare the existing
analytical Fourier reference with MANTA batches 4/32, at Gamma and three
momenta. Each warms up once and measures five full builds and all F/J/K
exports. Two independent complex Hermitian spin-density channels are used.

The fixture has nine pure s/p/d AOs, two primitives per shell, seven explicit
AO images and six Hartree reciprocal vectors. Exchange has six Q vectors at
Gamma and 60 over nine transfers at three momenta. Full Coulomb omits its
zero mode with no Madelung correction. Hcore is an explicitly supplied diagonal
matrix; this does not benchmark physical periodic SCF or nuclear construction.
The reference uses existing analytical Bloch Fourier factors, physical AO
scales, whole-mesh density contraction and the existing linear exchange
contraction. It also computes all three electronic-energy components.

Timing includes admission, snapshots, J/K generation, Fock/energy composition
and owned exports. Validation is outside build timing but inside total worker
time and RSS. JSON reports min/max and CPU times, memory and actual work.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`078a3614e411e951e630cfcae6c4f203fff7922521778579ac8f77ce317a8653`.

| Mesh | Provider | Batch | Median build ms | Median CPU ms | Total worker s | Peak process MiB |
|---|---|---:|---:|---:|---:|---:|
| gamma | reference | 32 | 1.328 | 1.328 | 1.364 | 145.47 |
| gamma | manta | 4 | 6.671 | 6.672 | 1.442 | 146.45 |
| gamma | manta | 32 | 4.331 | 4.151 | 1.531 | 145.72 |
| multi-k | reference | 32 | 10.000 | 9.250 | 1.565 | 145.70 |
| multi-k | manta | 4 | 28.154 | 26.933 | 1.675 | 145.72 |
| multi-k | manta | 32 | 20.090 | 17.400 | 1.762 | 146.05 |

Maximum Fock error is `2.76e-13`; maximum error in any energy component is
`2.51e-13`. At three momenta, batch 4/32 uses 1386/693 shell calls and
5544/2772 primitive-pair evaluations. Both consume 18144 vector-primitive
products and 240 exchange matrix products. Modeled aggregate peak storage
is 32.57 to 32.75 MiB, including conservative simultaneous reservations for
the two component workspaces. Process RSS includes Python and reference
libraries and is reported separately.

MANTA is slower on this small fixture. This benchmark establishes composition
accuracy, explicit work sharing and bounded storage; it makes no general
speedup, full-SCF or finite-support convergence claim.
