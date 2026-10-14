# Selected periodic RHF job evidence

Run `.venv/bin/python benchmarks/manta_periodic_rhf.py --job --exchange-q0 BvK-ewald --reciprocal-energy-cutoff 2 --repeats 5`
from a freshly rebuilt editable checkout. Six fresh single-thread workers
exercise Gamma, two-point and shifted two-point meshes with reciprocal
batches 4 and 32. The two-orbital skew-cell fixture, finite cutoffs and
independent quadrature/libint SCF oracle are shared with the
[cutoff-domain benchmark](manta_periodic_domains.md). These small sources
qualify execution and output behavior, not converged physical cutoffs.

Every timed call includes public `run_periodic_job` admission, domain
generation, integrals, Ewald terms, complete RHF with final confirmation,
owned matrix exports, complete scalar iteration history, log, manifest,
geometry, citations and QVF. Output
uses a temporary directory and the default artifact selection. The same
stem is reused, so these timings describe repeated local file delivery;
they do not measure durable storage or remote delivery. Each worker warms
up once, then measures five calls. Worker time and peak process RSS also
include imports and the independent validation outside the timed calls.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`5dd8fcc938b5bfb16ca1cb341185bd92e26ef3bec1e8658d0a87b4c2e209cc1d`.

| Mesh | Batch | Median job ms | Min/max ms | Median CPU ms | Worker s | Peak process MiB | Shell/pair calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| gamma | 4 | 150.22 | 140.64/179.23 | 136.31 | 3.331 | 173.61 | 296/296 |
| gamma | 32 | 124.82 | 121.84/125.89 | 109.39 | 2.011 | 171.50 | 56/56 |
| multi-k | 4 | 161.37 | 123.18/252.87 | 140.05 | 2.033 | 171.91 | 548/548 |
| multi-k | 32 | 125.54 | 124.14/593.01 | 112.28 | 3.103 | 174.98 | 92/92 |
| shifted | 4 | 115.16 | 107.94/120.60 | 100.24 | 1.635 | 173.33 | 548/548 |
| shifted | 32 | 362.18 | 214.48/374.04 | 261.52 | 3.124 | 173.09 | 92/92 |

Maximum independent errors are 1.30e-13 hartree in total energy,
9.54e-14 in Fock elements and 5.81e-35 in density. The symmetric fixture
makes the density comparison especially simple. Each run converges in two
iterations with three Fock calls including confirmation. Modeled peak
storage, including conservative output reservations, is 83.38 to 83.49 MiB;
this is distinct from process RSS. Every case reserves 100 scalar history
rows (78,400 bytes including native/typed exports and another 1,638,400 bytes
for Python records and serialization), while actually retaining two rows.
The large job-to-native timing difference
and spread reflect output and Python overhead for these tiny numerical
sources. They establish no algorithmic speedup or scaling result.

`tests/test_manta_periodic_job.py` additionally checks exact native-source
parity, QVF validation and density readback, shifted-basis gauge transport,
full-mesh input validation, budget and unsupported-capability refusal,
dry-run admission, complete iteration-history readback, and lifecycle
handling of calculation and delivery failures. An asymmetric
iteration-limited case verifies that QVF retains
the last evaluated density even when the terminal orbitals generate a
different density. These artifact checks complement the independently
qualified native physical sources; they do not extend the supported RHF
profile to other methods or boundaries.
