# Bounded periodic UHF numerical solver

Run `.venv/bin/python benchmarks/manta_periodic_uhf_solver.py --repeats 5`
from a freshly rebuilt editable checkout. Three fresh single-thread workers
use one, three and eight physical k points, three complex nonorthogonal
basis functions, two alpha electrons and one beta electron. The fixture is
an explicitly variational spin/k-coupled linear response, not a periodic
Gaussian Hamiltonian. It qualifies the numerical prerequisite for MANTA UHF;
it does not qualify physical periodic UHF jobs or convergence of cutoffs.

Each worker measures one cold call, then five native numerical solves. An
independent NumPy generalized fixed-point iteration validates the measured
result's density, Fock and total energy. Calls include bounded planning and
the tiny diagnostic Python boundary. RSS includes imports, fixture storage,
all repetitions and the independent reference; the modeled byte count does
not include those Python owners or allocator overhead. Scalar charged work
units are conservative admission charges, not measured floating-point counts.

The numerical leaf owns one shared orthogonalizer and separate D/F/C/energy
arrays for both spins. At full or truncated retained rank r its exact modeled
peak is `64 K n^2 + 48 K n r + 16 K r + 32 n^2 + 32 n r + 8 n` bytes.
Borrowed S/H, callback storage and declared external owners are additional.
The diagnostic's intentionally dense response matrix is limited to eight
k points and four basis functions. It is a bounded test oracle interface,
not a production representation of the MANTA Fock response.

The acceptance suite also covers empty/reversed spin populations, closed-shell
RHF parity, positive discarded overlap directions, incompatible retained
ranks, AO-phase transport, every capped epoch, cancellation, callback faults
and exact/one-byte-short resource admission. The spin-summed commutator RMS
divides by K, not 2K. The gap gate uses the smallest within-spin gap at fixed
populations and makes no claim about stability against spin transfer.

Measured on Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native core SHA-256:
`5a63addc355652f58744241144fdd738e047f58948517e99b626a607005dabad`.

| Physical k points | Cold ms | Median ms | Min/max ms | Median CPU ms | Peak process MiB | Fock calls | Owned/total modeled bytes |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.305 | 0.158 | 0.156/0.190 | 0.157 | 156.36 | 7 | 1656/7416 |
| 3 | 3.267 | 0.854 | 0.523/1.392 | 0.581 | 155.58 | 6 | 3768/52152 |
| 8 | 1.455 | 1.812 | 1.346/2.063 | 1.428 | 155.77 | 6 | 9048/345432 |

Maximum independent errors are 8.33e-12 in density elements, 5.56e-17 in
Fock elements and 4.44e-16 hartree in total energy. The one-point case takes
seven iterations and the other cases take six, with one coupled Fock call
per iteration. Charged work units are respectively 62,759,040, 167,283,968
and 456,041,728. These tiny runtimes include timer/scheduling variation and
provide neither an acceleration ratio nor a scaling claim. Components of
the numerical solve are not separately timed. Physical integral, Madelung,
source-confirmation and output costs belong to the subsequent UHF driver
benchmark; none is present in these numerical-model measurements.
