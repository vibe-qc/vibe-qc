# Generated fitted periodic KS benchmark

This bounded acceptance measures pure RI-J and all three fitted global-hybrid
choices through generated cutoff domains and atom-centred quadrature. It does
not qualify public `run_periodic_job` routing, physical range fitting, automatic
accuracy profiles, BIPOLE/CRYSTAL parity or symmetry reduction.

Reproduce from a fresh native build in this checkout:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python benchmarks/manta_periodic_ri_ks_domains.py
```

The script launches twelve separate one-thread workers. Each records a cold
native call and five repeated native plus five repeated selected scientific
calls. Every call regenerates the quadrature, domains, fitting/projectors and
SCF; no source owner is reused across solves. The selected interface constructs
and records scientific provenance. It does not include a full job, QVF export
or active file-output channel. Timing was collected without concurrent test
work from this task. Different sources define different finite Hamiltonians
and iteration counts; these timings are not a speedup comparison.

The cells have lattice rows `(6, .2, .1), (0, 6.2, .15), (0, 0, 6.4)` bohr and
a shifted `(2, 1, 1)` mesh. Be RKS and Li UKS use three s orbital and three s
auxiliary functions. The empty-beta H cases use two s orbital and four s
auxiliary functions. Pure cases use PBE except empty-beta BLYP; global cases
use PBE0 except empty-beta B88/LYP with 20% full-Coulomb exchange. These empty
spin witnesses avoid the separately documented PBE/libxc minority endpoint
conditioning. Global cases use the method-owned BvK correction.

Controls are reciprocal kinetic cutoff 6 hartree, orbital pair radius 3 bohr,
XC image radius 11 bohr, fitting omega 0.65 and auxiliary radius 8 bohr. The
atom-centred grid has 12 radial points, Lebedev order 17, partition radius
8 bohr and 1,320 points. These are finite-source witnesses, not recommended
accuracy settings. Each cell has one orbital translation, 168 Gamma vectors
and 27 canonical XC images. RI-J-only global exchange has 676 exact pair
vectors, including primed diagonal segments. RI-K has 340 off-diagonal pair
vectors and empty diagonal segments, owned by Gamma. Pure KS has no pair
table. RI-J retains one metric; RI-K retains three. All retained metric ranks
are three for Be/Li and four for H.

The independent reference uses libint raw metrics and three-centre couplings,
NumPy projector factorization/contractions, Gaussian Fourier direct responses,
Gaussian Bloch/libxc grid evaluation and SciPy SCF. It does not use the MANTA
Fock contractions or SCF iteration. The nuclear energy is independently
reconstructed. Every matrix, energy component and grid population is compared
at the actual evaluated density, and the converged total energy is checked
against the independently converged density.

| Source and spin | Energy (hartree) | Iterations | Native median (s) | Selected median (s) | Joint admission (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| pure-rks | -11.695694654271 | 9 | 0.0847 | 0.0871 | 84.554 |
| pure-uks | -6.397168375806 | 13 | 0.1231 | 0.1254 | 84.556 |
| pure-empty | -0.455558264868 | 5 | 0.0340 | 0.0345 | 84.490 |
| J-rks | -11.662245807108 | 8 | 0.0836 | 0.0819 | 100.241 |
| J-uks | -6.381394289474 | 11 | 0.1147 | 0.1147 | 100.245 |
| J-empty | -0.458875727913 | 9 | 0.0593 | 0.0623 | 101.167 |
| K-rks | -12.113927045356 | 8 | 0.1202 | 0.1193 | 100.488 |
| K-uks | -6.637925464823 | 11 | 0.1685 | 0.1673 | 100.490 |
| K-empty | -0.459017486543 | 9 | 0.0861 | 0.0857 | 101.406 |
| JK-rks | -11.710182061922 | 8 | 0.1202 | 0.1148 | 84.315 |
| JK-uks | -6.407052545188 | 11 | 0.1622 | 0.1679 | 84.317 |
| JK-empty | -0.458899465907 | 9 | 0.0815 | 0.0827 | 85.252 |

`J` means RI-J/exact-K, `K` means exact-J/RI-K, and `JK` means both fitted.
All states converge and confirm. Generated, explicit and selected D, F, H, S,
coefficients, orbital energies and total energies agree bitwise.

| Independent quantity | Maximum absolute error |
| --- | ---: |
| Overlap | 3.330669e-16 |
| HCORE | 3.552714e-15 |
| Converged density | 2.006539e-10 |
| Same-density Fock | 1.443290e-14 |
| Energy components | 2.550737e-14 |
| Converged total energy | 2.486900e-14 |
| Integrated grid populations | 7.105427e-15 |

Maximum native confirmation differences are
2.220446e-16 for F and
1.776357e-15 hartree for energy.
The coarse grid's maximum deviation from orbital electron populations ranges
from 0.015 to 0.248 electrons. Agreement with an independent
calculation on that grid does not establish quadrature convergence.

The count-only joint admission ranges from 84.315 to 101.406 MiB.
The executed native plans use actual source counts and range from
84.043 to 101.119 MiB; selected plans add their Python evidence
reservation and range from 84.156 to 101.232 MiB. These are
conservative ownership models, not RSS measurements. Retained grid bytes are
133,344; domain admission ranges from 104608.000 to 263936.000 bytes and retained
fitting owners from 109464.000 to 332664.000 bytes. Whole-worker peak RSS
is 173.156 to 177.969 MiB, including Python, libraries, exported
states and the independent oracle. Serialized selected evidence ranges from
7801.000 to 10590.000 bytes; serialized size alone is not its
live-memory bound.

`tests/test_manta_periodic_ri_ks_domains.py` additionally checks Gamma and
shifted integer-domain membership, both exact/RI transfer conventions,
conjugacy, all five exact/one-short native and selected caps, refusal before
source allocation, truncated/empty fitting projectors, capped states,
independent AO/auxiliary image shifts and rectangular overlap frames. Owned
FockBuild exports retain KS method energy and actual evaluated density.
BlockSpaceAction checks image transport; it grants no source-symmetry or
BIPOLE parity certificate. Twelve saved prechange generated direct KS states
retain all numerical arrays, plans and identities exactly.

Native core SHA-256: `c9b81d47943d442eb5bb04026f06827b3ff9c60630bf00156bdf0c0b1891e84d`.
Scientific output adapter SHA-256: `f0c0407ee49a8c981041ee16f055136b66d1e007ef3ff97b4239a979dec053e5`.
The script verifies its extension belongs to the current checkout and is newer
than every native source before measuring. This is focused finite-source
acceptance, not a full numerical lane or release gate.
