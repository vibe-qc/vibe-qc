# Bound-admitted finite periodic BUPO Coulomb evidence

Run `.venv/bin/python benchmarks/manta_periodic_bupo.py` from the rebuilt
editable checkout. Six fresh single-thread workers cover signed contracted
s-only and s/p sources at orders 4, 6 and 8. Each takes five measurements
after warm-up. Existing pair, hierarchy and direct Ewald sources are outside
timed setup. Setup includes the fixed bound-admitted partition; build includes
forward moments, repeated deterministic admission, direct blocks, the reverse
pass, complex J, energy and aggregate approximation bounds. Direct timing is
the matching complete native finite Ewald build.

Darwin arm64, Python 3.14.8, NumPy 2.5.3. Benchmark native SHA-256:
`126f218503d477a41f4e20a899882796583c4bdc533bf21c45b989a7af736987`.

| Basis | Order | Integral tolerance | Setup ms | BUPO build ms | Direct build ms | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|---:|
| s | 4 | 1e-2 | 0.609 | 26.576 | 30.734 | 102287872 |
| s | 6 | 1e-5 | 1.131 | 28.529 | 29.774 | 103359872 |
| s | 8 | 1e-7 | 2.539 | 32.149 | 28.152 | 106050352 |
| s/p | 4 | 1e-2 | 0.565 | 43.078 | 49.165 | 102532696 |
| s/p | 6 | 1e-5 | 1.168 | 45.215 | 49.060 | 103658328 |
| s/p | 8 | 1e-7 | 2.677 | 52.406 | 49.246 | 106435976 |

Each skew-cell source has six centers, 13 canonical pair images, 13 tree
nodes, 27 real Ewald images, 26 reciprocal vectors and three k points.
The basis has tight signed two-primitive contractions with exponent scale
3200 and support radius 0.07 bohr. The angular source replaces two s shells
with p shells. Origin and boundary-crossing pairs remain in the same source.
The fixture is explicit in `tests/test_manta_periodic_bupo.py`.

All six partitions accept eight far blocks (18 unordered pair interactions)
and use 33 direct blocks (73 interactions). Actual real quartet calls fall
from 2457 to 1971. Fourier shell calls rise from 52 to 292 because a mixed
partition streams complete direct pair blocks; it cannot use the full
source's two-pass reciprocal contraction unchanged. Setup and higher-order
M2L work can outweigh the saved quartets, as the order-8 timings show.
These small fixtures are not a production scaling claim.

Process peak RSS is 166.58-168.80 MiB, including imports and verification
arrays. It is distinct from the modeled native reservation, which includes
retained sources and conservative worst-case all-direct work. No global
four-index tensor, all-pair Fourier panel or per-edge M2L cache is retained.
Repeated fixed-partition admission trades work for bounded memory.

The largest benchmark J error against the direct source is 1.990e-13;
the largest energy error is 2.843e-14 Eh. The largest weighted half-trace
mismatch is 1.422e-14 Eh. Bounds, rather than numerical cancellation, admit
the far blocks. At order 6 the s-only energy/J bounds are 1.125e-6 Eh and
2.551e-6, and the s/p bounds are 2.361e-6 Eh and 8.101e-6. At order 8 the
corresponding bounds are 2.538e-8/5.769e-8 and 9.957e-8/3.705e-7.
The per-integral tolerance is not the aggregate energy accuracy.

The 26 focused checks cover complete scalar Gaussian integration,
Cartesian/pure angular comparison to the independently qualified direct
source, cubic/skew cells, complex densities, nonuniform/zero weights,
two-step weighted derivatives, linearity, bounds and density scaling.
Order-12 and tiny-density calculations retain the full moment action.
Periodic-image overlap refines even a central-cell far candidate. Zero
tolerance and an inadequate Gaussian envelope fall back to exact finite
Ewald. Moving near images between bare and complementary terms preserves
one kernel, including trace moments and the gauge. All five exact caps,
shape-before-payload refusal, immutable ownership and unaligned input pass.
The existing 71 direct-source and Gaussian-error tests pass; four saved
complete direct calculations remain bitwise unchanged after helper extraction.

The algorithm composes BUPO separation (Neese et al., 2025,
DOI 10.1021/acs.jpca.4c07415), Ewald1921 IV.1 Eqs.37-40
(DOI 10.1002/andp.19213690304) and Sun2017 q=0 coefficients, Eqs.16-17
(DOI 10.1063/1.4998644). Full raw Cartesian moments and native Gaussian
ledgers are required for the nonharmonic complementary kernel.
The scientific contract derives the fixed quadratic form and aggregation.

Qualification `finite-periodic-bupo-j` compares to the same retained finite
canonical Ewald source. The bounds exclude roundoff, omitted AO pair images
and infinite-domain/covariance error. Full electronic+nuclear charge policy,
production BUPO/RI-BUPO accuracy profiles, exchange and geometry derivatives
remain open. No suitable ORCA/CRYSTAL arbitrary-density folded finite source
is available; independent Gaussian integration, the previously qualified
direct kernel and density derivatives provide the references. No external
SCF parity or full MANTA completion is claimed. Shared FockBuild/UHF,
physical BIPOLE domains and source-qualified symmetry interfaces are preserved.

## Local calculation acceptance

Actual local `vq` job `e9bf566ea061` passed 42 complete Coulomb calculations
in 3.936 seconds, with one CPU and a 512 MiB reservation. It tested source
`b605432437d26bd34804020b1f4f1bf23cf754db` with the benchmark native SHA
above. The source was then rebased onto current main as
`4c5ceb97d21abc7e859267d292471308740546ec`. All 149 pinned MANTA
source/reference files and native bytes are unchanged. Incoming shared Fock
state, symmetry and BIPOLE changes are preserved; this runtime makes no
separate acceptance claim for those later native BIPOLE changes.

Eight scalar cases cover both cells and orders 0/4/6/8. Eight angular cases
cover both cells, Cartesian/pure representations and orders 0/6. Two
order-12 cases, 18 derivative base/perturbation calculations, three near
transfers, two exact fallbacks and one independent zero-weight case complete
the 42. Eleven cases use independent complete signed Gaussian integration;
angular cases reuse the already independently qualified direct finite kernel.
Four density directions are tested at steps 1e-4 and 5e-5.

All 42 fetched NPZ artifacts match the original queue workspace. Energies,
weighted traces, independent Gaussian component energies, J bounds and
directional derivatives were recomputed from the fetched arrays. Source
and native pins agree before execution, after execution and after rebasing.
The largest deliberately approximate order-zero discrepancies are
4.665e-5 Eh in energy and 1.785e-4 in J, within the reported Gaussian bounds.
Higher orders reduce the approximation error; small benchmark errors do not
replace those bounds or establish a production target.

The largest weighted half-trace mismatch is 4.548e-13 Eh. The largest
reported-energy derivative error is 1.269e-9 Eh, and the largest derivative
error recomputed from fetched weighted traces is 2.975e-9 Eh. Independent
Gaussian references differ from the direct source by at most 2.047e-12 Eh
in energy and 9.948e-13 in J. Comparison allowances are 2e-11 for J,
3e-11 Eh for energy/traces and 2e-8 Eh for derivatives; they are separate
from the analytical approximation bounds. Both exact fallback cases match
the complete direct response bitwise and report zero approximation error.

The 26 focused tests, 71 existing integration checks, six benchmark workers,
manifest/changelog/citation checks, Sphinx, staged privacy and diff checks
pass. Rebuilding after adding algorithm citations produced identical native
bytes. The test helper now propagates custom mesh weights into its independent
reference and evidence; the zero-weight calculation exercises that change.
No ordinary CI or separate review chat was required. The finite-source,
roundoff and external-reference limitations above remain in force.
