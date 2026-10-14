# Producer-owned periodic GDF coupling actions

The exact paired RI-J baseline now consumes forward/adjoint actions from
`PeriodicGDFTiles`. This isolates the raw finite coupling from spin/weight
and metric policy before a periodic RI-BUPO approximation is introduced.
Both actions traverse the identical ordered shell tiles. Physical GDF
sphere/capsule and reciprocal domains, zero modes, factor/rank and source
identities retain their definitions. No symmetry reduction or BIPOLE parity
is inferred from the new API.

## Scientific acceptance

The implementing runner submitted and retrieved a complete local `vq`
calculation at source `bd829b3aed11289256aeed5d3fe9c8c0a25c93bb` with native SHA256
`e86bfcdd16a5092957e707f27daf5a372de0de8354ac91d5f3e41449dc09ee16`. Before/after source and runtime pins cover
1583 files. The private operational record retains job,
host, outputs and their hashes; none are stored in the product checkout.

Seven paired public states plus seven matched retained-mode controls passed:
shifted Be RHF, H UHF with empty beta, Li UHF converged/capped, Be PBE RKS,
and Li PBE UKS converged/capped. Each state exercises the actual producer
actions inside the existing method. Capped states remain nonconverged and
unconfirmed. Shared Fock operators retain evaluated densities and method
energies; QVF densities match the evaluated spin density exactly.

The numerical references independently build finite metric/coupling blocks
with libint, integrate Gaussian Fourier terms, factor with NumPy and solve
SCF with SciPy. KS checks use independent grid contractions of the same libxc
functional. A separate NumPy-only assessment checks the fetched arrays and
126 managed output hashes. No matched ORCA/CRYSTAL case is
available for these custom finite supports. This evidence establishes the
selected finite operator, not converged basis, k mesh or real/reciprocal
cutoffs, and not the accuracy of an approximate periodic RI-BUPO method.

| Complete-state quantity | Maximum absolute difference | Acceptance gate |
| --- | ---: | ---: |
| Fock entries | 5.151435e-14 | 8e-11 |
| Energy at evaluated density / Eh | 9.414691e-14 | 8e-11 |
| Independently relaxed energy / Eh | 1.030287e-13 | 8e-11 |
| Density entries | 2.790789e-10 | 5e-9 |
| Paired versus retained-mode Fock | 2.220446e-15 | 8e-11 |
| Paired versus retained-mode energy / Eh | 2.664535e-15 | 8e-11 |
| Coulomb energy versus weighted trace / Eh | 8.881784e-16 | 8e-11 |

The same job checks six arbitrary complex coupling inputs: Gamma/nonzero q
with Coulomb, erfc and erf physical kernels. Maximum forward error is
2.411662e-13, adjoint error
2.343714e-13, and complex pairing defect
2.273737e-13. Independent finite Gaussian/Boys/
Fourier contractions meet the unchanged 4e-12 absolute plus 3e-13 relative
matrix gates; the complex pairing gate is 4e-12 absolute.

## Focused checks

The 34 new tests cover arbitrary non-Hermitian complex inputs, image shifts,
physical range kernels, signed angular contractions, optional metric storage,
screened finite sources, C/F layouts, linearity, zero inputs, immutable owned
exports and one-short resource refusal before reading numerical input.
The 25 existing paired RI tests and 142 GDF tile/RI channel/target tests pass,
including retained ranks zero through six and weighted energy derivatives.
Three inventory checks and the changelog format check also pass: 205 distinct
checks. The initial fixture run omitted explicit action budgets; correcting
the harness required no numerical-code or tolerance change. Full CI was not
requested for this development increment.

## Measured cost and scope

Three isolated processes each measured three warm calls with one numerical
thread on a shared host. The fixture has two k points, two AOs, six auxiliary
functions and retained rank six. Code/runtime hashes and samples are emitted
by `python benchmarks/manta_periodic_gdf_actions.py`.

| Operation | Median wall time / ms | Executed shell calls | Coupling passes |
| --- | ---: | ---: | ---: |
| Forward | 7.668 | 9,750 | 1 |
| Adjoint | 7.856 | 9,750 | 1 |
| Paired RI-J | 15.859 | 19,500 | 2 |

Each action consumes 48 coupling elements and executes
9,750 primitive products. Modeled peak storage
is 25,453,856 bytes for one action and
25,478,844 bytes for the unchanged conservative joint RI plan.
The action plan adds `64 * (nk*nao**2 + naux)` bytes of numerical workspace
and a backend margin around the existing single-tile source plan. It does
not retain full T, fitted B, or a second metric. A consumer must admit all
repeated actions and other live storage jointly. These measured costs are
not a new speedup or scaling claim.

The next approximation must preserve this paired source contract and qualify
its partition/error propagation against the native factor. Canonical BUPO
pair images cannot silently replace the GDF ordered sphere/capsule domains.
Automatic whole-method accuracy and approximate periodic RI-BUPO remain open
in the [full 0.18.x capability plan](../docs/design/manta_validation_plan.md).
