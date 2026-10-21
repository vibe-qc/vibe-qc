# Producer-qualified global GDF pair support

The shared `PeriodicGDFPairTree` supplies global ordered pair geometry, exact
pair-dependent auxiliary capsule masks, complete canonical auxiliary-image
unions and source-owned Bloch-frame factors. It uses the same finite GDF
sphere/capsule predicates and prepared Gaussian extents as the existing
coupling producer. The shared bubble builder groups the ordered inventory;
it does not substitute molecular triangular or canonical Ewald pair weights.

This is the finite-support foundation for global periodic RI-BUPO. Existing
SCF still uses the accepted per-pair auxiliary hierarchy and call-wide moment
actions. Global far-field contractions must next consume this source, and
must independently certify Gaussian interaction errors over wholly included
nodes. An included mask is not a far-field acceptance decision.

## Reproduction and measured preparation

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python benchmarks/manta_periodic_gdf_pair_tree.py --workers 3 --repeats 3
```

Three sequential independent processes on arm64 each measure the first tree,
three rebuilt trees sharing prepared extents, complete auxiliary unions and
mask partitions for every retained image. Source construction is outside
timing. This compact fixture has two orbital and two auxiliary s shells and
two k points. Times below are median first calls across processes or median
nine repeated calls, with one numerical thread. Native/content hashes are
recorded by the script; these are preparation costs, not SCF speedups.

| Order | First tree ms | Prepared tree ms | Complete unions ms | All selected masks ms | Admitted peak bytes |
|---|---:|---:|---:|---:|---:|
| 0 | 0.076 | 0.016 | 0.600 | 0.262 | 36662536 |
| 2 | 0.102 | 0.014 | 0.547 | 0.269 | 37526072 |
| 4 | 0.195 | 0.014 | 0.536 | 0.267 | 38763112 |

All orders retain 32 ordered pair images and 50 shared tree nodes. Across both
auxiliary shells, the complete union has 212 images and 2,602 included pair-image
interactions. The selected-image mask walks visit 8,195 nodes and refine 2,814;
305 included blocks group 909 interactions. Exhaustive independent checks
verify every included/excluded descendant and absence of duplicates. These
geometric groups still need Gaussian far-field certification before they can
reduce integral work.

Admission bounds 2,916 pair slots and 4,394 auxiliary-image candidates per shell
before source preparation. It includes source-owned preparation, shared-tree
copies and sorting, endpoint bands, integer-union construction and query/export
storage. The large difference between admitted capacity and this small actual
inventory is intentional. No auxiliary-by-pair membership tensor or full T/B
is retained. Queries and trees contain no density or metric solution.

## Numerical qualification

40 new tests check the independent ordered integer inventory, every mask and
partition, complete auxiliary unions, cutoff equality, atom and independent
AO/auxiliary image transformations, reciprocal relabeling, signed angular
extents, concurrent queries, source lifetimes and all resource caps. Complete
finite s-Gaussian coupling reconstructed through native masks and phases agrees
with independent libint. The affected shared bubble, Gaussian fitting, GDF,
derivative-domain and existing global-action regressions also pass: 361 source
checks in total.

The canonical pair frame shifts both endpoints by a bra-shell lattice vector.
The producer's pair and auxiliary phases multiply to the original ordered
Bloch phase, including the nonzero-q bra-frame term. Leaves evaluate the exact
original capsule predicate; conservative endpoint bands classify internal nodes
or require refinement. Gaussian radii and approximation tolerances never decide
finite support. No new symmetry, external-program parity or whole-method
accuracy qualification follows.

Global pair-tree far-field contraction, automatic accuracy and the remaining
approved MANTA 0.18.x capability matrix remain unfinished.


## Complete calculation acceptance

Actual local vq execution on source `1b17da0959ac7cc5a2be956f1a6bd4bf93cbb764`
passed four complete
finite-source calculations using the new pair-tree APIs, seven public HF/KS
states and six raw complex-action regressions. Native SHA-256 is
`5f1ee976866c1070e64cc39583559e9746912f2e6048ed51d15893b7ebe64e97`.
The 361 source checks and benchmark use the same native
bytes and matching implementation-content pins. All 1,874 source/reference
pins matched before and after calculation; 63 managed outputs were verified
by hash. A NumPy-only assessment of the fetched arrays passed. Documentation,
released-changelog, diff and staged privacy checks also passed.

Each new source calculation traverses the complete native auxiliary union and
capsule partitions, evaluates independent Gaussian/Boys short-range integrals
with the native frame factors, and adds separately evaluated Fourier and
finite-zero terms. The four cases cover Gamma/nonzero q and shifted/unshifted
atoms. Comparison uses libint short-range integrals and a separate finite
Gaussian/Fourier source. Both complex contractions agree with the exact
producer, and retained-metric Gram responses and quadratic energies agree
with the independent complete source. Maximum discrepancies are:

| Quantity | Maximum absolute discrepancy |
|---|---:|
| Short-range coupling | 7.773e-16 |
| Complete Coulomb coupling | 1.111e-15 |
| Finite metric | 2.665e-15 |
| Forward contraction | 1.807e-15 |
| Adjoint contraction | 2.842e-15 |
| Retained-metric response | 1.195e-14 |
| Quadratic energy, Eh | 1.421e-14 |

Public regressions cover shifted-mesh Be RHF, empty-beta H UHF, converged/capped
Li UHF, Be PBE RKS and converged/capped Li PBE UKS. These still use the existing
per-pair auxiliary hierarchy and call-wide moment actions. Shared Fock
application/energy, exact exchange/Madelung, XC ownership and evaluated-density
QVF remain consistent after integration of main's QVF changes. Capped states
remain explicitly nonconverged. Maximum Fock discrepancy is 4.968e-15, density
discrepancy 1.518e-09, and same-density/relaxed energy discrepancies are
9.770e-14/9.059e-14 Eh, below the existing 8e-11 Fock/energy and
5e-9 density gates.

No matched ORCA/CRYSTAL reference is available for these custom finite domains.
The justified references are independent Gaussian/Boys/Fourier integrals,
libint erfc, NumPy metric contractions and independent SciPy SCF; KS shares
libxc. Geometric masks have exhaustive independent coverage; the existing
approximate Gaussian partition is not independently reimplemented. This does
not qualify bulk parity, new symmetry reductions or whole-method accuracy.
Global pair-tree far contractions remain the next integration step.
