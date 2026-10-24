# Periodic bubble hierarchy evidence

Run `.venv/bin/python benchmarks/manta_periodic_bubbles.py` from the rebuilt
editable checkout. Six fresh single-thread workers cover cubic/skew cells
and Cartesian moment orders 0, 6 and 12. Five repetitions follow warm-up.
The existing native pair source is constructed once outside the timed region;
setup measures hierarchy construction over that retained source. Forward and
reverse timing includes pair moments, density/phase contraction and tree
translations. No translation matrix is retained per tree edge. Independent
references run outside the timed region, inside process wall time and RSS.

Darwin arm64, Python 3.14.8, NumPy 2.5.3. Native SHA-256:
`68c5eac3a3363dace1482900d49e35ad5c5faaf23ebac39a588d906d2fb5e59b`.

| Cell | Order | Setup ms | Forward ms | Reverse ms | Process peak MiB | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|---:|
| Cubic | 0 | 0.006 | 0.044 | 0.038 | 166.36 | 33698849 |
| Cubic | 6 | 0.009 | 1.213 | 1.224 | 166.31 | 34182393 |
| Cubic | 12 | 0.021 | 28.311 | 28.224 | 187.02 | 40321153 |
| Skew | 0 | 0.007 | 0.048 | 0.041 | 165.44 | 33698849 |
| Skew | 6 | 0.007 | 1.177 | 1.200 | 166.25 | 34182393 |
| Skew | 12 | 0.022 | 28.594 | 28.691 | 187.09 | 40321153 |

Each source has 11 canonical pair-image entries. The default benchmark tree
has 13 nodes and 23 translations per pass, including leaf translations.
Modeled storage conservatively includes existing source reservations, tree
copies, a full forward/reverse cycle, one translation panel and owned exports.
It is not measured RSS or FLOPs. This foundation adds aggregation work; these
measurements make no end-to-end acceleration or production BUPO claim.

| Cell | Order | Energy error (Eh) | Energy error bound (Eh) | Same-order Fourier error |
|---|---:|---:|---:|---:|
| Cubic | 0 | 1.300e+00 | 1.351e+03 | 2.701e-15 |
| Cubic | 6 | 5.178e-02 | 1.244e+02 | 3.820e-15 |
| Cubic | 12 | 1.293e-04 | 6.598e-02 | 3.382e-15 |
| Skew | 0 | 1.153e+00 | 1.352e+03 | 2.674e-15 |
| Skew | 6 | 5.570e-02 | 1.356e+02 | 3.468e-15 |
| Skew | 12 | 1.604e-04 | 7.212e-02 | 3.468e-15 |

The complete calculation is the Coulomb energy and its whole-mesh density
derivative for a declared finite reciprocal domain, with the zero mode
excluded. Independent ordered Gaussian Fourier integrals retain every
selected image and use no native moment or translation panel. Independent
Gauss-Hermite moments also reproduce the same-order calculation. Taylor
error is bounded with full absolute Gaussian moments and is distinct from
pair-image source ownership. Arbitrary complex Hermitian densities at three
non-TRIM momenta have nonuniform weights and no time-reversal relation.

The 32 hierarchy tests pass. They include all-node weighted adjoints, energy
derivatives through order 12, direct Gauss-Hermite root moments, full/tail
radial bounds, exhaustive disjoint partitions with accept/refine, exact
joint resource caps, ownership and malformed-array refusal. Three old
molecular trees (nodes, plans, partitions and identities) and a periodic
pair forward/adjoint fixture remain bitwise unchanged. Existing molecular
tree tests also pass.

The translation follows the raw-moment BUPO construction (Neese et al.,2025,
DOI 10.1021/acs.jpca.4c07415), composed with the q=0 Gaussian Bloch source
(Sun et al.,2017, DOI 10.1063/1.4998644). The scientific contract gives the
weighted adjoint and density-bound propagation. No matched ORCA/CRYSTAL
calculation implements this arbitrary-density finite Fourier profile; analytic
Gaussian integrals, independent ordered sums and density derivatives supply
the reference. Comparison roundoff is separate from the analytical bound.

Qualification `periodic-pair-image-hierarchy-and-adjoint` is not a production
periodic SCF or infinite Ewald profile. Omitted pair images, periodic Gaussian
kernel far admission, whole-method error, charge/self/background selection,
source-specific symmetry and moving-geometry derivatives remain required.
The molecular triangular inventory guard and shared Fock/BIPOLE interfaces
are unchanged.

## Complete local calculation acceptance

Source `398b9db9` matches validated snapshot
`37929157f56cc2aa711a92e88f6999a70fd8a48d`. Local `vq` job `d07743d3fb44`
used the native SHA above. Thirteen implementation/reference files and the
binary are pinned before/after execution and remain identical after rebase.
All 44 fetched numerical artifacts match the original queue workspace.
Their Fourier energies, independent reference energies, analytical bounds
and weighted density traces were recomputed from fetched arrays.

The run covers 24 complete finite-domain Coulomb cases: Cartesian/pure
representations, cubic/skew cells, leaf capacities 1 and 3, and orders
0, 6 and 12. Four order-8 cases also check energy derivatives at two
finite-difference steps (1e-4 and 5e-5), for 44 unique calculations overall.
Execution took 6.074 seconds. The largest same-order Fourier discrepancy is
4.464e-15 and the largest energy/weighted-trace mismatch is 8.882e-16 Eh.
The largest derivative mismatch is 9.007e-12 Eh.

Order-12 energy differences from exact Gaussian Fourier sums range from
3.114e-6 to 1.115e-3 Eh for these fixed partitions, bounded by 0.00891 to
25.14 Eh. Every case improves from order 0, but these fixed choices do not
meet a strict production accuracy target. A consuming kernel must refine
or evaluate failing blocks directly; hierarchy construction alone does not
supply that admission policy. The separate comparison allowance is 5e-12,
with 2e-11 for the same-order check and 2e-8 for the two-step derivative test.
These allowances are not part of the analytical truncation bound.

All 32 new hierarchy tests and 33 existing molecular-tree tests pass. The
saved molecular/periodic baseline is bitwise unchanged. Manifest, changelog,
method citation, Sphinx and staged privacy checks pass. Incoming BIPOLE
subtree-jump traversal and logarithm bracketing are preserved by the rebase
and are not claimed compiled
into this reused MANTA runtime. No shared Fock, physical domain or symmetry
contract was changed. The reference and scope limitations above remain in force.
