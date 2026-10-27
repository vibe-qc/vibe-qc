# Complete finite Ewald Coulomb evidence

Run `.venv/bin/python benchmarks/manta_periodic_ewald.py` from the rebuilt
editable checkout. Eight fresh single-thread workers cover cubic/skew cells,
signed contracted s-s and angular s-p bases, and reciprocal batches 1/32.
Each takes five measurements after warm-up. Existing periodic pair-source
setup is outside the timed region. Ewald setup admits and retains that source
and canonical domains; build includes the complete real, reciprocal and gauge
J response and component energies. Independent references run outside the
measurement region but inside worker wall time and process RSS.

Darwin arm64, Python 3.14.8, NumPy 2.5.3. Native SHA-256:
`08a1996cc05e297dda042bb5e4674e59267c5115e9616f1bc5a6befa63fa1d7c`.

| Cell | Basis | Batch | Setup ms | Build ms | Process peak MiB | Modeled peak bytes |
|---|---|---:|---:|---:|---:|---:|---:|
| Cubic | s-s | 1 | 0.048 | 18.374 | 166.89 | 17119912 |
| Cubic | s-s | 32 | 0.040 | 14.739 | 166.06 | 17123880 |
| Cubic | s-p | 1 | 0.020 | 1.889 | 166.08 | 17137744 |
| Cubic | s-p | 32 | 0.020 | 0.455 | 166.25 | 17141712 |
| Skew | s-s | 1 | 0.035 | 18.813 | 165.95 | 17119912 |
| Skew | s-s | 32 | 0.041 | 15.078 | 166.05 | 17123880 |
| Skew | s-p | 1 | 0.022 | 1.879 | 166.08 | 17137744 |
| Skew | s-p | 32 | 0.020 | 0.460 | 166.20 | 17141712 |

The s-s source has 11 canonical pair entries, 27 real images and 124
reciprocal vectors. The s-p source has seven canonical entries, three real
images and 124 reciprocal vectors. These deliberately different workloads
exercise contraction and angular digestion; their timings are not a basis
scaling comparison. Each build streams one quartet or reciprocal tile, with
no global four-index or all-pair Fourier cache. The resource model includes
the retained pair-source reservation, driver arrays and shifted shell copies;
its conservative full-basis Fourier bound per panel is not measured work.
Process RSS includes imports and reference arrays and is not the modeled
native workspace. This is an all-direct fallback foundation, with no BUPO
acceleration claim.

Every complex response element and each real/reciprocal/gauge energy is
compared with independent full Gaussian integration. The largest benchmark
matrix error is 5.774e-15 and energy error is 9.770e-15 Eh. The energy and
weighted half-trace agree to floating-point precision. The constant gauge
is included once and Gaussian electronic self integrals are retained.

The 31 focused tests pass. Signed contractions and Cartesian/pure p and d
components use Gaussian characteristic functions, bivariate Gauss-Hermite
polynomials and Laplace quadrature, with an independent quadrature refinement.
The scalar reference uses a stable erf difference near coincident centers
and an erfc difference at large distance. General complex Hermitian k-mesh
densities, nonuniform/zero weights, two-step energy derivatives, cross-k
responses, linearity, exact resource caps, ownership, unaligned inputs,
origin/time-reversal/reciprocal covariance and source refusal are checked.
After adding a conservative shifted-shell storage allowance, the five exact
resource-boundary tests were rerun and pass with this final runtime.
Three prior lattice matrices/identities and one pair contraction/identity
are bitwise unchanged by the shared domain-helper extraction.

The Ewald convention follows the declared scientific contract and Ewald1921
IV.1, Eqs.37-40 (DOI 10.1002/andp.19213690304). Gaussian integrals use MD;
q=0 coefficients follow Sun2017 Eqs.16-17 (DOI 10.1063/1.4998644).
No suitable ORCA/CRYSTAL calculation exposes the same arbitrary-density
canonical finite quadratic form. Independent Gaussian integrals, ordered
Bloch Fourier sums and density derivatives supply the reference. No external
production SCF parity is claimed.

Qualification `finite-canonical-pair-ewald-j` covers the retained source and
finite kernel. Canonical folding is periodic in the infinite Ewald limit;
finite real-domain covariance error remains explicit. Domain convergence in
the tested cases does not certify omitted AO pair images or a general accuracy
profile. Kernel-admitted bubble traversal, aggregated Gaussian/domain errors,
full electronic+nuclear charge policy, exchange, production SCF selection
and geometry derivatives remain required. Shared FockBuild/UHF, physical
BIPOLE domains and qualified symmetry interfaces are preserved.


## Local calculation acceptance

Source `82c094bc00aea1066e2e1546e75d1689153b5924` was exercised by actual
local `vq` job `f6a535449383` with the native SHA above. Twenty-four native
implementation/reference files and the binary were pinned before and after
execution. All 42 fetched numerical artifacts match the queue workspace;
component energies, independent reference energies, weighted density traces
and directional derivatives were recomputed from the fetched arrays.

The calculation covers eight signed contracted s-s cases (two cells, two
eta values and two density seeds), eight Cartesian/pure angular s-p cases,
four additional angular density directions at steps 1e-4 and 5e-5, two full
d-shell cases and four enlarged-domain comparisons. The derivative bases
and perturbations contribute 20 complete calculations, for 42 overall.
Execution took 6.394 seconds. Maximum finite-domain differences are
8.438e-15 for a J element, 1.333e-14 Eh for total energy and 1.599e-14 Eh
for a component energy. The largest independently recomputed weighted
half-trace mismatch is 6.218e-15 Eh. The largest two-step derivative
mismatch is 2.565e-11 Eh.

Enlarged real cubes of radius 3 and reciprocal cubes of radius 6 at eta
0.45 and 0.85, in both cells, agree with independent undamped ordered
Gaussian Fourier energies (reciprocal cube radius 7) within 5.463e-14 Eh.
This is convergence evidence for these sources, not a rigorous infinite-domain
bound. The declared comparison allowances are 3e-12 (plus 3e-12 relative)
for finite matrices/component energies, 5e-12 for weighted traces, 2e-8 for
finite-difference derivatives and 3e-10 for the enlarged-domain limit.
These allowances are numerical comparisons, not certified error budgets.

The existing 31 scientific/API tests and five updated resource checks pass;
the eight benchmark workers, manifest/changelog/citation checks, Sphinx,
staged privacy and diff checks pass. Incoming finite-erfc BIPOLE reference
tests are preserved. No ordinary CI or second review chat was required.
The finite-source and external-reference limitations above remain in force.
