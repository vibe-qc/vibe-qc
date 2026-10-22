# Full periodic Coulomb fitting enclosure

`benchmarks/manta_periodic_ri_fitting_bounds.py` measures certificate setup
and reuse at an unchanged RI source. Preparation of the shared RI/BUPO
sources and the preliminary SCF lie outside the certificate timing. Three
constructors share those prepared sources; subsequent checked/plain J builds
must be bitwise equal. No fitting-speed or asymptotic crossover is implied.

The reference includes infinite Ewald and omitted AO-image bounds. The
certificate spans the declared fixed weighted Hermitian density family and
compares the selected RI map with unfitted zero-average Coulomb. It does not
assume a positive variational fitting residual, and includes the selected
rank, coupling and finite-domain discrepancy together. Floating arithmetic,
rank stability and density relaxation remain excluded. Automatic complete-job
attachment and useful auxiliary/source selection remain unfinished.

## Accepted calculations

Local `vq` acceptance used source `9bfd02610edfacb6974b6f254c1d05acbb5d2199`, editable runtime
0.17.23 and native SHA256
`f816edbc5a558af85ccb497fc1c8901c1cabf151366c7d202790e78ae791f078`. All 1,901 source/reference pins and the native hash
remained unchanged. One CPU and 4096 MiB were reserved; measured
largest-process RSS was 415.66 MiB, including the independent
reference calculation. Full inputs are in `fitting_job` in
`tests/test_manta_periodic_ri_fitting_bounds.py`.

The two-k-point Be RHF and Li UHF calculations converge in 9 and 13
iterations. Before SCF, the explicit certificate probes 18 Hermitian
directions under a fixed weighted-density L1 limit of 1000. Loose 1e8
J/energy targets test admission; this is not a recommended accuracy preset.

| Method | Total energy (Eh) | Observed delta-J | Observed delta-E (Eh) | J bound | E bound (Eh) |
|---|---:|---:|---:|---:|---:|
| RHF | -8.74524508028746 | 0.0013343704 | 0.017232600 | 46.037939 | 23018.969 |
| UHF | -3.67643714771935 | 0.00091882946 | 0.0071269109 | 46.037939 | 23018.969 |

The observed differences compare fitted and full unfitted Coulomb at the
same final density. They are not differences between relaxed states.
Independent Gaussian Fourier sums converge between reciprocal cubes 36
and 40 to 1.34e-15 in J and 2.67e-15 Eh in Coulomb energy; sampled transforms
also agree with shifted-contour quadrature. Independent finite-source
HF Fock/energy checks pass at 8e-11, and SciPy SCF densities agree within
3.23e-10. QVF and shared Fock exports agree, and the checked/plain RI J
actions are bitwise identical. No matched ORCA/CRYSTAL custom-domain
reference is available; these independent numerical references qualify
only the stated scope.

The bounds are very conservative and dominated by the full-reference
remainder. The maxima of the finite-discrepancy and reference-remainder
J bounds are 0.137474 and 46.024467; their rowwise sum has maximum 46.037939.
Automatic useful fitting accuracy is still unfinished. Sixteen focused
tests and 118 shared RI/BUPO regressions pass, covering complex indefinite
densities, zero weights, entry envelopes, rank limits, all five resource
caps, target/frame refusals, source lifetime and owned exports.

## Certificate overhead

Two fresh processes each construct one cold and two warm certificates.
Warm checked/plain J build order alternates; all pairs pass bitwise equality.

| Measurement | Result |
|---|---:|
| Cold setup wall range (s) | 0.253652-0.256914 |
| Median warm setup wall / CPU (s), four samples | 0.249926 / 0.249756 |
| Median checked / plain J build wall (s) | 0.005350 / 0.005408 |
| Median checked / plain J build CPU (s) | 0.005349 / 0.005400 |
| Median paired J build wall delta (s) | -0.000095 |
| Joint modeled peak (bytes) | 364945986 |
| Benchmark peak RSS (MiB) | 259.53 |

Setup excludes shared source preparation and the preliminary SCF. The
joint model reserves both source plans and certificate scratch; process
RSS also includes Python, libraries and unmodeled test/reference work.
Reuse overhead is within timing noise. Up to `n_k n_AO^2` complete probe
pairs are required, so neither linear scaling nor a useful production
crossover is established.
