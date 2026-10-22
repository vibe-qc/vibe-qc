# Retained-source infinite-Ewald bounds

Run `.venv/bin/python benchmarks/manta_periodic_bupo_domain.py` from the
rebuilt editable checkout. Six fresh single-thread workers use the signed
contracted s-only/s-p fixtures from the finite BUPO benchmark, at order 6
and component-integral tolerance 1e-5. Centered real and reciprocal cube
radii are 1, 2 and 3. Existing pair/tree/direct-source setup is outside the
timed region. Each worker takes five measurements after warm-up.

Darwin arm64, Python 3.14.8, NumPy 2.5.3. Native SHA-256:
`89e2a68f7b5248a8fb19b833a5dd4c59611791a90aff5e43c4ad51ee9d5435a6`.

| Basis | Cube radius | Bounded setup ms | Bounded build ms | Finite-only build ms | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|
| s | 1 | 1.507 | 28.119 | 28.435 | 120212005 |
| s | 2 | 3.183 | 145.978 | 147.733 | 120289621 |
| s | 3 | 5.954 | 388.693 | 390.319 | 120462277 |
| s/p | 1 | 1.493 | 47.419 | 46.383 | 120512509 |
| s/p | 2 | 2.943 | 214.369 | 215.632 | 120590125 |
| s/p | 3 | 6.175 | 592.260 | 590.885 | 120762781 |

The bound option leaves J and energy bitwise unchanged. Millisecond timing
differences include noise and do not establish a speedup. Each source has
13 retained pair images and 52 Gaussian majorant terms. Every build visits
91 unordered component-pair blocks and 1456 majorant pairs for the domain
bound; 18 finite pair interactions remain in accepted BUPO far blocks.
Construction validates all majorant-center separations once. Domain geometry
and lattice metrics are prepared once, and no quadratic interaction cache
is retained. Process peak RSS is 167.20-170.05 MiB and includes imports and
verification arrays; it is distinct from the conservative native reservation.

| Basis | Cube radius | Real domain energy bound | Reciprocal domain energy bound | Total energy bound |
|---|---:|---:|---:|---:|
| s | 1 | 2.35304e4 | 1.49159e2 | 2.36796e4 |
| s | 2 | 7.85543e1 | 7.66932 | 8.62236e1 |
| s | 3 | 4.48769e-2 | 1.33467e-1 | 1.78345e-1 |
| s/p | 1 | 8.84478e4 | 4.08265e3 | 9.25305e4 |
| s/p | 2 | 2.96135e2 | 2.09918e2 | 5.06052e2 |
| s/p | 3 | 1.63395e-1 | 3.65314 | 3.81654 |

All energy bounds are in Eh. The total also includes the finite BUPO
contribution, 1.125e-6 to 3.066e-6 Eh in these cases. These conservative
small-domain bounds do not meet a production target. They are reported
without substituting measured errors or relaxing the threshold. Separate
atomic periodic-source tests enlarge both domains until energy and J bounds
are below 1e-8, at eta 0.4 and 0.7, and compare complete calculations with
an independent bare reciprocal Gaussian Coulomb limit.

Twenty-one new source, limit, aggregation and API checks pass, followed by
two domain-enlargement checks. The existing 74 Gaussian-domain/finite-BUPO
tests pass. Four saved complete finite BUPO outputs, error bounds and
source/partition identities remain bitwise unchanged. Tests include signed
scalar and Cartesian/pure angular sources, cubic/skew cells, complex mesh
densities, a refined reciprocal reference, boundary images and diagonal
half-energy factors. Independent dense contraction of primitive bounds
checks both canonical scatter arms, including same-AO nonzero images.
Density scaling, zero weights/density, positive underflow, all resource caps,
owned exports and explicit cube/separation refusal pass.

The scientific contract derives the full-source aggregation from native
Gaussian convolution bounds and Ewald1921 IV.1 Eqs.37-40. Qualification
`periodic-bupo-j-with-retained-source-ewald-bound` includes finite BUPO
approximation plus infinite-domain error for the same retained canonical
source. It also bounds the finite kernel covariance defect relative to the
limiting periodic kernel. Omitted AO pair images, roundoff, full
electronic+nuclear charge policy, RI fitting, production SCF selection and
geometry derivatives remain unqualified. No matched ORCA/CRYSTAL
arbitrary-density retained-source profile is available. Independent complete
Gaussian Coulomb references provide the comparison; external SCF parity is
not claimed. Shared FockBuild/UHF, BIPOLE and symmetry interfaces are preserved.

## Local calculation acceptance

Actual local `vq` job `a75ec6dd724e` passed 39 complete calculations in
5.920 seconds, using one CPU and a 512 MiB reservation. It exercised source
`99c1ade3976f7e54709b7075995484636ef8cc68` with the native SHA above.
All 153 pinned MANTA source/reference files and native bytes agree before
and after execution. All 39 fetched numerical artifacts match the original
queue workspace; native/direct weighted traces, independent reciprocal or
Gaussian component energies, J errors and derivatives were recomputed.

Eight moderate scalar/angular cases cover both cells and Cartesian/pure
representations against bare reciprocal Gaussian Coulomb limits. Six tight
mixed-partition cases cover both cells and orders 0/6/8 against enlarged
independent Gaussian Ewald calculations. Seven atomic domain-enlargement
calculations and 18 additional derivative base/perturbation calculations
complete the 39. Four density directions, including zero mesh weights, use
steps 1e-4 and 5e-5. Perturbed reciprocal references use the independently
integrated linear Coulomb action.

The largest deliberately small-domain discrepancies are 2.934e-2 Eh in
energy and 8.469e-3 in J, within the reported bounds. The largest native
weighted half-trace mismatch is 2.132e-14 Eh. The largest reported-energy
derivative error is 1.461e-10 Eh, and the derivative recomputed from fetched
weighted traces differs by at most 3.245e-10 Eh. Independent reference
energy recovery and weighted traces agree within 5.685e-14 Eh.
Comparison allowances are 3e-11 for J/energy/traces and 2e-8 for derivatives,
separate from the analytical approximation bounds.

The declared retained-source energy and J targets are both 1e-8. Eta 0.4
reaches them at cube radius 10, with energy bound 2.969e-15 Eh and J bound
5.913e-14. Eta 0.7 reaches them at radius 8, with bounds 2.650e-18 Eh and
5.279e-17. Values below floating-point comparison resolution do not include
roundoff certification. These results qualify the retained source, not an
untruncated AO density or a production accuracy profile.

The 23 new checks, 74 existing integration checks, six benchmark workers,
manifest/changelog/citation checks, Sphinx, staged privacy and diff checks
pass. An incoming changelog fragment's missing heading was repaired without
changing its content. No ordinary CI or separate reviewer was required.
The qualification and external-reference limitations above remain in force.
