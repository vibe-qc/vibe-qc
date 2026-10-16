# Full AO pair-image Coulomb bounds

Run `.venv/bin/python benchmarks/manta_periodic_image_bounds.py` from the
rebuilt editable checkout. Four fresh single-thread workers take five timed
measurements after warm-up. Existing pair/tree/direct-source setup is outside
the measured region. The baseline includes the retained-source infinite-Ewald
bound; the image option adds the complete omitted AO-image envelope.

Darwin arm64, Python 3.14.8, NumPy 2.5.3. Native SHA-256:
`19611ce9f24b14ae1823b5a097faebeea94da2c16bde93025ea6db2110620b79`.

| Case | Image setup ms | Retained setup ms | Image build ms | Retained build ms | Modeled peak bytes |
|---|---:|---:|---:|---:|---:|
| Sparse signed s | 0.250 | 0.233 | 44.764 | 45.117 | 126827734 |
| Sparse signed s/p | 0.249 | 0.227 | 90.848 | 90.643 | 126935214 |
| Cutoff signed s/p | 0.148 | 0.156 | 20.744 | 20.510 | 126919782 |
| Complete tight s | 0.964 | 0.948 | 121.677 | 120.084 | 128938454 |

J, energy and the finite partition are bitwise unchanged by the option.
Small timing differences include measurement noise; no speedup is claimed.
Process peak RSS is 166.80-168.22 MiB, including imports and verification
arrays, and is distinct from the conservative native reservation. Completion
enumeration and two AO envelope arrays are constructed once. Builds contract
the original weighted density without enumerating missing images again.

| Case | Total energy bound, Eh | Maximum total J bound |
|---|---:|---:|
| Sparse signed s | 2.02824e6 | 7.07270e5 |
| Sparse signed s/p | 5.28907e6 | 7.01530e5 |
| Cutoff signed s/p | 9.97427e8 | 1.58850e8 |
| Complete tight s | 6.05451e-18 | 1.02399e-17 |

The large bounds are deliberate and do not meet a production target. These
sparse/cutoff inventories omit important source and target components, and
the uniform absolute Coulomb envelope is conservative. The complete tight
case has one s primitive of exponent 4 in a cubic cell of side 6 bohr,
all pair images in cube 1, eta 0.4, real cube 4 and reciprocal cube 8.
Its energy and J bounds both meet 1e-8 after enlarging the pair inventory
from cube 0 to cube 1. Bounds below floating-point comparison resolution
exclude roundoff; measured comparisons retain a separate numerical allowance.

Seventeen focused checks pass, together with 49 existing finite-BUPO and
retained-domain checks. Independent references use every ordered shell pair
and image, analytical signed Gaussian Fourier integrals and bare reciprocal
Coulomb coefficients. Tests cover cubic/skew cells, s/p/d and Cartesian/pure
representations, complex densities without time-reversal assumptions,
independent image/reciprocal refinement, sparse interior holes, cutoff
exclusions, zero mesh weights, density scaling and zero/tiny densities.
Both source and target errors, diagonal-image multiplicities, fixed-partition
invariance, owned exports and joint caps are checked.

The scientific contract Section 5.9 derives the native full Gaussian image
sum, uniform Ewald interaction bound and density aggregation. Existing
primary reading of Ewald1921 IV.1 Eqs.37-40 is reused for the split kernel;
the conservative Gaussian envelope is a separately derived bound.
Qualification `periodic-bupo-j-with-full-pair-image-ewald-bound` applies to
the full electronic Coulomb source under the zero-average conducting kernel.
Roundoff, full electronic+nuclear charge/background/self policy, exchange,
RI fitting, automatic production SCF selection and geometry derivatives
remain outside this increment. Shared FockBuild/UHF, BIPOLE physical domains
and qualified symmetry interfaces are unchanged.

No matched ORCA/CRYSTAL arbitrary-density Coulomb component profile is
available. Independent complete Gaussian calculations provide the numerical
reference; external SCF parity is not claimed.

## Local calculation acceptance

Actual local `vq` job `e0ad9bb22dc2` passed 20 complete calculations in
102.023 seconds with one CPU, a 512 MiB reservation and a 600-second limit.
It exercised source `4d355ad97321573e81ceb5dcd5326b414cc6d3dd` and the
native SHA above. All 158 pinned implementation/reference files and native
bytes agree before and after execution. Every fetched NPZ matches the
original queue workspace. Weighted traces, independent reciprocal/component
energies, image and total errors, density mass aggregation and derivatives
were recomputed from the fetched arrays.

Six baseline calculations cover both cells and signed s, Cartesian s/p and
pure s/d sources. A zero-weight base and eight perturbations exercise two
density directions at steps 1e-4 and 5e-5. A central off-diagonal target
excluded by the cutoff, sparse interior holes, two image-enlargement cases
and one tight mixed BUPO case complete the 20. The mixed case is explicitly
compared with the previously qualified independent retained-source Gaussian
Ewald reference; its omitted-image mass is below comparison resolution.
Its finite response and partition remain bitwise unchanged.

Independent reference image cubes 2/3 and reciprocal cubes 9/11 agree to
8.882e-16 Eh and 4.164e-16 in J for the angular skew cases. The largest
deliberate source-omission errors are 1.63533 Eh and 1.34353 in J, contained
by the reported bounds. These cases demonstrate detection of poor source
inventories, not production accuracy. The largest native half-trace error
is 1.422e-14 Eh; independent reference recovery and traces differ by at most
4.619e-14 Eh. Reported-energy and independently recomputed trace derivatives
differ from the analytical response by at most 3.780e-11 and 3.447e-11 Eh,
respectively. Comparison allowances are 3e-11 for J/energy/traces and 2e-8
for derivatives, separate from the analytical bounds.

For the complete tight source, enlarging only the pair-image cube from 0
to 1 reduces the total energy/J bounds from 2.66362e-2/4.50112e-2 to
6.05451e-18/1.02399e-17. Both final bounds meet the declared 1e-8 targets.
The independent bare Gaussian reference uses reciprocal cube 16. This is
an electronic-source target for this fixture; roundoff and the remaining
method requirements listed above are excluded.

The fetched positive mass sums include subnormal values. Their verification
uses 90-digit decimal products before conversion to binary64; direct binary64
products prematurely rounded individual contributions. The comparison allows
16 minimum binary64 subnormal quanta in addition to relative rounding error.
No analytical or chemistry tolerance was relaxed and no calculation was
repeated for this verification arithmetic correction.

The 17 new checks, 49 existing checks, four benchmark workers, manifest,
changelog, citation, Sphinx, staged privacy and diff checks pass. No ordinary
CI or separate reviewer was required. The full MANTA capability matrix remains
unfinished, including full charge policy, production periodic BUPO/RI-BUPO
selection, gradients and the other declared method profiles.

Rebased source `550abe33bbbea8f29f91851e29cfc6c3fb459d2f` retains all 158 tested MANTA
implementation/reference files and the recorded native bytes unchanged.
Incoming BIPOLE leaf-layout work is preserved. That later performance path
is not claimed compiled or separately accepted by this MANTA calculation.
