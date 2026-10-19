# Full Gaussian infinite Ewald domain evidence

Run `.venv/bin/python benchmarks/manta_gaussian_domain.py` from the rebuilt
editable checkout. Six fresh single-thread workers cover cubic/skew cells
and integer cubes of radii 2, 4 and 6. Five bound constructions follow a
warm-up; one four-term pair majorant interacts with a two-term auxiliary
majorant. Canonical domain validation and sorting run once per construction.
Producer construction is outside the timed region. Complete smooth-charge
energies and independent Fourier references are outside the timed region,
inside process wall time and peak RSS.

Measurements use Darwin arm64, Python 3.14.8 and NumPy 2.5.3. Native SHA-256:
`a4318dc54197ff70c5d960eeba0bef0c1ab10eeadf1658631f0cbb0655c386d0`.

| Cell | Cube radius | Wall/CPU ms | Process peak MiB | Modeled peak bytes |
|---|---:|---:|---:|---:|
| Cubic | 2 | 0.030/0.030 | 174.20 | 8437904 |
| Cubic | 4 | 0.153/0.153 | 173.73 | 8553872 |
| Cubic | 6 | 0.501/0.501 | 173.52 | 8835728 |
| Skew | 2 | 0.031/0.031 | 173.59 | 8437904 |
| Skew | 4 | 0.162/0.162 | 173.45 | 8553872 |
| Skew | 6 | 0.541/0.541 | 173.61 | 8835728 |

The modeled peak includes the default 8 MiB backend margin and admitted
copies and exports. It is not process RSS or a measured FLOP count.
Benchmark timing includes eight majorant-pair envelope evaluations; the
reference energies below use two normalized smooth Gaussian charge clouds.

| Cell | Cube radius | Energy error (Eh) | Domain energy bound (Eh) |
|---|---:|---:|---:|
| Cubic | 2 | 4.387e-13 | 2.794e+00 |
| Cubic | 4 | 2.220e-15 | 5.090e-05 |
| Cubic | 6 | 2.665e-15 | 2.327e-12 |
| Skew | 2 | 1.179e-12 | 3.159e+00 |
| Skew | 4 | 3.331e-16 | 6.998e-05 |
| Skew | 6 | 0.000e+00 | 3.993e-12 |

The finite energy uses analytical Gaussian erfc/erf convolutions and the
matching Ewald reciprocal sum, including finite Gaussian self interactions.
The independent reference uses the full zero-average Coulomb Fourier sum.
Neutral signed cloud charges are contracted only after component bounds;
the bound uses their absolute values. No point-charge self is subtracted.

Tests cover positive pointwise majorants for every signed angular AO
component, Cartesian and pure representations, translated orbital pairs,
large-coordinate center rounding, immutable source identity, exact resource
caps and count-only array refusal. All four pair/auxiliary combinations
are compared with full Gaussian convolution and Fourier references. Two
signed angular cases use independent polynomial Gaussian quadrature.
Complete energies enlarge the cube to satisfy a declared 1e-8 Eh bound.

A separate complete calculation combines native moments and finite Ewald
coefficients with the finite Gaussian approximation ledger and the new
infinite-domain bound. It checks matching physical kernel identities and
compares to independent full Gaussian Fourier energies. Its errors remain
separated into finite approximation, omitted domain and comparison roundoff.

All 48 focused tests pass (47 initial tests plus the added native energy
integration case). Six previous extent inventories and four compact-domain
receipts are bitwise unchanged. The code shares positive extent envelopes
and scalar shell bounds without changing those accepted calculations.

The source qualification is `native-gaussian-infinite-ewald-domain-remainder`.
The real bound uses a Gaussian convolution decay; reciprocal bounds and
kernel identity retain the physical Ewald parameter. Ewald (1921), DOI
10.1002/andp.19213690304, section IV.1, Eqs. 37-40 provides the split.
The majorant and convolution bound is a separate derivation documented in
the scientific contract, using the Gaussian product theorem.

No matched ORCA/CRYSTAL calculation implements this finite Gaussian kernel
profile. Independent analytic convolution and converged Fourier/quadrature
references supply the numerical check. There is no electronic SCF or external
parity claim. AO pair-image omission, aggregate density/Fock error, roundoff,
charge-policy selection and production periodic BUPO/RI-BUPO remain open.

## Complete local calculation acceptance

Source `78291d04` matches the validated snapshot
`bca44ca9ac5bf4dc83bd7e9777e21fd9ecc0ecf6`; local `vq` job `0713d6bf2ef9`
used the native SHA above. Before/after and rebase checks pin ten scientific
implementation/reference files and the binary. All 51 fetched numerical
artifacts match the original queue workspace. All 49 energy artifacts,
including adaptive intermediate cases, were recomputed from fetched matrices
and charge vectors. The isolated job completed in 5.49 seconds.

The run contains 24 fixed complete neutral Gaussian energies (cubic/skew,
two Ewald parameters, original/translated representatives, three cube sizes),
eight adaptive targets, two signed angular component calculations and one
native moment/finite-kernel energy. Including adaptive intermediate points,
48 distinct domain-energy calculations were evaluated; earlier cases were
reused. All adaptive bounds are below 1e-8 Eh at cube radii 7 through 9.
The largest accepted bound is 1.321e-9 Eh and the largest energy difference
is 2.721e-15 Eh. The largest fixed-case energy difference is 3.302e-8 Eh.
A separate 5e-13 Eh comparison allowance covers numerical evaluation noise;
it is excluded from the analytical bound.

Two low-split translated cells have finite-energy periodicity defects of
3.143e-8 and 3.302e-8 Eh. The triangle bounds through the common infinite
kernel are 135.03 and 151.74 Eh using only the real-domain errors. The two
higher-split defects are at roundoff. These short-cutoff bounds are loose;
the adaptive cases demonstrate how to meet a useful target. No exact
finite-real-domain covariance is claimed.

The signed angular pair/auxiliary cases have maximum component differences
of 3.148e-6 and 1.126e-6 Eh, within their native component bounds. Refining
independent Gaussian quadrature from 72 to 104 nodes preserves the results
within the stated comparison allowance. The separate native energy differs
from its complete Fourier reference by 1.798e-10 Eh. Its combined analytical
bound is 5.662e10 Eh, dominated by the pre-existing finite-approximation
ledger; this integration check establishes compatible source/kernel error
accounting, not a useful full-method accuracy profile. Tightening that ledger
and qualifying aggregated traversal remain required before production use.

Focused tests, manifest/changelog/citation checks, documentation and staged
privacy checks pass. Incoming BIPOLE two-p-shell optimization is preserved in
the rebase; it is not claimed compiled into this reused MANTA runtime. Shared
FockBuild/JKBuilder interfaces and qualified symmetry/domain ownership remain
unchanged. The external-reference and scientific scope limitations above apply.
