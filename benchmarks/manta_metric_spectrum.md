# Stored metric spectral enclosure benchmark

`manta_metric_spectrum.py` measures the shared certificate on synthetic complex
Hermitian matrices with two, eight and 32 functions. The eigensolve occurs
outside the timed region. Three consecutive observations measure the residual,
orthogonality, eigenvalue and rank enclosure; this is not a complete QC timing
or an asymptotic performance qualification. Work scales cubically, and every
periodic RI owner now pays this audit cost once. No factor changes are made.

| Dimension | Wall seconds, three observations | Eigenvalue bound | Rank radius |
| --- | --- | --- | --- |
| 2 | 4.91599e-06, 2.37499e-06, 1.00001e-06 | 1.22125e-15 | 1.01874 |
| 8 | 2.7333e-05, 2.63751e-05, 2.60411e-05 | 1.03251e-13 | 0.556161 |
| 32 | 0.00162208, 0.00161717, 0.00165071 | 1.5038e-12 | 12.578 |

Complete periodic RI-BUPO checks use the existing reference-energy profiles,
with paired fixed selected-source controls. All sixteen calculations converge.
The table records one automatic calculation per profile; it is not a timing
distribution or a before/after speed claim.

| Method | Profile | Wall seconds | Iterations | Energy / Eh |
| --- | --- | --- | --- | --- |
| rhf | reference-energy | 2.913710 | 10 | -7.142711309871 |
| uhf | reference-energy | 2.825746 | 10 | -2.600317249155 |
| rks | reference-energy | 3.141376 | 9 | -9.281975051461 |
| uks | reference-energy | 3.189987 | 11 | -3.908126100059 |
| pbe0 | reference-energy | 3.077823 | 9 | -8.774718842496 |
| hse06 | reference-energy | 3.080846 | 9 | -8.956216604105 |
| rhf | tight | 11.832726 | 10 | -7.142707212914 |
| uhf | tight | 12.016111 | 10 | -2.600315310703 |

Scientific source `a070f8860811789fd07fcc48f814ff2ee585fec2`. Across 8 exported metric
receipts, all spectrum and rank flags qualify. The largest eigenvalue enclosure
is 5.07371922e-14; the smallest strict rank radius
is 1.32231251e-09. These are absolute metric units,
not SCF energy tolerances. Tight HF combined J/energy targets remain 1e-6/1e-4 Eh.

Fixed controls reproduce energies, densities, Fock matrices, physical source
identities and full-Coulomb certificates bitwise. Four HF states, fitted J and
reciprocal inventories also match accepted independent Gaussian/libint,
NumPy/SciPy and Fourier36/40 evidence exactly, allowing those references to be
reused. No matched local ORCA/CRYSTAL custom-domain reference is available.
KS uses fixed-source controls and retains that external-comparison limitation.

Independent Decimal eigenvalue witnesses and arbitrary Hermitian perturbations
check the spectral proof. Cutoff crossings, deficient eigenvectors, underflow,
overflow, all/full retained spaces and resource boundaries are covered.
The bounds apply to the exact stored matrix, excluding integral construction,
factor/inverse or Fock arithmetic, overlap rank and density relaxation. They
are diagnostic; automatic rank-aware selection and production-scale cost
qualification remain unfinished. See scientific contract Section 5.31.


## Rank-driven fitting admission

`manta_metric_spectrum.py --rank-admission --repetitions 1` now exercises
complete periodic searches whose initial J/energy bounds pass but whose
initial stored metric rank is unresolved. The optional 1e-10 perturbation
request forces selection to continue through the existing shared producer
APIs. Each search has a complete fixed selected-source control, and the
provenance records every rejected rank request. These are one-observation
fixture timings, not a production-size cost or speedup claim.

| Method | Search advanced by rank requirement | Trials | Wall seconds | Selected radius | Energy / Eh |
| --- | --- | --- | --- | --- | --- |
| rhf | fixed | 2 | 0.406294 | 0.00072091729 | -8.745245080287 |
| uhf | fixed | 2 | 0.444001 | 0.00072091729 | -3.676437147719 |
| rhf | reference | 2 | 0.404230 | 0.00072091729 | -8.745245080287 |
| rhf | domain | 2 | 0.147560 | 2.60474476e-06 | -7.233807671871 |
| rhf | auxiliary | 2 | 0.677852 | 0.000445744506 | -8.728381639757 |

Scientific source `fa746e2f2791aaa42f4a546c21fdea3ad1dafbc3`. All five searches and their controls
converge with bitwise matching energy, density, Fock, physical source and
full-Coulomb certificate. Reference readiness is allowed to pass while the
provisional rank fails, so the reference is not enlarged to repair a metric
rank problem. Four searches with an unattainable request refuse before SCF.

Eight existing soft/tight RI-BUPO profiles also converge with the new 1e-10
rank request and fixed selected-source controls, bringing the total to 26
complete calculations. Four HF states, fitted J and reciprocal inventories
match accepted independent Gaussian/libint, NumPy/SciPy and Fourier36/40
references exactly. All eight energies equal the prior accepted values.
No matched local ORCA/CRYSTAL custom-domain reference is available; KS retains
the fixed-source and shared-functional comparison limitation.

Strict equality with the certified radius is refused. None preserves the
old acceptance policy, while zero requires a resolved rank. The requested
margin is not a measured integral error or a total-energy tolerance. This
extends the earlier diagnostic with optional automatic admission; it does
not complete integral/factor/Fock arithmetic, all-q rank qualification,
relaxed accuracy or production-scale validation. See contract Section 5.32.


## Stored factor inverse error bounds

Scientific source `f7e3c00b47adf1cb34bc88264b9f8d70e796ad03`. Sixteen complete local periodic
calculations (eight profiles and fixed selected-source controls) converge.
All eight exported selected metric inverse certificates qualify. Physical
energies, densities, Fock matrices, sources and full-Coulomb certificate
identities remain bitwise equal to controls. Four HF states and fitted J
arrays also match accepted independent Gaussian/libint, NumPy/SciPy and
Fourier references exactly. No matched ORCA/CRYSTAL custom-domain reference
is available; KS retains the fixed-control comparison limitation.

| Method | Profile | Wall seconds | Inverse error bound |
| --- | --- | --- | --- |
| rhf | reference-energy | 4.725194 | 59624.185 |
| uhf | reference-energy | 4.507778 | 59624.185 |
| rks | reference-energy | 5.129166 | 94.7213132 |
| uks | reference-energy | 4.944817 | 94.7213132 |
| pbe0 | reference-energy | 5.042157 | 94.7213132 |
| hse06 | reference-energy | 4.457929 | 94.7213132 |
| rhf | tight | 18.339180 | 86917.294 |
| uhf | tight | 20.869079 | 86917.294 |

These are absolute inverse-operator units, not Eh. Bounds grow for nearly
dependent auxiliary functions; they are not fitted J or SCF energy error
bounds. Integral construction and coupling/Fock contraction roundoff remain
excluded. The full-Coulomb certificate still has `roundoff_included=false`.

The combined synthetic spectrum/factor audit uses `--inverse`; eigensolves
are outside the timed region. Three observations per dimension follow.
These small cases qualify accounting, not production scaling.

| Dimension | Wall seconds, three observations | Inverse error bound |
| --- | --- | --- |
| 2 | 1.0209042e-05, 4.6669738e-06, 1.7910497e-06 | 4.4703913e-15 |
| 8 | 6.9124973e-05, 6.5541943e-05, 6.5290951e-05 | 3.83574427e-13 |
| 32 | 0.0048388341, 0.004341042, 0.004856334 | 6.0846042e-14 |

Independent 100-digit Decimal two-by-two inverse witnesses cover actual
stored complex entries, perturbed vectors/factors, full/truncated/empty
spaces and scales 1e-100 through 1e100. Further checks cover clusters,
Gamma/non-Gamma owners, unresolved ranks, overflow and all admission caps.
See scientific contract Section 5.33 for the finite divided-difference
proof; no first-order approximation is treated as a finite error bound.


## Coupling-projected metric action admission

Scientific source `79564c6b1be9ad6ccfd67b40de5acc81b8395ce9`. Twenty-eight complete local
calculations and four deliberate pre-SCF refusals pass. Eight soft/tight
HF/KS profiles request the same numerical metric-action targets as their
full-Coulomb J/energy targets. Fixed selected-source controls reproduce
energy, density, Fock matrices and physical identities bitwise. Four HF
states, fitted J and reciprocal inventories also match accepted independent
Gaussian/libint, NumPy/SciPy and Fourier references exactly. No matched
ORCA/CRYSTAL custom-domain reference is available; KS retains its fixed-source
comparison limitation.

| Method | Profile | Wall seconds | Metric J bound | Metric energy bound / Eh |
| --- | --- | --- | --- | --- |
| rhf | reference-energy | 3.416713 | 5.36318505e-10 | 1.58130338e-07 |
| uhf | reference-energy | 3.375801 | 5.36318505e-10 | 1.58130338e-07 |
| rks | reference-energy | 3.821058 | 1.34079592e-10 | 3.95325816e-08 |
| uks | reference-energy | 3.779387 | 1.34079592e-10 | 3.95325816e-08 |
| pbe0 | reference-energy | 3.597991 | 1.34079592e-10 | 3.95325816e-08 |
| hse06 | reference-energy | 3.645003 | 1.34079592e-10 | 3.95325816e-08 |
| rhf | tight | 13.851030 | 5.36318524e-10 | 1.5813034e-07 |
| uhf | tight | 14.062252 | 5.36318523e-10 | 1.5813034e-07 |

Soft full-Coulomb and metric targets are 1e-3/5e-2; tighter HF targets
are 1e-6/1e-4. These are separate conditions, not additive contributions to
a verified total SCF error. Actual integral and Fock-contraction roundoff
remain excluded. The projected bound uses the actual coupling directions;
the preceding inverse-operator values are not J/energy tolerances.

Five further cases reject their initial unresolved metric-action bound,
advance the bounded metric/domain/auxiliary search and retain the actual
passing source. A sixth case exercises nonzero ordered multipole coupling.
All six match their fixed-source controls bitwise.

| Method | Search | Nonzero ordered multipoles | Trials | Metric J bound | Metric energy bound / Eh |
| --- | --- | --- | --- | --- | --- |
| rhf | fixed | False | 2 | 5.23748402e-13 | 1.54424146e-10 |
| uhf | fixed | False | 2 | 5.23748402e-13 | 1.54424146e-10 |
| rhf | reference | False | 2 | 5.23748402e-13 | 1.54424146e-10 |
| rhf | domain | False | 2 | 5.23748402e-13 | 1.54424146e-10 |
| rhf | auxiliary | False | 2 | 4.40844507e-06 | 0.00107960902 |
| rhf | fixed | True | 2 | 5.23748402e-13 | 1.54424146e-10 |

Those six transition cases use loose 1e8 fitting and metric-action targets
to isolate qualification from the existing physical-domain choice. Four
1e-100 metric-action requests refuse before SCF. Independent 100-digit
Decimal tests cover full/truncated/empty spectra, complex coupling columns,
arbitrary factor errors, J and energy envelopes. An ill-conditioned fixture
shows a projected bound more than 1,000 times tighter than the worst-direction
inverse estimate. All public and aggregate resource caps are exercised.

Reproduce transition cases with `manta_metric_spectrum.py --metric-action`.
Timings are single observations, not a production scaling claim. Explicit
screened/ordered-multipole tiles are supported; compressed/global moment
and pair-tree actions retain the new policy guard. Contract Section 5.34
gives the finite residual proof and the remaining arithmetic exclusions.
