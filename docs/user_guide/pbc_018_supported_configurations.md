# Bounded PBC 0.18 qualification configurations

This inventory records the demonstrated PBC configurations selected for the
bounded release acceptance in issue #557. It does not expand that release
scope or close the CRYSTAL23 parity programme (#593). Method availability,
input matching and successful SCF termination each establish less than a
matched numerical reference comparison.

The machine-readable source is
[`pbc_018_supported_configurations.json`](../../benchmarks/pbc_018_supported_configurations.json).
It contains seven converged calculation rows, a separate refusal boundary,
and explicit exclusions. This is an evidence inventory, not an executable
`run_suite.py` matrix or a blanket supported-method declaration. Each row's
`claim` identifies exactly what passed. The runner's #557 evidence note maps
`evidence_key` to immutable calculation archives, source/native identities,
queued jobs and original judgements. Operational receipts remain outside the
product repository.

## Included rows and measured results

Energies are Hartree per cell. All seven rows are neutral, three-dimensional,
zero-temperature singlets, with unreduced meshes. The unrestricted ECP rows
have seven electrons in each spin channel. No row here establishes magnetic,
low-dimensional, symmetry-reduced or SKALA acceptance.

| Row ID | Method and basis | Passed comparison | Measured cost |
| --- | --- | --- | --- |
| `lih-rsgdf-gamma` | RHF, STO-3G, SR/LR GDF, explicit metric cutoff 1e-13 | E = -8.335327447080662; matched PySCF 2.6.2 GDF error -1.31490e-9, below 1e-7 | 1317.001 s including setup; 7 iterations |
| `lih-rsgdf-222` | Same LiH configuration, 2 x 2 x 2 mesh | E = -7.922007391408032; matched fitted reference error -4.22104e-9, below 1e-7 | 11063.026 s including setup; 8 iterations |
| `diamond-gamma-pbe0-xc` | Restricted PBE0, def2-SVP, Gamma GDF | E = -75.08648342894216; 9.91167e-8 from the converged XC-domain reference; T2 splitting 3.33738e-6; both below 1e-5 | 2268.98 s for diamond plus diffuse-cell check; separate SCF timer unavailable |
| `gpw-rhf-h2-9` | RHF, H2/STO-3G, 9-bohr cube, 24 cubed grid | Same-model energy/Fock derivative error 4.90266e-11, below 2e-9 | 0.6743 s for SCF plus directional derivative |
| `gpw-rhf-h2-16` | Same model, 16-bohr cube, 48 cubed grid | Derivative error 4.50180e-11, below 2e-9 | 0.4604 s for SCF plus directional derivative |
| `cl2-uks-ecp-gamma` | Unrestricted PBE, Cl2/LANL2DZ, 20-bohr cube | Bond-gradient error 6.09538e-9 Ha/bohr, below 2e-6; molecular energy offset 6.02557e-6, below 1e-5 | Shared two-row command: 378.71 s, including displaced SCFs, gradients and molecular controls |
| `cl2-uks-ecp-311` | Same ECP configuration, 3 x 1 x 1 mesh | Bond-gradient error 6.10973e-9 Ha/bohr; same energy-offset gate passes | Same shared command; do not count twice |

LiH used four CPU slots. Diamond used eight; its whole-command CPU time was
17421.98 s and peak RSS 457.25 MiB. The ECP command used four slots, 669.70 s
CPU time and 5165.26 MiB peak RSS. The original ECP pytest receipt separately
reports 432.697 s. These timers have different scopes and are not interchangeable.
GPW used four numerical threads. These are individual observations, not a
same-hardware speed comparison with CRYSTAL23. LiH's original combined job
completed Gamma before timing out during 222 setup; the separately resumed
222 calculation passed on the same immutable runtime. The timeout remains
part of the evidence.

## Exact inputs and support conventions

The JSON gives Cartesian atoms, lattice matrices in bohr, actual fractional
reciprocal k coordinates and their normalized weights. Its convention is
`k_cart = 2*pi*inverse(A).T @ k_fractional`, with lattice vectors in columns.
Gamma is `(0,0,0)` with weight one. The 222 mesh is every triple from
`{0, 1/2}` with weight 1/8; 311 is `(0,0,0)`, `(1/3,0,0)`, `(2/3,0,0)` with
weight 1/3. These coordinates are resolved from the archived callers and
pinned mesh implementation. They are not an attestation of a CRYSTAL output
mesh; matching SHRINK dimensions alone remains insufficient (#645).

LiH has conventional fcc parameter 7.7158 bohr, Li at the origin and H at
`(a/2,a/2,a/2)`. Both rows explicitly use `def2-svp-jk`, SR fitting, splitting
parameter 0.4 bohr inverse, raw reciprocal precision 1e-10, metric cutoff
1e-13 and all 69 fitting modes. The one-electron domain is pair-complete;
its printed radius is 40.0925 bohr. Printed fitting AO/auxiliary radii are
81.42/64.52 bohr and the LR cutoff is 22.66 Ha. Those rounded diagnostics
must not replace the source-pinned automatic support policy. SCF energy and
commutator tolerances are 1e-11 and 1e-8. J and K are fitted, with neutral
3D Ewald electrostatics and the BvK Ewald exchange correction. This comparison
does not remove auxiliary fitting error or qualify a universal metric policy.

Diamond uses fcc parameter 6.740653086540123 bohr and atoms at the origin and
`(a/4,a/4,a/4)`. The automatic XC density radius is 18.501284106877893 bohr.
The periodic Becke/Gamma-torus grid uses 75 radial points and a 17 x 36
product angular rule, with partition image radius 10 bohr. The comparison
is with the same-method converged 25-bohr domain energy, not a CRYSTAL
result. A support-planning replay on the original source and native binary
recovers the missing non-XC domains:

| Diamond domain | Resolved value |
| --- | --- |
| One-electron AO pairs, pair-complete support | 23.532024462586605 bohr |
| One-electron planning enclosure, before overlap screening | 34.86711490084021 bohr |
| Fitting AO images | 43.9975328815882 bohr |
| Fitting auxiliary images | 57.98878984898395 bohr |
| Fitting LR reciprocal cutoff | 21.031901088329146 Ha |
| Nuclear cutoff / V_ne reciprocal cutoff | 25 bohr / 200 Ha |

The enclosure is not the executed one-electron cutoff. The replay also
reproduces the accepted XC radius exactly. These values complete the recorded
configuration; they do not establish arbitrary-cell convergence.

The later continuous-grid check (`qc465-continuous-grid`, source `f1ad7714`)
uses the same frozen diamond configuration and the original 25-bohr reference
of -75.08648352805882 Ha. Its converged energy is -75.08648292670634 Ha,
absolute error 6.01352e-7 Ha and T2 splitting 3.21507e-6 Ha. Both original
1e-5 gates pass and the XC radius is unchanged. This supplement establishes
affected-path source linkage; it does not replace the original result, its
diffuse-cell check or its measured cost, or qualify the final release binary.
The JSON records it separately under `followup_checks`.

The GPW rows retain the adjoint-symmetric legacy Gamma molecular-limit
exchange model, AO-image cutoff 25 bohr and Ewald nuclear convention with
the FFT Poisson zero mode omitted. Their accepted carrier calls
`run_periodic_rhf_gpw` without a functional: these are RHF, not PBE results.
The finite-difference direction and step are recorded in the matrix. Fixing
the derivative of this energy model does not establish physical bulk GPW
exchange (#534).

The ECP rows place Cl2 at the centre of a 20-bohr cube with bond length
1.99 angstrom. LANL2DZ removes ten core electrons per atom; nuclear and
Hartree electrostatics use the effective charges +7,+7. PBE has no exact
exchange contribution. The archived fixture used `pair_complete_1e=False`
and automatic truncation of the legacy cell-ball support. A support-planning
replay on the original source and native binary recovers a final AO radius of
15 bohr and nuclear radius of 25 bohr. Both meshes and every bond displacement
in the two-step finite difference accept those starting radii in one optimizer
evaluation, with minimum overlap eigenvalue above 0.05124. The Schwarz
thresholds remain 1e-12 for integrals and 1e-14 for derivatives. This recovers
metadata for the captured source-matched preview; it does not qualify a later
physical-support fixture with different domains. The molecular control shares this code's XC
implementation; it is not independent ORCA/CRYSTAL force validation. Earlier
#35 single-point route/refusal and metadata checks remain accepted under
their original scope, without being promoted to general ECP gradients (#579).

The ECP derivative consumer now also requires a receipt binding the terminal
spin densities, orbitals, energies and occupations to the SCF source. Edited
states and contradictory ECP flags are refused. Clearing both the receipt
and the flags still refuses a basis with an applicable ECP requirement or
sidecar potential; light all-electron atoms in the same basis family remain
allowed. Rerun the same configuration
when using older results without this terminal-state receipt; the historical
evidence above retains its original source and numerical scope.

## Refusal and exclusions

Issue #652 is excluded specifically for dense translated NaCl/LANL2DZ on
the GDF RHF/UHF routes and for general projector-tail completeness. Its
historical 25-bohr home-centered projector sphere missed partners when the
AO enclosure grew to 63.0539 bohr. Main contains the physical midpoint-domain
repair, but a recovered RHF example does not close the remaining shared UHF
acceptance or the density-grid artifact failure. The frozen matrix has no
NaCl row. Its only ECP rows are the Cl2 Ewald UKS preview with explicitly
retained legacy support, AO radius 15 bohr and projector radius 25 bohr.
Their evidence cannot qualify the NaCl/GDF route. This is an inventory
exclusion, not a claim that every such input is refused at runtime.

The two #179 LiH rows retain their original source `63031b08`, matched
PySCF 2.6.2 references, explicit metric cutoff 1e-13, 69 fitting modes and
1e-7 Ha tolerance. Their scoped PASS does not close the broader #179 issue
or establish default-metric, dense-cell or unfitted accuracy.

No #531 BIPOLE cancellation row is included. In particular, the 100-bohr
Ne residual of -9.208e-8 Ha fails the unchanged
`abs(delta_E) * L / N_e**2 < 1e-8` gate. The 50-bohr observation alone
does not qualify cancellation. The remaining programme must bound the
combined omitted EE, EN and NN contribution, including charge amplitudes
and image multiplicities, and retain both molecular-limit rungs and a
non-vacuum origin/atom-representative check. This inventory exclusion does
not alter a kernel, weaken the gate or require that programme to close for
the included configurations.

The `dense-mg-rijcosx` row is a qualified refusal (#555): Mg primitive fcc,
conventional parameter 4.51 angstrom, STO-3G, Gamma RHF, dense exchange-image
rank three. Direct and public entries refuse before integral construction.
The retained three-check boundary suite passed in 4.23 s. There is no
converged energy, fitting/XC result or corrected dense-cell COSX claim.

No #531 BIPOLE cancellation configuration is included. The 100-bohr Ne
residual of -9.208e-8 Ha still fails the unchanged coefficient gate
`abs(delta_E)*L/N_e^2 < 1e-8`. The narrow 50-bohr observation does not qualify
the larger-domain claim. The existing formulation analysis follows Dovesi
et al., *Physical Review B* **28**, 5781 (1983), section II,
[doi:10.1103/PhysRevB.28.5781](https://doi.org/10.1103/PhysRevB.28.5781):
control the combined electron-electron, electron-nuclear and nuclear-nuclear
Coulomb error with physical, relabelling-invariant supports. A monopole
estimate or an enlarged radius alone is not that certificate. This remains
separate research; the umbrella need not close to retain the other rows.

The #573 restricted starter matrix's 28-bohr overlap-support check is not
accepted total-energy parity. #648's input/ECP matching tools do not replace
actual reference operators, k coordinates/weights, grids, domains, zero
modes and numerical results. Those obligations and broader HF, DFT, hybrid,
SKALA, spin, dimensionality and performance coverage remain in #593.

## MANTA and remaining runner work

Reuse the independently owned
[MANTA frozen configurations and measurements](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md)
and [control guide](manta_release_controls.md). The four profiles are
`image-tight`, `ordinary-rhf`, `global-pbe0` and `compact-tight`, each with its
fixed-source control. Their exact geometry, custom bases, mesh shifts,
physical supports, fitting and XC settings are serialized by
`benchmarks/manta_release_controls.py:configuration`. That producer remains
the single source for those rows. Component certificates cannot be added
into a relaxed total-energy certificate, and no symmetry or bulk parity
claim is introduced here.

The last reviewed candidate omitted the already accepted `image-tight`
allowance commit `ebbf5957bf93bf1ea7af3bbaef954e3e0fa5f729`. The existing
[candidate inclusion check](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md#candidate-inclusion-check)
records the required bridge. This inventory does not replace the candidate
owner's inclusion check or authorize a release.

The runner can reuse the seven scoped PASS rows and dense refusal. The two
support-radius metadata gaps are resolved by separate queued planning replays,
with original native digests and relevant Python module hashes checked before
execution. The matrix links these supplements by `support_metadata_evidence_key`;
the #557 evidence note retains the immutable recovery receipts. These are
deterministic support reconstructions,
not newly observed SCF results. Original energies, gradient checks, tolerances,
costs and source identities remain unchanged. The continuous-grid diamond
supplement and the ECP terminal-state checks are separate from that metadata
recovery. Release qualification and candidate identity remain with the loop.
The broader #593 programme remains separate.
