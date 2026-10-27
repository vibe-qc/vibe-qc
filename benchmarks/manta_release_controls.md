# Bounded MANTA 0.18 control qualification

The source-tree driver `manta_release_controls.py` freezes the examples in
the [release-control guide](../docs/user_guide/manta_release_controls.md).
The initial, pre-continuous-grid acceptance source is
`7917b560ffd3d48fbbd1bf4f0ecb66da898e7b4d`; native SHA256 is
`735bfb8b52b54d8fbdf08e4b7e88c6fb239989ac9e1b296a0d2d10b2b216e48f`.
The current-grid integration below completes the
MANTA control contribution. It is not the frozen 0.18 release candidate and
does not replace candidate CI or publication checks.

Eight complete local queued calculations (four enabled cases and four
fixed-source controls) converge. Ten independently requested strict physical
targets refuse before creating computed QVF archives. Each enabled state,
energy, Fock, overlap and core Hamiltonian agrees bitwise with its control
and the saved accepted image/nuclear calculation. Source identities match
where recorded. Source, native, libxc and retained output hashes are pinned.

The following initial observations retain the earlier grid for PBE0.

| Profile | Energy / Eh | Iterations | Wall / s | CPU / s | Native modeled peak / MiB |
| --- | --- | --- | --- | --- | --- |
| image-tight | -8.74524614327748 | 9 | 0.9219 | 0.8769 | 1009.55 |
| ordinary-rhf | -8.74524508028746 | 9 | 0.1698 | 0.1585 | 876.94 |
| global-pbe0 | -9.47931399201 | 9 | 0.1524 | 0.1445 | 865.47 |
| compact-tight | 0.666715572701808 | 11 | 2.6289 | 2.6114 | 326.11 |

These are single complete-job observations, not scaling or speedup claims.
Total acceptance time, including controls and refusals, is 8.53 seconds;
peak process RSS is 320.97 MiB. Modeled memory is a conservative reservation,
not observed RSS. All jobs use one numerical thread and a 4-GiB reservation.

The compact case simultaneously meets every requested 1e-8 component target:

| Component | Certified bound |
| --- | --- |
| Overlap image matrix | 3.93637e-15 |
| Kinetic image matrix / Eh | 1.15849e-14 |
| Density-family kinetic energy / Eh | 3.90781e-13 |
| Exchange matrix / Eh | 8.93099e-12 |
| Density-family exchange energy / Eh | 3.00952e-10 |
| Nuclear potential matrix / Eh | 2.48442e-19 |
| Density-family nuclear attraction / Eh | 1.87507e-17 |
| Nuclear repulsion tail / Eh | 4.50780e-41 |

The tiny nuclear tail is an analytic omitted-domain bound excluding arithmetic;
FP64 subtraction cannot independently resolve it. Earlier high-precision and
Gaussian/Boys/Fourier/libint references remain its evidence. Current nuclear
bounds also reuse the subsequently accepted s-pair image-envelope improvement;
the unchanged physical state agrees with the earlier acceptance exactly.
For `image-tight`, overlap and kinetic matrix bounds are 7.56e-13 and 5.56e-13,
and the kinetic-energy bound is 3.69e-11 Eh. Its RI coupling bounds are
1.30e-9 for J and 2.49e-7 Eh for fixed-density Coulomb energy, below the
requested 1e-6 targets. The ordinary RHF/PBE0 physical targets remain deliberately
loose; their success is control-path qualification, not production accuracy.

Reuse the accepted
[image](manta_periodic_ri_image_accuracy.md),
[exchange](manta_periodic_ri_exchange_accuracy.md) and
[nuclear](manta_periodic_ri_nuclear_accuracy.md) independent witnesses for unchanged
components. No matched ORCA/CRYSTAL reference exists for these custom finite
domains, and PBE0 shares libxc. The new compact simultaneous-control case adds
integration coverage without changing numerical kernels or physical supports.
No whole-method relaxed energy, basis/k/XC convergence, bulk parity, source
symmetry reduction or production speedup claim follows.

An initial job refused before SCF because new example work caps were too small.
Count-only planning then recorded maxima of 9.71e10 shell calls, 1.75e18 scalar
terms and 1.73e18 kernel terms. The corrected explicit limits (1e12 shell and
primitive counts, 2e18 scalar/kernel terms) cover those conservative worst-case
plans. No scientific tolerance or result was changed to make acceptance pass.
The failed record is preserved. Six focused documentation checks passed;
that initial documentation-only qualification reused existing scientific suites.
The subsequent shared-grid compatibility check is recorded below. The final
containing candidate must still verify the accepted source
relationship and complete the normal release gates.


## Continuous periodic grid integration

The containing integration `9a868249930d7f239e50f41d0cb7f14c88e1fd5d` uses
native SHA256
`ec79f1039061fd1fecabb1a438642f6fe434729941f2a98aa5fba43078f8beca`.
The shared periodic partition now switches image
inclusion smoothly across an outer band. MANTA reuses that producer and its
band fraction; its count-only work and workspace admission now includes the
band. The HF cases do not use the XC grid, so their accepted evidence above
is retained. No alternate Fock assembly or symmetry-reduction permission is
introduced.

Only PBE0 was rerun through the local queue: the enabled calculation and
fixed-source control both converge in 9 iterations and agree
bitwise in energy, density, Fock, S and H. All three strict physical targets
still refuse before computed QVF creation. The current energy is
-9.47932169663847 Eh, a change of -7.70463e-06 Eh from
the earlier grid. S and H are unchanged bitwise. This grid change is recorded
explicitly rather than treating the old PBE0 state as current acceptance.

Independent Gaussian/Fourier/libint inputs, NumPy metric contractions and a
SciPy SCF solution give energy, Fock and density errors of
5.33e-15 Eh, 9.44e-16 and
2.04e-10, respectively. Gates are 8e-11 Eh for energy,
8e-11 for the Fock matrix and 5e-9 for density. The unified Fock operator and
total energy agree with the complete native state. This comparison shares
libxc and the generated grid; independent radial/angular/continuous-partition,
Fourier-moment and refined tensor-grid checks separately cover the grid.
No matched external-program bulk parity or converged material accuracy is
claimed.

The current enabled PBE0 observation takes 0.2229 seconds wall
and 0.1685 seconds CPU, with modeled peak reservation
865.47 MiB. The complete queued comparison,
including its control, refusals and independent reference, takes
0.76 seconds with process RSS
305.66 MiB. These are single-job observations.
All 28 focused quadrature checks pass in 10.45 seconds. The combined
qualification consists of
ten complete calculations and thirteen strict refusals across the two
stages. Retain both source/runtime records when qualifying a containing
release candidate; do not duplicate the unaffected HF calculations.


## Corrected 0.18 candidate linkage

The four frozen profiles pass on candidate `95c18c7e` plus the converter-header
repair `a92b4880` and the explicit work allowance described below, tested as
integration source `2d0fdb8ca06b5e37c1f6721667800e9954defdfe`. The local native SHA256 is
`b63d1324333bcd85373eb30afb72bfbdddd8ca153bb8809e3123297339c5f50e`; libxc SHA256 remains
`194ffe4dc47c817fa70560f5706e27710054cd25e830d1ed2e872c51ef819cf2`.
This is acceptance of these controls on that native build, not a claim that
another platform's binary was executed or that all release gates passed.

The physical fixture definitions, exchange and nuclear bounds, shared quadrature
and semilocal XC producer are unchanged. Later optional arithmetic ledgers also
increased unconditional RI preflight
reservations. That concrete admission change required a containing-candidate
check under the original caps; source ancestry alone was insufficient.
The initial `95c18c7e` build failed to import before any chemistry because
separately compiled registration units lacked converter headers. The existing
`a92b4880` repair restores those headers without numerical changes. Its
focused binding checks are reused; the failed import is retained and is not
counted as a calculation. The tested integration contains that repair
and the example allowance update, with no numerical method changes.

The repaired source initially refuses `image-tight` at its old 2e18 scalar-work
cap. Count-only planning now reserves 6.41002e18 scalar and 5.90784e18 kernel
terms; peak planned memory remains below 1.25 GB. The example therefore declares
7e18 scalar/kernel allowances for this profile only. The other profiles keep
2e18, and all memory, shell and primitive caps remain unchanged. These counts
are pessimistic reservations, not observed operations. The original refusal
is preserved and is checked again at the old cap; it is not counted as PASS.
No scientific tolerance, basis, mesh, physical support or selector changed.

Eight complete local queued calculations (four enabled and four controls)
converge, and all ten strict physical-target requests refuse before computed
QVF creation. Every enabled energy, density, Fock, overlap and core Hamiltonian
matches its saved accepted state bitwise. Physical source identities, exact
physical configurations and image/exchange/nuclear certificates also match.
The HF states reuse the initial qualification; PBE0 reuses the continuous-grid result,
not the earlier grid. The automatic coupling selector retains its accepted
selection. The only changed input is the declared `image-tight` work allowance.

| Profile | Energy / Eh | Iterations | Wall / s | Native modeled peak / MiB |
| --- | --- | --- | --- | --- |
| image-tight | -8.74524614327748 | 9 | 1.5554 | 1183.61 |
| ordinary-rhf | -8.74524508028746 | 9 | 0.3098 | 958.97 |
| global-pbe0 | -9.47932169663847 | 9 | 0.2229 | 947.50 |
| compact-tight | 0.666715572701808 | 11 | 4.6788 | 326.13 |

The full candidate comparison, including controls and refusals, takes
15.49 seconds with process RSS 290.44 MiB.
These are single one-thread observations under the original four-GiB cap;
modeled reservations are not measured RSS or a speedup claim.
The receipt verifies 2015 source/input pins and
106 output hashes. The retained result SHA256 is
`6e7bc26f1e49def49f7c85b8dd7bc328b7269d70500630fde5cb7c27262c06a5`.

The bitwise comparison links this candidate to the original independent
Gaussian/Boys/Fourier/libint and NumPy/SciPy evidence above, including the
continuous-grid PBE0 errors and their unchanged gates. Those references were
reused, not rerun and counted as new independent comparisons. No matched
ORCA/CRYSTAL custom-domain reference exists; KS shares libxc and grid generation,
with the separate grid witnesses retained. All finite-support, arithmetic,
rank, density-relaxation, basis, Brillouin-zone and XC exclusions remain.
This completes the frozen MANTA control check for that corrected source.
The release candidate must contain the converter repair and allowance update;
its final source and native identity must be reconciled with this receipt.
Overall candidate CI, packaging, publication and the release tag remain
separate requirements. A different candidate needs a new source-difference
assessment, with calculations repeated only for affected behavior.

### Candidate inclusion check

The 2026-10-24 source check of candidate
`2b2ed8838e50b18068567b1776d04324a4bae590` found the converter repair present,
but the work allowance update absent. This candidate does not yet contain
the complete accepted configuration. The missing main patch is
`ebbf5957bf93bf1ea7af3bbaef954e3e0fa5f729`; it changes only the example's
allowance and its changelog fragment. The already accepted old-cap refusal
applies, so it must not be reported as a passing `image-tight` example.

Of the 2015 accepted source/input file hashes, 2014 match this candidate.
The sole mismatch is `benchmarks/manta_release_controls.py`. All numerical
Python, native and build sources match the accepted integration, including
all 15 files in the converter repair. The complete tree difference consists
of the missing allowance, its changelog fragment and an unrelated periodic
grid-response test fixture correction. That correction changes no numerical
source or frozen input. Unified Fock, shared periodic producers and symmetry
behavior therefore retain the accepted source relationship.

After inclusion, the benchmark file must match accepted SHA256
`95969ae40f84aff7c6f88df005132ffd5081c095ef8dc1b7d6edaa89ce1a752f`.
Recheck the final source difference and retain the actual candidate native
receipt. Reuse the eight accepted calculations and eleven expected refusals
above when the relevant sources and inputs still match; a changed commit
identifier alone does not require another calculation. Source equivalence
does not establish another binary's identity or replace candidate CI.

### Prepared successor source verdict

Prepared source `7151963121829210d6df7706ebb329536c45ce05`, tree
`c4e721ee80bb6df68afaceca0f517735d997e777`, contains both required repairs.
Its driver is byte-identical to the accepted allowance patch `ebbf5957`.
All 2015 accepted source/input hashes match the corrected owner integration
`2d0fdb8c`. The entire tree difference from that integration comprises three
documentation files and the periodic grid-response test fixture. Numerical
Python, native and build sources, physical supports and selector inputs are
unchanged. This establishes **PASS for the bounded scientific controls and
their linkage to the prepared source** without another calculation.

| Evidence stage | Source | Reuse boundary |
| --- | --- | --- |
| Initial component and complete-method acceptance | `7917b560` | Retain the three unaffected HF states and original independent component witnesses; its old-grid PBE0 state is historical |
| Continuous-grid PBE0 integration | `9a868249` | Retain current-grid energy -9.47932169663847 Eh and independent energy/Fock/density errors 5.33e-15 Eh, 9.44e-16 Eh and 2.04e-10, below gates 8e-11 Eh, 8e-11 Eh and 5e-9 |
| Corrected candidate comparison | `2d0fdb8c` | Eight converged calculations, ten strict physical refusals and the expected old-cap refusal; exact states and physical certificates match retained acceptance |
| Prepared successor | `7151963` | All accepted source/input pins match; reuse the preceding scientific evidence with its original measured cost and native identity |

The corrected comparison used a native core built from `cfa5d1ea`, with the
core and libxc hashes recorded above. The subsequent allowance edit changed
no native source. That actual local binary remains the scientific receipt;
it is not renamed as a successor binary or deployed runtime. At this source
checkpoint the published candidate is still `2b2ed883`, which lacks the
allowance. The release owner retains candidate selection, the actual final
core/source/dependency receipt and the prescribed release gates. These
delivery requirements do not leave another MANTA source fix or calculation
pending for the unchanged prepared configuration.

The four profiles remain experimental. Fixed physical supports and the
recorded automatic RI coupling admission must be retained together; a later
explicit coupling profile is not this evidence or its timing. The component
budgets and exclusions in the [control guide](../docs/user_guide/manta_release_controls.md#what-the-bounds-mean)
remain binding. Broader provider coverage, derivatives, GPU/distributed
execution, symmetry reduction and whole-method automatic accuracy remain
subsequent MANTA increments outside this frozen release contribution.

### Selected candidate linkage

Selected candidate `5dfc97f0ec269e681c84bfda9531a0336b2f19f6` now contains
both the converter-header repair and the accepted `ebbf5957` allowance.
The driver and all 15 converter-repair files match the accepted owner
integration. This resolves the missing-allowance prerequisite recorded for
`2b2ed883`; that earlier candidate's refusal remains historical evidence.

Relative to the accepted owner integration, 2012 of 2015 source/input pins
match. The three changed Python modules are `periodic_ecp_source.py`,
`periodic_gradient_open_shell.py` and `periodic_uks_multi_k_ewald.py`.
Their changes bind the ECP UKS derivative receipt to the terminal state.
The four frozen profiles use all-electron RHF or restricted PBE0 energies,
without gradients. They do not enter that changed ECP/open-shell derivative
path. Their dispatch, shared producers, Fock construction, native kernels,
build recipes, physical supports and recorded coupling admission are
unchanged. The separate ECP acceptance remains with its owner.

The completed predecessor CI build also supplies a distinct Linux native
receipt for `2b2ed883`, with core SHA256
`c6c5d17cfd421cbfbe0fe543bc4b94915f7d043e8bc30a224a343465ab2addd1`.
The retained core artifact was downloaded and its hash verified. Its native
source tree `06a6d1f32df6b93e4cd363b1e7a3be439cfc42ff` matches the selected
candidate's native source tree. The receipt records Python 3.13.17, libint
2.13.1, libxc 7.0.0, linked dependency hashes and no stale-core override.
This establishes predecessor build provenance, not execution of the four
profiles on that Linux binary. The actual scientific comparison remains on
the local `b63d1324` core with its full hash and dependency identity above.
Neither binary is relabelled as the selected candidate's final runtime.

The bounded MANTA scientific and selected-source linkage verdict is **PASS**,
reusing the eight converged calculations, ten strict physical refusals,
expected old-cap refusal and original independent references. No affected
MANTA calculation or source fix remains for these unchanged configurations.
The selected candidate's own binary identity and CI are release-owner gates;
source equivalence does not replace them. MANTA remains experimental, with
the same component budgets, exclusions and single-job cost observations.
