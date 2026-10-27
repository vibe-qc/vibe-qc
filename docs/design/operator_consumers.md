# Operator producers, consumers and migration status

Baseline `f14e9ef5`, design #472. Paths in the static inventory are relative to
the repository. The family map below gives the interpretation and explicit
reason for each remaining bypass. Static sites include internal small
Rayleigh-Ritz problems and metric factorizations; they must not all be relabeled
Fock matrices. Dynamic dispatch and imported aliases were traced at the shared
helpers noted below. The inventory does not claim a formal whole-program call
graph or certification of every experimental method.

## Construction/state coverage and next increments

The family map below inventories eigensolution, not complete method-state
unification. The broader #472 continuation adds the following acceptance map.
Every deferred applicable family remains required work under #472; a passing
solver crossing is not a construction, energy, derivative or output migration.
Gamma 3D RHF/UHF and Gamma 3D/slab RHF/RKS/UKS now produce evaluated states.
Gamma Ewald ROHF also associates its physical spin build with its effective operator.
Remaining families retain their method-specific acceptance requirements.

| Producer/provider and execution | Representation | Construction/state status | Concrete follow-up under #472 |
|---|---|---|---|
| Gamma Ewald 3D RHF/UHF and slab 2D RHF, libint K and existing analytic-FT/grid or slab J | Real dense AO, total restricted or two spin densities, full Gamma mesh | Migrated: PATOM, SCF, terminal build, energy components, fixed physical/modified solver distinction, result and QVF identities, evaluated restart density; shared source receipt with preserved 3D HF identities and explicit slab Parry boundary | Preserve the same finite source in force/stress acceptance; no new derivative capability is implied |
| Gamma Ewald 3D and slab 2D RKS/UKS, libint exchange and libxc | Real total-density or two-spin AO KS, full Gamma mesh | Migrated: joint XC energy/potential, pure/global/screened exchange assembly, separate PATOM HF seed and spin solver histories, physical/solver parentage and coherent terminal/capped result/QVF | Multi-k construction remains deferred; no new derivative, provider or symmetry admission |
| Native Gamma GDF RHF/libxc RKS, bare fit in a home-cell-only 3D cell without active symmetry projection | Real total-density AO, owned fitted tensor and guarded libxc quadrature, full Gamma mesh | Migrated: separate full-HF PATOM, joint XC energy/potential, physical/modified Fock association, coherent terminal/capped density and QVF, finite-temperature complete-spectrum occupations and consistent entropy/free energy | External XC, RSGDF, private slab/torus, symmetry-projected and multi-k GDF construction remain deferred; fitted-source receipt grants no new provider or derivative capability |
| Molecular libint RHF/UHF/RKS/UKS, stored/direct/fitted/hybrid/range-separated paths (M1) | Native dense real Fock | Eigensolution migrated; common build/state adapter deferred | Native synchronous borrowed views and method-owned same-density HF/XC energy; independent spin recycling and stopped-state/output acceptance |
| Molecular MANTA direct/RI HF/KS (M1) | Native JKBuilder, dense SCF | Same native solver route; direct/RI export-to-build crossing tested | Share the native construction adapter; retain source/metric receipts and admitted physical capability matrix |
| Explicit screened molecular P0 HF (`symmetry_molecular_hf`) | Real full AO RHF/UHF; fixed stored or compact streamed orbit mask; optional source-bound Q/V or hierarchical pre-audit omission | Opt-in JKBuilder bridge to native SCF plus method-owned HF FockBuild evaluation; native UHF jointly requests total-density J and independent spin K channels; optional measured routing preserves the exact mask and keeps execution costs outside source/frame/density identities | Explicit run_rhf/run_uhf entry captures physical terminal/capped/NOITER FockBuild and density-defining QVF orbitals. Prepared sources reuse actual qualification and scalar cost profiles across independent states, with exact geometry/basis guards and fresh physical counters; retained-only audits require their exact mask. Explicit run_job and QVF requests now preserve the qualified source and physical output metadata; QVF re-execution qualifies a fresh producer. Exact method-local full-channel or joined RHF J/K evaluations now avoid terminal integral work when source and density bytes match; cold fallback remains. Ordered hierarchical sources now skip angular audits and unprofitable short-use probes; they cannot invoke reduced executors. Numerical mask/Fock identity is independent of qualification evidence. Default dispatch, zero-copy native capture, other producers and derivatives remain separate |
| Explicit native MD molecular HF (`symmetry_molecular_hf`) | Real full AO RHF/UHF with a fixed audited whole-orbit mask | Experimental low-level MANTA DirectJKBuilder qualification reuses the same JKBuilder bridge and physical FockBuild interface, with a distinct native-MD source label | Complete ordered-tile and stabilizer audit required. Ordered fallback uses identical MD tiles and mask; libint pre-audit bounds cannot qualify this source. Public prepared-source/job dispatch remains unchanged. No periodic, fitting, range-kernel or derivative admission |
| Explicit molecular Abelian RI HF (`symmetry_ri`) | Complex adapted AO/auxiliary factors; real native RHF/UHF density channels | Experimental MANTA-owned fitted-factor export, complete retained-space audit and sparse J/K; existing ScreenedMolecularHF evaluates the common physical FockBuild with a distinct fitted source identity | Native producer rank and finite Gram remain fixed; measured dense/blocked per-channel routing reuses scalar profiles with fresh physical counters and independent source identity. The blocked candidate uses a native auxiliary-panel executor; dense same-factor and low-level NumPy fallbacks retain arbitrary-density covariance checks. Explicit RHF/UHF method, job and QVF requests share the prepared source and evaluated natural-orbital result; fitting setup, physical contractions and exact-input terminal capture are reported separately. RI and quartet bridges share bounded method-local J/K snapshots, with selected-source/channel/density-byte guards and a counted cold fallback. Producer export avoids the raw coupling array, with an explicit ordered-tile fallback and separate rounded-payload identities. Default routing, producer-attested reduced fitting generation, periodic kernels and derivatives remain separate |
| Molecular and Gamma ROHF/ROKS (M2/P2) | Physical spin Focks, effective Roothaan operator and MO partitions | Gamma Ewald ROHF physical construction, PATOM, coupled effective/solver association, core-guess and equal-occupation block solves, and terminal/capped/zero-iteration result/QVF migrated; Gamma GPW ROHF/ROKS HCORE seed uses the shared metric solve and its SCF/terminal physical spin builds now carry joint energy/source identity with the terminal effective operator; Gamma GPW Roothaan solver steps now preserve the physical parent, shared frame and acceleration/shift metadata, with the last solve retained separately from the terminal build; density-defining orbital capture now retains accepted shared spin occupations or explicitly labelled spin-natural factors for mixed/initial states; Gamma runner, density-defining QVF orbitals and exact spin-density QVF/NPZ restart transport are migrated; other construction/state routes deferred | Retain ROHF constrained effective residual and occupation-block canonicalization, separately from physical UHF spin commutators; preserve real-only admission and complete retained roots. No additional ROKS route, multi-k, new source or symmetry admission |
| Gamma GPW RHF and libxc RKS including Dudarev DFT+U, and UHF including Dudarev DFT+U plus libxc UKS including Dudarev DFT+U, existing grid J and libint K | Real dense AO, full Gamma mesh | Migrated: separate full-HF PATOM, joint XC energy/potential including GGA and meta-GGA projections, pure/global-hybrid exchange, physical/DIIS roles, coherent terminal density/energy/output, functional/grid/policy source guards and finite-temperature occupations; RHF+U and RKS+U bind the existing active AO projectors and explicit Dudarev energy/potential to the same state; UHF owns coupled alpha/beta builds, shared-J/per-spin-K energy, physical terminal orbitals and preserved runner/QVF identities; UHF+U adds explicit per-spin Hubbard potentials and one spin-summed correction with an unchanged bare-UHF PATOM seed; UKS owns joint spin XC energy/potentials, including GGA cross-spin and meta-GGA terms, physical/solver roles and terminal density/QVF state; UKS+U composes the same spin Hubbard correction without mixing it into XC | External XC, augmented GAPW and multi-k remain separate increments. No MANTA, BIPOLE or symmetry dispatch changes; the derivative correction in #534 is accepted, without a physical bulk-exchange claim. |
| Native Gamma and multi-k periodic Ewald/GDF, RHF/RKS/UHF/UKS (P1/P2) | Real/complex dense AO and projected Focks | Shared eigensolution; remaining construction/state adapters deferred | Full/representative mesh identity, separate spin/k histories, XC joint energy/potential, density epochs and source/zero-mode receipts |
| Gamma external RKS, existing full-grid provider and libint exchange | Real total-density AO, full Gamma mesh | Shared restricted physical build and solver/terminal state; joint external XC energy/potential, resolved quadrature/domain and run-local provider identity | Real-provider qualification remains #603; no new spin, smearing, ECP or reduced-mesh admission |
| Gamma augmented GAPW bare zero-temperature RHF | Real AO, resolved analytic/block/projector augmentation with existing libint exchange | SCF and terminal FockBuild own the evaluated augmentation-functional energy and physical J/K/F; DIIS operators retain their physical build/density parent in the shared eigensolver; density-defining terminal orbitals and runner/QVF identity | #604 retains finite-temperature, +U and KS/spin construction and further evaluated/accepted-state transport. Run-local capture grants no cross-run provider equivalence or bulk parity |
| Remaining GPW/GAPW including hybrid/range-separated and fixed-density bands (P2) | AO and orthogonalized/projected dense blocks | Shared eigensolution; construction/state adapter deferred beyond the admitted Gamma GPW HF/libxc KS and ROHF/ROKS routes above | Preserve grid, image and augmentation supports; bind each energy and projected solver Fock to its physical parent; check forces/stress separately |
| BIPOLE RHF/RKS/UHF/UKS including supported exchange variants | Mutable per-run builder with finite real/reciprocal sources | Exported Si operator compatibility only; no common SCF build-state migration | Adapt resolved-domain records and terminal cold confirmation; keep physical exchange output support and CRYSTAL parity qualification separate |
| MANTA finite periodic RHF/UHF and joint grid/domain RKS/UKS | Bounded native spin/k dense storage and custom Jacobi solver | Evaluated-state/receipt semantics already native; exported operator crossing only | Add zero-copy native semantic adapters and explicitly costed exports; retain aggregate admission, overflow checks and stopped-state histories; public KS job work remains provider-owned |
| DFTB0/SCC-DFTB, PM6/OMx/GFN2, unrestricted/periodic variants (SE1/SE2) | Orthogonal/nonorthogonal effective Hamiltonians | Some eigensolution migrated; common method build/state deferred | Preserve charge/self-consistency energy and repulsive terms; test representative molecular/periodic/open-shell routes; SECCM stays experimental |
| Post-SCF CC/MP2/DLPNO/PNO and multireference reference consumers (C2) | Reference and semicanonical Fock blocks, bounded native pair kernels | Selected block solves migrated; evaluated-reference association deferred | Transport occupied/virtual/frozen partitions and source identity; do not add copies to bounded kernels; pair-density truncations retain their own role |
| Properties, gradients/stress, stability/response, embedding, restart and output (M2/P2/R2/A1) | Physical reference state plus method-specific derivative/response operators | Gamma UHF result/QVF/restart path migrated; remaining consumers deferred | Identify evaluated reference epochs before contractions; maintain derivative support/gauge and output consistency; non-Hermitian and indefinite pencils need distinct algebra |
| Experimental CCM/chi and related lines (P3) | Distinct finite Hamiltonians and qualified full-mesh reference states | Selected PNO solves migrated; common construction/state deferred | Adapt existing restricted periodic mean-field contract without changing support, mesh or experimental status |
| CI Hamiltonians, density/auxiliary metrics, Hessians and non-Hermitian CC Jacobians (C1/A1/R2/S1) | Distinct mathematical objects | Deliberately outside the Fock specialization | Share lower Hermitian algebra only where appropriate; keep root, metric and physical-role admission |

Symmetry's selective molecular J/K scheduler is a separate provider-side
capability, landed and opt-in. Its per-input
targets `('j', 'k', 'k')` for `[Dtotal, Dalpha, Dbeta]` return compact mapped
channels and per-input error bounds. This does not admit source reduction for
new MANTA/BIPOLE sources or orbital block reduction for a broken-symmetry
density. Geometry, source, density and retained-space admission stay separate.

The molecular quartet and RI method adapters calibrate automatic routes only
on first use of a channel signature. Native NOITER skips new probes using
its exact one-evaluation schedule; prepared sources retain no method state.
Calibration receipts include `targets`, even for zero-probe decisions.
`scf_elapsed_seconds` includes demand calibration, while `scf_seconds`
excludes it; probe receipts and terminal costs remain separate from physical
SCF counters and FockBuild identity.

Experimental `symmetry_products` adds complete matrix-irrep and multiplicity
resolution for ordinary unitary finite representations. Its invariant-map
application accepts arbitrary input vectors; composition executes reduced
multiplicity blocks. Complete finite-tensor reconstruction and explicit
recoupling guard the partner gauge. This is a reusable algebra seam for
integrals, response and correlation; it supplies no producer certificate,
new FockBuild state policy or public method dispatch. Dense coordinate
transforms and audit storage remain explicit costs.
Finite coordinate conversion uses the retained `basis_inverse`, including
reconstruction and composition; numerical orthogonality is not an exact
inverse. No included QC0.18 method calls this module. After release, producer
qualification stays with #635, molecular evaluated-state integration with
#608 and periodic source/state parity with #577. Reduced-pipeline optimization
must include full source audits, basis transport and reconstruction in its
elapsed-time and memory comparison before a consumer changes dispatch.

## Tracker pickup map

The full implementation goal is tracked by #472. The issues below cover the
remaining construction, evaluated-state and eigensolver work; completed rows
above remain controls. Claim a bounded slice in the relevant issue before
editing and preserve active ownership. Each issue defines its first increment,
compatibility constraints and numerical acceptance. These are work packages,
not additional review gates. The bounded release scope remains #557.

| Work package | Issue |
|---|---|
| Native borrowed/owned evaluated-Fock bridge | #594 |
| Molecular libint RHF/UHF evaluated-state migration | #595 |
| Molecular libint RKS/UKS evaluated-state migration | #596 |
| Molecular ROHF/ROKS physical and effective Fock association | #597 |
| Multi-k Ewald HF/KS FockBuild migration | #598 |
| Gamma GDF open-shell evaluated Fock migration | #599 |
| GDF source variants and projected-state adapters | #600 |
| Multi-k GDF HF/KS and restricted-open-shell state migration | #601 |
| Gamma GPW ROHF/ROKS evaluated-state migration, accepted in !345-!349 | #602 (closed) |
| GPW external-XC evaluated-state composition | #603 |
| Augmented GAPW physical Fock and state migration | #604 |
| Multi-k GPW/GAPW and fixed-density band Fock association | #605 |
| BIPOLE SCF evaluated-state adapter and terminal confirmation | #606 |
| MANTA periodic native Fock-state and solver semantic adapter | #607 |
| Molecular symmetry producers and shared evaluated-state completion | #608 |
| DFTB effective-Hamiltonian and evaluated-state migration | #609 |
| PM6/OMx/GFN and MSINDO shared operator-state migration | #610 |
| Non-DLPNO post-SCF reference and semicanonical Fock consumers | #611 |
| Properties, derivatives, stability and response reference association | #612 |
| Restart, runner and output completion for evaluated Fock states | #613 |
| Experimental chi Fock-state and eigensolver remainder | #614 |
| All-method operator inventory and 0.18 scope reconciliation | #615 |
| Molecular MANTA direct/RI evaluated-state integration | #616 |
| DLPNO/PAO/pair reference and eigensolver consumers | #580 |
| Periodic full/IBZ symmetry and method/source parity integration | #577 |
| Experimental AICCM real-Gamma evaluated-state increments RG-3/RG-4 | #519 |

The [full MANTA work tracker](manta_work_tracker.md#shared-interface-ownership)
is #617, with provider packages #618-#639 alongside libint. Its molecular and
periodic state integration reuse #616 and #607 above; #594 remains the shared
native ownership bridge. MANTA source, metric and error-bound work must not
create a competing Fock object or change eigensolver policy implicitly.
Record shared API changes and containing source/evidence on both affected
work items, preserving active ownership and bounded native execution.

Provider qualification and CRYSTAL parity remain with #593, #576 and #577;
MANTA source-specific symmetry is #635 and correlation exports remain with
#581. Role exclusions for non-Fock
algebra and remaining lower-level Hermitian sites are reconciled in #615.

## Release boundary and remaining algebra ownership

The full #472 migration and the bounded 0.18 qualification in #557 are different
completion criteria. An open migration issue does not by itself block the
release. Conversely, a shared eigensolver or accepted FockBuild capture does
not qualify a new physical source, derivative or reduced symmetry route.
The release owner must bind accepted evidence to the exact candidate; the
existence of a candidate branch is not acceptance or publication.

| Selected 0.18 obligation in #557 | Reusable operator evidence and remaining qualification owner |
|---|---|
| GPW derivative correction and dense-cell RIJCOSX refusal | #534 is closed with scoped acceptance; #555 has accepted refusal evidence recorded in #557. Preserve those criteria and establish candidate containment, rather than repeating the full migration. No physical bulk-exchange claim follows. |
| Included ECP routes and unsupported-gradient refusal | #35 owns affected-method acceptance and the containing runtime. Native Fock migration #594-#596 and MANTA composition #616 do not imply new ECP gradient support. |
| Bounded MANTA control profile | Reuse the selected provider evidence under #576/#617 and link exact source/native identities to the candidate. #607/#616 own state adapters, not certification of whole-method automatic accuracy. |
| Selected PBC examples, XC domain, fitting rank and Coulomb cancellation | #465/#179/#531 and the #557 claim matrix define included settings and tolerances. #593 owns broader parity; BIPOLE #606 and symmetry #577/#608 preserve their source-specific boundaries. |
| Spectroscopy and release publication | #545 acceptance is required only for included spectroscopy claims. #557 owns exact-candidate CI, packaging and the immutable release; ordinary migrations do not run release CI. |

### QC 0.18 evaluated-state safety reconciliation

The frozen inputs are the seven rows and explicit refusal in the
[bounded PBC inventory](../../benchmarks/pbc_018_supported_configurations.json)
and the four [MANTA control profiles](https://github.com/vibe-qc/vibe-qc/blob/main/benchmarks/manta_release_controls.md).
This reconciliation under #615 does not require completion of the all-method
#472 inventory. Fix a demonstrated defect in an included configuration;
retain other migrations for later releases. A code-path inspection is not a
new scientific acceptance result or a proof against arbitrary caller mutation.

| Frozen route | State-integrity finding and bounded disposition |
|---|---|
| LiH RSGDF RHF Gamma and 222; diamond Gamma GDF PBE0 | The restricted GDF terminal branch retains the evaluated density with its physical Fock and energy, bypassing trial mixing; canonical refill is not silently substituted as the evaluated density. No additional defect was demonstrated in these converged configurations. Reuse #179 and #465 evidence. The affected continuous-grid comparison is now accepted under #557: domain energy error 6.0136e-7 Ha and T2 splitting 3.2151e-6 Ha retain the original 1e-5 gates. Exact candidate binary qualification remains separate. |
| GPW RHF H2, 9 and 16 bohr cells | Reuse #534 energy/Fock directional-derivative acceptance. This is RHF, not a PBE control. Subsequent evaluated-build work does not turn full GPW/GAPW migration into a release requirement or establish physical bulk exchange. |
| Cl2 Ewald UKS ECP Gamma and 311 preview | Demonstrated included-consumer gap: mutable terminal spin densities, orbitals, energies or occupations can diverge from the retained derivative source, and contradictory ECP flags can bypass the intended source check. The PBC owner landed state binding and pre-kernel refusal in `722f7d6f`, with extended regressions in `66c9e784`. All 17 admission/copy and complete-force checks pass, including both frozen meshes. Accepted forces and unsupported-gradient boundaries are unchanged. |
| Four MANTA profiles | The native RHF route confirms Fock and energy by rebuilding at the returned density; the frozen driver compares source identity and energy/Fock/density/overlap/HCORE with the same-source certificate-disabled control. Reuse the owner-controlled continuous-grid and independent-reference evidence and exact candidate linkage. No new provider-state defect was demonstrated by this reconciliation. Component certificates are not a total relaxed-energy bound. |
| Dense-cell RIJCOSX | Preserve #555's qualified refusal. An unsupported-route refusal is the intended result, not a missing Fock migration. |

Candidate `5dfc97f0` contains the byte-identical ECP state-receipt module from
`722f7d6f`. Its producer and consumer differences are the NOITER and physical
projector-support paths, outside the frozen positive-iteration, legacy-support
Cl2 inputs. This is affected-path source containment, not whole-runtime
equality: the acceptance runtime and candidate native versions differ. Preserve
the original native ECP evidence and require the release owner to bind the
final binary and CI receipt. Dense NaCl/GDF projector-domain work under #652
remains outside these rows; it must not broaden the Cl2 claim.

The PBC owner completed the ECP repair acceptance under #557, result note
53206: 15 admission/copy checks and both original Cl2 PBE/LANL2DZ preview
calculations passed. Gamma and 311 periodic energies are respectively
-29.810831798121452 and -29.810831798121473 Ha. Molecular offsets are
6.0255749794e-6 and 6.0255749581e-6 Ha, below the unchanged 1e-5 Ha gate;
force-minus-Richardson errors are -5.9409351605e-9 and -5.9652009027e-9
Ha/bohr, below the unchanged 2e-6 Ha/bohr gate. Both retain the original
legacy support, spin populations and convergence settings. Edited orbital
energies and contradictory ECP flags refuse before derivative evaluation;
restored state and unchanged copies remain admissible.

Acceptance used base `f1ad771416b02c48c518cedb64403de5acc249e8` with the
three hash-pinned Python guard/consumer modules matching `722f7d6f` and
tests matching `66c9e784`. Native SHA256 is
`513f1892606499f1ac15f4e4db6b7e820681d649c5fb25719f2c5bdc47cd8244`.
The owner retains the complete immutable calculation and attempt receipts on
#557. These are same-code molecular and periodic finite-difference controls,
not independent matched periodic ORCA/CRYSTAL force acceptance. General ECP
forces/stress remain excluded. There is no remaining scientific case for this
bounded repair; exact candidate binary/CI qualification remains separate.

The compact correlation repair is complete at `69c7f5be`, integrated through
`fb99e9c8`: cyclic Fock/overlap identities and occupied-projector partition
records bind at source construction and are rechecked before selection and
admission, without densification or factor reads. Its 23 accepted checks
include both compact-projector and coefficient-based finite-torus MP2 controls;
maximum same-source canonical/character energy difference is 1.90e-15 Eh and
maximum residual is 3.32e-14. Detailed source/native receipts belong to #580.
This closes that bounded integrity increment, not factor certification or the
all-method programme. The prepared candidate `7151963` does not contain this
binding; experimental correlation is outside the frozen claims, so importing
this repair is not an added release gate. Candidate selection, builds and
publication remain with the release owner.

Gamma GPW ROHF/ROKS #602 is accepted through !345-!349: seed and effective
solves, physical builds, solver parentage, density-defining orbitals, and
runner/QVF/NPZ transport. The transport increment has eleven accepted checks
and a maximum direct/resumed energy difference of 2.665e-15 Eh. This is
same-source transport evidence, not an external bulk reference. Optional
sampled density-volume tolerance and bond-order output (related #547) remain
separate limitations. Public integer-shell admission is unchanged.

Gamma GAPW bare zero-temperature RHF #604 now includes SCF-loop construction
and DIIS solver parentage (source `6896f00f`, main integration `55f751b8`).
Nine state/transport checks and six stopped/converged comparison pairs passed;
maximum energy difference was 1.777e-15 Eh against the preceding implementation.
The explicit augmentation functional supplies the convergence energy. This
same-source evidence does not establish bulk exchange accuracy, migrate KS or
spin variants, or add a #557 release requirement.

Remaining algebra work must also have an explicit owner even when it is not
a Fock construction. This map supplements the method packages above; internal
Ritz solves and density/metric decompositions are not migration omissions.

| Family and concrete starting point | Owner and next bounded step |
|---|---|
| M1/M2 native HCORE frontier completion in `cpp/src/rhf.cpp` and related drivers | #594/#595 retain the seed occupation and symmetry frame when adding evaluated-state transport. Do not replace frontier completion with a partial SCF spectrum. |
| M2/R2 stability and response, including the native RHF stability call | #612 owns reference-state association and signed-curvature/callback acceptance. Hessian and response operators retain their mathematical kinds. |
| C1 determinant CI in `cpp/src/casci.cpp` and `cpp/src/selected_ci.cpp` | #615 retains the lower-level Hermitian integration work: qualify a named sigma boundary with multi-root, spin-sector, warm-start and bounded-memory checks before replacing a specialized solver. Projected Ritz decompositions remain internal. |
| S1 callback backend capability and storage in `solvers/operators.py` and native Davidson/LOBPCG/GPLHR | #615 owns explicit capability gaps and no-hidden-materialization acceptance. Existing strict refusals remain until a backend satisfies the advertised storage/root contract. |
| C2 reference and semicanonical blocks | #611 and #580 own non-DLPNO and DLPNO consumers respectively, including frozen partitions and workspace limits. Molecular DLPNO now refuses explicit RHF/UHF occupations that contradict its ECP-aware integer occupied/virtual partition before factor work; absent metadata keeps the legacy count contract. Full stationarity and factor certification remain separate. Pair-density truncation stays outside the Fock specialization. |
| A1 density/auxiliary metrics, geometry Hessians and R2 non-Hermitian pencils | Deliberately excluded from Fock construction. #615 records role exclusions; #612 owns applicable physical-reference association. No generic Hermitian replacement is authorized for a non-Hermitian pencil. |

#615 remains open for the named lower-level integration work and future
coverage updates. #603-#605 remain the external-XC, augmented and multi-k GPW
migration packages; completion of #602 does not close them. Keep development
on main and candidate qualification with its existing owner, without adding
these deferred packages to the frozen #557 gates.

## Family map (eigensolution)

| Key | Producers and consumers | Representation and contract | Migration status and next step |
|---|---|---|---|
| M1 | Molecular `fock.cpp` and integral/DF/XC builders feed `rhf.cpp`, `rks.cpp`, `uhf.cpp`, `uks.cpp`; wrappers include direct and fitted molecular paths | Dense real AO Fock, one or two spin densities; method-owned rank-reducing X; dense or real Davidson; MO coefficients feed density, energy, gradients and post-SCF | Main diagonalization lambdas migrated to `solve_fock_matrix`. Opt-in MANTA molecular RHF/UHF/RKS/UKS entry points reuse these same native drivers through `JKBuilder`; the integral provider does not choose a separate eigensolver. Separate HCORE frontier completion remains local because it builds a symmetry-consistent seed and full occupation frame. No change to that policy. Spin-specific warm caches and public per-step diagnostics are follow-up work. |
| M2 | `guess.py`, native `guess.cpp`, `rohf.py`, ROHF/ROKS helpers, `newton.cpp`, `soscf.cpp`, `trah.cpp`, `orbital_objective.cpp`, `degenerate_frontier.cpp`, restart code | Core/effective Fock blocks, overlap factors, orbital Hessians and seed densities are different operators; some complete spectra and some lowest stability directions | Real molecular ROHF/ROKS and Gamma GPW ROHF/ROKS effective-Fock solves now share `FockDiagonalizer`, with caller-owned occupation/level-shift policies and private Davidson recycling. Guest-Saunders closed/open/virtual Fock blocks use separate shared dense solves, preserve block ordering and respect rectangular retained frames. Initial guesses and other mixed-role consumers remain deferred: preserve seed ensembles and trust-region logic; Hessian/Ritz subproblems stay internal and must not enter a Fock API. Native stability Davidson remains a lower-level callback consumer pending signed-curvature acceptance tests. |
| P1 | `bloch.cpp`, `periodic_scf.cpp`, `_diag_in_orth_basis` in `periodic_rhf_multi_k_ewald.py`; real-space/bloch Fock builders | Complex Hermitian per-k AO F and S; adjoint transforms; no mixing of k-point weights into eigenvector normalization | Native Bloch and SCF Fock solves and Python helper migrated. Overlap factorization and density eigenvalues remain separate. Shared Python helper also serves `periodic_k_gdf.py` and multi-k RHF/RKS/UHF/UKS Ewald callers. `diagonalize_bloch` transitively serves k-point DFTB0/SCC-DFTB; their charge and occupation policies are unchanged. |
| P2 | Gamma Ewald HF/KS, `pbc_gdf.py`, `periodic_rhf_gdf.py`, GAPW J/augmentation/open-shell/post-SCF, periodic embedding, retained mean-field consumers | Mostly explicit real Gamma or complex k blocks. Independent X cutoffs and symmetrization; some per-driver Davidson branches. Embedding contains Green functions and density/metric decompositions too | Gamma RHF/RKS/UKS Ewald (including slab Gamma), Gamma GDF RHF/RKS and PBC-GDF RHF/RKS/UHF/UKS now share `FockDiagonalizer`, with separate spin recycling, unchanged rank policies and caller-owned occupations. Dense/Davidson energy, density, smearing and force checks accompany the migration. Gamma GPW/GAPW RHF/RKS/UHF/UKS and range-separated GAPW now also use this adapter for iterative SCF and final external-XC canonicalization, retaining their dense/Davidson defaults. Multi-k GPW RKS/UKS/ROKS, GAPW RKS and the fixed-density band-path helper use the shared dense boundary with explicit AO or orthogonal coordinates, spin/k-point/generation metadata and caller-owned global occupations. Projected DIIS matrices stay in their orthogonal frame; the driver lifts coefficients through X. Shared multi-k Ewald/GDF helper calls use P1; the private multi-k slab loop and embedding Fock sites still require driver-specific k-gauge and occupation tests. Gamma Ewald ROHF effective-Fock solves now share `FockDiagonalizer`, honor explicit Davidson controls and require all retained roots before J/K construction, preserving level shifts and alpha-energy occupation selection. Its dense HCORE seed remains separate. Gamma Ewald UHF and Gamma/slab UKS also use separate alpha/beta adapters for PATOM in-field, SCF and terminal consistency Focks; a separate one-electron HCORE solve preserves its configured solver. Active controls are validated before J/K, spinlock/quadratic updates require full retained spectra, and method-owned terminal density/physical Rayleigh-quotient reporting is preserved. Gamma augmented GAPW restricted HCORE now uses the shared dense one-electron solve with its existing overlap transform and occupation policy. Other grid-method HCORE/PATOM seed diagonalizations stay with the guess producers; Gamma GPW restricted-open-shell effective-Fock and Guest-Saunders solves use M2; orbital-transformation kernels retain their specialized rotation policy. GAPW augmentation metrics and embedding pencils need explicit producer contracts. Memory-bounded periodic RHF, including the finite MANTA periodic RHF driver, retains its custom Jacobi kernel because copying operator snapshots would violate its memory contract. Exported spectra/metrics are cross-checked against the common API and generalized LAPACK, with explicit retained-rank and shell-image gauge tests. Whole-mesh MANTA spin-Fock snapshots can feed per-spin/k operators; provider source receipts and zero-mode policies remain outside solver metadata. |
| P3 | Experimental `periodic/ccm` and `periodic/chi`: direct/fitted HF/KS, Bloch references, PNO, localization and post-HF assembly | Dense localized/cell/Bloch matrices, generalized overlaps, semicanonical Fock blocks and density eigensystems; domain and source provenance are essential | Unrestricted PNO Fock rotations in neutral CCM UMP2 and its UCCSD pair assembly transitively share the molecular `_pnos` boundary, with CCM pair-frame and spin labels. The remaining sites are deferred and still experimental: carry domain/source/basis metadata and check physical-support acceptance before migration. Density/Gram factorizations remain internal. No production claim follows from shared algebra. |
| SE1 | Native DFTB0/UDFTB0: semiempirical basis/H0/overlap builders to occupations, electronic energy and charges | Dense real H0 C = S C e, strict SPD metric, shared spectrum and separate spin occupations | Migrated to `solve_one_electron_metric`. Generalized eigensolution now uses the same residual-checked native dense boundary; no change to the Hamiltonian parameters or energy model. |
| SE2 | SCC-DFTB, GFN2, PM6/OMx, INDO/MSINDO, periodic and SECCM variants; native engines and Python fallback models | SCC charge-dependent H; NDDO/INDO orthogonal effective F; GFN generalized metric and multipole terms; real or complex Bloch blocks; some separate spin spectra | K-point consumers of `diagonalize_bloch` transitively migrated. Remaining inline sites deferred by model: validate converged charges, forces and restricted/unrestricted equivalence before replacing. SECCM remains experimental. INDO stability is a Hessian operator, not a Fock matrix. |
| R1 | TDDFT TDA sigma builder in `tddft.py` to excitation roots, transition moments and oscillator strengths | Real Hermitian excitation operator; analytic diagonal; matrix-free action; occupied-virtual indices, not AO coefficients | Iterative TDA migrated to `HermitianOperator(kind='response')`. Dense TDA/reference paths and positive-metric Casida transformations remain local complete-spectrum operations. Full response A/B structure must survive any later migration. Removed reliance on a convergence-flag workaround in the migrated solve: the current backend and recomputed residual must both pass. |
| R2 | `excited.py`, non-Hermitian response front ends, direct native bindings and declarations; uncategorized scanner candidates | Small dense CIS-like response or general operator interfaces; CC Jacobians can be non-Hermitian, real eigenvalues are not sufficient evidence of Hermiticity | Hermitian response can share the lower algebra after method-specific tests. Non-Hermitian, left/right-vector and structured-pencil contracts are explicitly deferred. Pybind compatibility symbols remain callable; they are not an endorsement of their matrix-free labels. Declaration-only sites are not runtime bypasses. |
| C1 | Native CASCI/selected-CI sigma engines; Python selected CI, CISD, DMRG and determinant-density gradient helpers | Determinant Hamiltonian/sigma, analytic diagonal, spin and active-space structure; DMRG effective local Hamiltonians; own block/restart/selection machinery | Retain specialized iterative kernels pending spin-completeness, determinant ordering, multi-root and memory evidence. They share the Hermitian mathematical contract but are not Fock operators. Replacing projected subspace eigensolves alone would not unify their solver algorithms. |
| C2 | CC/MP2/DLPNO and multireference PT2 semicanonicalizers, pair-density and effective-Hamiltonian consumers | Occupied/virtual Fock blocks, pair density eigensystems and effective CI Hamiltonians; spin and frozen-core partitions matter | Brueckner CCD active-occupied/virtual blocks, FNO-CCSD retained virtuals and reference UMP2/ROHF-MP2 per-spin occupied/virtual blocks now use shared dense solves with named MO coordinate partitions. Frozen columns, empty/singleton policies, within-block ordering and MP2 singles coupling are preserved. FNO solve arrays are released before the final native CC call. Dense molecular DLPNO-MP2/UMP2 retained PNO Fock blocks now share the boundary too, preserving pair-density thresholds, corrections and amplitude coupling with explicit pair/spin frames. Molecular DLPNO-UCCSD rotated-occupied `t1` active alpha/beta blocks also use the shared dense boundary, with admitted-source/frozen/spin coordinate labels, separate degenerate spin spaces and no solve for an empty block. The compiled-PNO UCCSD preparation path also shares retained pair-PNO Fock solves, adding pair indices to the admitted-source/frozen/spin frame. Pair-density truncation and its empty-block fallback are unchanged. Enabled singles-PNO preparation on the compiled route now uses named source/frozen/occupied/spin Fock solves too, preserving density selection and PAO projection; the default full singles space adds no solve. These Python preparation and triples solves release each snapshot before the next spin, pair or occupied orbital; no snapshot is added to the native residual loop. Cached spin-PAO Fock preparation for compiled and union UCCSD also shares the boundary with source/frozen/atom-domain/spin labels, retaining the local overlap-rank policy and coefficient cache without AO metric snapshots. Union pairs and singles reuse the same per-domain/spin coefficients instead of rebuilding identical PAOs; their combined-spin column ordering and empty-domain refusals are unchanged. Retained union pair-PNO and enabled singles-PNO Fock rotations now share the boundary with source/frozen/pair-or-occupied labels and explicit combined alpha/beta or PAO-spin coordinates. Density selection and degenerate spin mixing are preserved; solve results are released immediately. Default full union singles also share one named source/frozen/combined-spin solve and reuse only its full-space coefficients and energies across occupied orbitals; amplitudes remain independent. Rotated-occupied `t1` retained TNO Fock blocks also share the checked boundary, naming source/frozen/rotated-triple/combined-spin coordinates and preserving density selection and screening refusal. Local T0 and iterative T1 TNO construction, native TNO kernels, other PAO preparation, native/CC pair-local and multireference Fock sites are deferred pending domain/gauge, denominator and workspace acceptance tests; copying snapshots cannot be introduced into bounded kernels without memory evidence. Density truncation and effective Hamiltonians keep their roles. Non-Hermitian CC amplitude/Jacobian solves are outside this Hermitian implementation. |
| A1 | AO/auxiliary metric orthogonalization, density/natural orbitals, properties, Hessians/phonons, localization, geometry trust models, output and grid quadrature | Small symmetric tensors, density/Gram/metric matrices or force-constant matrices, with their own units and root semantics | Deliberately retained. MANTA RI auxiliary Coulomb-metric factorization and retained-metric response also stay here: the complete metric projector and rank cutoff belong to the producer. Direct and RI spin J/K crossing tests feed explicit Hermitian Fock snapshots without moving source receipts into solver metadata. These are not orbital optimizations and frequently need a full decomposition or matrix function. Routing them through iterative Fock policy adds cost and risks changing cutoffs or units. They may use the lower Hermitian interface later if shared diagnostics provide a measured benefit. |
| S1 | Python registry and native Davidson/LOBPCG/JD/GPLHR implementation and small projected eigensolves | Backend algorithm implementation, not a producer of physical Fock operators | New strict entry enforces capabilities. Historical direct adapters remain for compatibility: real and complex callback Davidson reconstruct A only below the native restart floor, while LOBPCG/JD/GPLHR native callback adapters reconstruct A. Complex Davidson now shares its explicit iteration through operator applications, and real/complex block callbacks are available. Python strict interface refuses these paths. Small projected dense solves inside an iterative algorithm are intentional. |

## Concrete findings

- `solvers/_common.py` also defines `Hamiltonian`, `SolverOptions` and
  `SolverResult` for correlated-method dispatch. They are not the eigenproblem
  options/results. Renaming or merging them would break imports and conflate
  physical models with numerical solves. The new module uses explicit imports
  from `solvers.eigensolver` and adds distinct operator names.
- `SCFStep` previously cast Fock to float64, used transpose for complex
  transforms, solved with transposed X for guesses, and mutated caller options.
  It now adapts to the strict operator solve, preserves complex data and uses
  original-frame warm starts, including rectangular X.
- Historical generalized reduction used transpose rather than adjoint and
  returned reduced residuals beside AO coefficients. The complex transform
  and residual frame are corrected; strict rank reduction also rejects an
  indefinite metric and a root request larger than retained rank.
- Dense SM/LM selectors were treated as SA/LA. Both-ends with one root could
  request an invalid empty subset. Dense selection now enumerates and selects
  the actual spectrum; the native iterative extremal kernels are not given
  unsupported magnitude requests.
- `run_job` accepted LOBPCG and set `use_davidson=True`. The request now raises.
  Other runner methods still own their solver routing; adding a universal
  runner keyword needs a separate per-method migration, not a registry name.
- `EigenProblem` allowed both matrix and callback, with inconsistent precedence
  among strategies. The new operator accepts one representation, plus optional
  vector/block callbacks describing the same action.
- Historical `_matvec_diagonal` probes all unit vectors without a supplied
  diagonal. Strict operators require an explicit preconditioner diagonal.
  Transformed sparse/callback Davidson now accepts explicit X and
  `reduced_diagonal = diag(X^H A X)`; it never infers this from diag(A).
  Callback actions own their returned vectors/blocks, including when the
  producer reuses scratch storage. Density-dependent captures remain the
  producer's responsibility.
  Complex Davidson now applies vectors/blocks directly. LOBPCG callback
  implementations still allocate dense matrices and are rejected by strict
  capabilities; both Davidson variants reject their small-problem fallback
  through the strict callback interface.
- Native Davidson can raise a requested subspace budget and use a dense solve
  below the restart floor. The strict callback path refuses the latter;
  existing explicit methods retain the documented behavior. Hard memory budgets
  therefore remain the domain of the specialized bounded kernels.
- Python Lanczos used changes in Ritz values to set convergence even when the
  final residual failed. Convergence now requires the measured residual and
  requested root count. It remains an unrestarted exploratory implementation.
- GPLHR-derived harmonic candidates are distance-selected, but zero residual
  does not certify absence of a closer omitted root. This remains an explicit
  experimental limitation, exposed by `selection_verified=False`.

## Public and internal boundaries

`run_job`, `run_periodic_job`, method wrappers, numerical problem descriptors
and the additive operator API are user-facing. Low-level pybind exports are
also reachable and must retain compatibility even when their use is discouraged.
SCF lambdas, projected eigensystems, integral-builder matrices and native
`hermitian_operator.hpp` are implementation details. The uniform result
formatter is `vibeqc.output.eigensolver.eigensolver_diagnostics`; extending
historical SCF/QVF schemas is deferred explicitly.

The static inventory below can be regenerated with
`python scripts/audit_eigen_consumers.py`. It includes an entry for every file
found by the stated scan, with a family key above. Line numbers are navigation
aids for this increment and will drift. Entries indicate semantic-audit or
retention status, not test coverage; validation commands are in the guide.

## Static source inventory

| Source | Family/status key | Audited static sites (line:symbol) |
|---|---|---|
| `cpp/include/vibeqc/bloch.hpp` | R2 | 92:diagonalize_bloch |
| `cpp/include/vibeqc/bounded_periodic_rhf_solver.hpp` | P2 | 152:solve_bounded_periodic_rhf, 175:solve_bounded_periodic_rhf |
| `cpp/include/vibeqc/davidson.hpp` | S1 | 142:davidson_solve, 152:davidson_solve_matvec, 162:davidson_solve_matmat, 172:davidson_solve_hermitian, 180:davidson_solve_hermitian_matvec, 186:davidson_solve_hermitian_matmat |
| `cpp/include/vibeqc/gplhr.hpp` | S1 | 67:gplhr_solve, 71:gplhr_solve_matvec |
| `cpp/include/vibeqc/hermitian_operator.hpp` | S1 | 28:solve_hermitian_matrix, 31:solve_hermitian_matrix, 38:solve_fock_matrix, 42:solve_fock_matrix, 49:solve_one_electron_metric |
| `cpp/include/vibeqc/jd.hpp` | S1 | 76:jd_solve, 82:jd_solve_matvec |
| `cpp/include/vibeqc/lobpcg.hpp` | S1 | 80:lobpcg_solve, 89:lobpcg_solve_matvec |
| `cpp/include/vibeqc/semiempirical/methods/indo/ccm_stability.hpp` | SE2 | 40:SelfAdjointEigenSolver, 70:davidson_solve_matvec, 137:SelfAdjointEigenSolver |
| `cpp/include/vibeqc/semiempirical/methods/indo/indo_engine.hpp` | SE2 | 2124:SelfAdjointEigenSolver, 2139:SelfAdjointEigenSolver, 2182:SelfAdjointEigenSolver, 2188:SelfAdjointEigenSolver, 2236:SelfAdjointEigenSolver, 2239:SelfAdjointEigenSolver, 2250:SelfAdjointEigenSolver, 2285:SelfAdjointEigenSolver, 2323:SelfAdjointEigenSolver, 2333:SelfAdjointEigenSolver, 2544:SelfAdjointEigenSolver, 2597:SelfAdjointEigenSolver, 2600:SelfAdjointEigenSolver, 2611:SelfAdjointEigenSolver |
| `cpp/src/bindings.cpp` | R2 | 4034:davidson_solve, 4039:davidson_solve_matvec, 4045:davidson_solve_hermitian, 4046:davidson_solve_hermitian, 4051:davidson_solve_matmat, 4056:davidson_solve_hermitian_matmat, 4057:davidson_solve_hermitian_matmat, 4062:davidson_solve_hermitian_matvec, 4063:davidson_solve_hermitian_matvec, 4111:lobpcg_solve, 4116:lobpcg_solve_matvec, 4149:jd_solve, 4179:gplhr_solve, 9681:diagonalize_bloch |
| `cpp/src/bloch.cpp` | P1 | 121:diagonalize_bloch, 126:diagonalize_bloch, 130:SelfAdjointEigenSolver, 132:diagonalize_bloch, 138:diagonalize_bloch, 147:solve_fock_matrix |
| `cpp/src/bounded_periodic_rhf_solver.cpp` | P2 | 332:solve_bounded_periodic_rhf |
| `cpp/src/bounded_periodic_rhf_solver_bindings.cpp` | P2 | 164:solve_bounded_periodic_rhf |
| `cpp/src/casci.cpp` | C1 | 370:SelfAdjointEigenSolver, 453:SelfAdjointEigenSolver |
| `cpp/src/ccsd.cpp` | C2 | 3190:SelfAdjointEigenSolver |
| `cpp/src/davidson.cpp` | S1 | 67:SelfAdjointEigenSolver, 224:SelfAdjointEigenSolver, 469:SelfAdjointEigenSolver, 613:SelfAdjointEigenSolver, 750:SelfAdjointEigenSolver, 778:davidson_solve, 783:davidson_solve_matvec, 802:davidson_solve_matvec, 821:davidson_solve_matmat, 836:davidson_solve_hermitian, 848:davidson_solve_hermitian_matvec, 867:davidson_solve_hermitian_matmat |
| `cpp/src/degenerate_frontier.cpp` | M2 | 56:SelfAdjointEigenSolver |
| `cpp/src/df.cpp` | A1 | 303:SelfAdjointEigenSolver |
| `cpp/src/dlpno_ccsd.cpp` | C2 | 1459:SelfAdjointEigenSolver, 1475:SelfAdjointEigenSolver |
| `cpp/src/dlpno_uccsd.cpp` | C2 | 1322:SelfAdjointEigenSolver, 1333:SelfAdjointEigenSolver |
| `cpp/src/geomopt_kernels.cpp` | A1 | 200:SelfAdjointEigenSolver |
| `cpp/src/gplhr.cpp` | S1 | 88:GeneralizedEigenSolver, 171:gplhr_solve, 176:gplhr_solve_matvec |
| `cpp/src/grid.cpp` | A1 | 41:SelfAdjointEigenSolver |
| `cpp/src/guess.cpp` | M2 | 531:SelfAdjointEigenSolver, 542:SelfAdjointEigenSolver, 1048:SelfAdjointEigenSolver, 1283:SelfAdjointEigenSolver, 1369:SelfAdjointEigenSolver, 1477:GeneralizedSelfAdjointEigenSolver, 2244:SelfAdjointEigenSolver, 2706:SelfAdjointEigenSolver |
| `cpp/src/hermitian_operator.cpp` | S1 | 30:davidson_solve, 32:davidson_solve_hermitian, 41:SelfAdjointEigenSolver, 76:solve_hermitian_matrix, 80:solve_hermitian_matrix, 84:solve_fock_matrix, 89:solve_fock_matrix, 94:solve_one_electron_metric |
| `cpp/src/jd.cpp` | S1 | 196:SelfAdjointEigenSolver, 332:SelfAdjointEigenSolver, 373:jd_solve, 378:jd_solve_matvec |
| `cpp/src/linear_dependence.cpp` | A1 | 43:SelfAdjointEigenSolver |
| `cpp/src/lobpcg.cpp` | S1 | 134:SelfAdjointEigenSolver, 233:SelfAdjointEigenSolver, 285:SelfAdjointEigenSolver, 303:lobpcg_solve, 308:lobpcg_solve_matvec, 314:lobpcg_solve_matvec, 318:lobpcg_solve_matvec |
| `cpp/src/manta_periodic_rhf.cpp` | P2 | 347:solve_bounded_periodic_rhf |
| `cpp/src/manta_ri_metric.cpp` | A1 | 52:SelfAdjointEigenSolver |
| `cpp/src/newton.cpp` | M2 | 792:SelfAdjointEigenSolver |
| `cpp/src/orbital_objective.cpp` | M2 | 25:SelfAdjointEigenSolver, 38:SelfAdjointEigenSolver |
| `cpp/src/periodic_gaussian_rhf.cpp` | P2 | 455:solve_bounded_periodic_rhf |
| `cpp/src/periodic_mean_field_state.cpp` | A1 | 942:SelfAdjointEigenSolver |
| `cpp/src/periodic_rhf.cpp` | P2 | 38:SelfAdjointEigenSolver, 250:SelfAdjointEigenSolver, 254:davidson_solve, 284:SelfAdjointEigenSolver |
| `cpp/src/periodic_scf.cpp` | P1 | 68:SelfAdjointEigenSolver, 115:SelfAdjointEigenSolver, 136:solve_fock_matrix, 443:SelfAdjointEigenSolver |
| `cpp/src/rhf.cpp` | M1 | 308:solve_fock_matrix, 332:SelfAdjointEigenSolver |
| `cpp/src/rks.cpp` | M1 | 881:solve_fock_matrix, 907:SelfAdjointEigenSolver |
| `cpp/src/scf_restart.cpp` | M2 | 62:SelfAdjointEigenSolver |
| `cpp/src/selected_ci.cpp` | C1 | 386:SelfAdjointEigenSolver, 441:SelfAdjointEigenSolver |
| `cpp/src/semiempirical/dftb0.cpp` | SE1 | 62:solve_one_electron_metric, 142:solve_one_electron_metric |
| `cpp/src/semiempirical/gradient.cpp` | SE2 | 1824:GeneralizedSelfAdjointEigenSolver |
| `cpp/src/semiempirical/kpoints_dftb0.cpp` | SE2 | 248:diagonalize_bloch |
| `cpp/src/semiempirical/kpoints_scc_dftb.cpp` | SE2 | 181:diagonalize_bloch |
| `cpp/src/semiempirical/methods/nddo/omx_fock.cpp` | SE2 | 1008:SelfAdjointEigenSolver, 1162:SelfAdjointEigenSolver, 1163:SelfAdjointEigenSolver, 1203:SelfAdjointEigenSolver, 1204:SelfAdjointEigenSolver |
| `cpp/src/semiempirical/methods/nddo/periodic_pm6.cpp` | SE2 | 753:SelfAdjointEigenSolver |
| `cpp/src/semiempirical/methods/nddo/pm6_fock.cpp` | SE2 | 1350:SelfAdjointEigenSolver, 1953:SelfAdjointEigenSolver, 1954:SelfAdjointEigenSolver |
| `cpp/src/semiempirical/methods/xtb/gfn2_driver.cpp` | SE2 | 520:SelfAdjointEigenSolver, 1003:GeneralizedSelfAdjointEigenSolver |
| `cpp/src/semiempirical/methods/xtb/periodic_gfn2.cpp` | SE2 | 475:GeneralizedSelfAdjointEigenSolver |
| `cpp/src/semiempirical/methods/xtb/ugfn2.cpp` | SE2 | 84:GeneralizedSelfAdjointEigenSolver |
| `cpp/src/semiempirical/periodic_dftb0.cpp` | SE2 | 147:GeneralizedSelfAdjointEigenSolver, 338:GeneralizedSelfAdjointEigenSolver |
| `cpp/src/semiempirical/periodic_scc_dftb.cpp` | SE2 | 327:GeneralizedSelfAdjointEigenSolver, 471:GeneralizedSelfAdjointEigenSolver, 607:GeneralizedSelfAdjointEigenSolver, 729:GeneralizedSelfAdjointEigenSolver |
| `cpp/src/semiempirical/scc_dftb.cpp` | SE2 | 128:GeneralizedSelfAdjointEigenSolver, 355:GeneralizedSelfAdjointEigenSolver, 474:GeneralizedSelfAdjointEigenSolver, 642:GeneralizedSelfAdjointEigenSolver |
| `cpp/src/semiempirical/seccm/dftb0.cpp` | SE2 | 138:SelfAdjointEigenSolver, 158:GeneralizedSelfAdjointEigenSolver |
| `cpp/src/semiempirical/seccm/gfn2.cpp` | SE2 | 733:GeneralizedSelfAdjointEigenSolver, 1666:SelfAdjointEigenSolver |
| `cpp/src/semiempirical/seccm/omx.cpp` | SE2 | 925:SelfAdjointEigenSolver |
| `cpp/src/semiempirical/seccm/pm6.cpp` | SE2 | 654:SelfAdjointEigenSolver |
| `cpp/src/semiempirical/seccm/scc_dftb.cpp` | SE2 | 162:GeneralizedSelfAdjointEigenSolver, 356:SelfAdjointEigenSolver |
| `cpp/src/semiempirical/seccm/seccm_common.h` | SE2 | 341:SelfAdjointEigenSolver, 394:SelfAdjointEigenSolver |
| `cpp/src/soscf.cpp` | M2 | 155:SelfAdjointEigenSolver |
| `cpp/src/trah.cpp` | M2 | 47:SelfAdjointEigenSolver |
| `cpp/src/uhf.cpp` | M1 | 385:solve_fock_matrix, 414:SelfAdjointEigenSolver |
| `cpp/src/uks.cpp` | M1 | 1158:solve_fock_matrix |
| `python/vibeqc/aux_basis.py` | A1 | 2498:eigh, 2833:eigh, 3021:eigh, 4065:eigh, 4413:eigh, 5309:eigh, 6263:eigh, 6495:eigh, 6577:eigh, 7067:eigh |
| `python/vibeqc/bands.py` | R2 | 359:diagonalize_bloch, 461:diagonalize_bloch, 833:diagonalize_bloch |
| `python/vibeqc/basis_optimization/gradients.py` | A1 | 182:eigh |
| `python/vibeqc/basis_optimization/ld_diagnostics.py` | A1 | 135:eigvalsh, 416:eigvalsh |
| `python/vibeqc/bond_analysis.py` | A1 | 158:eigh, 234:eigh |
| `python/vibeqc/cc.py` | C2 | 840:solve_dense_fock, 845:solve_dense_fock, 1499:eigh, 1522:solve_dense_fock |
| `python/vibeqc/coop_cohp.py` | R2 | 355:diagonalize_bloch, 594:diagonalize_bloch |
| `python/vibeqc/cphf.py` | R2 | 832:eigvalsh |
| `python/vibeqc/density_fitting.py` | R2 | 560:eigvalsh, 587:eigvalsh |
| `python/vibeqc/dlpno/_ccsd_ref.py` | C2 | 526:solve_dense_fock |
| `python/vibeqc/dlpno/ccsd.py` | C2 | 250:eigh, 264:eigh |
| `python/vibeqc/dlpno/ccsd_local_solver.py` | C2 | 728:eigh, 745:eigh, 754:eigh |
| `python/vibeqc/dlpno/mp2.py` | C2 | 518:eigh, 538:solve_dense_fock |
| `python/vibeqc/dlpno/pao.py` | C2 | 149:eigh, 534:eigh, 542:eigh |
| `python/vibeqc/dlpno/triples_local.py` | C2 | 575:eigh, 581:eigh, 800:eigh, 806:eigh |
| `python/vibeqc/dlpno/uccsd.py` | C2 | 267:eigh, 280:eigh |
| `python/vibeqc/dlpno/uccsd_local_solver.py` | C2 | 219:eigh, 232:eigh, 530:eigh, 549:eigh, 1353:eigh |
| `python/vibeqc/dlpno/ump2.py` | C2 | 181:eigh, 189:solve_dense_fock, 209:_pnos, 233:_pnos, 234:_pnos, 399:_pnos, 464:_pnos, 465:_pnos |
| `python/vibeqc/eda.py` | A1 | 289:eigh, 414:eigh, 426:eigvalsh |
| `python/vibeqc/entanglement.py` | A1 | 122:eigvalsh, 207:eigh, 213:eigvalsh, 299:eigvalsh |
| `python/vibeqc/excited.py` | R2 | 102:eigh |
| `python/vibeqc/geomopt/internal_coords.py` | A1 | 373:eigh |
| `python/vibeqc/geomopt/trust.py` | A1 | 90:eigvalsh, 493:eigh |
| `python/vibeqc/geomopt/ts.py` | A1 | 177:eigh, 193:eigh, 279:eigh |
| `python/vibeqc/gradient/_caspt2.py` | A1 | 73:eigh, 443:eigvalsh, 508:eigh, 919:eigh, 1086:eigvalsh, 1121:eigh |
| `python/vibeqc/gradient/_caspt2_cilag.py` | A1 | 69:eigh |
| `python/vibeqc/gradient/_casscf.py` | A1 | 456:eigh, 813:eigh, 1028:eigh |
| `python/vibeqc/gradient/_ic_caspt2_density.py` | A1 | 394:eigh |
| `python/vibeqc/gradient/_ms_caspt2_grad.py` | C2 | 217:eigh, 286:eigh, 348:eigvalsh, 387:eigh |
| `python/vibeqc/gradient/_ms_caspt2_nac.py` | C2 | 165:eigh, 409:eigh |
| `python/vibeqc/gradient/_nevpt2.py` | A1 | 62:eigh, 306:eigh, 667:eigvalsh, 695:eigh, 944:eigvalsh, 979:eigh |
| `python/vibeqc/gradient/_nevpt2_cilag.py` | A1 | 76:eigh |
| `python/vibeqc/guess.py` | M2 | 388:eigvalsh, 617:eigh, 628:eigh, 655:eigh, 686:eigh, 894:eigh, 947:eigh |
| `python/vibeqc/guess_read.py` | M2 | 73:eigvalsh |
| `python/vibeqc/hessian.py` | A1 | 884:eigh, 888:eigh |
| `python/vibeqc/hessian_analytic.py` | A1 | 222:eigvalsh, 387:eigvalsh, 454:eigh, 458:eigh |
| `python/vibeqc/hessian_analytic_rks.py` | A1 | 237:eigvalsh, 421:eigvalsh, 478:eigh, 482:eigh |
| `python/vibeqc/hessian_analytic_uhf.py` | A1 | 168:eigvalsh, 169:eigvalsh, 352:eigvalsh, 353:eigvalsh, 421:eigh, 425:eigh |
| `python/vibeqc/hessian_analytic_uks.py` | A1 | 278:eigvalsh, 279:eigvalsh, 612:eigvalsh, 613:eigvalsh, 677:eigh, 681:eigh |
| `python/vibeqc/hessian_caspt2.py` | A1 | 191:eigh |
| `python/vibeqc/hessian_casscf.py` | A1 | 263:eigh |
| `python/vibeqc/iao.py` | A1 | 121:eigh |
| `python/vibeqc/kerker.py` | A1 | 466:eigh, 480:eigh |
| `python/vibeqc/linear_dependence.py` | A1 | 291:eigvalsh, 306:eigh |
| `python/vibeqc/nbo.py` | A1 | 171:eigh, 181:eigh, 309:eigh, 318:eigh |
| `python/vibeqc/orthogonalisation.py` | A1 | 144:eigh, 286:eigvalsh, 317:eigh, 411:eigvalsh |
| `python/vibeqc/output/formats/molden.py` | A1 | 270:eigh |
| `python/vibeqc/output/formats/population.py` | A1 | 470:eigh, 887:eigh, 1257:eigh |
| `python/vibeqc/parity.py` | R2 | 172:eigvalsh, 173:eigvalsh, 174:eigvalsh, 175:eigvalsh |
| `python/vibeqc/pbc_bipole.py` | P2 | 1030:_diag_in_orth_basis, 1530:_diag_in_orth_basis, 1947:_diag_in_orth_basis |
| `python/vibeqc/pbc_bipole_common.py` | P2 | 510:eigh |
| `python/vibeqc/pbc_bipole_rks.py` | P2 | 1113:_diag_in_orth_basis, 1645:_diag_in_orth_basis, 2037:_diag_in_orth_basis, 2284:_diag_in_orth_basis |
| `python/vibeqc/pbc_bipole_uhf.py` | P2 | 829:_diag_in_orth_basis, 830:_diag_in_orth_basis, 1309:_diag_in_orth_basis, 1313:_diag_in_orth_basis, 1796:_diag_in_orth_basis, 1800:_diag_in_orth_basis |
| `python/vibeqc/pbc_bipole_uks.py` | P2 | 1031:_diag_in_orth_basis, 1032:_diag_in_orth_basis, 1609:_diag_in_orth_basis, 1613:_diag_in_orth_basis, 2136:_diag_in_orth_basis, 2137:_diag_in_orth_basis, 2462:_diag_in_orth_basis, 2463:_diag_in_orth_basis |
| `python/vibeqc/pbc_gdf.py` | P2 | 1521:FockDiagonalizer, 2252:FockDiagonalizer, 2254:FockDiagonalizer, 3141:FockDiagonalizer, 3143:FockDiagonalizer |
| `python/vibeqc/periodic/ccm/bloch_reference.py` | P3 | 253:eigh, 321:eigh |
| `python/vibeqc/periodic/ccm/dft.py` | P3 | 251:eigh, 368:eigh |
| `python/vibeqc/periodic/ccm/direct.py` | P3 | 1472:eigh, 1742:eigh |
| `python/vibeqc/periodic/ccm/dlpno.py` | P3 | 294:eigh, 307:eigh |
| `python/vibeqc/periodic/ccm/dlpno_ccsd.py` | P3 | 85:eigh, 91:eigh |
| `python/vibeqc/periodic/ccm/dlpno_uccsd.py` | P3 | 109:eigh, 113:eigh, 129:eigh, 133:eigh, 142:eigvalsh |
| `python/vibeqc/periodic/ccm/dlpno_ump2.py` | P3 | 144:_pnos, 190:_pnos, 194:_pnos |
| `python/vibeqc/periodic/ccm/properties.py` | P3 | 452:eigh |
| `python/vibeqc/periodic/ccm/ri.py` | P3 | 153:eigh, 1086:eigh, 1158:eigh |
| `python/vibeqc/periodic/ccm/scf.py` | P3 | 339:eigh, 472:eigh |
| `python/vibeqc/periodic/ccm/system.py` | P3 | 638:eigvalsh |
| `python/vibeqc/periodic/ccm/uhf.py` | P3 | 221:eigh |
| `python/vibeqc/periodic/chi/localization.py` | P3 | 140:eigh, 255:eigh |
| `python/vibeqc/periodic/chi/operator_diagnostics.py` | P3 | 259:eigvalsh |
| `python/vibeqc/periodic/chi/pno.py` | P3 | 1058:eigh, 1084:eigh, 1134:eigh, 1162:eigh, 1299:eigh, 1300:eigh |
| `python/vibeqc/periodic/chi/posthf.py` | P3 | 966:eigh, 976:eigh, 1032:eigh, 1042:eigh, 1322:eigh, 1416:eigh, 1629:eigh, 1630:eigh |
| `python/vibeqc/periodic_embedding/embedded_localise.py` | P2 | 193:eigh, 206:eigh |
| `python/vibeqc/periodic_embedding/runner.py` | A1 | 179:eigvalsh |
| `python/vibeqc/periodic_embedding/scf2step.py` | P2 | 371:eigh |
| `python/vibeqc/periodic_embedding/substrate_gf.py` | P2 | 385:eigh, 418:eigh |
| `python/vibeqc/periodic_gapw_augment.py` | P2 | 3303:_eigh_safe, 3305:_eigh_safe, 3332:_eigh_safe, 3336:FockDiagonalizer, 4142:_eigh_safe, 4146:FockDiagonalizer, 4149:FockDiagonalizer, 4186:_eigh_safe, 4665:_eigh_safe, 4669:FockDiagonalizer, 4672:FockDiagonalizer, 4699:_eigh_safe, 5438:eigh, 5582:eigh, 5902:solve_dense_fock |
| `python/vibeqc/periodic_gapw_hessian.py` | A1 | 364:eigh |
| `python/vibeqc/periodic_gapw_j.py` | P2 | 202:eigvalsh, 2581:_eigh_safe, 2583:_eigh_safe, 2595:_eigh_safe, 2604:_eigh_safe, 2611:FockDiagonalizer, 3497:eigh, 3623:eigh, 3952:solve_dense_fock, 3975:solve_dense_fock, 4212:solve_dense_fock, 4336:davidson_solve, 4342:eigh, 4346:eigh |
| `python/vibeqc/periodic_gapw_open_shell.py` | P2 | 820:_eigh_safe, 825:FockDiagonalizer, 828:FockDiagonalizer, 872:_eigh_safe, 1464:eigh, 2272:eigh, 2429:eigh, 2487:eigh, 2674:solve_dense_fock, 2697:solve_dense_fock, 2700:solve_dense_fock |
| `python/vibeqc/periodic_gapw_ot.py` | P2 | 281:_eigh_safe, 608:_eigh_safe |
| `python/vibeqc/periodic_gapw_phonon.py` | A1 | 488:eigh |
| `python/vibeqc/periodic_gapw_postscf.py` | P2 | 627:eigh, 629:solve_dense_fock |
| `python/vibeqc/periodic_gapw_range_sep.py` | P2 | 452:_eigh_safe, 454:_eigh_safe, 461:_eigh_safe, 465:FockDiagonalizer |
| `python/vibeqc/periodic_gdf_gradient.py` | A1 | 905:eigh, 1289:eigh, 2311:eigh |
| `python/vibeqc/periodic_k_density.py` | A1 | 79:eigvalsh, 114:eigh |
| `python/vibeqc/periodic_k_gdf.py` | P2 | 784:eigh, 1291:eigvalsh, 4408:eigh, 5597:eigvalsh, 6072:_diag_in_orth_basis, 6387:_diag_in_orth_basis, 6572:_diag_in_orth_basis, 6647:_diag_in_orth_basis, 6755:_diag_in_orth_basis, 6852:_diag_in_orth_basis, 7859:eigvalsh, 8208:_diag_in_orth_basis, 8295:_diag_in_orth_basis, 8484:_diag_in_orth_basis, 8485:_diag_in_orth_basis, 8563:_diag_in_orth_basis, 8566:_diag_in_orth_basis, 8628:_diag_in_orth_basis, 8629:_diag_in_orth_basis, 8706:_diag_in_orth_basis, 8707:_diag_in_orth_basis, 8934:_diag_in_orth_basis, 8935:_diag_in_orth_basis |
| `python/vibeqc/periodic_rhf_ewald.py` | P2 | 207:eigh, 224:eigh, 561:FockDiagonalizer, 1129:FockDiagonalizer |
| `python/vibeqc/periodic_rhf_gdf.py` | P2 | 1031:FockDiagonalizer, 1658:eigh, 1798:eigh |
| `python/vibeqc/periodic_rhf_multi_k_ewald.py` | P1 | 336:eigh, 352:eigh, 375:solve_operator, 926:_diag_in_orth_basis, 1445:_diag_in_orth_basis, 1648:_diag_in_orth_basis, 1793:_diag_in_orth_basis |
| `python/vibeqc/periodic_rks_ewald.py` | P2 | 574:FockDiagonalizer |
| `python/vibeqc/periodic_rks_multi_k_ewald.py` | P2 | 804:_diag_in_orth_basis, 1252:_diag_in_orth_basis, 1447:_diag_in_orth_basis, 1545:_diag_in_orth_basis |
| `python/vibeqc/periodic_rohf_ewald.py` | M2 | 339:FockDiagonalizer, 347:solve_operator, 349:solve_dense_fock, 603:solve_dense_fock |
| `python/vibeqc/periodic_rohf_gdf.py` | M2 | 502:eigvalsh, 630:_diag_in_orth_basis, 676:_diag_in_orth_basis, 908:_diag_in_orth_basis, 963:_diag_in_orth_basis |
| `python/vibeqc/periodic_rohf_multi_k_ewald.py` | M2 | 422:_diag_in_orth_basis, 749:_diag_in_orth_basis |
| `python/vibeqc/periodic_roks_multi_k_ewald.py` | M2 | 563:_diag_in_orth_basis, 928:_diag_in_orth_basis |
| `python/vibeqc/periodic_uhf_ewald.py` | P2 | 452:FockDiagonalizer, 456:FockDiagonalizer, 487:FockDiagonalizer |
| `python/vibeqc/periodic_uhf_multi_k_ewald.py` | P2 | 707:_diag_in_orth_basis, 708:_diag_in_orth_basis, 1024:_diag_in_orth_basis, 1028:_diag_in_orth_basis, 1327:_diag_in_orth_basis, 1331:_diag_in_orth_basis, 1380:_diag_in_orth_basis, 1383:_diag_in_orth_basis, 1490:_diag_in_orth_basis, 1494:_diag_in_orth_basis |
| `python/vibeqc/periodic_uks_ewald.py` | P2 | 562:davidson_solve, 570:eigh |
| `python/vibeqc/periodic_uks_multi_k_ewald.py` | P2 | 732:_diag_in_orth_basis, 733:_diag_in_orth_basis, 1045:_diag_in_orth_basis, 1049:_diag_in_orth_basis, 1473:_diag_in_orth_basis, 1477:_diag_in_orth_basis, 1665:_diag_in_orth_basis, 1668:_diag_in_orth_basis, 1941:_diag_in_orth_basis, 1945:_diag_in_orth_basis |
| `python/vibeqc/properties.py` | A1 | 276:eigh, 484:eigh, 921:eigh, 934:eigh |
| `python/vibeqc/qtaim.py` | A1 | 320:eigvalsh, 574:eigh |
| `python/vibeqc/relocalize.py` | A1 | 119:eigvalsh, 672:eigh |
| `python/vibeqc/rohf.py` | M2 | 514:solve_dense_fock, 524:solve_dense_fock, 534:solve_dense_fock, 555:eigh, 1034:FockDiagonalizer, 1455:FockDiagonalizer, 1633:eigh |
| `python/vibeqc/runner.py` | A1 | 3942:eigh, 5037:eigh, 5214:eigvalsh |
| `python/vibeqc/semiempirical/methods/msindo.py` | SE2 | 1988:eigh, 2029:eigh, 2039:eigh, 2076:eigh, 2090:eigh, 2147:eigh, 2151:eigh, 2271:eigh, 2274:eigh, 2282:eigh, 2283:eigh, 2322:eigh, 2369:eigh, 2372:eigh, 2390:eigh, 2391:eigh, 3188:eigh, 3511:eigh, 3642:eigh, 3923:eigh |
| `python/vibeqc/semiempirical/methods/msindo_ccm_stability.py` | SE2 | 35:eigh, 54:eigsh, 102:eigvalsh |
| `python/vibeqc/semiempirical/methods/msindo_cis.py` | SE2 | 541:eigh, 811:eigh, 817:eigh |
| `python/vibeqc/solvers/_casci.py` | C1 | 330:eigh |
| `python/vibeqc/solvers/_caspt2_cases.py` | R2 | 1076:eigh, 1082:eigh |
| `python/vibeqc/solvers/_caspt2_rdm.py` | R2 | 612:eigh, 619:eigh, 659:eigh, 4814:eigh, 4924:eigh, 4991:eigh, 5004:eigh, 5057:eigh |
| `python/vibeqc/solvers/_cisd.py` | C1 | 267:eigh |
| `python/vibeqc/solvers/_dmrg.py` | C1 | 257:eigvalsh, 357:eigh, 381:eigh |
| `python/vibeqc/solvers/_hamiltonian.py` | A1 | 190:eigh, 449:eigh, 458:eigh, 505:eigh |
| `python/vibeqc/solvers/_mrci.py` | R2 | 262:eigh |
| `python/vibeqc/solvers/_mrpt.py` | C2 | 411:eigh, 844:eigh, 853:eigh, 865:eigh |
| `python/vibeqc/solvers/_ms_caspt2.py` | C2 | 258:eigh, 340:eigh, 573:eigh |
| `python/vibeqc/solvers/_scf.py` | S1 | 31:solve_operator, 105:solve_operator |
| `python/vibeqc/solvers/_selected_ci.py` | C1 | 828:eigh |
| `python/vibeqc/solvers/_v2rdm.py` | A1 | 117:eigh, 196:eigh, 286:eigvalsh, 294:eigvalsh, 302:eigvalsh |
| `python/vibeqc/solvers/eigensolver.py` | S1 | 501:solve_operator, 545:eigh, 642:eigh, 738:davidson_solve_matmat, 740:davidson_solve_matvec, 743:davidson_solve, 836:davidson_solve_hermitian_matmat, 838:davidson_solve_hermitian_matvec, 841:davidson_solve_hermitian, 1166:eigh, 1236:solve_operator, 1256:solve_scf_step |
| `python/vibeqc/solvers/operators.py` | S1 | 286:eigh, 356:solve_eigenproblem |
| `python/vibeqc/tddft.py` | R1 | 745:solve_operator, 1000:eigh, 1164:eigh, 1178:eigh, 1617:eigh, 1745:eigh, 1754:eigh, 2115:eigh |
| `python/vibeqc/thermo.py` | R2 | 215:eigvalsh |

The experimental molecular quartet factory also consumes MANTA's native
`MolecularSourceSupport` certificate before evaluating its source. This binds
exact radial channels and operation-compatible shell inventory with an explicit
center tolerance; arithmetic equivariance still requires the full numerical
audit. The qualification identity is reported separately from physical
source/mask, AO frame, density and evaluated-Fock identities. This does not
admit a state reduction, a periodic source, a BUPO partition or a derivative.
