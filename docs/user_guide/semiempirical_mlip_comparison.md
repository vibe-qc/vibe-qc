---
myst:
  html_meta:
    "description": "A production-oriented comparison of vibe-qc's DFTB, GFN2-xTB, PM6/OMx, MSINDO, and MACE method surfaces."
    "og:title": "vibe-qc, semiempirical and MACE method comparison"
---

# Semiempirical and MACE method comparison

**Status date: 2026-07-26.** This note compares the fast-method surface in
vibe-qc for development and production planning: DFTB0/SCC-DFTB, GFN2-xTB,
PM6/OM1/OM2/OM3, MSINDO, and MACE. It is intentionally conservative: a
method can be executable and still not be production-validated.

## Abstract

vibe-qc now contains two different kinds of fast energy engines. The
semiempirical engines are electronic-structure methods implemented inside
vibe-qc. They build Hamiltonians, solve SCF-like equations when required,
and can expose electronic quantities such as densities or charges. MACE is
different: it is an external pre-trained machine-learning interatomic
potential (MLIP). It predicts a DFT-fitted potential-energy surface from a
neural network and provides no wavefunction, orbitals, density, or charge
model.

The production conclusion is therefore not "which method is best" but
"which method is valid for this question." Use MACE for fast structures,
forces, stress, molecular dynamics, and relaxations inside a trained domain.
Use MSINDO when the system lies inside its validated molecular scope and a
semiempirical electronic model is needed. Use OM2/OM3 (and PM6, within its
documented integral-model residuals) for molecular NDDO work. Use DFTB as
a screening and preoptimization tool. Treat GFN2-xTB and OM1 as
experimental until their gates lift.

## Feature matrix

| Engine | Model type | Energy scale | Electronic outputs | Gradients / stress | Periodic support | Open shell | Current scope | Production recommendation |
|---|---|---|---|---|---|---|---|---|
| DFTB0 / UDFTB0 | non-self-consistent tight binding | vibe-qc in-house semiempirical total | density-like TB quantities, limited charges | analytic DFTB0 gradients; stress paths available | Gamma + k-point paths | yes | 91 in-house estimated elements | Screening and preoptimization only; parameters are not DFTB+ parity parameters |
| SCC-DFTB / USCC | self-consistent-charge tight binding | vibe-qc in-house semiempirical total | charges, densities, SCC trace | fixed-charge approximate gradients; stress paths available | Gamma + k-point paths | yes | 91 in-house estimated elements | Screening/preoptimization; do not publish quantitative energies without external validation |
| GFN2-xTB | self-consistent tight binding with multipole electrostatics | GFN2-style semiempirical total | charges and SCC state | analytic molecular gradient; gapped Gamma-periodic gradient and all-nine stress parity, still experimental | Gamma path experimental; general k-point paths fail closed | yes | 86 fetched GFN2 parameters | Keep experimental gate; remaining gates are faithful periodic AES electrostatics, periodic molecular-limit parity, broader difficult fixtures, and xTB parity |
| PM6 / UPM6 | NDDO SCF | PM6-like total, not MOPAC heat of formation | density, SCF convergence, spin counts for UPM6 | finite difference | Gamma periodic path, experimental | molecular yes | 75 bundled chemical-element parameter records | Molecular development: H-only and H-heavy s/p interactions are source-correct; full heavy-heavy tensor parity remains open |
| PM7 / UPM7 | NDDO SCF | unavailable | none | unavailable | unavailable | no | 75 bundled chemical-element parameter records | Gated until Stewart's feathered electrostatics and all coupled NDDO terms are implemented |
| OM2 / OM3 | orthogonalization-corrected NDDO SCF (Dral 2016 eqs 6-20) | OMx total; published-benchmark relative energetics (H3- 0.1 kcal/mol, ethane barrier ±0.6); minima ~0.1-0.2 A long | density, SCF convergence, PES wells, published matrix elements + barriers | finite difference | Bloch-periodic route gated; topology-bound OMx-SECCM experimental | yes (UHF) | H, C, N, O, F | Validated molecular development/prescreening; external implementation parity remains open |
| OM1 | orthogonalization-corrected NDDO SCF | OMx total; X-H bonds ~0.3 A short, close contacts can collapse (analytic ECP missing) | density, SCF convergence, documented-limitation pins | finite difference | Bloch-periodic and topology-bound OM1-SECCM routes gated | yes (UHF) | H, C, N, O, F | Experimental (warns); unblocks when Kolb-Thiel 1993 ECP lands |
| MSINDO | INDO SCF over Slater orbitals | reference-MSINDO total | charges, density, binding energy, spin UHF state | FD plus analytic INDO molecular/CCM; analytic closed-shell NDDO molecular | CCM 1D/2D/3D single-point; closed-shell through `run_job` | INDO UHF s/p/d within validated fixtures; NDDO closed-shell | H-Xe INDO; NDDO energy/gradient H, Li-F, Na-Cl, including Al-Cl SPDD | Production within documented INDO and closed-shell NDDO scope |
| MACE-MPA / MACE-OFF | external pre-trained MLIP | model-specific reference-shifted DFT surface, not an electronic total | none | analytic forces; periodic stress and variable-cell relaxation | periodic direct drivers for materials models | spin not modeled as electronic state | model-dependent trained domain | Operational for fixed-model structures and forces; scientific accuracy claims require external observables; never compare absolute energies to SCF/semiempirical totals |

## Method-by-method comparison

### DFTB0 and SCC-DFTB

DFTB is the fastest native electronic route in vibe-qc. DFTB0 has no charge
self-consistency loop, so it is especially useful for cheap geometry
preconditioning. SCC-DFTB adds charge response and is often more chemically
reasonable for polar systems. The limitation is parameter provenance: the
current tables are in-house estimates, not a published DFTB parameter set
with fitted repulsive potentials. That makes these methods useful for
workflow acceleration, not for quantitative publication claims.

Production guidance: keep using DFTB for preoptimization and regression
coverage of fast periodic infrastructure. Do not promote the present
parameter set to quantitative DFTB until fitted repulsives and external DFTB+
parity tests exist.

### GFN2-xTB

GFN2-xTB is the most ambitious native fast method in the stack. The deep
H0/overlap bug was fixed on 2026-06-01. Molecular AES dipole/quadrupole
terms, the GAM3 third-order term, and post-SCF native D4-style dispersion
have landed.
The native Gamma-periodic gradient and stress now differentiate the same
image-summed energy functional, including H0, overlap, AES, coordination, and
response terms. A gapped polar cell pins every gradient and all nine stress
components against central differences. This closes the derivative mismatch,
not the method-wide production gate.
The method remains gated because the defining GFN2 production surface still
requires periodic AES image-cell multipole Ewald terms, stricter periodic
molecular-limit parity, difficult periodic/polar SCC fixtures, and a closed
parity matrix against external `xtb` over molecules, ions, radicals, and
periodic test cells.

Production guidance: keep `GFN2ExperimentalWarning`. Use it for development
and targeted comparisons, not production numbers. The next production
milestone should be parity-first, not threshold tuning.

### PM6 and OMx

PM6 and OMx share the NDDO family shape: minimal valence basis, parameterized
one- and two-electron approximations, and SCF over a small Hamiltonian. PM6
has wider element reach through the bundled MOPAC-derived parameter cache.
OM1/OM2/OM3 have a narrower H/C/N/O/F surface but include the
orthogonalization corrections that distinguish the OMx family.

The molecular PM6/OM2/OM3 PES shape is pinned by
`tests/test_semiempirical_pes.py` (proper wells for
H2O/N2/CO/CH4, geometry-opt smoke), open-shell SCF converges (UPM6 +
UOMx), and the OMx relative energetics reproduce the published
benchmarks, Weber & Thiel 2000 H3- bending curve to 0.1 kcal/mol,
matrix elements to <0.01 eV, and the ethane rotation barrier (the
signature ORT observable: OM2 3.1 vs published 2.8, OM3 3.0 vs 2.4
kcal/mol), see `TestOMxV2AgainstWeber2000`. Two documented accuracy
residuals remain: PM6 falls back to a spherical Klopman-Ohno integral
model where MOPAC diatomic multipole data is absent, and the OMx
two-centre integrals use Klopman-scaled STO monopole densities in place
of the published scaled-Gaussian multipole integrals (omx_fock.cpp
documents every approximation inline). Both bias bond minima
systematically ~0.1-0.2 Å long while leaving the PES shape and relative
energetics intact. OM1 is the one experimental member: its
analytically-evaluated core-valence ECP (Kolb & Thiel 1993) is not
implemented yet, X-H bonds come out ~0.3 Å short, close non-bonded
contacts can variationally collapse (the uncapped F2 term), and
`run_job(method="om1")` warns accordingly.

Usage guidance: OM2/OM3 are suitable for H/C/N/O/F molecular development and
prescreening within the documented residuals, not as external-parity
production engines. PM6 is likewise a development route outside H-only and
H-heavy s/p systems until the full heavy-heavy tensor and out-of-process MOPAC
parity are complete; quote its results as PM6-like totals, not MOPAC heats of
formation. Method-paper citations are emitted on every run.

### MSINDO

MSINDO is the strongest production semiempirical path today within its
documented scope. The implementation is independent, validated
out-of-process against a reference MSINDO build, and reproduces total
energies to microhartree scale for the supported closed-shell and UHF test
sets. The tradeoff is scope rather than missing infrastructure: NDDO remains
closed-shell only, CCM through `run_job` is closed-shell single-point only,
ROHF is not implemented, and open-shell/heavy-element claims are tied to named
oracle fixtures rather than a blanket promise for every near-degenerate case.

Production guidance: MSINDO can be used for molecular energies and charges
inside the validated element/spin scope. Use the analytic molecular and CCM
gradients where documented; use finite differences or the documented fallback
near state/root degeneracies. The next production steps are scope expansion,
COSMO/GEPOL hardening, and moving remaining Python hot loops into C++.

### MACE

MACE provides a fast route to structures and forces when the system is
inside a trained domain. It is not an electronic-structure method:
there is no SCF, no basis, no density, no orbitals, no charges, and no band
structure. Its energy is a model-specific, reference-shifted DFT-surface
quantity. For water, the default MACE-MPA-0 and MACE-OFF23 models sit on very
different absolute energy scales, so absolute energies across MACE models or
between MACE and SCF/semiempirical methods are not meaningful.

Production guidance: use the registered MACE models for optimization,
molecular dynamics, stress, cell relaxation, and relative energies at fixed
composition and fixed model. Keep the ASL gate strict for academic-only
MACE-OFF23 weights. For materials-family work, default to the MIT
`medium-mpa-0` model. Treat runtime success and upstream-calculator parity as
implementation evidence, not external accuracy validation.

## Runnable comparison protocol

[`examples/mlip/05_mace_pm6_relative_curve.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/mlip/05_mace_pm6_relative_curve.py)
is the smallest safe cross-family comparison. It evaluates the same
fixed-angle, fixed-composition H₂O stretch with the registered MIT
`medium-mpa-0` model and PM6. It then subtracts each method's own sampled
minimum:

```text
r_oh_angstrom,mace_delta_ev,pm6_delta_ev
```

This supports comparison of curve shape, sampled minimum, and response to
the same displacement. It deliberately omits a mixed absolute-energy
difference. Neither curve is an external reference, so agreement is not an
accuracy result and disagreement is a diagnostic. Add a cited experimental
or electronic-structure observable before making an accuracy claim.

For periodic materials, start with
[`examples/mlip/02_periodic_mace_silicon.py`](https://github.com/vibe-qc/vibe-qc/blob/main/examples/mlip/02_periodic_mace_silicon.py).
Its default two-point strain probe retains one MACE model and reports
relative energy, gradient, stress, and provenance. Expand to an EOS only
after the compact probe is nonzero, stable, and compared with an external
reference protocol.

## Production implementation guidance

1. Keep method families separate in the user interface. Semiempirical methods
   are native electronic-structure engines; MACE is an external MLIP. The
   shared `energy()` / `gradient()` duck type is useful, but the provenance and
   energy interpretation are different.

2. Do not compare absolute energies across families. DFTB, GFN2, PM6, OMx,
   MSINDO, ab initio SCF, and MACE all use different references and fitted
   targets. Compare geometries, force directions, relative energies within one
   method/model, normalized fixed-composition curve shapes, or externally
   validated observables.

3. Keep every user-facing method in the citation database. The 2026-06-04
   audit found and fixed missing PM6 and OMx routes. Future method additions
   should update `python/vibeqc/output/citations/database.toml` and
   `tests/test_citations.py` in the same merge.

4. Preserve the no-runtime-wrapper rule for QC programs. xTB, MOPAC, DFTB+,
   and reference MSINDO belong in `examples/regression/...` subprocess runners,
   never imported into `python/vibeqc/` or the C++ core.

5. Gate production claims on parity, not convergence. For semiempirical
   methods, a converged SCF is necessary but not sufficient. The acceptance
   bar should include external energies, gradients or finite-difference
   checks, rotation/translation invariance, size consistency, charge signs,
   open-shell cases, and periodic molecular-limit tests where relevant.

6. Prefer small shippable milestones. Each milestone should leave `main`
   taggable: method available only inside its honest scope, docs and
   changelog aligned, citations emitted, tests green, and experimental paths
   visibly gated.

## Suggested next implementation sequence

1. **GFN2-xTB:** finish periodic AES image-cell multipole Ewald, tighten
   periodic molecular-limit parity, and keep difficult polar/periodic SCC
   fixtures strict; then close the xTB parity matrix before removing
   the experimental warning.

2. **PM6:** close the MOPAC parity gap on a small molecule set, documenting
   energy convention differences explicitly. Only then expand claims around
   the 75-record chemical-element cache.

3. **PM7:** implement Stewart's feathered electrostatics consistently in the
   electron-electron, electron-core, and core-core channels, then validate
   molecular and periodic pairs against MOPAC before lifting the gate.

4. **OMx:** add external reference fixtures for OM1/OM2/OM3. The corrected
   `run_omx_v2` kernel is the only public molecular native path; retain that
   single-path invariant while expanding parity fixtures.

5. **MSINDO:** keep direct energy, gradient, CCM, and COSMO B-matrix/vector-op
   production calls on the existing C++ kernels, port remaining COSMO/GEPOL,
   post-SCF, and root-tracking hot loops, and keep the closed-shell/open-shell
   scope tied to oracle-backed fixtures.

6. **MACE:** add a Python <=3.13 CI lane for the real MACE e2e tests and keep
   the model registry current as upstream weights and licenses evolve.
