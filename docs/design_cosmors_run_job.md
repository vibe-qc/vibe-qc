# Design, COSMO-RS thermodynamics from `run_job`

**Status:** accepted, 2026-09-30. Proposed 2026-09-29; the recommended
protocol (section 2a) measured 2026-09-30; the maintainer took the three
decisions of section 5 on 2026-09-30, each as proposed. The maintainer decided
that `run_job` should compute and report COSMO-RS thermodynamics, with this
note first. **M0 is implemented** (`run_job(cosmors=vq.CosmoRSOptions(...))`,
`vibeqc.solvation.cosmors.job`, documented in the user guide's "COSMO-RS
solvation free energies"). **M1 is implemented** (2026-10-11:
`CosmoRSOptions(solvent="water")`, `vibeqc.solvation.cosmors.solvents`), with
one refinement of the proposal below: the solvent is computed at the
*solute's* protocol rather than at a fixed recommended one, so Q3's agreement
holds by construction whatever level the solute runs at, and the recommended
protocol is a recommendation the docs make rather than a constraint the code
imposes. The Direct COSMO-RS nuclear gradient M2 waits on exists (same date);
M2 itself, the chemical potential on the Direct COSMO-RS density, is not
implemented. Ring atoms can be perceived on request
(`n_ring_atoms="auto"`); the default of Q4 is unchanged.

The COSMO-RS layer already exists as a library in
`vibeqc.solvation.cosmors`: sigma profiles, sigma potentials, chemical
potentials, the gas-phase reference, solvation free energies, activity and
partition coefficients, and the COSMOSPACE solver. What is missing is a job
that runs the calculations those functions need, in the right order, on
consistent surfaces, and reports the result through `vibeqc.output` with its
citations. That is orchestration, not new physics, and this note is about the
orchestration.

## 1. Scope

**Lands at M0.** `run_job` computes a COSMO-RS solvation free energy
`dG_solv` (kcal/mol) for the solute in one pure solvent, together with the two
chemical potentials it is the difference of, when the caller asks for it. The
`.out` gets a `## COSMO-RS` block, the result object carries the numbers, and
the `cosmors` citation route fires from the job plan for the first time.

**Deferred.** Mixtures and mole fractions; activity and partition
coefficients as job outputs (they stay library calls); the COSMOSPACE solver as
a job option; temperature scans; geometry optimization on a COSMO-RS surface
(that waits on a Direct COSMO-RS gradient, which the optional smoothed
hydrogen-bond term, `Parameterization.hb_smoothing`, now makes possible);
periodic systems.

## 2. What the thermodynamics need

`solvation_free_energy(mu_S, mu_gas)` is the end of a chain, and every link
has a precondition:

| quantity | library call | needs |
|---|---|---|
| solute surface and descriptors | `build_conductor_surface`, `segment_descriptors` | a conductor SCF of the solute: `epsilon = inf`, `cavity="fine"`, the parameterization's own radii, `radii_scale=1.0`, no probe |
| solute sigma profile | `sigma_profile` | the descriptors |
| `mu_S`, chemical potential in the solvent | `chemical_potential(profile, potential, params, solvent_area=...)` | a solvent sigma potential and the solvent molecule's surface area |
| `mu_gas`, gas-phase reference | `gas_phase_chemical_potential(surface, descriptors, params, ideal_screening_energy_hartree=..., n_ring_atoms=...)` | the ideal screening energy `E_gas - E_COSMO` (positive), and a ring-atom count |
| `dG_solv` | `solvation_free_energy(mu_S, mu_gas)` | both of the above |

`run_cpcm_scf` already computes the gas-phase reference energy of the solute
as `SolventResult.e_gas`, so the ideal screening energy is
`result.e_gas - result.energy` of the conductor run, positive: the gas phase
lies above the ideally screened state by that much. Klamt's papers quote the
same quantity with the opposite sign, and eq. 21 is printed for that
convention; the gas reference had that sign, and its eta term, wrong until
#435, which is why the vapour-pressure test of Q6 exists. The job therefore adds no
SCF of its own for the solute. The solvent potential is the one input that
costs a separate calculation.

## 2a. The recommended protocol

Klamt's 1998 parameterization, `KLAMT_1998`, was fitted against
DMol/BPW91/DNP surfaces. A surface computed at another level carries its own
sigma distribution, and the parameterization's energies are only as good as
the overlap. Measured on water, with the fine cavity and the parameterization's
radii, the level of theory decides how far a surface polarizes: RHF/STO-3G
puts the 1st to 99th percentile of sigma at -0.012 to +0.014 e/A^2,
BPW91/def2-TZVP at -0.017 to +0.018. The narrow profile shows in every number
built on it. In an RHF/STO-3G water potential acetate's COSMO-RS term is
+11.7 kcal/mol, unfavourable for an anion in water; at BPW91/def2-TZVP it is
-15.0, and ammonium's moves from -11.0 to -19.5.

**The recommended protocol is therefore BPW91/def2-TZVP** (`functional=
"bpw91"`, the fine cavity, the parameterization's radii, no probe), the nearest
match to DMol/BPW91/DNP that vibe-qc can run, and M1's computed solvents use
it. It is not an exact match and cannot be one, since DNP is a numerical basis.
Nor does it end extrapolation for ions: at BPW91/def2-TZVP acetate puts 6.9
percent of its area beyond the fitted grid and ammonium 8.2. That is what an
ion's surface is, which is why Q5 reports the fraction.

## 3. Pre-flight questions

### Q1, a `run_job` keyword or a separate driver?

**Proposal: a keyword**, `run_job(..., cosmors=CosmoRSOptions(...))`. A
separate `run_cosmors` driver would have to reproduce `run_job`'s output
plumbing, citation plan and result shape. A keyword gets all three for free and
keeps one place where a job's citations are assembled, which is the defect
class `fd9387d` fixed. The conductor solvent model the chain needs is then
implied by the request: passing `cosmors=` with no `solvent=` runs the
conductor SCF on the fine cavity at the parameterization's radii, and passing
an incompatible `solvent=` alongside it is refused rather than silently
overridden.

### Q2, where does the solvent potential come from?

**M0: the caller passes one**, a `SigmaPotential` built with
`solvent_sigma_potential`, which already refuses a screened solvent run and
foreign radii and records the solvent's protocol. **M1: a solvent name**,
computed on demand at the recommended protocol and cached in a per-user cache
directory, keyed by solvent, protocol and parameterization. The cache location
has to be portable (a platform user-cache path), never a path in the checkout.

### Q3, what happens when the protocols disagree?

Three protocols meet here: the parameterization's fitted one, the solvent
potential's and the solute's. **Proposal:** refuse a solute whose method or
basis differs from the solvent potential's, because the two sigma
distributions are then not comparable and the error is silent; record, as
`SigmaPotential.protocol` already does, a solute and solvent that agree with
each other but not with the parameterization, and print that transfer in the
`## COSMO-RS` block.

### Q4, where does the ring-atom count come from?

`mu_gas` subtracts `omega_ring * n_ring_atoms` (Klamt 1998, eq. 21), and
vibe-qc has no bond perception to count ring atoms from. **Proposal for M0:**
the caller passes `n_ring_atoms` in `CosmoRSOptions`, defaulting to `0`, and
the output states the value used. Perceiving rings from covalent radii is a
separate, general-purpose change and should not be buried in this one.

### Q5, what does the output say?

A `## COSMO-RS` block through `vibeqc.output`: `mu_S`, `mu_gas`, `dG_solv` in
kcal/mol, the temperature, the parameterization and the three protocols, and
the `fraction_outside_grid` of the solute's surface. That last line matters:
the Direct COSMO-RS convergence panel measured that an ion's surface runs past
the fitted sigma range, at every level of theory tried (section 2a), and a free
energy extrapolated there should say so where the number is read.
The citation route is `cosmors`; `cosmospace` fires only once the COSMOSPACE
solver is a job option.

### Q6, how is it validated?

**Exact identities, now:** a solute in itself has an activity coefficient of
exactly one; `log K` between two phases is antisymmetric; `dG_solv` from the
job equals the library chain run by hand on the same surfaces, to machine
precision. **Against experiment, one check exists already:** eight
pure-liquid vapour pressures from Klamt 1995 Table 2, which exercise every term
of the gas reference, reproduce to rms 1.2 kcal/mol at RHF/STO-3G (#435); a job
that reports `dG_solv` must keep passing it. A broad accuracy claim needs a
benchmark dataset such as FreeSolv, which the literature library does not
hold, so M0 claims consistency, not accuracy, and the docs say so.

## 4. Milestones

* **M0 (done).** `CosmoRSOptions` with a caller-supplied solvent potential and
  `n_ring_atoms`; the conductor SCF implied by the request; the chain of
  section 2; the `## COSMO-RS` block, result fields and the `cosmors`
  citation; tests on the identities of Q6 and on the protocol refusal of Q3.
* **M1 (done).** Solvent by name, computed at the solute's protocol, with the
  per-user cache of conductor surfaces (`$XDG_CACHE_HOME/vibeqc/cosmors`).
* **M2.** After the Direct COSMO-RS gradient: the chemical potential on the Direct COSMO-RS
  density rather than the conductor one, and COSMOSPACE as an option.

## 5. Decisions

Taken by the maintainer on 2026-09-30, each as proposed:

1. **A keyword on `run_job`** (Q1), `run_job(..., cosmors=CosmoRSOptions(...))`,
   not a separate driver.
2. **Refuse a solute and solvent at different protocols** (Q3); record, and
   print, a pair that agrees with itself but not with the parameterization.
3. **`n_ring_atoms` from the caller at M0** (Q4), default `0`, stated in the
   output; ring perception is separate work.
