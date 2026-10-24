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
imposes. The Direct COSMO-RS nuclear gradient M2 waits on exists (same date).
**M2a is implemented** (2026-10-21: `CosmoRSOptions(density="direct")`, the
solvation free energy on the Direct COSMO-RS density, section 6), and so is
**M2b** (2026-10-22: `activity_coefficient=True`, with COSMOSPACE as the test
oracle, section 7). Ring atoms can be perceived on request
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
* **M2a (done).** The solvation free energy on the Direct COSMO-RS density,
  Klamt & Diedenhofen 2015 eq. 9 (section 6).
* **M2b (done).** The infinite-dilution activity coefficient as a job output,
  COSMOSPACE as its independent oracle (section 7).

## 5. Decisions

Taken by the maintainer on 2026-09-30, each as proposed:

1. **A keyword on `run_job`** (Q1), `run_job(..., cosmors=CosmoRSOptions(...))`,
   not a separate driver.
2. **Refuse a solute and solvent at different protocols** (Q3); record, and
   print, a pair that agrees with itself but not with the parameterization.
3. **`n_ring_atoms` from the caller at M0** (Q4), default `0`, stated in the
   output; ring perception is separate work.

## 6. M2: the Direct COSMO-RS density and COSMOSPACE

Klamt & Diedenhofen 2015 (doi:10.1021/jp511158y) gives the free energy on the
Direct COSMO-RS density as eq. 9,

    dG_solv = E_DCRS - (E_gas + omega n_ring + eta_gas) - RT ln(rho_S V_G / M_S)

with `E_DCRS` its eq. 4: the conductor COSMO energy, plus `f_pol sum_i s_i
mu_S(sigma_i)`, plus the combinatorial term, at `f_pol = 1`. vibe-qc's Direct
COSMO-RS energy is eq. 4 without the combinatorial term, which does not depend
on the density. So eq. 9 is the conductor chain of section 2 with one
substitution: `E_COSMO + sum a mu` becomes the self-consistent energy.
`ideal_screening_energy` is then `E_gas` minus the conductor functional at the
Direct COSMO-RS density, and the gas reference of eq. 21 is reused unchanged.

The signs were checked before implementing (#435's rule). The ring term raises
the gas side by about 0.21 kcal/mol per ring atom in all three papers that
state it: Klamt 1998 eq. 21 (`- omega n` with omega = -0.21 kcal/mol), Klamt
2000 eq. 19 (`+ omega n` with omega = +0.89 kJ/mol) and Klamt 2015 eq. 9.
Cyclohexane being less volatile than n-hexane points the same way. The last
term of eq. 9 converts to 1 mol/L in both phases: water's -2.05 kcal/mol from
its vapour pressure, minus `RT ln(55.3 * 24.79)`, gives -6.33 against the
experimental -6.3.

**Decisions, taken by the maintainer on 2026-10-21, each as proposed:**

* **D1.** `eta` stays the `klamt1998` value, not refitted for Direct
  COSMO-RS. Klamt refitted his own by -0.2 kcal/mol (-6.85 against -6.65 for
  the infinite-epsilon model) and reports a mean unsigned error of 0.67
  kcal/mol unrefitted against 0.66 refitted, over 2346 solvation free
  energies. A refit is a fitting exercise and needs data the library lacks.
  The output says the constant is not refitted.
* **D2.** COSMOSPACE becomes a job option (M2b) reporting an
  infinite-dilution activity coefficient, on the conductor density only, and
  never feeding `dG_solv`. Superseded in part by D2a and D2b (section 7),
  after the premise that it differs from the sigma potential turned out
  wrong.
* **D3.** One sigma potential per solvent, with dispersion in the
  per-element gas reference (Sinnecker 2006, Klamt 1998), not the 2015
  paper's element-specific potentials, which need a parameter set vibe-qc
  does not have.
* **D4.** The 1 mol/L conversion, eq. 9's last term, waits for the FreeSolv
  work and will cover both densities. It needs cited solvent densities.

**Recorded, not decided:** the 2015 paper applies the outlying-charge
correction and uses separate gas and conductor optima; vibe-qc does neither,
on either density, as at M0. The ε-dependent model of eq. 5 and 6 stays
refused, since it needs COSMOtherm's `c0` and its own `eta`.

**The two densities differ by more than relaxation.** The sigma potential is
evaluated on raw segment sigma, as Sinnecker 2006 eq. 17 and Klamt 2015 eq. 4
write it; the conductor chain integrates the averaged profile of Klamt 1998
eq. 11. The shift splits exactly into the relaxation `G[D_DCRS] - G[D_COSMO]`
of the one functional and that averaging difference, and the test pins the
split. Water in its own RHF/STO-3G potential: -0.04 and -0.23 kcal/mol.

**Validated** by the null-solvent limit (every field equal to the conductor
chain's), eq. 9 composed by hand on the job's own run, the exact split above
with a relaxation that is not positive, and the Klamt 1995 Table 2
vapour-pressure panel on the direct density.

**Corrections of 2026-10-22.** The first panel numbers (rms 1.09, quoted in
the v0.17.25 release notes) put each solute on a 0.30 angstrom cavity against
a potential built at 0.35. Two things followed from fixing that:

* The direct density converges with the cavity grid only near 0.25
  angstrom, because raw segment sigma depends on the segment size: in a
  water potential at RHF/STO-3G, benzene's direct `dG_solv` is +0.69, +0.93,
  +1.19, +1.19, +1.18 kcal/mol at 0.40, 0.35, 0.30, 0.25, 0.20 angstrom,
  while the conductor form moves by 0.01. The job records the spacing and
  notes a direct run coarser than 0.25 (`DIRECT_CONVERGED_SPACING_ANG`).
* At 0.25 angstrom and RHF/STO-3G the panel gives rms 1.12 (direct) against
  1.23 (conductor). At the recommended BPW91/def2-TZVP it gives **rms 2.93
  (direct) against 1.76 (conductor)**, means -2.10 and -1.43: water,
  methanol and ethanol are overpolarized by 1.5 to 3.2 kcal/mol on the direct
  density, about half of it the density's own relaxation. Klamt 2015 corrects
  this direction with a refitted `eta_gas` and element-specific potentials;
  D1 and D3 keep neither, so the direct density stays a tool for the
  solute's polarization, not for quantitative free energies.

## 7. M2b: activity coefficients, and what COSMOSPACE is

**COSMOSPACE is the sigma-potential equation.** Klamt, Krooshof & Taylor
2002 (doi:10.1002/aic.690481023) say in the abstract that the solution was
"initially developed for" COSMO-RS in implicit form, and Appendix A derives
the COSMO-RS self-consistency equation from the same pair statistics. With
`G_nu = gamma_nu exp(a_eff u_nu,nu / 2kT)`, eq. 16 is the fixed point
`sigma_potential_segments` solves, Theta being the area fraction, and the
self-energies cancel in eq. 13's difference. Measured on three real pairs,
the COSMOSPACE residual equals the COSMO-RS segment form to 5e-13. D2's
premise, an "unfitted" side output differing from COSMO-RS, was therefore
wrong, as was the module docstring that said so.

**D2a (2026-10-22): the combinatorial term is `klamt1998`'s own**,
`-lambda kT ln A`, not Staverman-Guggenheim: the 2002 paper leaves the r, q
normalization for COSMO surfaces to a "manuscript in preparation", so SG has
no published construction on these cavities.

**D2b (2026-10-22): `CosmoRSOptions(activity_coefficient=True)`** reports
`ln gamma_inf = (mu_S^X - mu_X^X) / RT` from the same chain as `dG_solv`,
the solute's own potential built exactly as a solvent's, conductor density
only. COSMOSPACE stays a library solver and is the test oracle: carrying the
solute's segments both as members of the pure-solute ensemble and as
sigma_perp = 0 probes, it reproduces the job's profile integral to the
binning error (0.1 to 0.5 percent).

**Also measured:** evaluating the solute on its segments with its own
sigma_perp, as the fitted misfit term would allow, makes the RHF/STO-3G
vapour panel worse (rms 1.30 against 1.23), and binning alone moves it by
0.004. The profile chain stays as it is.

## 8. What production would need

COSMO, the conductor-like reaction field with analytic gradients, is
production. COSMO-RS thermodynamics and Direct COSMO-RS are experimental and
say so at runtime (`CosmoRSExperimentalWarning`, `method_status =
"experimental"` in the `.system`). The engineering is complete; what keeps
them experimental is accuracy at the recommended protocol (rms 1.76 kcal/mol
and a mean of -1.43 on the conductor density), which comes from transferring
constants fitted on DMol/BPW91/DNP. Promotion needs, in order: an
experimental training set the project can redistribute or reference (FreeSolv
is not in the literature library; its supplementary PDF only describes the
tarball); a maintainer decision to fit a parameter set to vibe-qc's protocol,
at minimum `eta` and the dispersion constants, validated on data the fit did
not see; and Validator acceptance through `vq` at that protocol.
