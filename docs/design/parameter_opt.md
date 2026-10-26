# Parameter-opt: generalized semiempirical parameter optimization

Status: research design and implementation roadmap, 2026-10-26. No fitter,
new parameter set or scientific accuracy claim is delivered by this document.
The proposed package name is `vibeqc.parameter_opt`; the user-facing name is
**parameter-opt**. All interfaces below are proposed, not executable APIs.

## Decision and scope

Build one fitting framework with adapters for each Hamiltonian, reference
protocol and observable. Start with molecular DFTB repulsion and a small
PM6 parameter block. Extend to SCC-DFTB, GFN2 and MSINDO after their adapters
pass isolation and derivative checks. The abstraction can accommodate other
semiempirical methods; it cannot make an unsupported Hamiltonian, observable,
element or boundary condition valid. OMx is a maintainer retirement target,
so it is an adapter-contract example, not a planned implementation milestone.
SECCM remains experimental and comes after molecular acceptance.

Use a constrained nonlinear optimizer for Hamiltonian parameters, an exact
conditional linear/quadratic solve for eligible additive terms, and a separate
budgeted selector for new reference calculations. MINUIT2 is an optional local
optimizer and profiling tool. MILP selects reference jobs or discrete model
options; it does not solve a general nonlinear SCF parameter fit.

The proposed research method is **identifiability-guided multifidelity variable
projection** (IMVP). Its testable idea is to spend expensive reference work on
uncertain parameter directions while distinguishing parameter uncertainty from
irreducible model error. Variable projection, active learning, regularization
and multifidelity learning are established ideas. Their integration here is a
research proposal, not a verified first-in-literature claim.

## Evidence and consequences for the design

Relevant local papers and SI were inspected alongside the primary online
sources below. This is a focused design review, not an exhaustive novelty search.

| Source and section | Finding used here | Design consequence or departure |
|---|---|---|
| Stewart, PM6, 2007, *Theory*, *Reference data*, *Use of Ab-Initio results*, [DOI](https://doi.org/10.1007/s00894-007-0233-4) | Reference error, approximation error and incomplete optimization are distinct; calculated data supplement experiments | Diagnose all three separately. Do not inherit historical reference levels as today's accuracy targets |
| Bannwarth et al., GFN2-xTB, 2019, parameter-fitting discussion following Eq. 45; SI sections 1.5.1-1.5.2, Eqs. 53-55, [DOI](https://doi.org/10.1021/acs.jctc.8b01176) | Mixed CCSD(T) energy and DFT structure/force targets; density-dependent dispersion contributes to the self-consistent model | Use property-specific reference protocols. A dispersion coefficient is not conditionally linear if changing it changes the density |
| Froitzheim et al., g-xTB, local 2025 preprint; SI section 1.1.2, Eqs. 4-5 and discussion pointing to section 1.5, [preprint](https://doi.org/10.26434/chemrxiv-2025-bjxvt) | Energy reference-state changes and atomic increments matter for fitting total energies | Make energy gauges explicit. This design defaults to balanced energy differences, not raw all-electron energies versus valence-only energies |
| Wang, Martinez and Pande, ForceBalance, 2014, *Computational Methods*, objective and regularization discussion, [paper](https://doi.org/10.1021/jz500737m) | Scaled parameters, weighted targets, regularization and trust-region optimization | Reuse these principles for electronic-structure adapters; do not claim them as new |
| Ammothum Kandy et al., CCS, 2021, abstract and published curvature-switch description, [paper](https://doi.org/10.1021/acs.jctc.0c01156) | Constrained spline repulsion is a convex quadratic fitting problem for fixed electronic terms; silicon illustrates model-form limitations | Prefer QP for that subproblem. Full equations and SI must be obtained before implementing its exact spline convention |
| Golub and Pereyra, variable projection, 2003, [review](https://doi.org/10.1088/0266-5611/19/2/201) | Separable nonlinear least squares eliminates linear coefficients | Use the separable formulation below; exact constrained derivative details remain an implementation prerequisite |
| Ramakrishnan et al., delta learning, 2015, section II, Eq. 1 and section III.1, [paper](https://arxiv.org/html/1503.04987) | Learn the difference between a baseline and a target calculation | Use paired calculations to estimate reference-level discrepancies; distinguish imputed targets from measured ones |
| Podryabinkin and Shapeev, 2017, sections 2.2 and 3, [paper](https://arxiv.org/html/1611.09346) | Parameter-sensitive D-optimal selection can guide acquisition | Extend to local nonlinear sensitivities and explicit reference cost. The linear-model results do not guarantee SCF reliability |
| Guo et al., 2018, triples equations and benchmark discussion, pp. 011101-2 to 011101-4, [DOI](https://doi.org/10.1063/1.5011798) | Iterative triples differ materially from the semicanonical T0 approximation | Record triples variant, PNO thresholds and basis convergence; DLPNO-CCSD(T) is not an uncertainty-free label |
| James and Winkler, Minuit2 manual, parameter limits, error definition and error reliability sections, [manual](https://root.cern.ch/root/htmldoc/guides/minuit2/Minuit2.html) | MIGRAD, HESSE and MINOS address minimization, local curvature and profiles | Use optionally for smooth, moderate-size problems. Curvature errors are not generalization guarantees |

Reading limits: the CCS full text was inaccessible through the web reader;
the variable-projection review was bibliographically verified but not read in
full. Before their implementation milestones, acquire full papers and SI through
the configured literature library's documented `downloaded/` intake and catalog
workflow. No companion-library files are changed by this design task. Do not
claim exact reproduction from abstracts. Dataset redistribution and parameter
licenses require their own checks before packaging data.

## Reference data as a protocol graph

A fidelity is a complete protocol, not the label "DFT" or "high accuracy".
Store geometry and cell hashes, elements, charge, multiplicity, state tracking,
boundary conditions, basis/ECP, frozen core, relativistic treatment, functional,
exact-exchange/range-separation choices, dispersion, integration grid, SCF and
correlation tolerances, PNO/triples settings, k mesh, units and code/version.
Store convergence diagnostics, source/native runtime identity, provenance,
license, uncertainties and the content hash of each raw result. A changed
protocol creates a new dataset version.

Proposed reference roles:

| Role | Initial use | Qualification |
|---|---|---|
| Economical DFT or composite DFT | Broad geometries, distortions, forces and vibrational data | Establish representative grid, basis and SCF convergence |
| Hybrid or range-separated hybrid DFT | Polarization, barriers, charge transfer and bridging calculations | Choose per chemical domain using anchor comparisons; a hybrid is not automatically superior |
| DLPNO-CCSD(T), preferably an explicitly identified iterative-triples variant where appropriate | Sparse relative-energy anchors | Tighten PNO thresholds, examine basis convergence/CBS, compare canonical CCSD(T) on affordable cases |
| Multireference or justified experimental/literature data | Cases outside single-reference applicability | Separate protocol and uncertainty; do not force stretched bonds or problematic transition-metal states into a CCSD(T) label |
| Periodic DFT/hybrid calculations | Equations of state, stresses, phase differences | Converge k sampling, basis and electrostatics; molecular cluster CC data are not automatically periodic bulk labels |

The fidelity graph is property-specific, not a universal accuracy ranking. Pair
low/high calculations on the same geometry and electronic state. Record a
DLPNO energy on a DFT geometry as that composite protocol. Do not infer
high-level forces from high-level energies or combine them as a single
conservative high-level surface without a differentiable, validated correction.

Prefer reaction, conformer, interaction and atomization energies with explicit
stoichiometry and atomic states. Apply BSSE conventions consistently for
interactions. Heat of formation, electronic energy, enthalpy and free energy
are distinct observables with explicit thermal/ZPE conventions. Atomic charges
are partition-scheme-dependent auxiliary data; dipoles and response properties
are better cross-method constraints when supported. Orbital gaps are not
fundamental or optical gaps.

Split data before fitting by molecule/scaffold, reaction family and material
prototype. Keep all conformers, distortions and fidelities of a parent in one
split. Fit discrepancy models and preprocessing only on training folds. Reserve
validation for model selection and a locked test set for final reporting.
Active acquisition draws from a separate training pool, never the test set.

## Mathematical contract

Let `x` describe a fixed geometry/cell/state, `z` a discrete model choice,
`theta` nonlinear electronic parameters and `a` eligible linear coefficients.
Only when the adapter proves separability use

```math
E(x;\theta,a,z)=E_{\mathrm{el}}(x;\theta,z)+B(x;\theta,z)a.
```

Rows for energy differences are stoichiometric combinations of rows of `B`.
Force rows are `-dB/dR`; stress rows use a documented strain/volume convention.
Examples of eligible coefficients are additive repulsive spline coefficients
with fixed knots/cutoffs and fixed electronic parameters. Hubbard parameters,
orbital exponents and self-consistent dispersion generally belong in `theta`.
Relaxed geometries and frequencies are not linear observables even if energy
is linear in `a`; fit fixed-geometry data in the first implementation.

Map physical parameters to dimensionless coordinates with recorded scales,
positive transforms where needed, tied parameters and explicit bounds. Use
method-specific stability constraints rather than universal signs for all
parameter types. Gauge-fix atomic offsets or remove them with balanced energy
differences; do not let them hide reaction errors.

For target block `g`, whiten residuals with `W_g`, constructed from reference
covariance plus a fixed practical tolerance floor. Use declared class weights
`w_g` and block size `n_g` so a large number of force components cannot silently
override reaction energies:

```math
L(\theta,a)=\sum_g\frac{w_g}{n_g}\sum_i
 \rho\!\left([W_g(f_g(\theta,a)-\widetilde y_g)]_i\right)
 +\lambda_\theta\|D(\theta-\theta_0)\|^2
 +\lambda_a\|La\|^2.
```

Start with squared loss; offer Huber loss with documented scale. Use covariance
blocks for related energy differences, or an independent contrast basis.
Do not double-count a raw energy and all its derived differences as independent
measurements. Freeze tolerances during each fit; a fitted uncertainty must not
be allowed to inflate freely to erase errors. Select regularization and class
weights on validation and report per-class metrics, not only the scalar loss.

For fixed `theta,z`, solve

```math
a^*(\theta,z)=\arg\min_{Ca\leq d,\;Ha=h}L(\theta,a),\qquad
\Phi(\theta,z)=L(\theta,a^*(\theta,z)).
```

With squared residuals and linear constraints this is a convex QP. Use QR/SVD
for unconstrained least squares; avoid explicit normal-equation inversion.
Robust loss requires a compatible convex solver or converged IRLS, not a claim
of a single exact QP solve. Enforce cutoff value/derivative continuity and only
physically justified shape constraints. Moving knots and geometry relaxation
invalidate the simple fixed-design-matrix shortcut.

Optimize the reduced objective with bounded trust-region least squares or a
constrained optimizer appropriate to the remaining constraints. Differentiate
the reduced residual, including `a*` response, through the solve/KKT conditions
where smooth; finite-difference the fully re-solved objective initially.
Active-set changes can be nonsmooth. Envelope derivatives of the scalar
objective require the relevant regularity assumptions; they do not justify
ignoring coefficient response in its residual Jacobian.

## IMVP: selecting calculations that can change the fit

1. Fit a conservative initial parameter subset and conditionally solve `a`.
   Use multiple starts only within an explicit evaluation budget.
2. Compute the scaled, whitened reduced residual Jacobian `J` and its SVD.
   Report singular values, parameter correlations and data-only rank separately
   from regularization. Freeze or tie unsupported directions rather than
   announcing precise values dominated by priors.
3. Diagnose whether systematic residuals align with the local column space of
   `J`. The component outside that space cannot be removed by an infinitesimal
   parameter update. Verify this diagnosis with restarts, wider perturbations
   and independent reference checks; it is not proof of global impossibility.
4. For supported paired fidelities learn `delta_HL(x)=y_H(x)-y_L(x)` with grouped
   cross-validation. An imputed high-level label is `y_L+delta_hat`, carrying
   propagated/calibrated uncertainty and a measured/imputed flag. Actual high
   labels replace imputations, rather than adding duplicate evidence. A mixed
   property objective without imputation remains the simpler baseline.
5. For each candidate job, estimate sensitivity to weak parameter directions,
   reference discrepancy uncertainty, chemical coverage and evaluation cost.
   With a regularized information matrix `F`, a possible local score is
   `logdet(F + J_j.T W_j J_j) - logdet(F)`, divided by expected cost for greedy
   selection. Here `J_j` is an unwhitened candidate Jacobian and `W_j` its
   inverse noise covariance. Use the appropriate expected noise for the requested fidelity;
   use surrogate sensitivities only as estimates. Update scores after batches.
6. Select a diverse affordable batch, run reference jobs through `vq`, ingest
   only qualified results and refit. Compare predicted with realized gains.
   Persist the dataset/objective version at every iteration.
7. Stop on budget, small validation improvement across predeclared batches, or
   stable parameter rank. Persistent structured residuals trigger a model-form
   experiment or a narrower declared domain, not endless parameter tuning.

The proposed advance is the joint use of reference discrepancy and reduced
parameter identifiability to decide which calculation to buy next. Neither
large residuals alone nor parameter sensitivity alone can establish whether a
new DLPNO label will help. Ensemble/bootstrap disagreement is a practical
acquisition signal, not a guaranteed uncertainty bound outside training support.

## Optimizers and the precise role of MILP

| Problem | Initial solver strategy | Limitation |
|---|---|---|
| Smooth residual fit | Scaled trust-region least squares | Local solution; requires reliable residuals/derivatives |
| Linear additive block | QR/SVD or constrained QP | Only truly separable terms |
| Moderate nonlinear parameter block | Optional MINUIT2 MIGRAD, then HESSE and selected MINOS profiles | Costly profiles; limits, priors and model misspecification weaken statistical interpretations |
| Rough/non-smooth fit | Bounded derivative-free trust region; limited multistart/global exploration | Evaluation cost; no universal global guarantee |
| Reference batch selection | Greedy information gain, then optional MILP with frozen scores | Linear score surrogate, not exact joint log-determinant optimization |
| Discrete functional-form selection | Small outer enumeration; later mixed-integer surrogate search | Nonlinear full model still needs reevaluation and held-out selection |

For frozen job utilities `u_j`, costs `c_j`, binary selections `b_j` and coverage
matrix `A`, the optional acquisition MILP is

```math
\max_b\sum_j u_j b_j,\quad
\sum_j c_jb_j\leq C,\quad Ab\geq q,\quad b_j\in\{0,1\}.
```

Add at-most-one constraints for duplicate alternatives and dependencies for
missing paired baseline calculations. Set finite time/gap limits, record the
incumbent and bound, and detect infeasible coverage before job submission.
Precluster near-duplicates or update scores between batches because additive
utilities overcount correlated jobs. [SciPy's MILP contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html)
is linear; joint information gain is nonlinear. Quadratic residuals with binary
model selection are MIQP only if the remaining model is quadratic; full SCF
fitting is generally MINLP. No MILP optimality certificate applies to it.

For MINUIT2 explicitly record objective normalization and `errordef`. A
penalized robust loss with arbitrary class weights is not automatically a
chi-squared likelihood. Report grouped bootstrap and held-out errors alongside
curvature/profile diagnostics, and do not label HESSE errors predictive error
bars. Optional numerical backends must use lazy imports and licensing review.

## Software architecture and failure behavior

The existing `SemiempiricalModel` defines energy and nuclear-gradient behavior;
it is not a parameter derivative interface. Existing parameter access spans
native DFTB objects, GFN2 provenance-aware loading, NDDO parameter sets and
MSINDO parameter access. Reuse these without mutating shared default tables.

Proposed modules under `python/vibeqc/parameter_opt/`:

| Module | Contract |
|---|---|
| `schema.py` | `ParameterSpec`, immutable `ParameterSet`, ties, units, bounds, priors, linear/nonlinear tags and license/provenance |
| `adapters/` | `capabilities()`, `make_model(parameters, system)`, `evaluate(request)`, optional parameter Jacobian and linear-design blocks |
| `datasets.py` | Immutable observations, protocol graph, grouped splits, covariance and derived-observable dependencies |
| `objective.py` | Reproducible residual blocks, constraints, class metrics and failure accounting |
| `solvers/` | Conditional solves, trust-region and optional Minuit2/MILP backends |
| `acquisition.py` | Budget accounting, coverage, discrepancy models and rank diagnostics |
| `execution.py` | Existing `vq` submission/collection integration, content-addressed evaluation cache and restart state |
| `validation.py` | Frozen baseline comparisons, domain tests and parameter uncertainty summaries |

A reference provider emits job specifications and imports normalized results.
It must not import ORCA, CRYSTAL or another QC engine into `python/vibeqc/` or
`cpp/`. Compute the fitted model in vibe-qc; external reference calculations are
separate jobs. User-facing manifests, parameter artifacts and reports go through
`vibeqc.output`; operational logs and site configuration stay outside checkouts.

Cache keys include source/native identity, parameter hash, full system/state,
protocol, requested observables and convergence settings. Evaluate in isolated
workers; no shared mutable parameter defaults. Resume only with matching
objective/dataset hashes. Record attempts and costs even for failed jobs.

SCF failure, invalid overlap, wrong state, unsupported properties and
non-finite results are structured failures. Never drop failed residual rows or
substitute zero. Reject infeasible trials or apply an explicitly defined
feasibility procedure while retaining a fixed target list. Separate data errors
from candidate failures. A fixed, bounded retry policy may help convergence;
changing the physical method to force convergence creates a new protocol.

Analytic parameter derivatives can follow later through SCF implicit response:
if `G(P,theta)=0`, solve `G_P dP/dtheta = -G_theta` or its adjoint. This requires
converged differentiable branches and a well-conditioned response. State
crossings, occupation changes and SCF bifurcations need explicit handling.
Nuclear forces require mixed derivatives when used in parameter fitting.
Check all parameter derivatives against step-size-controlled finite differences;
finite differences cost roughly `2p` model evaluations per geometry for `p`
parameters, with further cost for numerical nuclear derivatives.

## Roadmap with measurable exit criteria

Dependencies are explicit; these are work packages, not release-date promises.
Development landing and later scientific acceptance retain their existing roles.

| Stage | Deliverable and dependency | Exit criterion |
|---|---|---|
| P0: contracts and dataset rules | Schema, capabilities, units, energy gauges, tiny synthetic dataset; no native changes | Reject incompatible protocols; deterministic grouped splits; round-trip hashes; two candidate sets cannot contaminate each other |
| P1: first end-to-end fit | DFTB fixed-electronic repulsion adapter, squared loss, conditional solve, restart/output; depends P0 | Recover an identifiable synthetic parameter set; agree with joint least squares; exercise failure/restart paths; complete one real fit/evaluation workflow through `vq` |
| P2: nonlinear and second family | Small PM6 block, bounded trust region, parameter finite differences; depends P1 | Derivative checks across step sizes; fixed failed-case accounting; independent ORCA/reference comparison with energy conventions matched; preserve experimental labels |
| P3: reference protocols | DFT/hybrid/DLPNO normalized records, paired labels, grouped evaluation; depends P0 and P2 for fits | Demonstrate provenance and unit rejection, basis/threshold convergence checks and same-geometry differences; publish baseline versus fitted per-class metrics |
| P4: optional solvers | MINUIT2 adapter and constrained spline QP; depends P2 and completed full-text/SI review | Compare minima/evaluation counts on identical budgets; demonstrate active constraints, rank deficiency and failure diagnostics; verify export/reload predictions |
| P5: IMVP acquisition | Paired discrepancy model, reduced-Jacobian rank, greedy selector; depends P3-P4 | At equal reference cost compare with random, diversity-only and residual-only selection over at least five seeds; report all seeds and clustered uncertainty |
| P6: budgeted MILP | Frozen-score selection with costs/coverage/dependencies; depends P5 | Match exhaustive selection on tiny cases; handle infeasible budgets; account for actual job costs and show whether it improves on greedy |
| P7: broader adapters and periodic work | SCC electronic, GFN2 and MSINDO first; periodic/SECCM separately; depends molecular acceptance | Each adapter passes parameter isolation, state/derivative checks and external scientific comparison; periodic stress/cell/k conventions verified; no blanket promotion |
| P8: publication and packaging | Versioned parameter artifacts, domain/model cards, citations and license decisions; depends each applicable acceptance | Locked test report, independent reload reproduction and domain limits; no accuracy claim without measured evidence |

Prioritize P0-P3 before speculative global search or model-form expansion.
A usable initial framework does not depend on MILP, MINUIT2, analytic parameter
gradients or a learned correction. Keep optional extras separate from core use.

## Pilot and falsification plan

Start with a proposed CHNO molecular pilot, limited to supported closed-shell
routes: approximately 100 parent systems and 1,000 distorted/conformer/interaction
geometries, then a 100-geometry paired high-level anchor subset. These are
planning quantities, not acquired data or an accuracy promise. Assign grouped
70/15/15 training/validation/test splits before calculations; use sufficiently
independent families and expand if a split lacks necessary coverage. Pilot
actual runtimes on ten representative jobs before choosing a total core-hour
budget. Include charged cases only after route and reference qualification.

Use fixed-electronic DFTB repulsion to test separability, then a small PM6 block
to test generic nonlinear adapters. Include equilibrium, distorted, short-range
repulsive and separated-fragment cases. Add periodic material prototypes only
in P7. Published benchmark data may inform validation only after checking prior
parameter-training overlap; use fresh chemical families for transfer claims.

Predeclare two separate goals: fitting accuracy and reliable geometry/force
behavior. As initial *research targets*, use held-out relative-energy MAE below
2 kcal/mol, force RMSE below 0.05 eV/angstrom and bond-length MAE below 0.02
angstrom within the declared pilot domain. Also require no increased SCF failure
rate or unexplained per-class degradation relative to the original parameters.
Report tails, worst cases and confidence intervals; revise targets openly before
final test access if feasibility study shows the model form cannot meet them.
No universal chemical-accuracy target is implied.

Compare original parameters, ordinary regularized joint fitting, variable
projection alone, multifidelity without active selection and full IMVP. Use the
same starting sets, splits, reference budget and model-evaluation budget. Record
wall/core time, high-level labels purchased, failure rate, per-property errors,
rank/condition diagnostics and prediction-interval coverage. Bootstrap parent
groups, not individual correlated force components.

The IMVP hypothesis fails if its advantage disappears at equal total cost or
across seeds, if discrepancy imputation worsens high-level held-out predictions,
or if projected sensitivities fail to predict realized information gain. Retain
the simpler fitting framework in that case. If all optimizers reach the same
structured error floor, test a narrowly scoped physical correction or constrain
the advertised domain. A learned residual energy is a separately named optional
model with conservative derivatives and its own acceptance, not silently part
of a reparameterized legacy method.
