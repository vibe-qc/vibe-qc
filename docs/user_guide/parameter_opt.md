# Parameter-opt: first molecular energy fits

Parameter-opt is an experimental Python framework for fitting selected
semiempirical parameters to qualified electronic-energy differences. Issue
#676 implements the first P0-P3 source slice of the
[design roadmap](../design/parameter_opt.md). It does not promote DFTB or PM6
readiness and does not provide a generally validated parameter set.

## Supported scope

- Immutable parameter records with finite bounds, units, scaling, Gaussian
  priors, direct ties, canonical hashes and JSON round trips.
- Molecular geometries in bohr, explicit charge, multiplicity and state.
- Qualified DFT, hybrid and DLPNO electronic-energy records in hartree, with
  complete protocol metadata, raw-result digests and source/runtime identity.
- Atom- and charge-balanced energy contrasts. Shared reference uncertainties
  generate covariance between contrasts; independent practical tolerance floors
  keep the covariance nonsingular. Distinct raw records are assumed independent.
- Deterministic parent-group splits. Connected reaction parents, conformers and
  paired fidelities stay together; identical geometries cannot use different
  parent labels. Small splits may be empty and must be checked by the caller.
- DFTB fixed-decay Born-Mayer amplitudes, solved with bounded linear least
  squares; PM6 `uss`, `betas` and `gss` blocks for H, C, N, O and F, solved with
  scaled bounded nonlinear least squares. Both adapters construct fresh native
  parameter sets for every candidate. Nuclear forces are not fitting targets yet.
- Conditional linear elimination for adapters that provide exact `linear_terms`,
  including mixed nonlinear/linear problems. Priors remain centered on the
  original specification across restarts.

The DFTB adapter uses the existing in-house Hamiltonian. Specified pairs replace
the default repulsion with `A exp(-B R)`; other pairs retain their built-in
potentials. `A` is fitted in hartree and `B` is a fixed positive inverse-bohr
value. This is neither spline-fitted production DFTB nor a Slater-Koster table
optimizer. PM6 parameters are in eV even though computed energies are in
hartree. PM6 energies are not MOPAC heats of formation.

## Basic API

```python
from vibeqc.parameter_opt import (
    Dataset, DFTBRepulsion, FitProblem, ParameterSet,
)

# dataset_document is a versioned Dataset.to_dict() electronic-contrast document.
data = Dataset.from_dict(dataset_document)
split = data.split(seed=17)
if not split["train"] or not split["test"]:
    raise ValueError("expand the parent-family dataset before transfer testing")
train = Dataset(split["train"])
adapter = DFTBRepulsion(((1, 1, 1.5),))
parameters = ParameterSet.initial(adapter.specs(initial=1.0, upper=100.0))
problem = FitProblem(adapter, parameters, train)
result = problem.fit(checkpoint="fit.json", max_nfev=100)
fitted = parameters.replace(result["values"])

# Explicitly evaluate the untouched test data; fit() only reports its own data.
test = FitProblem(adapter, parameters, Dataset(split["test"]))
print(test.metrics(fitted))
```

For PM6, use `PM6Block(((1, "betas"),))` and its `specs(width=2.0)` to choose
an interval around the existing parameter. There is no hidden parameter download
or external quantum-chemistry import in these adapters. SciPy, already a core
numerical dependency, supplies both solvers. MINUIT2 and MILP remain later work.

`ModelAdapter` describes the minimal custom-adapter interface. `identity()` must
include every execution setting and implementation/parameter identity that can
change results. `energy()` returns one finite energy in hartree for a `System`.
An adapter with linear parameters must additionally return `(offset, columns)`
from `linear_terms(parameters, system)`, in the order of all linear specs,
such that `energy = offset + columns @ linear_values` exactly at fixed nonlinear
parameters. Density-dependent or relaxed-geometry terms generally do not qualify.

## References and qualification

`ReferenceProtocol` takes a family, method, basis, code, version and
`settings_json`. Every non-synthetic computed protocol requires `scf`,
`relativity`, `ecp` and `state_protocol`. DFT/hybrid also require `functional`,
`grid` and `dispersion`; hybrid requires `exchange`; DLPNO requires `pno`,
`triples`, `frozen_core` and `aux_basis`. Explicit `"none"` values are appropriate
when an approximation is absent. Presets must be tied to the code version;
record numerical overrides in the same settings object.

`ReferenceRecord` binds the protocol to a system, energy, nonnegative uncertainty,
raw-result SHA256 and source identity. Nonconverged records are refused.
An uncertainty of zero means no estimated stochastic reference error; it does
not establish an exact physical reference. Use declared practical tolerances and
separate basis/threshold studies to describe remaining systematic uncertainty.

`read_references(path)` accepts a normalized JSON document with schema
`vibeqc.parameter-opt.references.v1`, `coordinate_unit: "bohr"`,
`energy_unit: "hartree"` and a `records` list of dataclass-shaped records.
It refuses implicit conversions and missing protocol settings. Raw ORCA or
CRYSTAL parsing is the reference-job producer's responsibility. The producer
must preserve the actual geometry/state, method settings and convergence result;
a final energy line alone is insufficient.

`reference_request(system, protocol)` returns a hashed portable calculation spec.
It is not itself a runnable `vq` input. Submit the corresponding existing
calculation script with `vq`, collect its output, then normalize qualified
records. This first slice has no autonomous reference submission or campaign
scheduler. `paired_delta(low, high)` refuses different geometries, charge,
state or parent labels. Its uncertainty assumes independent reference errors.
It does not manufacture high-level forces or imputed reference labels.

`convergence_check(coarse, fine, varied=["basis"], tolerance=...)` reports the
absolute energy sensitivity at one identical physical system. Only declared
protocol settings may differ; use `"aux_basis"` too when changing that basis.
The result is `within_tolerance`, not a universal scientific PASS. Compare
relative-energy sensitivities and representative geometries before claiming
basis convergence for a fitting domain.

## Objective, failures and restart

Each contrast has a tolerance floor, positive weight and category. The squared
residual objective divides category weights by category counts so dense groups
do not silently dominate. The covariance consists of independent raw-reference
errors propagated through contrast coefficients plus diagonal tolerance floors.
With importance matrix `D` and covariance `C`, the data loss is
`(D residual).T @ inverse(C) @ (D residual)`; the implementation uses a Cholesky
solve. Priors contribute squared, dimensionless deviations from initial values.
Duplicate contrasts, including sign/scalar copies, are refused.

A failed SCF, invalid native calculation or non-finite energy raises
`EvaluationFailure`. The initial version deliberately stops the fit and writes
a failure checkpoint, preserving the fixed residual list and best valid
candidate if available. It does not hide failures by dropping data, return a
false convergence status or automatically change SCF settings. Correct the
cause or narrow bounds before restarting with a new objective identity.

Checkpoints and reports are atomic JSON written through `vibeqc.output`.
They include dataset/objective, native module, adapter source and baseline
parameter hashes; fitted parameters and specs; loss, status, citations,
evaluation counts, data-only rank diagnostics and baseline/fitted per-category
metrics. Rank is a local numerical diagnostic, not a predictive uncertainty estimate. Evaluation counts
measure adapter calls, including failed calls, not CPU time. The in-memory cache
is specific to the problem and candidate/system identity. A restart requires
identical hashes and warm-starts the best candidate; it does not restore SciPy
optimizer history or the cache. `max_nfev` is the nonlinear solver function-evaluation
limit for this invocation (excluding extra finite-difference Jacobian calls), not a total reference-job or native-evaluation budget.
Inspect `status`: reaching the iteration budget is not convergence.

## Queued first-fit example

[The H2 example](../../examples/parameter_opt/fit_h2.py) consumes normalized
reference curves and fits either a single DFTB amplitude or PM6 hydrogen
resonance parameter. Put it and `references.json` in an external run directory:

```sh
vq submit localhost --dir fit-input --cpus 1 -- /path/to/python fit_h2.py \
  --references references.json --adapter dftb --output dftb-fit.json
```

Select the interpreter containing the intended source/native build. Repeat with
`--adapter pm6 --output pm6-fit.json`. All supplied data are training data in
this example; different H2 distances are not independent molecular test groups.
Use the general API and disjoint chemical parents for transferability claims.

## Remaining roadmap work

P0-P3 now have a functional energy-fitting implementation, not full scientific
acceptance across the planned CHNO domain. Broader external comparisons, locked
held-out chemical families and reference convergence remain issue #676 work.
Force/stress targets, nonlinear SCF response, spline/QP constraints, uncertainty
calibration, active acquisition, persistent distributed caches, automatic `vq`
reference production, MINUIT2 and MILP are outside this initial slice. Do not
interpret optimizer convergence or synthetic recovery as scientific acceptance.
